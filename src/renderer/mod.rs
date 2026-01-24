mod init;
mod texture;
mod utils;
mod blackbody;
use std::{collections::VecDeque, path::Path, sync::Arc};

use anyhow::{Context, Result, anyhow};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice,
    PushKernelArg,
};
use init::{
    RendererBuffers, allocate_buffers, build_cuda_kernels, build_textures, compute_launch_dims,
    validate_bloom_settings,
};
use texture::CudaTextureLut;
use utils::build_kerr_params;

use crate::Config;
pub struct FrameView<'a> {
    pub index: usize,
    pub data: &'a [u32],
}
struct FrameBuffer {
    hdr_buffer: CudaSlice<f32>,
    bloom_buffer: CudaSlice<f32>,
    image_gpu: CudaSlice<u32>,
    host_image: PinnedHostSlice<u32>,
    lut_error_flag: CudaSlice<u32>,
    lut_error_host: PinnedHostSlice<u32>,
    ready_event: Option<CudaEvent>,
}
impl FrameBuffer {
    fn new(buffers: RendererBuffers) -> Self {
        Self {
            hdr_buffer: buffers.hdr_buffer,
            bloom_buffer: buffers.bloom_buffer,
            image_gpu: buffers.image_gpu,
            host_image: buffers.host_image,
            lut_error_flag: buffers.lut_error_flag,
            lut_error_host: buffers.lut_error_host,
            ready_event: None,
        }
    }

    fn is_ready(&self) -> bool {
        self.ready_event
            .as_ref()
            .is_some_and(CudaEvent::is_complete)
    }
}
pub struct CudaRenderer {
    stream: Arc<CudaStream>,
    trace_kernel: CudaFunction,
    bloom_kernel: CudaFunction,
    post_kernel: CudaFunction,
    frames: Vec<FrameBuffer>,
    pending_frames: VecDeque<usize>,
    next_frame: usize,
    lut_texture: CudaTextureLut,
    disk_texture: CudaTextureLut,
    lut_size: i32,
    lut_max_temp: f32,
    disk_inner: f32,
    disk_outer: f32,
    width: u32,
    height: u32,
    block_dim: (u32, u32, u32),
    grid_dim: (u32, u32, u32),
    bloom_enabled: bool,
    bloom_intensity: f32,
    bloom_radius: f32,
    bloom_radius_int: i32,
}
impl CudaRenderer {
    pub fn new(config: &Config, cuda_dir: &Path) -> Result<Self> {
        let u_width = config.window.width;
        let u_height = config.window.height;
        let context = CudaContext::new(0).context("初始化 CUDA 上下文失败")?;
        let stream = context.default_stream();
        let kerr_params = build_kerr_params(config)?;
        let (bloom_enabled, bloom_intensity, bloom_radius, bloom_radius_int) =
            validate_bloom_settings(config)?;
        let (lut_texture, disk_texture, lut_size, lut_max_temp) =
            build_textures(&stream, config, &kerr_params)?;
        let (trace_kernel, bloom_kernel, post_kernel) =
            build_cuda_kernels(&context, &stream, config, cuda_dir, &kerr_params)?;
        let frames = vec![
            FrameBuffer::new(allocate_buffers(&context, &stream, u_width, u_height)?),
            FrameBuffer::new(allocate_buffers(&context, &stream, u_width, u_height)?),
        ];
        let pending_frames = VecDeque::with_capacity(frames.len());
        let (block_dim, grid_dim) = compute_launch_dims(config, u_width, u_height);
        Ok(Self {
            stream,
            trace_kernel,
            bloom_kernel,
            post_kernel,
            frames,
            pending_frames,
            next_frame: 0,
            lut_texture,
            disk_texture,
            lut_size,
            lut_max_temp,
            disk_inner: kerr_params.disk_inner,
            disk_outer: config.kernel.disk.outer_radius,
            width: u_width,
            height: u_height,
            block_dim,
            grid_dim,
            bloom_enabled,
            bloom_intensity,
            bloom_radius,
            bloom_radius_int,
        })
    }

    pub fn submit_render(
        &mut self,
        cam_pos: [f32; 3],
        fwd: [f32; 3],
        rgt: [f32; 3],
        up: [f32; 3],
        fov_scale: f32,
    ) -> Result<bool> {
        if self.pending_frames.len() == self.frames.len() {
            return Ok(false);
        }
        let frame_count = self.frames.len();
        let frame_index = self.next_available_frame()?;
        let launch_config = LaunchConfig {
            grid_dim: self.grid_dim,
            block_dim: self.block_dim,
            shared_mem_bytes: 0,
        };
        let width = i32::try_from(self.width).context("Window width exceeds i32 range")?;
        let height = i32::try_from(self.height).context("Window height exceeds i32 range")?;
        let bloom_active =
            self.bloom_enabled && self.bloom_radius_int > 0 && self.bloom_intensity > 0.0;
        let bloom_sigma = self.bloom_radius / 3.0;
        let bloom_flag = i32::from(bloom_active);
        let post_shared_mem_bytes = self.post_shared_mem_bytes(bloom_active)?;
        let post_launch_config = LaunchConfig {
            grid_dim: self.grid_dim,
            block_dim: self.block_dim,
            shared_mem_bytes: post_shared_mem_bytes,
        };
        let frame = &mut self.frames[frame_index];
        self.stream
            .memset_zeros(&mut frame.lut_error_flag)
            .context("清零 LUT 错误标记失败")?;
        unsafe {
            let mut launch = self.stream.launch_builder(&self.trace_kernel);
            let vecs = [cam_pos, fwd, rgt, up];
            launch.arg(&mut frame.hdr_buffer);
            launch.arg(&width);
            launch.arg(&height);
            for v in &vecs {
                launch.arg(&v[0]).arg(&v[1]).arg(&v[2]);
            }
            launch
                .arg(&self.lut_texture.texture)
                .arg(&self.lut_size)
                .arg(&self.lut_max_temp)
                .arg(&mut frame.lut_error_flag)
                .arg(&self.disk_texture.texture)
                .arg(&self.disk_inner)
                .arg(&self.disk_outer)
                .arg(&fov_scale)
                .launch(launch_config)
                .context("Failed to launch kernel")?;
        }
        self.stream
            .memcpy_dtoh(&frame.lut_error_flag, &mut frame.lut_error_host)
            .context("读取 LUT 温度错误标记失败")?;
        if bloom_active {
            unsafe {
                let mut launch = self.stream.launch_builder(&self.bloom_kernel);
                launch
                    .arg(&frame.hdr_buffer)
                    .arg(&mut frame.bloom_buffer)
                    .arg(&width)
                    .arg(&height)
                    .arg(&self.bloom_radius_int)
                    .arg(&bloom_sigma)
                    .arg(&bloom_flag)
                    .launch(launch_config)
                    .context("Failed to launch bloom kernel")?;
            }
        }
        unsafe {
            let mut launch = self.stream.launch_builder(&self.post_kernel);
            launch
                .arg(&frame.hdr_buffer)
                .arg(&frame.bloom_buffer)
                .arg(&mut frame.image_gpu)
                .arg(&width)
                .arg(&height)
                .arg(&self.bloom_radius_int)
                .arg(&bloom_sigma)
                .arg(&self.bloom_intensity)
                .arg(&bloom_flag)
                .launch(post_launch_config)
                .context("Failed to launch post process kernel")?;
        }
        self.stream
            .memcpy_dtoh(&frame.image_gpu, &mut frame.host_image)
            .context("Failed to copy image back")?;
        let event = self
            .stream
            .record_event(None)
            .context("Failed to record frame event")?;
        frame.ready_event = Some(event);
        self.pending_frames.push_back(frame_index);
        self.next_frame = (frame_index + 1) % frame_count;
        Ok(true)
    }

    pub fn ready_frame(&mut self) -> Result<Option<FrameView<'_>>> {
        let lut_max_temp = self.lut_max_temp;
        let Some(frame_index) = self.pending_frames.front().copied() else {
            return Ok(None);
        };
        let frame = &mut self.frames[frame_index];
        if !frame.is_ready() {
            return Ok(None);
        }
        let lut_error_state = frame
            .lut_error_host
            .as_slice()
            .context("读取 LUT 温度错误标记失败")?;
        if lut_error_state.len() < 2 {
            return Err(anyhow!("LUT 错误标记缓冲区大小异常"));
        }
        if lut_error_state[0] != 0 {
            let temp = f32::from_bits(lut_error_state[1]);
            return Err(anyhow!(
                "颜色温度超过 lut_max_temp: {temp} > {lut_max_temp}"
            ));
        }
        let host_image = frame
            .host_image
            .as_slice()
            .context("Failed to sync image buffer")?;
        Ok(Some(FrameView {
            index: frame_index,
            data: host_image,
        }))
    }

    pub fn finish_frame(&mut self, index: usize) -> Result<()> {
        let expected = self.pending_frames.front().copied().context("帧队列为空")?;
        if expected != index {
            return Err(anyhow!("帧队列顺序异常: expect={expected} got={index}"));
        }
        self.pending_frames.pop_front();
        self.frames[index].ready_event = None;
        Ok(())
    }

    fn next_available_frame(&self) -> Result<usize> {
        if self.frames.is_empty() {
            return Err(anyhow!("帧缓冲区未初始化"));
        }
        for offset in 0..self.frames.len() {
            let index = (self.next_frame + offset) % self.frames.len();
            if !self.pending_frames.iter().any(|&pending| pending == index) {
                return Ok(index);
            }
        }
        Err(anyhow!("没有可用帧缓冲区"))
    }

    fn post_shared_mem_bytes(&self, bloom_active: bool) -> Result<u32> {
        if !bloom_active {
            return Ok(0);
        }
        let block_x = self.block_dim.0 as usize;
        let block_y = self.block_dim.1 as usize;
        let radius = usize::try_from(self.bloom_radius_int).context("Bloom 半径超出 usize 范围")?;
        let tile_h = block_y
            .checked_add(radius.saturating_mul(2))
            .context("Bloom 共享内存尺寸溢出")?;
        let tile_len = block_x
            .checked_mul(tile_h)
            .context("Bloom 共享内存尺寸溢出")?;
        let bytes = tile_len
            .checked_mul(std::mem::size_of::<f32>() * 4)
            .context("Bloom 共享内存尺寸溢出")?;
        u32::try_from(bytes).context("Bloom 共享内存尺寸超过 u32")
    }
}
