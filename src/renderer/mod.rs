mod init;
mod texture;
mod utils;
use std::{path::Path, sync::Arc};

use anyhow::{Context, Result, anyhow};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use init::{
    RendererBuffers, allocate_buffers, build_cuda_kernels, build_textures, compute_launch_dims,
    validate_bloom_settings,
};
use texture::CudaTextureLut;
use utils::build_kerr_params;

use crate::Config;
pub struct CudaRenderer {
    stream: Arc<CudaStream>,
    trace_kernel: CudaFunction,
    bloom_kernel: CudaFunction,
    post_kernel: CudaFunction,
    hdr_buffer: CudaSlice<f32>,
    bloom_buffer: CudaSlice<f32>,
    image_gpu: CudaSlice<u32>,
    host_image: PinnedHostSlice<u32>,
    lut_texture: CudaTextureLut,
    disk_texture: CudaTextureLut,
    lut_size: i32,
    lut_max_temp: f32,
    lut_error_flag: CudaSlice<u32>,
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
        let RendererBuffers {
            hdr_buffer,
            bloom_buffer,
            image_gpu,
            host_image,
            lut_error_flag,
        } = allocate_buffers(&context, &stream, u_width, u_height)?;
        let (block_dim, grid_dim) = compute_launch_dims(config, u_width, u_height);
        Ok(Self {
            stream,
            trace_kernel,
            bloom_kernel,
            post_kernel,
            hdr_buffer,
            bloom_buffer,
            image_gpu,
            host_image,
            lut_texture,
            disk_texture,
            lut_size,
            lut_max_temp,
            lut_error_flag,
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

    pub fn render(
        &mut self,
        cam_pos: [f32; 3],
        fwd: [f32; 3],
        rgt: [f32; 3],
        up: [f32; 3],
        fov_scale: f32,
    ) -> Result<&[u32]> {
        let launch_config = LaunchConfig {
            grid_dim: self.grid_dim,
            block_dim: self.block_dim,
            shared_mem_bytes: 0,
        };
        self.stream
            .memset_zeros(&mut self.lut_error_flag)
            .context("清零 LUT 错误标记失败")?;
        let width = i32::try_from(self.width).context("Window width exceeds i32 range")?;
        let height = i32::try_from(self.height).context("Window height exceeds i32 range")?;
        unsafe {
            let mut launch = self.stream.launch_builder(&self.trace_kernel);
            let vecs = [cam_pos, fwd, rgt, up];
            launch.arg(&mut self.hdr_buffer);
            launch.arg(&width);
            launch.arg(&height);
            for v in &vecs {
                launch.arg(&v[0]).arg(&v[1]).arg(&v[2]);
            }
            launch
                .arg(&self.lut_texture.texture)
                .arg(&self.lut_size)
                .arg(&self.lut_max_temp)
                .arg(&mut self.lut_error_flag)
                .arg(&self.disk_texture.texture)
                .arg(&self.disk_inner)
                .arg(&self.disk_outer)
                .arg(&fov_scale)
                .launch(launch_config)
                .context("Failed to launch kernel")?;
        }
        let mut lut_error_state = [0u32; 2];
        self.stream
            .memcpy_dtoh(&self.lut_error_flag, &mut lut_error_state)
            .context("读取 LUT 温度错误标记失败")?;
        if lut_error_state[0] != 0 {
            let temp = f32::from_bits(lut_error_state[1]);
            return Err(anyhow!(
                "颜色温度超过 lut_max_temp: {temp} > {}",
                self.lut_max_temp
            ));
        }
        let bloom_active =
            self.bloom_enabled && self.bloom_radius_int > 0 && self.bloom_intensity > 0.0;
        let bloom_sigma = self.bloom_radius / 3.0;
        let bloom_flag = i32::from(bloom_active);
        if bloom_active {
            unsafe {
                let mut launch = self.stream.launch_builder(&self.bloom_kernel);
                launch
                    .arg(&self.hdr_buffer)
                    .arg(&mut self.bloom_buffer)
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
                .arg(&self.hdr_buffer)
                .arg(&self.bloom_buffer)
                .arg(&mut self.image_gpu)
                .arg(&width)
                .arg(&height)
                .arg(&self.bloom_radius_int)
                .arg(&bloom_sigma)
                .arg(&self.bloom_intensity)
                .arg(&bloom_flag)
                .launch(launch_config)
                .context("Failed to launch post process kernel")?;
        }
        self.stream
            .memcpy_dtoh(&self.image_gpu, &mut self.host_image)
            .context("Failed to copy image back")?;
        self.host_image()
    }

    pub fn host_image(&self) -> Result<&[u32]> {
        self.host_image
            .as_slice()
            .context("Failed to sync image buffer")
    }
}
