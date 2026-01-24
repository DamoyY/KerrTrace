use std::{fs, path::Path, sync::Arc};

use anyhow::{Context, Result, anyhow};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
        PinnedHostSlice, PushKernelArg,
    },
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use texture::{CudaTextureLut, create_disk_texture, create_lut_texture};
use utils::{
    build_cuda_defines, build_kerr_params, generate_blackbody_lut, generate_disk_temperature_lut,
};

use crate::Config;
mod texture;
mod utils;
#[repr(C)]
#[derive(Clone, Copy)]
struct KerrParams {
    a: f32,
    m: f32,
    aa: f32,
    inv_m: f32,
    a_norm: f32,
    rh: f32,
    disk_inner: f32,
}
unsafe impl DeviceRepr for KerrParams {}
struct RendererBuffers {
    hdr_buffer: CudaSlice<f32>,
    bloom_buffer: CudaSlice<f32>,
    image_gpu: CudaSlice<u32>,
    host_image: PinnedHostSlice<u32>,
    lut_error_flag: CudaSlice<u32>,
}
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
        let kerr_params = build_kerr_params(&config.kernel)?;
        let (bloom_enabled, bloom_intensity, bloom_radius, bloom_radius_int) =
            Self::validate_bloom_settings(config)?;
        let (lut_texture, disk_texture, lut_size, lut_max_temp) =
            Self::build_textures(&stream, config, &kerr_params)?;
        let (trace_kernel, bloom_kernel, post_kernel) =
            Self::build_cuda_kernels(&context, &stream, config, cuda_dir, &kerr_params)?;
        let RendererBuffers {
            hdr_buffer,
            bloom_buffer,
            image_gpu,
            host_image,
            lut_error_flag,
        } = Self::allocate_buffers(&context, &stream, u_width, u_height)?;
        let (block_dim, grid_dim) = Self::compute_launch_dims(config, u_width, u_height);
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

    fn validate_bloom_settings(config: &Config) -> Result<(bool, f32, f32, i32)> {
        let bloom_enabled = config.bloom.enabled;
        let bloom_intensity = config.bloom.intensity;
        if !bloom_intensity.is_finite() || bloom_intensity < 0.0 {
            return Err(anyhow!("Bloom 强度无效: {bloom_intensity}"));
        }
        let bloom_radius = config.bloom.radius;
        if !bloom_radius.is_finite() || bloom_radius < 0.0 {
            return Err(anyhow!("Bloom 半径无效: {bloom_radius}"));
        }
        let max_radius = f64::from(i32::MAX);
        let radius_f64 = f64::from(bloom_radius);
        if radius_f64 > max_radius {
            return Err(anyhow!("Bloom 半径过大: {bloom_radius}"));
        }
        let bloom_radius_int = if bloom_radius <= 0.0 {
            0
        } else {
            let radius_ceil = bloom_radius.ceil();
            if f64::from(radius_ceil) > max_radius {
                return Err(anyhow!("Bloom 半径过大: {bloom_radius}"));
            }
            unsafe { radius_ceil.to_int_unchecked::<i32>() }
        };
        Ok((
            bloom_enabled,
            bloom_intensity,
            bloom_radius,
            bloom_radius_int,
        ))
    }

    fn build_textures(
        stream: &Arc<CudaStream>,
        config: &Config,
        kerr_params: &KerrParams,
    ) -> Result<(CudaTextureLut, CudaTextureLut, i32, f32)> {
        let (lut_cpu, lut_max_temp) = generate_blackbody_lut(
            config.blackbody.lut_size,
            config.blackbody.lut_max_temp,
            config.blackbody.wavelength_start,
            config.blackbody.wavelength_end,
            config.blackbody.wavelength_step,
        )
        .context("生成黑体查找表失败")?;
        let disk_lut = generate_disk_temperature_lut(
            kerr_params,
            config.kernel.disk.outer_radius,
            config.blackbody.lut_size,
        )
        .context("生成吸积盘温度分布表失败")?;
        let lut_size =
            i32::try_from(config.blackbody.lut_size).context("LUT size exceeds i32 range")?;
        let lut_texture = create_lut_texture(stream.as_ref(), &lut_cpu, config.blackbody.lut_size)
            .context("创建 LUT 纹理失败")?;
        let disk_texture =
            create_disk_texture(stream.as_ref(), &disk_lut, config.blackbody.lut_size)
                .context("创建吸积盘纹理失败")?;
        Ok((lut_texture, disk_texture, lut_size, lut_max_temp))
    }

    fn build_cuda_kernels(
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        config: &Config,
        cuda_dir: &Path,
        kerr_params: &KerrParams,
    ) -> Result<(CudaFunction, CudaFunction, CudaFunction)> {
        let kernel_source =
            fs::read_to_string(cuda_dir.join("kernel.cu")).context("读取 kernel.cu 失败")?;
        let defines = build_cuda_defines(&config.kernel, config.blackbody.wavelength_step);
        let full_source = format!("{defines}\n{kernel_source}");
        let ptx_opts = CompileOptions {
            include_paths: vec![cuda_dir.to_string_lossy().to_string()],
            use_fast_math: Some(config.cuda.use_fast_math),
            name: Some("kernel".to_string()),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(&full_source, ptx_opts).context("编译 PTX 失败")?;
        let module = context.load_module(ptx).context("加载 PTX 模块失败")?;
        let trace_kernel = module
            .load_function("trace_kernel")
            .context("获取 trace kernel 失败")?;
        let bloom_kernel = module
            .load_function("bloom_horizontal")
            .context("获取 bloom kernel 失败")?;
        let post_kernel = module
            .load_function("post_process")
            .context("获取 post process kernel 失败")?;
        let mut params_symbol = module
            .get_global("c_params", stream)
            .context("获取常量内存符号 c_params 失败")?;
        let mut params_view = unsafe {
            params_symbol
                .transmute_mut::<KerrParams>(1)
                .ok_or_else(|| anyhow!("c_params 常量内存大小不足"))?
        };
        stream
            .memcpy_htod(std::slice::from_ref(kerr_params), &mut params_view)
            .context("复制 Kerr 参数到常量内存失败")?;
        Ok((trace_kernel, bloom_kernel, post_kernel))
    }

    fn allocate_buffers(
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        width: u32,
        height: u32,
    ) -> Result<RendererBuffers> {
        let pixel_count = (width as usize)
            .checked_mul(height as usize)
            .context("图像尺寸超出 usize 范围")?;
        let hdr_len = pixel_count
            .checked_mul(4)
            .context("HDR 缓冲区尺寸超出 usize 范围")?;
        let hdr_buffer = stream
            .alloc_zeros::<f32>(hdr_len)
            .context("allocate HDR 缓冲区失败")?;
        let bloom_buffer = stream
            .alloc_zeros::<f32>(hdr_len)
            .context("allocate Bloom 缓冲区失败")?;
        let image_gpu = stream
            .alloc_zeros::<u32>(pixel_count)
            .context("allocate 图像缓存失败")?;
        let host_image = unsafe { context.alloc_pinned::<u32>(pixel_count) }
            .context("allocate 固定主机内存失败")?;
        let lut_error_flag = stream
            .alloc_zeros::<u32>(2)
            .context("allocate LUT 错误标记失败")?;
        Ok(RendererBuffers {
            hdr_buffer,
            bloom_buffer,
            image_gpu,
            host_image,
            lut_error_flag,
        })
    }

    const fn compute_launch_dims(
        config: &Config,
        width: u32,
        height: u32,
    ) -> ((u32, u32, u32), (u32, u32, u32)) {
        let block_x = config.renderer.block_dim[0];
        let block_y = config.renderer.block_dim[1];
        let grid_x = width.div_ceil(block_x);
        let grid_y = height.div_ceil(block_y);
        ((block_x, block_y, 1), (grid_x, grid_y, 1))
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
                    .arg(&self.bloom_radius)
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
                .arg(&self.bloom_radius)
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
