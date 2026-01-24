use std::{fs, path::Path, sync::Arc};

use anyhow::{Context, Result, anyhow};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, PinnedHostSlice},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};

use super::{
    texture::{CudaTextureLut, create_disk_texture, create_lut_texture},
    utils::{build_cuda_defines, generate_blackbody_lut, generate_disk_temperature_lut},
};
use crate::Config;
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct KerrParams {
    pub(super) a: f32,
    pub(super) m: f32,
    pub(super) aa: f32,
    pub(super) inv_m: f32,
    pub(super) a_norm: f32,
    pub(super) rh: f32,
    pub(super) disk_inner: f32,
    pub(super) inv_w_2: f32,
    pub(super) inv_h_2: f32,
    pub(super) aspect_ratio: f32,
    pub(super) disk_noise_scale: f32,
    pub(super) disk_noise_strength: f32,
    pub(super) disk_noise_winding: f32,
    pub(super) disk_noise_enabled: i32,
    pub(super) disk_noise_detail: i32,
}
unsafe impl DeviceRepr for KerrParams {}
pub(super) struct RendererBuffers {
    pub(super) hdr_buffer: CudaSlice<f32>,
    pub(super) bloom_buffer: CudaSlice<f32>,
    pub(super) image_gpu: CudaSlice<u32>,
    pub(super) host_image: PinnedHostSlice<u32>,
    pub(super) lut_error_flag: CudaSlice<u32>,
}
pub(super) fn validate_bloom_settings(config: &Config) -> Result<(bool, f32, f32, i32)> {
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
pub(super) fn build_textures(
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
    let disk_texture = create_disk_texture(stream.as_ref(), &disk_lut, config.blackbody.lut_size)
        .context("创建吸积盘纹理失败")?;
    Ok((lut_texture, disk_texture, lut_size, lut_max_temp))
}
pub(super) fn build_cuda_kernels(
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
pub(super) fn allocate_buffers(
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
    let host_image =
        unsafe { context.alloc_pinned::<u32>(pixel_count) }.context("allocate 固定主机内存失败")?;
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
pub(super) const fn compute_launch_dims(
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
