use super::{
    blackbody::generate_blackbody_lut, defines::build_cuda_defines,
    disk::generate_disk_temperature_lut, texture::CudaTextureLut,
};
use crate::config::Config;
use alloc::{string::String, sync::Arc};
use anyhow::{Context as _, Result, ensure};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaStream, DeviceRepr, LaunchConfig, sys::CUdevice_attribute,
    },
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use num_traits::ToPrimitive as _;
use std::{fs, path::Path};
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
pub(super) fn launch_settings(
    config: &Config,
    context: &CudaContext,
) -> Result<(LaunchConfig, LaunchConfig, i32, i32)> {
    let [block_x, block_y] = config.renderer.block_dim;
    let threads = block_x.checked_mul(block_y).context("线程块尺寸溢出")?;
    let max_threads =
        context.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?;
    ensure!(
        i32::try_from(threads)? <= max_threads,
        "线程块超过 GPU 限制"
    );
    let launch = LaunchConfig {
        grid_dim: (
            config.window.width.div_ceil(block_x),
            config.window.height.div_ceil(block_y),
            1,
        ),
        block_dim: (block_x, block_y, 1),
        shared_mem_bytes: 0,
    };
    let radius = config
        .bloom
        .radius
        .ceil()
        .to_i32()
        .context("Bloom 半径超出 i32")?;
    let active = config.bloom.enabled && radius > 0_i32 && config.bloom.intensity > 0.0_f32;
    let mut post = launch;
    if active {
        let halo = u32::try_from(radius)?
            .checked_mul(2)
            .context("Bloom halo 溢出")?;
        let rows = block_y.checked_add(halo).context("Bloom 行数溢出")?;
        let cells = rows
            .checked_mul(block_x)
            .context("Bloom 共享内存尺寸溢出")?;
        post.shared_mem_bytes = cells.checked_mul(16).context("Bloom 共享内存字节数溢出")?;
        let max_shared = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?;
        ensure!(
            i32::try_from(post.shared_mem_bytes)? <= max_shared,
            "Bloom 共享内存超出 GPU 限制"
        );
    }
    Ok((launch, post, radius, i32::from(active)))
}
pub(super) fn build_textures(
    stream: &Arc<CudaStream>,
    config: &Config,
    kerr_params: &KerrParams,
) -> Result<(CudaTextureLut, CudaTextureLut, i32, f32)> {
    let lut_cpu = generate_blackbody_lut(&config.blackbody)?;
    let disk_lut = generate_disk_temperature_lut(
        kerr_params,
        config.kernel.disk.outer_radius,
        config.blackbody.lut_size,
    )?;
    let size = config.blackbody.lut_size;
    let lut_texture = CudaTextureLut::new(stream.context(), &lut_cpu, size, 4)?;
    let disk_texture = CudaTextureLut::new(stream.context(), &disk_lut, size, 1)?;
    Ok((
        lut_texture,
        disk_texture,
        i32::try_from(size)?,
        config.blackbody.lut_max_temp,
    ))
}
pub(super) fn build_cuda_kernels(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    config: &Config,
    cuda_dir: &Path,
    kerr_params: &KerrParams,
) -> Result<(CudaFunction, CudaFunction, CudaFunction)> {
    let source = fs::read_to_string(cuda_dir.join("kernel.cu")).context("读取 kernel.cu 失败")?;
    let defines = build_cuda_defines(&config.kernel);
    let full_source = format!("{defines}\n{source}");
    let mut options = vec![String::from("--warning-as-error=all-warnings")];
    if config.cuda.use_fast_math {
        options.push(String::from("--use_fast_math"));
    }
    let ptx_opts = CompileOptions {
        include_paths: vec![cuda_dir.to_str().context("CUDA 路径不是 UTF-8")?.into()],
        name: Some(String::from("kernel")),
        options,
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(&full_source, ptx_opts).context("编译 PTX 失败")?;
    let module = context.load_module(ptx).context("加载 PTX 模块失败")?;
    let trace = module.load_function("trace_kernel")?;
    let bloom = module.load_function("bloom_horizontal")?;
    let post = module.load_function("post_process")?;
    let mut symbol = module.get_global("c_params", stream)?;
    ensure!(
        symbol.len() == core::mem::size_of::<KerrParams>(),
        "KerrParams ABI 大小不匹配"
    );
    let mut view =
        unsafe { symbol.transmute_mut::<KerrParams>(1) }.context("KerrParams 对齐或大小异常")?;
    stream.memcpy_htod(core::slice::from_ref(kerr_params), &mut view)?;
    Ok((trace, bloom, post))
}
