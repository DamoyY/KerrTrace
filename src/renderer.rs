mod blackbody;
mod defines;
mod disk;
mod frame;
#[cfg(test)]
#[path = "../tests/integration/invariants.rs"]
mod invariants;
#[cfg(test)]
#[path = "../tests/unit/luminosity.rs"]
mod luminosity;
mod pipeline;
#[cfg(test)]
#[path = "../tests/unit/spectral.rs"]
mod spectral;
mod submit;
mod texture;
use crate::config::Config;
use alloc::{collections::VecDeque, sync::Arc};
use anyhow::{Context as _, Result};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig};
use frame::FrameBuffer;
use pipeline::{build_cuda_kernels, build_textures, launch_settings};
use std::path::Path;
use texture::CudaTextureLut;
pub struct CudaRenderer {
    stream: Arc<CudaStream>,
    trace_kernel: CudaFunction,
    bloom_kernel: CudaFunction,
    post_kernel: CudaFunction,
    available: VecDeque<FrameBuffer>,
    pending: VecDeque<FrameBuffer>,
    displayed: Option<FrameBuffer>,
    lut_texture: CudaTextureLut,
    disk_texture: CudaTextureLut,
    lut_size: i32,
    lut_max_temp: f32,
    disk_inner: f32,
    disk_outer: f32,
    width: i32,
    height: i32,
    launch_config: LaunchConfig,
    post_launch_config: LaunchConfig,
    bloom_intensity: f32,
    bloom_sigma: f32,
    bloom_radius: i32,
    bloom_active: i32,
}
impl CudaRenderer {
    pub fn new(config: &Config, cuda_dir: &Path) -> Result<Self> {
        config.validate()?;
        let context = CudaContext::new(0).context("初始化 CUDA 上下文失败")?;
        let stream = context.default_stream();
        let kerr_params = disk::build_kerr_params(config)?;
        let (launch_config, post_launch_config, bloom_radius, bloom_active) =
            launch_settings(config, &context)?;
        let (lut_texture, disk_texture, lut_size, lut_max_temp) =
            build_textures(&stream, config, &kerr_params)?;
        let (trace_kernel, bloom_kernel, post_kernel) =
            build_cuda_kernels(&context, &stream, config, cuda_dir, &kerr_params)?;
        let available = core::iter::repeat_with(|| {
            FrameBuffer::new(&stream, config.window.width, config.window.height)
        })
        .take(2)
        .collect::<Result<VecDeque<_>>>()?;
        Ok(Self {
            stream,
            trace_kernel,
            bloom_kernel,
            post_kernel,
            available,
            pending: VecDeque::with_capacity(2),
            displayed: None,
            lut_texture,
            disk_texture,
            lut_size,
            lut_max_temp,
            disk_inner: kerr_params.disk_inner,
            disk_outer: config.kernel.disk.outer_radius,
            width: i32::try_from(config.window.width)?,
            height: i32::try_from(config.window.height)?,
            launch_config,
            post_launch_config,
            bloom_intensity: config.bloom.intensity,
            bloom_sigma: config.bloom.radius / 3.0_f32,
            bloom_radius,
            bloom_active,
        })
    }
    pub fn ready_frame(&self) -> Result<Option<&[u32]>> {
        self.pending
            .front()
            .map_or_else(|| Ok(None), |frame| frame.ready_data(self.lut_max_temp))
    }
    pub fn finish_frame(&mut self) -> Result<()> {
        let frame = self.pending.pop_front().context("没有待呈现的帧")?;
        frame.ready_event.synchronize()?;
        if let Some(previous) = self.displayed.replace(frame) {
            self.available.push_back(previous);
        }
        Ok(())
    }
    pub fn displayed_frame(&self) -> Result<Option<&[u32]>> {
        self.displayed
            .as_ref()
            .map_or_else(|| Ok(None), |frame| frame.ready_data(self.lut_max_temp))
    }
}
impl Drop for CudaRenderer {
    fn drop(&mut self) {
        if let Err(error) = self.stream.synchronize() {
            log::error!("释放渲染资源前等待 CUDA 完成失败: {error}");
        }
    }
}
