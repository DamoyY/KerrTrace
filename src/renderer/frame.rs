use alloc::sync::Arc;
use anyhow::{Context as _, Result, ensure};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, PinnedHostSlice};
pub(super) struct FrameBuffer {
    pub(super) hdr_buffer: CudaSlice<f32>,
    pub(super) bloom_buffer: CudaSlice<f32>,
    pub(super) image_gpu: CudaSlice<u32>,
    pub(super) host_image: PinnedHostSlice<u32>,
    pub(super) lut_error_flag: CudaSlice<u32>,
    pub(super) lut_error_host: PinnedHostSlice<u32>,
    pub(super) ready_event: CudaEvent,
}
impl FrameBuffer {
    pub(super) fn new(stream: &Arc<CudaStream>, width: u32, height: u32) -> Result<Self> {
        let pixels = width.checked_mul(height).context("图像尺寸溢出")?;
        let pixel_count = usize::try_from(pixels).context("图像尺寸超出 usize")?;
        let hdr_len = pixel_count.checked_mul(4).context("HDR 缓冲区尺寸溢出")?;
        let context = stream.context();
        Ok(Self {
            hdr_buffer: stream.alloc_zeros(hdr_len)?,
            bloom_buffer: stream.alloc_zeros(hdr_len)?,
            image_gpu: stream.alloc_zeros(pixel_count)?,
            host_image: unsafe { context.alloc_pinned(pixel_count) }?,
            lut_error_flag: stream.alloc_zeros(2)?,
            lut_error_host: unsafe { context.alloc_pinned(2) }?,
            ready_event: context.new_event(None)?,
        })
    }
    pub(super) fn ready_data(&self, max_temp: f32) -> Result<Option<&[u32]>> {
        if !self
            .ready_event
            .try_is_complete()
            .context("查询 CUDA 帧状态失败")?
        {
            return Ok(None);
        }
        let errors = self.lut_error_host.as_slice()?;
        let &[flag, temperature] = errors else {
            anyhow::bail!("LUT 错误缓冲区长度异常");
        };
        ensure!(
            flag == 0,
            "颜色温度超过 lut_max_temp: {} > {max_temp}",
            f32::from_bits(temperature)
        );
        Ok(Some(self.host_image.as_slice()?))
    }
}
