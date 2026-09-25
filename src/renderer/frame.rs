use alloc::sync::Arc;
use anyhow::{Context as _, Result};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, PinnedHostSlice};
pub(super) struct FrameBuffer {
    pub(super) hdr_buffer: CudaSlice<f32>,
    pub(super) bloom_buffer: CudaSlice<f32>,
    pub(super) image_gpu: CudaSlice<u32>,
    pub(super) host_image: PinnedHostSlice<u32>,
    pub(super) device_error: CudaSlice<u32>,
    pub(super) host_error: PinnedHostSlice<u32>,
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
            device_error: stream.alloc_zeros(2)?,
            host_error: unsafe { context.alloc_pinned(2) }?,
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
        let errors = self.host_error.as_slice()?;
        let &[flag, bits] = errors else {
            anyhow::bail!("设备错误缓冲区长度异常");
        };
        let value = f32::from_bits(bits);
        match flag {
            0 => {}
            1 => anyhow::bail!("颜色温度超过 lut_max_temp: {value} > {max_temp}"),
            2 => anyhow::bail!("积分重试耗尽: normalized_error={value}"),
            3 => anyhow::bail!("积分步长无效或位置停滞: step={value}"),
            4 => anyhow::bail!("积分步数耗尽: radius={value}"),
            5 => anyhow::bail!("静止相机应位于静止极限之外且光线初值有限: energy={value}"),
            6 => anyhow::bail!("吸积盘辐射计算无效: value={value}"),
            _ => anyhow::bail!("未知设备错误: code={flag}, value={value}"),
        }
        Ok(Some(self.host_image.as_slice()?))
    }
}
