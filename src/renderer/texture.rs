use alloc::sync::Arc;
use anyhow::{Context as _, Result, ensure};
use core::{mem, ptr};
use cudarc::{driver::CudaContext, runtime::sys};
pub(super) struct CudaTextureLut {
    pub(super) texture: sys::cudaTextureObject_t,
    array: sys::cudaArray_t,
    context: Arc<CudaContext>,
}
impl CudaTextureLut {
    pub(super) fn new(
        context: &Arc<CudaContext>,
        data: &[f32],
        width: usize,
        channels: usize,
    ) -> Result<Self> {
        ensure!(
            width > 0 && matches!(channels, 1 | 4),
            "纹理尺寸或通道数无效"
        );
        let expected_len = width.checked_mul(channels).context("纹理尺寸溢出")?;
        ensure!(data.len() == expected_len, "纹理数据长度不匹配");
        let device = i32::try_from(context.ordinal()).context("CUDA 设备编号超出 i32")?;
        unsafe { sys::cudaSetDevice(device) }
            .result()
            .context("设置 CUDA 设备失败")?;
        context.bind_to_thread()?;
        let bits = if channels == 4 { 32_i32 } else { 0_i32 };
        let descriptor = sys::cudaChannelFormatDesc {
            x: 32,
            y: bits,
            z: bits,
            w: bits,
            f: sys::cudaChannelFormatKind::cudaChannelFormatKindFloat,
        };
        let mut resource = Self {
            texture: 0,
            array: ptr::null_mut(),
            context: Arc::clone(context),
        };
        unsafe {
            sys::cudaMallocArray(
                &raw mut resource.array,
                &raw const descriptor,
                width,
                0,
                sys::cudaArrayDefault,
            )
        }
        .result()
        .context("分配一维 CUDA 纹理数组失败")?;
        unsafe {
            sys::cudaMemcpyToArray(
                resource.array,
                0,
                0,
                data.as_ptr().cast(),
                mem::size_of_val(data),
                sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
            )
        }
        .result()
        .context("复制纹理数据失败")?;
        resource.texture = create_texture_object(resource.array)?;
        Ok(resource)
    }
}
impl Drop for CudaTextureLut {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind_to_thread() {
            log::error!("纹理释放前绑定 CUDA 上下文失败: {error}");
            return;
        }
        if self.texture != 0 {
            let status = unsafe { sys::cudaDestroyTextureObject(self.texture) };
            if let Err(error) = status.result() {
                log::error!("释放 CUDA 纹理对象失败: {error}");
            }
        }
        if !self.array.is_null() {
            let status = unsafe { sys::cudaFreeArray(self.array) };
            if let Err(error) = status.result() {
                log::error!("释放 CUDA 纹理数组失败: {error}");
            }
        }
    }
}
fn create_texture_object(array: sys::cudaArray_t) -> Result<sys::cudaTextureObject_t> {
    let mut resource: sys::cudaResourceDesc = unsafe { mem::zeroed() };
    resource.resType = sys::cudaResourceType::cudaResourceTypeArray;
    resource.res.array = sys::cudaResourceDesc__bindgen_ty_1__bindgen_ty_1 { array };
    let mut descriptor: sys::cudaTextureDesc = unsafe { mem::zeroed() };
    descriptor.addressMode = [sys::cudaTextureAddressMode::cudaAddressModeClamp; 3];
    descriptor.filterMode = sys::cudaTextureFilterMode::cudaFilterModeLinear;
    descriptor.readMode = sys::cudaTextureReadMode::cudaReadModeElementType;
    descriptor.normalizedCoords = 1_i32;
    let mut texture = 0_u64;
    unsafe {
        sys::cudaCreateTextureObject(
            &raw mut texture,
            &raw const resource,
            &raw const descriptor,
            ptr::null(),
        )
    }
    .result()
    .context("创建 CUDA 纹理对象失败")?;
    Ok(texture)
}
