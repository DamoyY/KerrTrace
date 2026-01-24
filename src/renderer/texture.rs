use std::{ffi::c_void, mem, ptr};

use anyhow::{Context, Result, anyhow};
use cudarc::{
    driver::CudaStream,
    runtime::{result::RuntimeError, sys as cuda_sys},
};
pub(super) struct CudaTextureLut {
    pub(super) texture: cuda_sys::cudaTextureObject_t,
    pub(super) array: cuda_sys::cudaArray_t,
}
impl Drop for CudaTextureLut {
    fn drop(&mut self) {
        let mut errors = Vec::new();
        unsafe {
            if self.texture != 0 {
                let err = cuda_sys::cudaDestroyTextureObject(self.texture);
                if err != cuda_sys::cudaError_t::cudaSuccess {
                    errors.push(format!(
                        "cudaDestroyTextureObject failed: {}",
                        RuntimeError(err)
                    ));
                }
            }
            if !self.array.is_null() {
                let err = cuda_sys::cudaFreeArray(self.array);
                if err != cuda_sys::cudaError_t::cudaSuccess {
                    log::error!("cudaFreeArray failed: {}", RuntimeError(err));
                }
            }
        }
        if let Some(message) = (!errors.is_empty()).then(|| errors.join("; ")) {
            log::error!("CUDA resource cleanup failed: {message}");
        }
    }
}
pub(super) fn create_lut_texture(
    stream: &CudaStream,
    lut_data: &[f32],
    lut_size: usize,
) -> Result<CudaTextureLut> {
    create_texture_impl(
        stream,
        lut_data,
        lut_size,
        1,
        lut_size * 4,
        &[32, 32, 32, 32],
        &TextureMessages {
            zero_size: "LUT size 为 0，无法创建纹理",
            len_mismatch_prefix: "LUT 数据长度不匹配",
            bytes_overflow: "LUT 数据大小溢出",
            copy_fail: "复制 LUT 到 CUDA 数组失败",
            free_fail: "释放 LUT 数组失败",
        },
    )
}
pub(super) fn create_disk_texture(
    stream: &CudaStream,
    lut_data: &[f32],
    lut_size: usize,
) -> Result<CudaTextureLut> {
    create_texture_impl(
        stream,
        lut_data,
        lut_size,
        0,
        lut_size,
        &[32, 0, 0, 0],
        &TextureMessages {
            zero_size: "吸积盘纹理尺寸为 0，无法创建纹理",
            len_mismatch_prefix: "吸积盘纹理数据长度不匹配",
            bytes_overflow: "吸积盘纹理数据大小溢出",
            copy_fail: "复制吸积盘纹理到 CUDA 数组失败",
            free_fail: "释放吸积盘纹理数组失败",
        },
    )
}
struct TextureMessages<'a> {
    zero_size: &'a str,
    len_mismatch_prefix: &'a str,
    bytes_overflow: &'a str,
    copy_fail: &'a str,
    free_fail: &'a str,
}
fn create_texture_impl(
    stream: &CudaStream,
    data: &[f32],
    width: usize,
    height: usize,
    expected_len: usize,
    channel_bits: &[i32; 4],
    messages: &TextureMessages<'_>,
) -> Result<CudaTextureLut> {
    if width == 0 {
        return Err(anyhow!("{}", messages.zero_size));
    }
    if data.len() != expected_len {
        return Err(anyhow!(
            "{}: 期望 {}, 实际 {}",
            messages.len_mismatch_prefix,
            expected_len,
            data.len()
        ));
    }
    let device = i32::try_from(stream.context().ordinal())
        .context("CUDA device ordinal exceeds i32 range")?;
    cuda_runtime_result(
        unsafe { cuda_sys::cudaSetDevice(device) },
        "设置 CUDA device 失败",
    )?;
    let channel_desc = unsafe {
        cuda_sys::cudaCreateChannelDesc(
            channel_bits[0],
            channel_bits[1],
            channel_bits[2],
            channel_bits[3],
            cuda_sys::cudaChannelFormatKind::cudaChannelFormatKindFloat,
        )
    };
    let mut array: cuda_sys::cudaArray_t = ptr::null_mut();
    cuda_runtime_result(
        unsafe {
            cuda_sys::cudaMallocArray(
                &raw mut array,
                &raw const channel_desc,
                width,
                height,
                cuda_sys::cudaArrayDefault,
            )
        },
        "cudaMallocArray 失败",
    )?;
    let bytes = data
        .len()
        .checked_mul(mem::size_of::<f32>())
        .context(messages.bytes_overflow.to_string())?;
    let copy_result = cuda_runtime_result(
        unsafe {
            cuda_sys::cudaMemcpyToArray(
                array,
                0,
                0,
                data.as_ptr().cast::<c_void>(),
                bytes,
                cuda_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
            )
        },
        messages.copy_fail,
    );
    if let Err(err) = copy_result {
        let free_result = cuda_runtime_result(
            unsafe { cuda_sys::cudaFreeArray(array) },
            messages.free_fail,
        );
        if let Err(free_err) = free_result {
            return Err(anyhow!("{err}; {free_err}"));
        }
        return Err(err);
    }
    let texture = match create_texture_object(array) {
        Ok(texture) => texture,
        Err(err) => {
            let free_result = cuda_runtime_result(
                unsafe { cuda_sys::cudaFreeArray(array) },
                messages.free_fail,
            );
            if let Err(free_err) = free_result {
                return Err(anyhow!("{err}; {free_err}"));
            }
            return Err(err);
        }
    };
    Ok(CudaTextureLut { texture, array })
}
fn create_texture_object(array: cuda_sys::cudaArray_t) -> Result<cuda_sys::cudaTextureObject_t> {
    let mut res_desc: cuda_sys::cudaResourceDesc = unsafe { mem::zeroed() };
    res_desc.resType = cuda_sys::cudaResourceType::cudaResourceTypeArray;
    res_desc.res.array = cuda_sys::cudaResourceDesc__bindgen_ty_1__bindgen_ty_1 { array };
    let mut tex_desc: cuda_sys::cudaTextureDesc = unsafe { mem::zeroed() };
    tex_desc.addressMode = [
        cuda_sys::cudaTextureAddressMode::cudaAddressModeClamp,
        cuda_sys::cudaTextureAddressMode::cudaAddressModeClamp,
        cuda_sys::cudaTextureAddressMode::cudaAddressModeClamp,
    ];
    tex_desc.filterMode = cuda_sys::cudaTextureFilterMode::cudaFilterModeLinear;
    tex_desc.readMode = cuda_sys::cudaTextureReadMode::cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;
    let mut texture: cuda_sys::cudaTextureObject_t = 0;
    cuda_runtime_result(
        unsafe {
            cuda_sys::cudaCreateTextureObject(
                &raw mut texture,
                &raw const res_desc,
                &raw const tex_desc,
                ptr::null(),
            )
        },
        "cudaCreateTextureObject 失败",
    )?;
    Ok(texture)
}
fn cuda_runtime_result(err: cuda_sys::cudaError_t, context: &str) -> Result<()> {
    err.result().map_err(|e| anyhow!("{context}: {e}"))
}
