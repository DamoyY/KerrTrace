use std::{ffi::c_void, fs, mem, path::Path, ptr, sync::Arc};

use anyhow::{Context, Result, anyhow};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
    },
    nvrtc::{CompileOptions, compile_ptx_with_opts},
    runtime::{result::RuntimeError, sys as cuda_sys},
};
use utils::{
    build_cuda_defines, build_kerr_params, generate_blackbody_lut, generate_disk_temperature_lut,
};

use crate::Config;
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
struct CudaTextureLut {
    texture: cuda_sys::cudaTextureObject_t,
    array: cuda_sys::cudaArray_t,
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
pub struct CudaRenderer {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    image_gpu: CudaSlice<u8>,
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
}
impl CudaRenderer {
    pub fn new(config: &Config, cuda_dir: &Path) -> Result<Self> {
        let u_width = config.window.width;
        let u_height = config.window.height;
        let context = CudaContext::new(0).context("初始化 CUDA 上下文失败")?;
        let stream = context.default_stream();
        let kerr_params = build_kerr_params(&config.kernel)?;
        let (lut_cpu, lut_max_temp) = generate_blackbody_lut(
            config.blackbody.lut_size,
            config.blackbody.lut_max_temp,
            config.blackbody.wavelength_start,
            config.blackbody.wavelength_end,
            config.blackbody.wavelength_step,
        )
        .context("生成黑体查找表失败")?;
        let disk_lut = generate_disk_temperature_lut(
            &kerr_params,
            config.kernel.disk.outer_radius,
            config.blackbody.lut_size,
        )
        .context("生成吸积盘温度分布表失败")?;
        let lut_size =
            i32::try_from(config.blackbody.lut_size).context("LUT size exceeds i32 range")?;
        let lut_texture = create_lut_texture(&stream, &lut_cpu, config.blackbody.lut_size)
            .context("创建 LUT 纹理失败")?;
        let disk_texture = create_disk_texture(&stream, &disk_lut, config.blackbody.lut_size)
            .context("创建吸积盘纹理失败")?;
        let kernel_source =
            fs::read_to_string(cuda_dir.join("kernel.cu")).context("读取 kernel.cu 失败")?;
        let defines = build_cuda_defines(&config.kernel);
        let full_source = format!("{defines}\n{kernel_source}");
        let ptx_opts = CompileOptions {
            include_paths: vec![cuda_dir.to_string_lossy().to_string()],
            use_fast_math: Some(config.cuda.use_fast_math),
            name: Some("kernel".to_string()),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(&full_source, ptx_opts).context("编译 PTX 失败")?;
        let module = context.load_module(ptx).context("加载 PTX 模块失败")?;
        let kernel = module.load_function("kernel").context("获取核函数失败")?;
        let image_gpu = stream
            .alloc_zeros::<u8>((u_width * u_height * 4) as usize)
            .context("allocate 图像缓存失败")?;
        let block_x = config.renderer.block_dim[0];
        let block_y = config.renderer.block_dim[1];
        let grid_x = u_width.div_ceil(block_x);
        let grid_y = u_height.div_ceil(block_y);
        let mut params_symbol = module
            .get_global("c_params", &stream)
            .context("获取常量内存符号 c_params 失败")?;
        let mut params_view = unsafe {
            params_symbol
                .transmute_mut::<KerrParams>(1)
                .ok_or_else(|| anyhow!("c_params 常量内存大小不足"))?
        };
        stream
            .memcpy_htod(std::slice::from_ref(&kerr_params), &mut params_view)
            .context("复制 Kerr 参数到常量内存失败")?;
        Ok(Self {
            stream,
            kernel,
            image_gpu,
            lut_texture,
            disk_texture,
            lut_size,
            lut_max_temp,
            disk_inner: kerr_params.disk_inner,
            disk_outer: config.kernel.disk.outer_radius,
            width: u_width,
            height: u_height,
            block_dim: (block_x, block_y, 1),
            grid_dim: (grid_x, grid_y, 1),
        })
    }

    pub fn render(
        &mut self,
        cam_pos: [f32; 3],
        fwd: [f32; 3],
        rgt: [f32; 3],
        up: [f32; 3],
        fov_scale: f32,
    ) -> Result<Vec<u8>> {
        let launch_config = LaunchConfig {
            grid_dim: self.grid_dim,
            block_dim: self.block_dim,
            shared_mem_bytes: 0,
        };
        let width = i32::try_from(self.width).context("Window width exceeds i32 range")?;
        let height = i32::try_from(self.height).context("Window height exceeds i32 range")?;
        unsafe {
            let mut launch = self.stream.launch_builder(&self.kernel);
            let vecs = [cam_pos, fwd, rgt, up];
            launch.arg(&mut self.image_gpu);
            launch.arg(&width);
            launch.arg(&height);
            for v in &vecs {
                launch.arg(&v[0]).arg(&v[1]).arg(&v[2]);
            }
            launch
                .arg(&self.lut_texture.texture)
                .arg(&self.lut_size)
                .arg(&self.lut_max_temp)
                .arg(&self.disk_texture.texture)
                .arg(&self.disk_inner)
                .arg(&self.disk_outer)
                .arg(&fov_scale)
                .launch(launch_config)
                .context("Failed to launch kernel")?;
        }
        let host_image = self
            .stream
            .clone_dtoh(&self.image_gpu)
            .context("Failed to copy image back")?;
        Ok(host_image)
    }
}
fn cuda_runtime_result(err: cuda_sys::cudaError_t, context: &str) -> Result<()> {
    err.result().map_err(|e| anyhow!("{context}: {e}"))
}
fn create_lut_texture(
    stream: &CudaStream,
    lut_data: &[f32],
    lut_size: usize,
) -> Result<CudaTextureLut> {
    if lut_size == 0 {
        return Err(anyhow!("LUT size 为 0，无法创建纹理"));
    }
    if lut_data.len() != lut_size * 4 {
        return Err(anyhow!(
            "LUT 数据长度不匹配: 期望 {}, 实际 {}",
            lut_size * 4,
            lut_data.len()
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
            32,
            32,
            32,
            32,
            cuda_sys::cudaChannelFormatKind::cudaChannelFormatKindFloat,
        )
    };
    let mut array: cuda_sys::cudaArray_t = ptr::null_mut();
    cuda_runtime_result(
        unsafe {
            cuda_sys::cudaMallocArray(
                &raw mut array,
                &raw const channel_desc,
                lut_size,
                1,
                cuda_sys::cudaArrayDefault,
            )
        },
        "cudaMallocArray 失败",
    )?;
    let bytes = lut_data
        .len()
        .checked_mul(mem::size_of::<f32>())
        .context("LUT 数据大小溢出")?;
    let copy_result = cuda_runtime_result(
        unsafe {
            cuda_sys::cudaMemcpyToArray(
                array,
                0,
                0,
                lut_data.as_ptr().cast::<c_void>(),
                bytes,
                cuda_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
            )
        },
        "复制 LUT 到 CUDA 数组失败",
    );
    if let Err(err) = copy_result {
        let free_result = cuda_runtime_result(
            unsafe { cuda_sys::cudaFreeArray(array) },
            "释放 LUT 数组失败",
        );
        if let Err(free_err) = free_result {
            return Err(anyhow!("{err}; {free_err}"));
        }
        return Err(err);
    }
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
    let create_result = cuda_runtime_result(
        unsafe {
            cuda_sys::cudaCreateTextureObject(
                &raw mut texture,
                &raw const res_desc,
                &raw const tex_desc,
                ptr::null(),
            )
        },
        "cudaCreateTextureObject 失败",
    );
    if let Err(err) = create_result {
        let free_result = cuda_runtime_result(
            unsafe { cuda_sys::cudaFreeArray(array) },
            "释放 LUT 数组失败",
        );
        if let Err(free_err) = free_result {
            return Err(anyhow!("{err}; {free_err}"));
        }
        return Err(err);
    }
    Ok(CudaTextureLut { texture, array })
}
fn create_disk_texture(
    stream: &CudaStream,
    lut_data: &[f32],
    lut_size: usize,
) -> Result<CudaTextureLut> {
    if lut_size == 0 {
        return Err(anyhow!("吸积盘纹理尺寸为 0，无法创建纹理"));
    }
    if lut_data.len() != lut_size {
        return Err(anyhow!(
            "吸积盘纹理数据长度不匹配: 期望 {}, 实际 {}",
            lut_size,
            lut_data.len()
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
            32,
            0,
            0,
            0,
            cuda_sys::cudaChannelFormatKind::cudaChannelFormatKindFloat,
        )
    };
    let mut array: cuda_sys::cudaArray_t = ptr::null_mut();
    cuda_runtime_result(
        unsafe {
            cuda_sys::cudaMallocArray(
                &raw mut array,
                &raw const channel_desc,
                lut_size,
                0,
                cuda_sys::cudaArrayDefault,
            )
        },
        "cudaMallocArray 失败",
    )?;
    let bytes = lut_data
        .len()
        .checked_mul(mem::size_of::<f32>())
        .context("吸积盘纹理数据大小溢出")?;
    let copy_result = cuda_runtime_result(
        unsafe {
            cuda_sys::cudaMemcpyToArray(
                array,
                0,
                0,
                lut_data.as_ptr().cast::<c_void>(),
                bytes,
                cuda_sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
            )
        },
        "复制吸积盘纹理到 CUDA 数组失败",
    );
    if let Err(err) = copy_result {
        let free_result = cuda_runtime_result(
            unsafe { cuda_sys::cudaFreeArray(array) },
            "释放吸积盘纹理数组失败",
        );
        if let Err(free_err) = free_result {
            return Err(anyhow!("{err}; {free_err}"));
        }
        return Err(err);
    }
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
    let create_result = cuda_runtime_result(
        unsafe {
            cuda_sys::cudaCreateTextureObject(
                &raw mut texture,
                &raw const res_desc,
                &raw const tex_desc,
                ptr::null(),
            )
        },
        "cudaCreateTextureObject 失败",
    );
    if let Err(err) = create_result {
        let free_result = cuda_runtime_result(
            unsafe { cuda_sys::cudaFreeArray(array) },
            "释放吸积盘纹理数组失败",
        );
        if let Err(free_err) = free_result {
            return Err(anyhow!("{err}; {free_err}"));
        }
        return Err(err);
    }
    Ok(CudaTextureLut { texture, array })
}
