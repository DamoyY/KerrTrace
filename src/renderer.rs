use std::{fmt::Display, fs, path::Path, sync::Arc};

use anyhow::{Context, Result};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};

use crate::{
    blackbody::generate_blackbody_lut,
    config::{Config, KernelConfig},
};
pub struct CudaRenderer {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    image_gpu: CudaSlice<u8>,
    lut: CudaSlice<f32>,
    lut_size: i32,
    lut_max_temp: f32,
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
        let (lut_cpu, lut_max_temp) = generate_blackbody_lut(
            config.blackbody.lut_size,
            config.blackbody.lut_max_temp,
            config.blackbody.wavelength_start,
            config.blackbody.wavelength_end,
            config.blackbody.wavelength_step,
        )
        .context("生成黑体查找表失败")?;
        let lut_size =
            i32::try_from(config.blackbody.lut_size).context("LUT size exceeds i32 range")?;
        let lut = stream
            .clone_htod(&lut_cpu)
            .context("复制 LUT 到 GPU 失败")?;
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
        Ok(Self {
            stream,
            kernel,
            image_gpu,
            lut,
            lut_size,
            lut_max_temp,
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
                .arg(&self.lut)
                .arg(&self.lut_size)
                .arg(&self.lut_max_temp)
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

fn push_define(lines: &mut Vec<String>, key: &str, value: impl Display) {
    lines.push(format!("#define {key} {value}"));
}

fn push_define_f32(lines: &mut Vec<String>, key: &str, value: f32) {
    lines.push(format!("#define {key} {value:.10}"));
}

fn build_cuda_defines(config: &KernelConfig) -> String {
    let mut lines = Vec::with_capacity(15);
    let ints = [
        ("CONFIG_SSAA_SAMPLES", config.ssaa_samples),
        ("CONFIG_SKY_GRID_DIVISIONS", config.sky.grid_divisions),
        ("CONFIG_INTEGRATOR_MAX_STEPS", config.integrator.max_steps),
        (
            "CONFIG_INTEGRATOR_MAX_ATTEMPTS",
            config.integrator.max_attempts,
        ),
    ];
    for (key, value) in ints {
        push_define(&mut lines, key, value);
    }

    let floats = [
        ("CONFIG_EXPOSURE_SCALE", config.exposure_scale),
        ("CONFIG_SKY_LINE_THICKNESS", config.sky.line_thickness),
        ("CONFIG_SKY_INTENSITY", config.sky.intensity),
        ("CONFIG_BH_SPIN", config.black_hole.spin),
        ("CONFIG_BH_MASS", config.black_hole.mass),
        ("CONFIG_DISK_OUTER_RADIUS", config.disk.outer_radius),
        (
            "CONFIG_DISK_TEMPERATURE_SCALE",
            config.disk.temperature_scale,
        ),
        (
            "CONFIG_INTEGRATOR_INITIAL_STEP",
            config.integrator.initial_step,
        ),
        ("CONFIG_INTEGRATOR_TOLERANCE", config.integrator.tolerance),
        (
            "CONFIG_TRANSMITTANCE_CUTOFF",
            config.integrator.transmittance_cutoff,
        ),
        ("CONFIG_HORIZON_EPSILON", config.integrator.horizon_epsilon),
        ("CONFIG_ESCAPE_RADIUS", config.integrator.escape_radius),
    ];
    for (key, value) in floats {
        push_define_f32(&mut lines, key, value);
    }
    let mut output = lines.join("\n");
    output.push('\n');
    output
}
