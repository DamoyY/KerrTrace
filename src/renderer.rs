use std::{fmt::Display, fs, path::Path, sync::Arc};

use anyhow::{Context, Result, anyhow};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use log::error;

use crate::{Config, KernelConfig};
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
fn generate_blackbody_lut(
    size: usize,
    max_temp: f32,
    wavelength_start: f32,
    wavelength_end: f32,
    wavelength_step: f32,
) -> Result<(Vec<f32>, f32)> {
    if size < 2 {
        error!("黑体查找表尺寸过小: {size}");
        return Err(anyhow!("黑体查找表尺寸必须至少为 2"));
    }
    if !wavelength_step.is_finite() || wavelength_step <= 0.0 {
        error!("波长步进无效: {wavelength_step}");
        return Err(anyhow!("波长步进必须为正数且有限"));
    }
    let mut lut_data = Vec::with_capacity(size * 3);
    let denom_usize = size - 1;
    let denom_u32 =
        u32::try_from(denom_usize).map_err(|_| anyhow!("黑体查找表尺寸超出 u32 范围: {size}"))?;
    let denom_f = f64::from(denom_u32);
    let temps: Vec<f32> = (0..size)
        .map(|i| {
            let i_u32 =
                u32::try_from(i).map_err(|_| anyhow!("黑体查找表索引超出 u32 范围: {i}"))?;
            let ratio = f64::from(i_u32) / denom_f;
            let temp = ratio * f64::from(max_temp);
            f32_from_f64_checked(temp).map_err(|err| anyhow!("温度值转换失败: {err}"))
        })
        .collect::<Result<Vec<f32>>>()?;
    let mut lambdas = Vec::new();
    let mut l = wavelength_start;
    loop {
        if l > wavelength_end {
            break;
        }
        lambdas.push(l);
        l += wavelength_step;
    }
    let (xs, ys, zs) = get_xyz_sensitivity(&lambdas);
    for &t in &temps {
        if t == 0.0 {
            lut_data.push(0.0);
            lut_data.push(0.0);
            lut_data.push(0.0);
            continue;
        }
        let intensities = planck_law(&lambdas, t);
        let mut x_sum = 0.0;
        let mut y_sum = 0.0;
        let mut z_sum = 0.0;
        for i in 0..lambdas.len() {
            x_sum += intensities[i] * xs[i];
            y_sum += intensities[i] * ys[i];
            z_sum += intensities[i] * zs[i];
        }
        let (r, g, b) = xyz_to_rgb(x_sum, y_sum, z_sum);
        lut_data.push(r);
        lut_data.push(g);
        lut_data.push(b);
    }
    Ok((lut_data, max_temp))
}
fn gaussian(x: f32, alpha: f32, mu: f32, sigma1: f32, sigma2: f32) -> f32 {
    let sigma = if x < mu { sigma1 } else { sigma2 };
    let t = (x - mu) / sigma;
    alpha * (-0.5 * t * t).exp()
}
fn get_xyz_sensitivity(lambdas: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut xs = Vec::with_capacity(lambdas.len());
    let mut ys = Vec::with_capacity(lambdas.len());
    let mut zs = Vec::with_capacity(lambdas.len());
    for &l in lambdas {
        xs.push(
            gaussian(l, 1.056, 599.8, 37.9, 31.0)
                + gaussian(l, 0.362, 442.0, 16.0, 26.7)
                + gaussian(l, -0.065, 501.1, 20.4, 26.2),
        );
        ys.push(gaussian(l, 0.821, 568.8, 46.9, 40.5) + gaussian(l, 0.286, 530.9, 16.3, 31.1));
        zs.push(gaussian(l, 1.217, 437.0, 11.8, 36.0) + gaussian(l, 0.681, 459.0, 26.2, 13.8));
    }
    (xs, ys, zs)
}
fn planck_law(lambdas: &[f32], t: f32) -> Vec<f32> {
    let c2 = 1.4388e7;
    let mut val = Vec::with_capacity(lambdas.len());
    if t < 1e-8 {
        return vec![0.0; lambdas.len()];
    }
    for &l in lambdas {
        let exponent = c2 / (l * t);
        if exponent > 80.0 {
            val.push(0.0);
        } else {
            let v = (1.0 / l.powi(5)) / exponent.exp_m1();
            val.push(v * 1e15);
        }
    }
    val
}
fn xyz_to_rgb(x_val: f32, y_val: f32, z_val: f32) -> (f32, f32, f32) {
    let red = (3.240_454f32).mul_add(
        x_val,
        (-1.537_138_5f32).mul_add(y_val, -0.498_531_4f32 * z_val),
    );
    let green =
        (-0.969_266f32).mul_add(x_val, (1.876_010_8f32).mul_add(y_val, 0.041_556f32 * z_val));
    let blue = (0.055_643_4f32).mul_add(
        x_val,
        (-0.204_025_9f32).mul_add(y_val, 1.057_225_2f32 * z_val),
    );
    (red, green, blue)
}
fn f32_from_f64_checked(value: f64) -> Result<f32> {
    if !value.is_finite() {
        return Err(anyhow!("数值不是有限值: {value}"));
    }
    let min = f64::from(f32::MIN);
    let max = f64::from(f32::MAX);
    if value < min || value > max {
        return Err(anyhow!("数值超出 f32 范围: {value}"));
    }
    let parsed: f32 = value
        .to_string()
        .parse()
        .map_err(|err| anyhow!("解析 f32 失败: {err}"))?;
    if !parsed.is_finite() {
        return Err(anyhow!("转换后数值不是有限值: {parsed}"));
    }
    Ok(parsed)
}
