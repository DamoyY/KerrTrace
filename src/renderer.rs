use std::{fs, path::Path, sync::Arc};
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
        let width = u_width as i32;
        let height = u_height as i32;
        let context = CudaContext::new(0).context("Failed to initialize CUDA context")?;
        let stream = context.default_stream();
        let (lut_cpu, lut_max_temp) = generate_blackbody_lut(
            config.blackbody.lut_size,
            config.blackbody.lut_max_temp,
            config.blackbody.wavelength_start,
            config.blackbody.wavelength_end,
            config.blackbody.wavelength_step,
        );
        let lut_size = config.blackbody.lut_size as i32;
        let lut = stream
            .clone_htod(&lut_cpu)
            .context("Failed to copy LUT to GPU")?;
        let kernel_source =
            fs::read_to_string(cuda_dir.join("kernel.cu")).context("Failed to read kernel.cu")?;
        let defines = build_cuda_defines(&config.kernel);
        let full_source = format!("{}\n{}", defines, kernel_source);
        let ptx_opts = CompileOptions {
            include_paths: vec![cuda_dir.to_string_lossy().to_string()],
            use_fast_math: Some(config.cuda.use_fast_math),
            name: Some("kernel".to_string()),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(&full_source, ptx_opts).context("Failed to compile PTX")?;
        let module = context
            .load_module(ptx)
            .context("Failed to load PTX module")?;
        let kernel = module
            .load_function("kernel")
            .context("Failed to get kernel function")?;
        let image_gpu = stream
            .alloc_zeros::<u8>((u_width * u_height * 4) as usize)
            .context("Failed to allocate image buffer")?;
        let block_x = config.renderer.block_dim[0];
        let block_y = config.renderer.block_dim[1];
        let grid_x = (u_width + block_x - 1) / block_x;
        let grid_y = (u_height + block_y - 1) / block_y;
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
        let width = self.width as i32;
        let height = self.height as i32;
        let cam_x = cam_pos[0];
        let cam_y = cam_pos[1];
        let cam_z = cam_pos[2];
        let fwd_x = fwd[0];
        let fwd_y = fwd[1];
        let fwd_z = fwd[2];
        let rgt_x = rgt[0];
        let rgt_y = rgt[1];
        let rgt_z = rgt[2];
        let up_x = up[0];
        let up_y = up[1];
        let up_z = up[2];
        let lut_size = self.lut_size;
        let lut_max_temp = self.lut_max_temp;
        unsafe {
            self.stream
                .launch_builder(&self.kernel)
                .arg(&mut self.image_gpu)
                .arg(&width)
                .arg(&height)
                .arg(&cam_x)
                .arg(&cam_y)
                .arg(&cam_z)
                .arg(&fwd_x)
                .arg(&fwd_y)
                .arg(&fwd_z)
                .arg(&rgt_x)
                .arg(&rgt_y)
                .arg(&rgt_z)
                .arg(&up_x)
                .arg(&up_y)
                .arg(&up_z)
                .arg(&self.lut)
                .arg(&lut_size)
                .arg(&lut_max_temp)
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
fn build_cuda_defines(config: &KernelConfig) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "#define CONFIG_SSAA_SAMPLES {}",
        config.ssaa_samples
    ));
    lines.push(format!(
        "#define CONFIG_EXPOSURE_SCALE {:.10}",
        config.exposure_scale
    ));
    lines.push(format!(
        "#define CONFIG_SKY_GRID_DIVISIONS {}",
        config.sky.grid_divisions
    ));
    lines.push(format!(
        "#define CONFIG_SKY_LINE_THICKNESS {:.10}",
        config.sky.line_thickness
    ));
    lines.push(format!(
        "#define CONFIG_SKY_INTENSITY {:.10}",
        config.sky.intensity
    ));
    lines.push(format!(
        "#define CONFIG_BH_SPIN {:.10}",
        config.black_hole.spin
    ));
    lines.push(format!(
        "#define CONFIG_BH_MASS {:.10}",
        config.black_hole.mass
    ));
    lines.push(format!(
        "#define CONFIG_DISK_OUTER_RADIUS {:.10}",
        config.disk.outer_radius
    ));
    lines.push(format!(
        "#define CONFIG_DISK_TEMPERATURE_SCALE {:.10}",
        config.disk.temperature_scale
    ));
    lines.push(format!(
        "#define CONFIG_INTEGRATOR_INITIAL_STEP {:.10}",
        config.integrator.initial_step
    ));
    lines.push(format!(
        "#define CONFIG_INTEGRATOR_TOLERANCE {:.10}",
        config.integrator.tolerance
    ));
    lines.push(format!(
        "#define CONFIG_INTEGRATOR_MAX_STEPS {}",
        config.integrator.max_steps
    ));
    lines.push(format!(
        "#define CONFIG_INTEGRATOR_MAX_ATTEMPTS {}",
        config.integrator.max_attempts
    ));
    lines.push(format!(
        "#define CONFIG_TRANSMITTANCE_CUTOFF {:.10}",
        config.integrator.transmittance_cutoff
    ));
    lines.push(format!(
        "#define CONFIG_HORIZON_EPSILON {:.10}",
        config.integrator.horizon_epsilon
    ));
    lines.push(format!(
        "#define CONFIG_ESCAPE_RADIUS {:.10}",
        config.integrator.escape_radius
    ));
    lines.join("\n") + "\n"
}
