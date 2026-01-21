use std::{fs::File, path::Path};

use anyhow::{Context, Result};
use serde::Deserialize;
#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub window: WindowConfig,
    pub camera: CameraConfig,
    pub controls: ControlsConfig,
    pub renderer: RendererConfig,
    pub hud: HudConfig,
    pub blackbody: BlackbodyConfig,
    pub cuda: CudaConfig,
    pub kernel: KernelConfig,
}
#[derive(Debug, Deserialize, Clone)]
pub struct WindowConfig {
    pub width: u32,
    pub height: u32,
    pub vsync: bool,
}
#[derive(Debug, Deserialize, Clone)]
pub struct CameraConfig {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
    pub pitch_limit: [f32; 2],
    pub fov_limit: [f32; 2],
    pub zoom_speed: f32,
}
#[derive(Debug, Deserialize, Clone)]
pub struct ControlsConfig {
    pub move_speed: f32,
    pub sprint_multiplier: f32,
    pub mouse_sensitivity: f32,
}
#[derive(Debug, Deserialize, Clone)]
pub struct RendererConfig {
    pub block_dim: [u32; 2],
    pub position_epsilon: f32,
    pub spin: f32,
    pub save_first_frame: bool,
    pub first_frame_path: String,
}
#[derive(Debug, Deserialize, Clone)]
pub struct HudConfig {
    pub font_name: String,
    pub font_size: u32,
    pub margin: [u32; 2],
    pub anchor_x: String,
    pub anchor_y: String,
    pub width: u32,
    pub color: [u8; 4],
}
#[derive(Debug, Deserialize, Clone)]
pub struct BlackbodyConfig {
    pub lut_size: usize,
    pub lut_max_temp: f32,
    pub wavelength_start: f32,
    pub wavelength_end: f32,
    pub wavelength_step: f32,
}
#[derive(Debug, Deserialize, Clone)]
pub struct CudaConfig {
    pub use_fast_math: bool,
}
#[derive(Debug, Deserialize, Clone)]
pub struct KernelConfig {
    pub ssaa_samples: u32,
    pub exposure_scale: f32,
    pub sky: SkyConfig,
    pub black_hole: BlackHoleConfig,
    pub disk: DiskConfig,
    pub integrator: IntegratorConfig,
}
#[derive(Debug, Deserialize, Clone)]
pub struct SkyConfig {
    pub grid_divisions: u32,
    pub line_thickness: f32,
    pub intensity: f32,
}
#[derive(Debug, Deserialize, Clone)]
pub struct BlackHoleConfig {
    pub spin: f32,
    pub mass: f32,
}
#[derive(Debug, Deserialize, Clone)]
pub struct DiskConfig {
    pub outer_radius: f32,
    pub temperature_scale: f32,
}
#[derive(Debug, Deserialize, Clone)]
pub struct IntegratorConfig {
    pub initial_step: f32,
    pub tolerance: f32,
    pub max_steps: u32,
    pub max_attempts: u32,
    pub transmittance_cutoff: f32,
    pub horizon_epsilon: f32,
    pub escape_radius: f32,
}
pub fn load_config<P: AsRef<Path>>(path: P) -> Result<Config> {
    let file = File::open(path).context("Failed to open config file")?;
    let config: Config = serde_yaml::from_reader(file).context("Failed to parse config file")?;
    Ok(config)
}
