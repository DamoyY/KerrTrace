pub(crate) mod scene;
mod validation;
use alloc::string::String;
use anyhow::{Context as _, Result};
use core::num::NonZeroU32;
use scene::KernelConfig;
use serde::Deserialize;
use std::{fs, path::Path};
use validation::validate;
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub window: WindowConfig,
    pub camera: CameraConfig,
    pub controls: ControlsConfig,
    pub renderer: RendererConfig,
    pub bloom: BloomConfig,
    pub hud: HudConfig,
    pub blackbody: BlackbodyConfig,
    pub cuda: CudaConfig,
    pub kernel: KernelConfig,
    pub disk_noise: DiskNoiseConfig,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowConfig {
    pub width: u32,
    pub height: u32,
    pub frame_limit: Option<NonZeroU32>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraConfig {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
    pub pitch_limit: [f32; 2],
    pub fov_limit: [f32; 2],
    pub zoom_speed: f32,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlsConfig {
    pub move_speed: f32,
    pub sprint_multiplier: f32,
    pub mouse_sensitivity: f32,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RendererConfig {
    pub block_dim: [u32; 2],
    pub save_first_frame: bool,
    pub first_frame_path: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BloomConfig {
    pub enabled: bool,
    pub intensity: f32,
    pub radius: f32,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HudConfig {
    pub font_size: u32,
    pub margin: [u32; 2],
    pub color: [u8; 4],
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlackbodyConfig {
    pub lut_size: usize,
    pub lut_max_temp: f32,
    pub wavelength_start: f32,
    pub wavelength_end: f32,
    pub wavelength_step: f32,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaConfig {
    pub use_fast_math: bool,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiskNoiseConfig {
    pub enabled: bool,
    pub scale: f32,
    pub strength: f32,
    pub winding: f32,
    pub detail: u32,
}
pub fn load(path: &Path) -> Result<Config> {
    let source =
        fs::read_to_string(path).with_context(|| format!("无法读取配置文件 {}", path.display()))?;
    parse(&source)
}
pub fn parse(source: &str) -> Result<Config> {
    let config = serde_saphyr::from_str::<Config>(source).context("解析配置文件失败")?;
    validate(&config)?;
    Ok(config)
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        validate(self)
    }
}
pub(crate) mod numbers;
