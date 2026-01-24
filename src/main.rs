mod app;
mod hud;
mod math;
mod renderer;
use std::{fmt, fs::File, path::Path};

use anyhow::{Context, Result};
use app::App;
use serde::{
    Deserialize,
    de::{self, Deserializer, Visitor},
};
use winit::event_loop::{ControlFlow, EventLoop};
#[derive(Debug, Deserialize, Clone)]
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
    pub save_first_frame: bool,
    pub first_frame_path: String,
}
#[derive(Debug, Deserialize, Clone)]
pub struct BloomConfig {
    pub enabled: bool,
    pub intensity: f32,
    pub radius: f32,
}
#[derive(Debug, Deserialize, Clone)]
pub struct HudConfig {
    pub font_size: u32,
    pub margin: [u32; 2],
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
pub struct DiskNoiseConfig {
    pub enabled: bool,
    pub scale: f32,
    pub strength: f32,
    pub winding: f32,
    pub detail: u32,
}
#[derive(Debug, Deserialize, Clone)]
pub struct KernelConfig {
    pub spp: u32,
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
    #[serde(deserialize_with = "deserialize_u32_scientific")]
    pub max_steps: u32,
    pub max_attempts: u32,
    pub transmittance_cutoff: f32,
    pub horizon_epsilon: f32,
    pub escape_radius: f32,
}
fn deserialize_u32_scientific<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: Deserializer<'de>,
{
    struct U32Visitor;
    impl Visitor<'_> for U32Visitor {
        type Value = u32;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a non-negative integer")
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            u32::try_from(value).map_err(|_| E::custom("integer out of range for u32"))
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            if value < 0 {
                return Err(E::custom("integer must be non-negative"));
            }
            u32::try_from(value).map_err(|_| E::custom("integer out of range for u32"))
        }

        fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            if !value.is_finite() {
                return Err(E::custom("number must be finite"));
            }
            if value.fract() != 0.0 {
                return Err(E::custom("number must be an integer"));
            }
            if value < 0.0 || value > f64::from(u32::MAX) {
                return Err(E::custom("number out of range for u32"));
            }
            value
                .to_string()
                .parse::<u32>()
                .map_err(|_| E::custom("number out of range for u32"))
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            let parsed: f64 = value
                .parse()
                .map_err(|_| E::custom("failed to parse number"))?;
            self.visit_f64(parsed)
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            self.visit_str(&value)
        }
    }
    deserializer.deserialize_any(U32Visitor)
}
fn load_config<P: AsRef<Path>>(path: P) -> Result<Config> {
    let file = File::open(path).context("Failed to open config file")?;
    let config: Config = serde_yaml::from_reader(file).context("Failed to parse config file")?;
    Ok(config)
}
fn main() -> Result<()> {
    env_logger::init();
    let config = load_config("config.yaml")?;
    let event_loop = EventLoop::new()?;
    let control_flow = if config.window.vsync {
        ControlFlow::Wait
    } else {
        ControlFlow::Poll
    };
    event_loop.set_control_flow(control_flow);
    let mut app = App::new(config);
    event_loop.run_app(&mut app)?;
    Ok(())
}
