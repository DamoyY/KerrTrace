use serde::Deserialize;
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelConfig {
    pub spp: u32,
    pub exposure_scale: f32,
    pub sky: SkyConfig,
    pub black_hole: BlackHoleConfig,
    pub disk: DiskConfig,
    pub integrator: IntegratorConfig,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkyConfig {
    pub grid_divisions: u32,
    pub line_thickness: f32,
    pub intensity: f32,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlackHoleConfig {
    pub spin: f32,
    pub mass: f32,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiskConfig {
    pub outer_radius: f32,
    pub temperature_scale: f32,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegratorConfig {
    pub initial_step: f32,
    pub tolerance: f32,
    pub max_steps: u32,
    pub max_attempts: u32,
    pub transmittance_cutoff: f32,
    pub horizon_epsilon: f32,
    pub escape_radius: f32,
}
