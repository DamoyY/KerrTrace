use crate::config::scene::KernelConfig;
use core::fmt::Display;
fn push_define(lines: &mut Vec<String>, key: &str, value: impl Display) {
    lines.push(format!("#define {key} {value}"));
}
fn push_define_f32(lines: &mut Vec<String>, key: &str, value: f32) {
    lines.push(format!("#define {key} {value:.9e}f"));
}
pub(super) fn build_cuda_defines(config: &KernelConfig, wavelength_step: f32) -> String {
    let mut lines = Vec::with_capacity(15);
    let ints = [
        ("CONFIG_SPP", config.spp),
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
        ("CONFIG_BLACKBODY_WAVELENGTH_STEP", wavelength_step),
    ];
    for (key, value) in floats {
        push_define_f32(&mut lines, key, value);
    }
    let mut output = lines.join("\n");
    output.push('\n');
    output
}
