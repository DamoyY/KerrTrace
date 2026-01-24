use std::fmt::Display;

use anyhow::{Result, anyhow};
use log::error;

use super::init::KerrParams;
use crate::{
    Config, KernelConfig,
    math::{ensure_finite_f32, f32_from_f64_with_context},
};
fn calc_isco(a_norm: f32, prograde: bool, mass: f32) -> f32 {
    let aa = a_norm * a_norm;
    let z1 = (1.0 - aa)
        .cbrt()
        .mul_add((1.0 + a_norm).cbrt() + (1.0 - a_norm).cbrt(), 1.0);
    let z2 = (3.0f32).mul_add(aa, z1 * z1).sqrt();
    let sign = if prograde { -1.0 } else { 1.0 };
    let term_inside = (3.0 - z1) * (2.0f32.mul_add(z2, 3.0 + z1));
    mass * (3.0 + z2 + sign * term_inside.max(0.0).sqrt())
}
fn calc_novikov_thorne_factor(r: f32, a_norm: f32, r_isco: f32, inv_m: f32) -> f32 {
    if r <= r_isco {
        return 0.0;
    }
    let r_norm = r * inv_m;
    let r_isco_norm = r_isco * inv_m;
    let x = r_norm.sqrt();
    let x_ms = r_isco_norm.sqrt();
    let angle_base = (-a_norm).acos() * (1.0 / 3.0);
    let ang_step = 2.094_395_2_f32;
    let roots = [
        2.0 * angle_base.cos(),
        2.0 * (angle_base - ang_step).cos(),
        2.0 * (angle_base + ang_step).cos(),
    ];
    let mut sum_log = 0.0f32;
    for i in 0..3 {
        let xi = roots[i];
        let denom = xi * (xi - roots[(i + 1) % 3]) * (xi - roots[(i + 2) % 3]);
        if denom <= 0.0 {
            continue;
        }
        let coef = 3.0 * (xi - a_norm) * (xi - a_norm) / denom;
        let val = (x - xi) / (x_ms - xi);
        if val > 0.0 {
            sum_log += coef * val.ln();
        }
    }
    let q = (1.5 * a_norm).mul_add(-(x / x_ms).ln(), x - x_ms) - sum_log;
    let geometric_denom = r_norm * (2.0f32).mul_add(a_norm, r_norm.mul_add(x, -(3.0 * x)));
    (q / geometric_denom).max(0.0)
}
pub(super) fn build_kerr_params(config: &Config) -> Result<KerrParams> {
    let window_width = config.window.width;
    if window_width == 0 {
        return Err(anyhow!("窗口宽度不能为 0"));
    }
    let window_height = config.window.height;
    if window_height == 0 {
        return Err(anyhow!("窗口高度不能为 0"));
    }
    let spin = config.kernel.black_hole.spin;
    let mass = config.kernel.black_hole.mass;
    if !spin.is_finite() {
        return Err(anyhow!("黑洞自旋不是有限值: {spin}"));
    }
    if !mass.is_finite() {
        return Err(anyhow!("黑洞质量不是有限值: {mass}"));
    }
    if mass == 0.0 {
        return Err(anyhow!("黑洞质量不能为 0"));
    }
    let inv_m = 1.0 / mass;
    let aa = spin * spin;
    let a_norm = spin * inv_m;
    let rh = mass + mass.mul_add(mass, -aa).max(0.0).sqrt();
    let disk_inner = calc_isco(a_norm, true, mass);
    if !rh.is_finite() || !disk_inner.is_finite() {
        return Err(anyhow!("Kerr 参数计算产生了非有限值"));
    }
    let noise = &config.disk_noise;
    if !noise.scale.is_finite() {
        return Err(anyhow!("吸积盘噪声缩放不是有限值: {}", noise.scale));
    }
    if !noise.strength.is_finite() {
        return Err(anyhow!("吸积盘噪声强度不是有限值: {}", noise.strength));
    }
    if !noise.winding.is_finite() {
        return Err(anyhow!("吸积盘噪声缠绕系数不是有限值: {}", noise.winding));
    }
    let detail = i32::try_from(noise.detail)
        .map_err(|_| anyhow!("吸积盘噪声层数超出 i32 范围: {}", noise.detail))?;
    let noise_enabled = i32::from(noise.enabled);
    let width_f64 = f64::from(window_width);
    let height_f64 = f64::from(window_height);
    let aspect_ratio = f32_from_f64_with_context(width_f64 / height_f64, "窗口纵横比")?;
    Ok(KerrParams {
        a: spin,
        m: mass,
        aa,
        inv_m,
        a_norm,
        rh,
        disk_inner,
        inv_w_2: f32_from_f64_with_context(2.0 / width_f64, "窗口宽度倒数")?,
        inv_h_2: f32_from_f64_with_context(2.0 / height_f64, "窗口高度倒数")?,
        aspect_ratio,
        disk_noise_scale: noise.scale,
        disk_noise_strength: noise.strength,
        disk_noise_winding: noise.winding,
        disk_noise_enabled: noise_enabled,
        disk_noise_detail: detail,
    })
}
fn push_define(lines: &mut Vec<String>, key: &str, value: impl Display) {
    lines.push(format!("#define {key} {value}"));
}
fn push_define_f32(lines: &mut Vec<String>, key: &str, value: f32) {
    lines.push(format!("#define {key} {value:.10}"));
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
fn lut_denom_u32(size: usize, label: &str) -> Result<u32> {
    let denom_usize = size - 1;
    u32::try_from(denom_usize).map_err(|_| anyhow!("{label}尺寸超出 u32 范围: {size}"))
}
fn ratio_from_index(i: usize, denom: u32, label: &str) -> Result<f64> {
    let i_u32 = u32::try_from(i).map_err(|_| anyhow!("{label}索引超出 u32 范围: {i}"))?;
    Ok(f64::from(i_u32) / f64::from(denom))
}
pub(super) fn generate_disk_temperature_lut(
    params: &KerrParams,
    disk_outer: f32,
    size: usize,
) -> Result<Vec<f32>> {
    if size < 2 {
        error!("吸积盘温度表尺寸过小: {size}");
        return Err(anyhow!("吸积盘温度表尺寸必须至少为 2"));
    }
    let disk_outer = ensure_finite_f32(disk_outer, "吸积盘外半径")?;
    let disk_inner = ensure_finite_f32(params.disk_inner, "吸积盘内半径")?;
    if disk_outer <= disk_inner {
        return Err(anyhow!(
            "吸积盘外半径必须大于内半径: outer={disk_outer}, inner={disk_inner}"
        ));
    }
    let denom_u32 = lut_denom_u32(size, "吸积盘温度表")?;
    let mut data = Vec::with_capacity(size);
    let inner = f64::from(disk_inner);
    let span = f64::from(disk_outer - disk_inner);
    for i in 0..size {
        let ratio = ratio_from_index(i, denom_u32, "吸积盘温度表")?;
        let r = f32_from_f64_with_context(ratio.mul_add(span, inner), "吸积盘半径")?;
        let f = calc_novikov_thorne_factor(r, params.a_norm, disk_inner, params.inv_m);
        let f = ensure_finite_f32(f, "吸积盘通量因子")?;
        let v = f.sqrt().sqrt();
        data.push(ensure_finite_f32(v, "吸积盘温度因子")?);
    }
    Ok(data)
}
