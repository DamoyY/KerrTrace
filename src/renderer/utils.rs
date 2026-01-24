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
    Ok(KerrParams {
        a: spin,
        m: mass,
        aa,
        inv_m,
        a_norm,
        rh,
        disk_inner,
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
pub(super) fn generate_blackbody_lut(
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
    let wavelength_step = ensure_finite_f32(wavelength_step, "波长步进").map_err(|_| {
        error!("波长步进无效: {wavelength_step}");
        anyhow!("波长步进必须为正数且有限")
    })?;
    if wavelength_step <= 0.0 {
        error!("波长步进无效: {wavelength_step}");
        return Err(anyhow!("波长步进必须为正数且有限"));
    }
    let mut lut_data = Vec::with_capacity(size * 4);
    let denom_u32 = lut_denom_u32(size, "黑体查找表")?;
    let temps: Vec<f32> = (0..size)
        .map(|i| {
            let ratio = ratio_from_index(i, denom_u32, "黑体查找表")?;
            let temp = ratio * f64::from(max_temp);
            f32_from_f64_with_context(temp, "温度值")
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
        x_sum *= wavelength_step;
        y_sum *= wavelength_step;
        z_sum *= wavelength_step;
        let (r, g, b) = xyz_to_rgb(x_sum, y_sum, z_sum);
        lut_data.push(r);
        lut_data.push(g);
        lut_data.push(b);
        lut_data.push(0.0);
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
    if t <= 0.0 {
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
