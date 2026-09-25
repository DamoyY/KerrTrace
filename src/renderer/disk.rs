use super::pipeline::KerrParams;
use crate::{
    config::Config,
    config::numbers::{ensure_finite_f32, f32_from_f64_with_context},
};
use anyhow::{Result, anyhow, ensure};
use log::error;
fn calc_isco(a_norm: f32, mass: f32) -> Result<f32> {
    let spin = f64::from(a_norm.abs());
    let aa = spin * spin;
    let z1 = (1.0 - aa)
        .cbrt()
        .mul_add((1.0 + spin).cbrt() + (1.0 - spin).cbrt(), 1.0);
    let z2 = 3.0_f64.mul_add(aa, z1 * z1).sqrt();
    let term_inside = (3.0_f64 - z1) * 2.0_f64.mul_add(z2, 3.0_f64 + z1);
    f32_from_f64_with_context(
        f64::from(mass) * (3.0_f64 + z2 - term_inside.sqrt()),
        "同向 ISCO",
    )
}
fn calc_novikov_thorne_factor(r: f32, a_norm: f32, r_isco: f32, inv_m: f32) -> Result<f32> {
    if r <= r_isco {
        return Ok(0.0);
    }
    let spin = f64::from(a_norm.abs());
    let r_norm = f64::from(r) * f64::from(inv_m);
    let r_isco_norm = f64::from(r_isco) * f64::from(inv_m);
    let x = r_norm.sqrt();
    let x_ms = r_isco_norm.sqrt();
    let angle_base = (-spin).acos() / 3.0_f64;
    let ang_step = core::f64::consts::TAU / 3.0_f64;
    let roots = [
        2.0_f64 * angle_base.cos(),
        2.0_f64 * (angle_base - ang_step).cos(),
        2.0_f64 * (angle_base + ang_step).cos(),
    ];
    let mut sum_log = 0.0_f64;
    for xi in roots {
        let coef = 0.25_f64 * xi * xi.mul_add(xi, -1.0_f64);
        let log_ratio = ((x - x_ms) / (x_ms - xi)).ln_1p();
        sum_log = coef.mul_add(log_ratio, sum_log);
    }
    let q = (-1.5_f64 * spin).mul_add(((x - x_ms) / x_ms).ln_1p(), x - x_ms) - sum_log;
    let geometric_denom =
        r_norm * r_norm * 2.0_f64.mul_add(spin, r_norm.mul_add(x, -(3.0_f64 * x)));
    ensure!(
        q >= 0.0_f64 && geometric_denom > 0.0_f64,
        "薄盘通量无效: Q={q}, denominator={geometric_denom}"
    );
    f32_from_f64_with_context(q / geometric_denom, "吸积盘通量因子")
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
    ensure!(
        mass > 0.0_f32 && spin.abs() < mass,
        "黑洞参数应满足 mass > |spin|"
    );
    let inv_m = 1.0 / mass;
    let aa = spin * spin;
    let a_norm = spin * inv_m;
    let rh = mass + mass.mul_add(mass, -aa).max(0.0).sqrt();
    let disk_inner = calc_isco(a_norm, mass)?;
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
        .map_err(|error| anyhow!("吸积盘噪声层数超出 i32 范围: {} ({error})", noise.detail))?;
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
fn lut_denom_u32(size: usize, label: &str) -> Result<u32> {
    let denom_usize = size
        .checked_sub(1)
        .ok_or_else(|| anyhow!("{label}尺寸必须至少为 2"))?;
    u32::try_from(denom_usize)
        .map_err(|error| anyhow!("{label}尺寸超出 u32 范围: {size} ({error})"))
}
fn ratio_from_index(i: usize, denom: u32, label: &str) -> Result<f64> {
    let i_u32 =
        u32::try_from(i).map_err(|error| anyhow!("{label}索引超出 u32 范围: {i} ({error})"))?;
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
    let disk_outer_value = ensure_finite_f32(disk_outer, "吸积盘外半径")?;
    let disk_inner = ensure_finite_f32(params.disk_inner, "吸积盘内半径")?;
    if disk_outer_value <= disk_inner {
        return Err(anyhow!(
            "吸积盘外半径必须大于内半径: outer={disk_outer_value}, inner={disk_inner}"
        ));
    }
    let denom_u32 = lut_denom_u32(size, "吸积盘温度表")?;
    let mut data = Vec::with_capacity(size);
    let inner = f64::from(disk_inner);
    let span = f64::from(disk_outer_value - disk_inner);
    for i in 0..size {
        let ratio = ratio_from_index(i, denom_u32, "吸积盘温度表")?;
        let r = f32_from_f64_with_context(ratio.mul_add(span, inner), "吸积盘半径")?;
        let flux_factor = calc_novikov_thorne_factor(r, params.a_norm, disk_inner, params.inv_m)?;
        let finite_flux_factor = ensure_finite_f32(flux_factor, "吸积盘通量因子")?;
        let v = finite_flux_factor.sqrt().sqrt();
        data.push(ensure_finite_f32(v, "吸积盘温度因子")?);
    }
    Ok(data)
}
