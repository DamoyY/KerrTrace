use super::disk::{build_kerr_params, generate_disk_temperature_lut};
use crate::config;
use anyhow::{Result, ensure};
fn reference_flux(radius: f64, spin: f64, inner: f64) -> f64 {
    let start = inner.sqrt();
    let end = radius.sqrt();
    let step = (end - start) / 4096.0_f64;
    let integrand = |x: f64| {
        (8.0_f64 * spin)
            .mul_add(x, (-6.0_f64 * x).mul_add(x, x.powi(4)))
            .mul_add(1.0_f64, -3.0_f64 * spin * spin)
            / (x * 2.0_f64.mul_add(spin, (-3.0_f64).mul_add(x, x.powi(3))))
    };
    let mut integral = integrand(start) + integrand(end);
    for index in 1_u32..4096 {
        let weight = if index.is_multiple_of(2) {
            2.0_f64
        } else {
            4.0_f64
        };
        integral = weight.mul_add(integrand(f64::from(index).mul_add(step, start)), integral);
    }
    integral * step
        / (3.0_f64 * radius * radius * 2.0_f64.mul_add(spin, (-3.0_f64).mul_add(end, end.powi(3))))
}
#[test]
fn corotating_flux_matches_independent_quadrature() -> Result<()> {
    let mut settings = config::parse(include_str!("../../assets/settings.yaml"))?;
    for spin in [
        0.0_f32,
        0.00001_f32,
        0.45_f32,
        -0.45_f32,
        0.4999_f32,
        -0.4999_f32,
    ] {
        settings.kernel.black_hole.spin = spin;
        let params = build_kerr_params(&settings)?;
        for radius in [10.0_f32, 30.0_f32, 100.0_f32] {
            let data = generate_disk_temperature_lut(&params, radius * params.m, 2)?;
            let &[edge, temperature] = data.as_slice() else {
                anyhow::bail!("温度表尺寸错误");
            };
            ensure!(edge.abs() < f32::EPSILON, "ISCO 通量不为零");
            let expected = reference_flux(
                f64::from(radius),
                f64::from(params.a_norm.abs()),
                f64::from(params.disk_inner) / f64::from(params.m),
            );
            let actual = f64::from(temperature).powi(4);
            ensure!(
                (actual / expected - 1.0_f64).abs() < 2e-5_f64,
                "通量不匹配: spin={spin}, r/M={radius}, actual={actual}, expected={expected}"
            );
        }
    }
    Ok(())
}
#[test]
fn isco_and_far_field_follow_corotating_disk() -> Result<()> {
    let mut settings = config::parse(include_str!("../../assets/settings.yaml"))?;
    for spin in [-0.45_f32, 0.45_f32] {
        settings.kernel.black_hole.spin = spin;
        let params = build_kerr_params(&settings)?;
        ensure!(
            (params.disk_inner / params.m - 2.320_883_f32).abs() < 2e-5_f32,
            "同向 ISCO 不正确"
        );
        let mut fluxes = Vec::new();
        for radius in [1e6_f32, 2e6_f32] {
            let data = generate_disk_temperature_lut(&params, radius * params.m, 2)?;
            let Some(&temperature) = data.last() else {
                anyhow::bail!("温度表为空");
            };
            fluxes.push(f64::from(temperature).powi(4));
        }
        let &[near, far] = fluxes.as_slice() else {
            anyhow::bail!("通量数量错误");
        };
        ensure!((near / far - 8.0_f64).abs() < 0.02_f64, "远场通量不是 r^-3");
    }
    Ok(())
}
