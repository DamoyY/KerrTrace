use super::{blackbody::generate_blackbody_lut, defines::build_cuda_defines};
use crate::config;
use anyhow::{Context as _, Result, ensure};
#[test]
fn shader_defines_preserve_small_positive_parameters() -> Result<()> {
    let mut settings = config::parse(include_str!("../../assets/settings.yaml"))?;
    settings.kernel.integrator.tolerance = 1e-12_f32;
    let defines = build_cuda_defines(&settings.kernel, settings.blackbody.wavelength_step);
    let definition = defines
        .lines()
        .find(|line| line.starts_with("#define CONFIG_INTEGRATOR_TOLERANCE "))
        .context("缺少容差宏")?;
    let literal = definition.split_whitespace().last().context("缺少宏值")?;
    let parsed: f32 = literal.trim_end_matches('f').parse()?;
    ensure!(
        (parsed / settings.kernel.integrator.tolerance - 1.0_f32).abs() < 1e-6_f32,
        "CUDA 宏丢失小数值精度: {literal}"
    );
    Ok(())
}
#[test]
fn blackbody_spectrum_is_finite_and_starts_at_zero() -> Result<()> {
    let settings = config::parse(include_str!("../../assets/settings.yaml"))?;
    let data = generate_blackbody_lut(&settings.blackbody)?;
    ensure!(
        data.len() == settings.blackbody.lut_size * 4,
        "LUT 尺寸不匹配"
    );
    ensure!(
        data.iter().all(|value| value.is_finite()),
        "LUT 包含非有限值"
    );
    ensure!(
        data.get(..4)
            .context("LUT 为空")?
            .iter()
            .all(|value| value.abs() < f32::EPSILON),
        "零温度不是零辐射"
    );
    ensure!(data.iter().any(|&value| value > 0.0_f32), "光谱全黑");
    Ok(())
}
