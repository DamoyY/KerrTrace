use crate::{config::BlackbodyConfig, config::numbers::f32_from_f64};
use anyhow::{Context as _, Result, ensure};
use glam::{Mat3, Vec3};
use num_traits::ToPrimitive as _;
const XYZ_TO_RGB: Mat3 = Mat3::from_cols(
    Vec3::new(3.240_454_f32, -0.969_266_f32, 0.055_643_4_f32),
    Vec3::new(-1.537_138_5_f32, 1.876_010_8_f32, -0.204_025_9_f32),
    Vec3::new(-0.498_531_4_f32, 0.041_556_f32, 1.057_225_2_f32),
);
fn gaussian(wavelength: f32, alpha: f32, center: f32, left: f32, right: f32) -> f32 {
    let sigma = if wavelength < center { left } else { right };
    let offset = (wavelength - center) / sigma;
    alpha * (-0.5_f32 * offset * offset).exp()
}
fn sensitivity(wavelength: f32) -> Vec3 {
    Vec3::new(
        gaussian(wavelength, 1.056_f32, 599.8_f32, 37.9_f32, 31.0_f32)
            + gaussian(wavelength, 0.362_f32, 442.0_f32, 16.0_f32, 26.7_f32)
            + gaussian(wavelength, -0.065_f32, 501.1_f32, 20.4_f32, 26.2_f32),
        gaussian(wavelength, 0.821_f32, 568.8_f32, 46.9_f32, 40.5_f32)
            + gaussian(wavelength, 0.286_f32, 530.9_f32, 16.3_f32, 31.1_f32),
        gaussian(wavelength, 1.217_f32, 437.0_f32, 11.8_f32, 36.0_f32)
            + gaussian(wavelength, 0.681_f32, 459.0_f32, 26.2_f32, 13.8_f32),
    )
}
pub(super) fn generate_blackbody_lut(config: &BlackbodyConfig) -> Result<Vec<f32>> {
    let steps = ((f64::from(config.wavelength_end) - f64::from(config.wavelength_start))
        / f64::from(config.wavelength_step))
    .floor()
    .to_u32()
    .context("波长采样数量超出 u32")?;
    let samples = (0..=steps)
        .map(|index| {
            let wavelength = f32_from_f64(f64::from(index).mul_add(
                f64::from(config.wavelength_step),
                f64::from(config.wavelength_start),
            ))?;
            Ok((
                wavelength,
                sensitivity(wavelength),
                1e15_f32 / wavelength.powi(5),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let size = u32::try_from(config.lut_size).context("LUT 尺寸超出 u32")?;
    let denominator = size.checked_sub(1).context("LUT 尺寸过小")?;
    ensure!(denominator > 0, "LUT 尺寸至少为 2");
    let capacity = config.lut_size.checked_mul(4).context("LUT 容量溢出")?;
    let mut data = Vec::with_capacity(capacity);
    for index in 0..size {
        let temperature = f32_from_f64(
            f64::from(index) / f64::from(denominator) * f64::from(config.lut_max_temp),
        )?;
        let xyz = samples
            .iter()
            .fold(Vec3::ZERO, |sum, &(wavelength, response, scale)| {
                let exponent = 1.4388e7_f32 / (wavelength * temperature);
                if index == 0 || exponent > 80.0_f32 {
                    sum
                } else {
                    response.mul_add(Vec3::splat(scale / exponent.exp_m1()), sum)
                }
            });
        let rgb = XYZ_TO_RGB * (xyz * config.wavelength_step);
        ensure!(rgb.is_finite(), "黑体 LUT 产生非有限值: T={temperature}");
        data.extend_from_slice(&[rgb.x, rgb.y, rgb.z, 0.0_f32]);
    }
    Ok(data)
}
