use anyhow::{Result, anyhow};
use log::error;

pub fn generate_blackbody_lut(
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
    if !wavelength_step.is_finite() || wavelength_step <= 0.0 {
        error!("波长步进无效: {wavelength_step}");
        return Err(anyhow!("波长步进必须为正数且有限"));
    }
    let mut lut_data = Vec::with_capacity(size * 3);
    let denom_usize = size - 1;
    let denom_u32 =
        u32::try_from(denom_usize).map_err(|_| anyhow!("黑体查找表尺寸超出 u32 范围: {size}"))?;
    let denom_f = f64::from(denom_u32);
    let temps: Vec<f32> = (0..size)
        .map(|i| {
            let i_u32 =
                u32::try_from(i).map_err(|_| anyhow!("黑体查找表索引超出 u32 范围: {i}"))?;
            let ratio = f64::from(i_u32) / denom_f;
            let temp = ratio * f64::from(max_temp);
            f32_from_f64_checked(temp).map_err(|err| anyhow!("温度值转换失败: {err}"))
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
        let (r, g, b) = xyz_to_rgb(x_sum, y_sum, z_sum);
        lut_data.push(r);
        lut_data.push(g);
        lut_data.push(b);
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
    if t < 1e-8 {
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

fn f32_from_f64_checked(value: f64) -> Result<f32> {
    if !value.is_finite() {
        return Err(anyhow!("数值不是有限值: {value}"));
    }
    let min = f64::from(f32::MIN);
    let max = f64::from(f32::MAX);
    if value < min || value > max {
        return Err(anyhow!("数值超出 f32 范围: {value}"));
    }
    let parsed: f32 = value
        .to_string()
        .parse()
        .map_err(|err| anyhow!("解析 f32 失败: {err}"))?;
    if !parsed.is_finite() {
        return Err(anyhow!("转换后数值不是有限值: {parsed}"));
    }
    Ok(parsed)
}
