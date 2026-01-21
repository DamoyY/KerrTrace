pub fn generate_blackbody_lut(
    size: usize,
    max_temp: f32,
    wavelength_start: f32,
    wavelength_end: f32,
    wavelength_step: f32,
) -> (Vec<f32>, f32) {
    let mut lut_data = Vec::with_capacity(size * 3);
    let temps: Vec<f32> = (0..size)
        .map(|i| (i as f32 / (size - 1) as f32) * max_temp)
        .collect();
    let mut lambdas = Vec::new();
    let mut l = wavelength_start;
    while l <= wavelength_end + 0.0001 {
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
    (lut_data, max_temp)
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
            let v = (1.0 / l.powi(5)) / (exponent.exp() - 1.0);
            val.push(v * 1e15);
        }
    }
    val
}
fn xyz_to_rgb(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let r = 3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z;
    let g = -0.969_266_0 * x + 1.876_010_8 * y + 0.041_556_0 * z;
    let b = 0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z;
    (r, g, b)
}
