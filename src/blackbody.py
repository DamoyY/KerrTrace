import numpy as np
import cupy as cp


def generate_blackbody_lut(
    size,
    max_temp,
    wavelength_start,
    wavelength_end,
    wavelength_step,
):
    def gaussian(x, alpha, mu, sigma1, sigma2):
        sigma = np.where(x < mu, sigma1, sigma2)
        t = (x - mu) / sigma
        return alpha * np.exp(-0.5 * t * t)

    def get_xyz_sensitivity(lambda_nm):
        x = (
            gaussian(lambda_nm, 1.056, 599.8, 37.9, 31.0)
            + gaussian(lambda_nm, 0.362, 442.0, 16.0, 26.7)
            + gaussian(lambda_nm, -0.065, 501.1, 20.4, 26.2)
        )
        y = gaussian(lambda_nm, 0.821, 568.8, 46.9, 40.5) + gaussian(
            lambda_nm, 0.286, 530.9, 16.3, 31.1
        )
        z = gaussian(lambda_nm, 1.217, 437.0, 11.8, 36.0) + gaussian(
            lambda_nm, 0.681, 459.0, 26.2, 13.8
        )
        return x, y, z

    def planck_law(lambda_nm, T):
        c2 = 1.4388e7
        if T < 1e-8:
            return np.zeros_like(lambda_nm)
        exponent = c2 / (lambda_nm * T)
        overflow_mask = exponent > 80.0
        val = np.zeros_like(lambda_nm)
        safe_mask = ~overflow_mask
        if np.any(safe_mask):
            val[safe_mask] = (1.0 / (lambda_nm[safe_mask] ** 5)) / (
                np.exp(exponent[safe_mask]) - 1.0
            )
        return val * 1e15

    def xyz_to_rgb(X, Y, Z):
        r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
        g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
        b = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
        return r, g, b

    temps = np.linspace(0, max_temp, size, dtype=np.float32)
    lut_data = np.zeros((size, 3), dtype=np.float32)
    lambdas = np.arange(
        wavelength_start,
        wavelength_end + wavelength_step,
        wavelength_step,
        dtype=np.float32,
    )
    xs, ys, zs = get_xyz_sensitivity(lambdas)
    for i, T in enumerate(temps):
        if T == 0:
            continue
        intensities = planck_law(lambdas, T)
        X = np.sum(intensities * xs)
        Y = np.sum(intensities * ys)
        Z = np.sum(intensities * zs)
        r, g, b = xyz_to_rgb(X, Y, Z)
        lut_data[i] = [r, g, b]
    return cp.array(lut_data, dtype=cp.float32), max_temp
