#pragma once
#include "emission.cuh"
#include "geometry.cuh"
#include "interpolation.cuh"
__device__ void accumulate_disk(const RayState &previous, const RayState &s, const RayDerivs &k1, const RayDerivs &k3,
                                const RayDerivs &k4, const RayDerivs &k5, const RayDerivs &k6, const RayDerivs &k7,
                                float h_used, float kt, float pt, cudaTextureObject_t lut_tex, int lut_size,
                                float max_temp, cudaTextureObject_t disk_tex, float disk_inner, float disk_outer,
                                unsigned int *error_flag, float3 &color, float &transmittance)
{
    const KerrParams &p = c_params;
    float prev_x = previous.x, prev_y = previous.y, prev_z = previous.z;
    float prev_px = previous.px, prev_py = previous.py, prev_pz = previous.pz;
    if ((prev_y < 0.0f && s.y >= 0.0f) || (prev_y > 0.0f && s.y <= 0.0f))
    {
        float t_low = 0.0f;
        float t_high = 1.0f;
        float y_low = prev_y;
        float y_high = s.y;
        float denom = y_low - y_high;
        float t = denom != 0.0f ? __fdividef(y_low, denom) : 0.5f;
        t = fminf(fmaxf(t, t_low), t_high);
        for (int j = 0; j < 4; j++)
        {
            float y_t = dp_dense_output(prev_y, s.y, k1.dy, k3.dy, k4.dy, k5.dy, k6.dy, k7.dy, h_used, t);
            if (y_t == 0.0f)
            {
                t_low = t;
                t_high = t;
                break;
            }
            if ((y_t >= 0.0f) == (y_low >= 0.0f))
            {
                t_low = t;
                y_low = y_t;
            }
            else
            {
                t_high = t;
                y_high = y_t;
            }
            float dy_dt = dp_dense_output_derivative(prev_y, s.y, k1.dy, k3.dy, k4.dy, k5.dy, k6.dy, k7.dy, h_used, t);
            t = dy_dt != 0.0f ? t - __fdividef(y_t, dy_dt) : 0.5f * (t_low + t_high);
            t = fminf(fmaxf(t, t_low), t_high);
        }
        float x_hit = dp_dense_output(prev_x, s.x, k1.dx, k3.dx, k4.dx, k5.dx, k6.dx, k7.dx, h_used, t);
        float z_hit = dp_dense_output(prev_z, s.z, k1.dz, k3.dz, k4.dz, k5.dz, k6.dz, k7.dz, h_used, t);
        float px_hit = dp_dense_output(prev_px, s.px, k1.dpx, k3.dpx, k4.dpx, k5.dpx, k6.dpx, k7.dpx, h_used, t);
        float py_hit = dp_dense_output(prev_py, s.py, k1.dpy, k3.dpy, k4.dpy, k5.dpy, k6.dpy, k7.dpy, h_used, t);
        float pz_hit = dp_dense_output(prev_pz, s.pz, k1.dpz, k3.dpz, k4.dpz, k5.dpz, k6.dpz, k7.dpz, h_used, t);
        float r_hit = ks_r_from_xyz(x_hit, 0.0f, z_hit, p.aa);
        if (r_hit >= disk_inner && r_hit <= disk_outer)
        {
            float sqrt_M = __fsqrt_rn(p.M), r_sqrt = __fsqrt_rn(r_hit);
            float disc_denom = 1.0f - 3.0f * p.M / r_hit + 2.0f * p.a * sqrt_M / (r_hit * r_sqrt);
            if (disc_denom > 0.0f)
            {
                float omega = -sqrt_M / (r_hit * r_sqrt + p.a * sqrt_M);
                float r2_hit = r_hit * r_hit;
                float denom_hit = r2_hit + p.aa;
                float inv_denom = __fdividef(1.0f, denom_hit);
                float lx = (r_hit * x_hit - p.a * z_hit) * inv_denom;
                float ly = 0.0f;
                float lz = (r_hit * z_hit + p.a * x_hit) * inv_denom;
                float ldotp = fmaf(lx, px_hit, fmaf(ly, py_hit, lz * pz_hit));
                float ldotxi = lx * z_hit - lz * x_hit;
                float denomH = r2_hit * r2_hit;
                float H = p.M * r_hit * r2_hit * __fdividef(1.0f, denomH);
                float L = (z_hit * px_hit - x_hit * pz_hit) + 2.0f * H * ldotxi * (ldotp + kt);
                float energy = -pt;
                float l_over_e = __fdividef(L, energy);
                float g = 1.0f / (rsqrtf(disc_denom) * (1.0f - omega * l_over_e));
                float denom = disk_outer - disk_inner;
                float u = denom > 0.0f ? (r_hit - disk_inner) / denom : 0.0f;
                float sampled = tex1D<float>(disk_tex, u);
                float T = CONFIG_DISK_TEMPERATURE_SCALE * sampled * g;
                if (p.disk_noise_enabled && p.disk_noise_detail > 0 && p.disk_noise_strength != 0.0f)
                {
                    float sin_th, cos_th, phi_hit;
                    ks_bl_from_xyz(x_hit, 0.0f, z_hit, p, r_hit, sin_th, cos_th, phi_hit);
                    float r3 = r_hit * r_hit * r_hit;
                    float omega = 1.0f / (p.a + __fsqrt_rn(r3 * p.inv_M));
                    float spiral_phase = phi_hit + p.disk_noise_winding * omega;
                    float s_spiral, c_spiral;
                    __sincosf(spiral_phase, &s_spiral, &c_spiral);
                    float nu = r_hit * c_spiral * p.disk_noise_scale;
                    float nv = r_hit * s_spiral * p.disk_noise_scale;
                    float noise_val = fbm(nu, nv, p.disk_noise_detail);
                    float modulator = 1.0f + p.disk_noise_strength * (noise_val - 0.5f);
                    T *= modulator;
                }
                if (isfinite(T) && T > 0.0f)
                {
                    float3 d_col = fetch_color_from_lut(T, lut_tex, lut_size, max_temp, error_flag);
                    float luma = 0.2126f * d_col.x + 0.7152f * d_col.y + 0.0722f * d_col.z;
                    float luma_scaled = luma / CONFIG_BLACKBODY_WAVELENGTH_STEP;
                    float alpha = 1.0f - __expf(-luma_scaled * 0.5f);
                    color.x += d_col.x * alpha * transmittance;
                    color.y += d_col.y * alpha * transmittance;
                    color.z += d_col.z * alpha * transmittance;
                    transmittance *= (1.0f - alpha);
                }
            }
        }
    }
}
