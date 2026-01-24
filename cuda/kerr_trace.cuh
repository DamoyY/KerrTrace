#ifndef KERR_TRACE_CUH
#define KERR_TRACE_CUH
#include "kerr_math.cuh"
__device__ float3 trace_ray(
    float3 cam_pos,
    float3 ray_dir,
    cudaTextureObject_t lut_tex,
    int lut_size,
    float max_temp,
    cudaTextureObject_t disk_tex,
    float disk_inner,
    float disk_outer,
    unsigned int *error_flag)
{
    const KerrParams &p = c_params;
    float3 ray_n = normalize(ray_dir);
    RayState s;
    s.x = cam_pos.x;
    s.y = cam_pos.y;
    s.z = cam_pos.z;
    float a2 = p.aa;
    float rho2 = fmaf(s.x, s.x, fmaf(s.y, s.y, s.z * s.z));
    float u = rho2 - a2;
    float s_term = __fsqrt_rn(fmaf(u, u, 4.0f * a2 * s.y * s.y));
    float r2 = 0.5f * (u + s_term);
    float r = __fsqrt_rn(r2);
    float denom = r2 + a2;
    float inv_denom = __fdividef(1.0f, denom);
    float lx = (r * s.x - p.a * s.z) * inv_denom;
    float ly = __fdividef(s.y, r);
    float lz = (r * s.z + p.a * s.x) * inv_denom;
    float denomH = fmaf(r2, r2, a2 * s.y * s.y);
    float H = p.M * r2 * r * __fdividef(1.0f, denomH);
    float ldotn = fmaf(lx, ray_n.x, fmaf(ly, ray_n.y, lz * ray_n.z));
    float g_tt = -1.0f + 2.0f * H;
    float g_tn = 2.0f * H * ldotn;
    float g_nn = 1.0f + 2.0f * H * ldotn * ldotn;
    float disc = g_tn * g_tn - g_tt * g_nn;
    float kt = (-g_tn - __fsqrt_rn(disc)) / g_tt;
    float pt = g_tt * kt + g_tn;
    float scale = kt + ldotn;
    s.px = ray_n.x + 2.0f * H * lx * scale;
    s.py = ray_n.y + 2.0f * H * ly * scale;
    s.pz = ray_n.z + 2.0f * H * lz * scale;
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float transmittance = 1.0f;
    float h = CONFIG_INTEGRATOR_INITIAL_STEP;
    const float tol = CONFIG_INTEGRATOR_TOLERANCE;
    static const float a21 = 0.2f;
    static const float a31 = 3.0f / 40.0f, a32 = 9.0f / 40.0f;
    static const float a41 = 0.3f, a42 = -0.9f, a43 = 1.2f;
    static const float a51 = -11.0f / 54.0f, a52 = 2.5f, a53 = -70.0f / 27.0f, a54 = 35.0f / 27.0f;
    static const float a61 = 1631.0f / 55296.0f, a62 = 175.0f / 512.0f, a63 = 575.0f / 13824.0f, a64 = 44275.0f / 110592.0f, a65 = 253.0f / 4096.0f;
    static const float b1 = 37.0f / 378.0f, b3 = 250.0f / 621.0f, b4 = 125.0f / 594.0f, b6 = 512.0f / 1771.0f;
    static const float dc1 = 37.0f / 378.0f - 2825.0f / 27648.0f, dc3 = 250.0f / 621.0f - 18575.0f / 48384.0f, dc4 = 125.0f / 594.0f - 13525.0f / 55296.0f, dc5 = -277.0f / 14336.0f, dc6 = 512.0f / 1771.0f - 0.25f;
    for (int i = 0; i < CONFIG_INTEGRATOR_MAX_STEPS && transmittance > CONFIG_TRANSMITTANCE_CUTOFF; i++)
    {
        RayDerivs k1, k2, k3, k4, k5, k6;
        RayState next_s;
        float error;
        int attempts = 0;
        bool accepted = false;
        while (!accepted && attempts < CONFIG_INTEGRATOR_MAX_ATTEMPTS)
        {
            k1 = get_derivs(s, pt);
            k2 = get_derivs({s.x + h * a21 * k1.dx, s.y + h * a21 * k1.dy, s.z + h * a21 * k1.dz, s.px + h * a21 * k1.dpx, s.py + h * a21 * k1.dpy, s.pz + h * a21 * k1.dpz}, pt);
            k3 = get_derivs({s.x + h * (a31 * k1.dx + a32 * k2.dx), s.y + h * (a31 * k1.dy + a32 * k2.dy), s.z + h * (a31 * k1.dz + a32 * k2.dz), s.px + h * (a31 * k1.dpx + a32 * k2.dpx), s.py + h * (a31 * k1.dpy + a32 * k2.dpy), s.pz + h * (a31 * k1.dpz + a32 * k2.dpz)}, pt);
            k4 = get_derivs({s.x + h * (a41 * k1.dx + a42 * k2.dx + a43 * k3.dx), s.y + h * (a41 * k1.dy + a42 * k2.dy + a43 * k3.dy), s.z + h * (a41 * k1.dz + a42 * k2.dz + a43 * k3.dz), s.px + h * (a41 * k1.dpx + a42 * k2.dpx + a43 * k3.dpx), s.py + h * (a41 * k1.dpy + a42 * k2.dpy + a43 * k3.dpy), s.pz + h * (a41 * k1.dpz + a42 * k2.dpz + a43 * k3.dpz)}, pt);
            k5 = get_derivs({s.x + h * (a51 * k1.dx + a52 * k2.dx + a53 * k3.dx + a54 * k4.dx), s.y + h * (a51 * k1.dy + a52 * k2.dy + a53 * k3.dy + a54 * k4.dy), s.z + h * (a51 * k1.dz + a52 * k2.dz + a53 * k3.dz + a54 * k4.dz), s.px + h * (a51 * k1.dpx + a52 * k2.dpx + a53 * k3.dpx + a54 * k4.dpx), s.py + h * (a51 * k1.dpy + a52 * k2.dpy + a53 * k3.dpy + a54 * k4.dpy), s.pz + h * (a51 * k1.dpz + a52 * k2.dpz + a53 * k3.dpz + a54 * k4.dpz)}, pt);
            k6 = get_derivs({s.x + h * (a61 * k1.dx + a62 * k2.dx + a63 * k3.dx + a64 * k4.dx + a65 * k5.dx), s.y + h * (a61 * k1.dy + a62 * k2.dy + a63 * k3.dy + a64 * k4.dy + a65 * k5.dy), s.z + h * (a61 * k1.dz + a62 * k2.dz + a63 * k3.dz + a64 * k4.dz + a65 * k5.dz), s.px + h * (a61 * k1.dpx + a62 * k2.dpx + a63 * k3.dpx + a64 * k4.dpx + a65 * k5.dpx), s.py + h * (a61 * k1.dpy + a62 * k2.dpy + a63 * k3.dpy + a64 * k4.dpy + a65 * k5.dpy), s.pz + h * (a61 * k1.dpz + a62 * k2.dpz + a63 * k3.dpz + a64 * k4.dpz + a65 * k5.dpz)}, pt);
            next_s = {
                s.x + h * (b1 * k1.dx + b3 * k3.dx + b4 * k4.dx + b6 * k6.dx),
                s.y + h * (b1 * k1.dy + b3 * k3.dy + b4 * k4.dy + b6 * k6.dy),
                s.z + h * (b1 * k1.dz + b3 * k3.dz + b4 * k4.dz + b6 * k6.dz),
                s.px + h * (b1 * k1.dpx + b3 * k3.dpx + b4 * k4.dpx + b6 * k6.dpx),
                s.py + h * (b1 * k1.dpy + b3 * k3.dpy + b4 * k4.dpy + b6 * k6.dpy),
                s.pz + h * (b1 * k1.dpz + b3 * k3.dpz + b4 * k4.dpz + b6 * k6.dpz)};
            float scale_x = tol * fmaxf(fabsf(s.x), 1.0f);
            float scale_y = tol * fmaxf(fabsf(s.y), 1.0f);
            float scale_z = tol * fmaxf(fabsf(s.z), 1.0f);
            float scale_px = tol * fmaxf(fabsf(s.px), 1.0f);
            float scale_py = tol * fmaxf(fabsf(s.py), 1.0f);
            float scale_pz = tol * fmaxf(fabsf(s.pz), 1.0f);
            float err_x = fabsf(h * (dc1 * k1.dx + dc3 * k3.dx + dc4 * k4.dx + dc5 * k5.dx + dc6 * k6.dx)) / scale_x;
            float err_y = fabsf(h * (dc1 * k1.dy + dc3 * k3.dy + dc4 * k4.dy + dc5 * k5.dy + dc6 * k6.dy)) / scale_y;
            float err_z = fabsf(h * (dc1 * k1.dz + dc3 * k3.dz + dc4 * k4.dz + dc5 * k5.dz + dc6 * k6.dz)) / scale_z;
            float err_px = fabsf(h * (dc1 * k1.dpx + dc3 * k3.dpx + dc4 * k4.dpx + dc5 * k5.dpx + dc6 * k6.dpx)) / scale_px;
            float err_py = fabsf(h * (dc1 * k1.dpy + dc3 * k3.dpy + dc4 * k4.dpy + dc5 * k5.dpy + dc6 * k6.dpy)) / scale_py;
            float err_pz = fabsf(h * (dc1 * k1.dpz + dc3 * k3.dpz + dc4 * k4.dpz + dc5 * k5.dpz + dc6 * k6.dpz)) / scale_pz;
            error = fmaxf(fmaxf(fmaxf(err_x, err_y), fmaxf(err_z, err_px)), fmaxf(err_py, err_pz));
            if (error <= 1.0f)
                accepted = true;
            h = h * 0.9f * __powf(error, -0.2f);
            attempts++;
        }
        float prev_x = s.x, prev_y = s.y, prev_z = s.z;
        float prev_px = s.px, prev_py = s.py, prev_pz = s.pz;
        s = next_s;
        float r_now = ks_r_from_xyz(s.x, s.y, s.z, p.aa);
        if (r_now < p.rh + CONFIG_HORIZON_EPSILON)
            break;
        if ((prev_y < 0.0f && s.y >= 0.0f) || (prev_y > 0.0f && s.y <= 0.0f))
        {
            RayDerivs ke = get_derivs(s, pt);
            float denom_y = s.y - prev_y;
            float t = denom_y != 0.0f ? __fdividef(-prev_y, denom_y) : 0.0f;
            float x_hit = lerp_hermite(prev_x, k1.dx * h, s.x, ke.dx * h, t);
            float z_hit = lerp_hermite(prev_z, k1.dz * h, s.z, ke.dz * h, t);
            float px_hit = lerp_hermite(prev_px, k1.dpx * h, s.px, ke.dpx * h, t);
            float py_hit = lerp_hermite(prev_py, k1.dpy * h, s.py, ke.dpy * h, t);
            float pz_hit = lerp_hermite(prev_pz, k1.dpz * h, s.pz, ke.dpz * h, t);
            float sin_th, cos_th, phi_hit, r_hit;
            ks_bl_from_xyz(x_hit, 0.0f, z_hit, p, r_hit, sin_th, cos_th, phi_hit);
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
        if (r_now > CONFIG_ESCAPE_RADIUS)
        {
            RayDerivs ke = get_derivs(s, pt);
            float3 ed = normalize(make_float3(ke.dx, ke.dy, ke.dz));
            color.x += get_sky_color(ed, transmittance).x;
            color.y += get_sky_color(ed, transmittance).y;
            color.z += get_sky_color(ed, transmittance).z;
            break;
        }
    }
    return color;
}
#endif
