#ifndef KERR_CUH
#define KERR_CUH
struct KerrParams
{
    float a, M, aa, inv_M, A_norm, rh, disk_inner;
};
__constant__ KerrParams c_params;
struct RayState
{
    float r, theta, phi, pr, pth, pph;
};
struct RayDerivs
{
    float dr, dth, dph, dpr, dpth;
};
__device__ __forceinline__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float length(float3 a) { return __fsqrt_rn(dot(a, a)); }
__device__ __forceinline__ float3 normalize(float3 a)
{
    float inv = rsqrtf(dot(a, a));
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}
__device__ __forceinline__ float fast_cbrtf(float x)
{
    return __powf(x, 1.0f / 3.0f);
}
__device__ __forceinline__ float calc_isco(float A_norm, bool prograde, float M)
{
    float aa = A_norm * A_norm;
    float Z1 = 1.0f + fast_cbrtf(1.0f - aa) * (fast_cbrtf(1.0f + A_norm) + fast_cbrtf(1.0f - A_norm));
    float Z2 = __fsqrt_rn(3.0f * aa + Z1 * Z1);
    float sign = prograde ? -1.0f : 1.0f;
    float term_inside = (3.0f - Z1) * (3.0f + Z1 + 2.0f * Z2);
    return M * (3.0f + Z2 + sign * __fsqrt_rn(fmaxf(term_inside, 0.0f)));
}
__device__ __forceinline__ RayDerivs get_derivs(const RayState &s)
{
    const KerrParams &p = c_params;
    float sin_th, cos_th;
    __sincosf(s.theta, &sin_th, &cos_th);
    float s2 = sin_th * sin_th;
    float Sigma = s.r * s.r + p.aa * cos_th * cos_th;
    float Delta = s.r * s.r - 2.0f * p.M * s.r + p.aa;
    float inv_Sigma = __fdividef(1.0f, Sigma);
    float inv_Delta = __fdividef(1.0f, Delta);
    float K = (s.r * s.r + p.aa) - p.a * s.pph;
    RayDerivs d;
    d.dr = inv_Sigma * Delta * s.pr;
    d.dth = inv_Sigma * s.pth;
    d.dph = inv_Sigma * ((p.a * K * inv_Delta) + __fdividef(s.pph, s2) - p.a);
    d.dpr = -inv_Sigma * ((s.r - p.M) * s.pr * s.pr + (K * inv_Delta) * ((s.r - p.M) * K * inv_Delta - 2.0f * s.r));
    d.dpth = inv_Sigma * sin_th * cos_th * (__fdividef(s.pph * s.pph, s2 * s2) - p.aa);
    return d;
}
__device__ __forceinline__ float3 get_sky_color(float3 dir, float transmittance)
{
    const float PI = 3.14159265f;
    float theta_sky = atan2f(dir.z, dir.x);
    float phi_lat_sky = asinf(fminf(fmaxf(dir.y, -1.0f), 1.0f));
    float grid_spacing = PI / (float)CONFIG_SKY_GRID_DIVISIONS;
    float line_thickness = CONFIG_SKY_LINE_THICKNESS;
    float d_theta = fmodf(fabsf(theta_sky), grid_spacing);
    float d_phi = fmodf(fabsf(phi_lat_sky), grid_spacing);
    bool is_grid = (d_theta < line_thickness || d_theta > (grid_spacing - line_thickness)) ||
                   (d_phi < line_thickness || d_phi > (grid_spacing - line_thickness)) ||
                   (fabsf(phi_lat_sky) < line_thickness * 3.0f);

    float intensity = is_grid ? CONFIG_SKY_INTENSITY * transmittance : 0.0f;
    return make_float3(intensity, intensity, intensity);
}
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
    float r_init = length(cam_pos);
    float th_init = acosf(__fdividef(cam_pos.y, r_init));
    float ph_init = atan2f(cam_pos.x, cam_pos.z);
    float st, ct, sp, cp;
    __sincosf(th_init, &st, &ct);
    __sincosf(ph_init, &sp, &cp);
    float3 e_r = make_float3(st * sp, ct, st * cp);
    float3 e_th = make_float3(ct * sp, -st, ct * cp);
    float3 e_ph = make_float3(cp, 0.0f, -sp);
    float v_r_loc = dot(ray_n, e_r);
    float v_th_loc = dot(ray_n, e_th);
    float v_ph_loc = dot(ray_n, e_ph);
    RayState s;
    s.r = r_init;
    s.theta = th_init;
    s.phi = ph_init;
    float s2 = st * st;
    float r2 = s.r * s.r;
    float Sigma = r2 + p.aa * ct * ct;
    float Delta = r2 - 2.0f * p.M * s.r + p.aa;
    float r2_a2 = r2 + p.aa;
    float A = r2_a2 * r2_a2 - p.aa * Delta * s2;
    float inv_A = __fdividef(1.0f, A);
    float alpha = __fsqrt_rn(Sigma * Delta * inv_A);
    float omega = (2.0f * p.M * s.r * p.a) * inv_A;
    float sqrt_Sigma = __fsqrt_rn(Sigma);
    float sqrt_Delta = __fsqrt_rn(Delta);
    float sqrt_A = __fsqrt_rn(A);
    float pr_cov = __fdividef(sqrt_Sigma, sqrt_Delta) * v_r_loc;
    float pth_cov = sqrt_Sigma * v_th_loc;
    float pph_cov = __fdividef(sqrt_A, sqrt_Sigma) * st * v_ph_loc;
    float E = alpha + omega * pph_cov;
    float inv_E = __fdividef(1.0f, E);
    s.pr = pr_cov * inv_E;
    s.pth = pth_cov * inv_E;
    s.pph = pph_cov * inv_E;
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
            k1 = get_derivs(s);
            k2 = get_derivs({s.r + h * a21 * k1.dr, s.theta + h * a21 * k1.dth, 0, s.pr + h * a21 * k1.dpr, s.pth + h * a21 * k1.dpth, s.pph});
            k3 = get_derivs({s.r + h * (a31 * k1.dr + a32 * k2.dr), s.theta + h * (a31 * k1.dth + a32 * k2.dth), 0, s.pr + h * (a31 * k1.dpr + a32 * k2.dpr), s.pth + h * (a31 * k1.dpth + a32 * k2.dpth), s.pph});
            k4 = get_derivs({s.r + h * (a41 * k1.dr + a42 * k2.dr + a43 * k3.dr), s.theta + h * (a41 * k1.dth + a42 * k2.dth + a43 * k3.dth), 0, s.pr + h * (a41 * k1.dpr + a42 * k2.dpr + a43 * k3.dpr), s.pth + h * (a41 * k1.dpth + a42 * k2.dpth + a43 * k3.dpth), s.pph});
            k5 = get_derivs({s.r + h * (a51 * k1.dr + a52 * k2.dr + a53 * k3.dr + a54 * k4.dr), s.theta + h * (a51 * k1.dth + a52 * k2.dth + a53 * k3.dth + a54 * k4.dth), 0, s.pr + h * (a51 * k1.dpr + a52 * k2.dpr + a53 * k3.dpr + a54 * k4.dpr), s.pth + h * (a51 * k1.dpth + a52 * k2.dpth + a53 * k3.dpth + a54 * k4.dpth), s.pph});
            k6 = get_derivs({s.r + h * (a61 * k1.dr + a62 * k2.dr + a63 * k3.dr + a64 * k4.dr + a65 * k5.dr), s.theta + h * (a61 * k1.dth + a62 * k2.dth + a63 * k3.dth + a64 * k4.dth + a65 * k5.dth), 0, s.pr + h * (a61 * k1.dpr + a62 * k2.dpr + a63 * k3.dpr + a64 * k4.dpr + a65 * k5.dpr), s.pth + h * (a61 * k1.dpth + a62 * k2.dpth + a63 * k3.dpth + a64 * k4.dpth + a65 * k5.dpth), s.pph});
            next_s = {
                s.r + h * (b1 * k1.dr + b3 * k3.dr + b4 * k4.dr + b6 * k6.dr),
                s.theta + h * (b1 * k1.dth + b3 * k3.dth + b4 * k4.dth + b6 * k6.dth),
                s.phi + h * (b1 * k1.dph + b3 * k3.dph + b4 * k4.dph + b6 * k6.dph),
                s.pr + h * (b1 * k1.dpr + b3 * k3.dpr + b4 * k4.dpr + b6 * k6.dpr),
                s.pth + h * (b1 * k1.dpth + b3 * k3.dpth + b4 * k4.dpth + b6 * k6.dpth),
                s.pph};
            error = fmaxf(fmaxf(fabsf(h * (dc1 * k1.dr + dc3 * k3.dr + dc4 * k4.dr + dc5 * k5.dr + dc6 * k6.dr)) / (tol * fabsf(s.r)),
                                fabsf(h * (dc1 * k1.dth + dc3 * k3.dth + dc4 * k4.dth + dc5 * k5.dth + dc6 * k6.dth)) / (tol * fabsf(s.theta))),
                          fmaxf(fabsf(h * (dc1 * k1.dpr + dc3 * k3.dpr + dc4 * k4.dpr + dc5 * k5.dpr + dc6 * k6.dpr)) / (tol * fabsf(s.pr)),
                                fabsf(h * (dc1 * k1.dpth + dc3 * k3.dpth + dc4 * k4.dpth + dc5 * k5.dpth + dc6 * k6.dpth)) / (tol * fabsf(s.pth))));
            if (error <= 1.0f)
                accepted = true;
            h = h * 0.9f * __powf(error, -0.2f);
            attempts++;
        }
        float prev_r = s.r, prev_th = s.theta;
        s = next_s;
        if (s.r < p.rh + CONFIG_HORIZON_EPSILON)
            break;
        if ((prev_th < 1.570796327f && s.theta >= 1.570796327f) || (prev_th > 1.570796327f && s.theta <= 1.570796327f))
        {
            RayDerivs ke = get_derivs(s);
            float t = (1.570796327f - prev_th) / (s.theta - prev_th);
            float t2 = t * t, t3 = t2 * t;
            float r_hit = (2 * t3 - 3 * t2 + 1) * prev_r + (t3 - 2 * t2 + t) * (k1.dr * h) + (-2 * t3 + 3 * t2) * s.r + (t3 - t2) * (ke.dr * h);
            if (r_hit >= disk_inner && r_hit <= disk_outer)
            {
                float sqrt_M = __fsqrt_rn(p.M), r_sqrt = __fsqrt_rn(r_hit);
                float disc_denom = 1.0f - 3.0f * p.M / r_hit + 2.0f * p.a * sqrt_M / (r_hit * r_sqrt);
                if (disc_denom > 0.0f)
                {
                    float g = 1.0f / (rsqrtf(disc_denom) * (1.0f - (-sqrt_M / (r_hit * r_sqrt + p.a * sqrt_M)) * s.pph));
                    float denom = disk_outer - disk_inner;
                    float u = denom > 0.0f ? (r_hit - disk_inner) / denom : 0.0f;
                    float sampled = tex1D<float>(disk_tex, u);
                    float T = CONFIG_DISK_TEMPERATURE_SCALE * sampled * g;
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
        if (s.r > CONFIG_ESCAPE_RADIUS)
        {
            float sn, cn, spn, cpn;
            __sincosf(s.theta, &sn, &cn);
            __sincosf(s.phi, &spn, &cpn);
            float f = 1.0f - 2.0f * p.M / s.r;
            float3 ed = normalize(make_float3(f * s.pr * sn * spn + s.pth / s.r * cn * spn + s.pph / (s.r * sn) * cpn,
                                              f * s.pr * cn - s.pth / s.r * sn,
                                              f * s.pr * sn * cpn + s.pth / s.r * cn * cpn - s.pph / (s.r * sn) * spn));
            color.x += get_sky_color(ed, transmittance).x;
            color.y += get_sky_color(ed, transmittance).y;
            color.z += get_sky_color(ed, transmittance).z;
            break;
        }
    }
    return color;
}
#endif
