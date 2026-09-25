#ifndef GEODESICS_CUH
#define GEODESICS_CUH
#include "emission.cuh"
#include "geometry.cuh"
#include "integration.cuh"
#include "interpolation.cuh"
#include "radiation.cuh"
__device__ float3 trace_ray(float3 cam_pos, float3 ray_dir, cudaTextureObject_t lut_tex, int lut_size, float max_temp,
                            cudaTextureObject_t disk_tex, float disk_inner, float disk_outer, unsigned int *error_flag)
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
    const float base_tol = CONFIG_INTEGRATOR_TOLERANCE;
    const float tol_max_scale = 5.0f;
    const float tol_potential_near = 0.3f;
    const float tol_potential_far = 0.05f;
    RayDerivs fsal_k1;
    bool have_fsal = false;
    for (int i = 0; i < CONFIG_INTEGRATOR_MAX_STEPS && transmittance > CONFIG_TRANSMITTANCE_CUTOFF; i++)
    {
        RayDerivs k1, k3, k4, k5, k6, k7;
        RayState next_s;
        float error;
        float r_curr = ks_r_from_xyz(s.x, s.y, s.z, p.aa);
        float r2_curr = r_curr * r_curr;
        float denomH = fmaf(r2_curr, r2_curr, p.aa * s.y * s.y);
        float H = p.M * r_curr * r2_curr * __fdividef(1.0f, denomH);
        float t = __fdividef(H - tol_potential_far, tol_potential_near - tol_potential_far);
        t = fminf(fmaxf(t, 0.0f), 1.0f);
        float tol = base_tol * (tol_max_scale - tol_max_scale * t);
        int attempts = 0;
        bool accepted = false;
        float h_used = h;
        bool k1_ready = false;
        if (have_fsal)
        {
            k1 = fsal_k1;
            k1_ready = true;
        }
        while (!accepted && attempts < CONFIG_INTEGRATOR_MAX_ATTEMPTS)
        {
            float h_step = h;
            if (!k1_ready)
            {
                k1 = get_derivs(s, pt);
                k1_ready = true;
            }
            next_s = integrate_step(s, pt, h_step, tol, k1, k3, k4, k5, k6, k7, error);
            if (error <= 1.0f)
                accepted = true;
            h = h_step * 0.9f * __powf(error, -0.2f);
            h_used = h_step;
            attempts++;
        }
        RayState previous = s;
        s = next_s;
        if (accepted)
        {
            fsal_k1 = k7;
            have_fsal = true;
        }
        else
        {
            have_fsal = false;
        }
        float r_now = ks_r_from_xyz(s.x, s.y, s.z, p.aa);
        if (r_now < p.rh + CONFIG_HORIZON_EPSILON)
            break;
        accumulate_disk(previous, s, k1, k3, k4, k5, k6, k7, h_used, kt, pt, lut_tex, lut_size, max_temp, disk_tex,
                        disk_inner, disk_outer, error_flag, color, transmittance);
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
