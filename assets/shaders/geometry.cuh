#pragma once
struct KerrParams
{
    float a, M, aa, inv_M, A_norm, rh, disk_inner;
    float inv_w_2, inv_h_2, aspect_ratio;
    float disk_noise_scale, disk_noise_strength, disk_noise_winding;
    int disk_noise_enabled, disk_noise_detail;
};
__constant__ KerrParams c_params;
struct RayState
{
    float x, y, z, px, py, pz;
};
struct RayDerivs
{
    float dx, dy, dz, dpx, dpy, dpz;
};
__device__ __forceinline__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 normalize(float3 a)
{
    float inv = rsqrtf(dot(a, a));
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}
__device__ __forceinline__ unsigned int pcg32(unsigned long long state)
{
    unsigned long long x = state * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned int xorshifted = (unsigned int)(((x >> 18u) ^ x) >> 27u);
    unsigned int rot = (unsigned int)(x >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31));
}
__device__ __forceinline__ float rand01(unsigned long long state)
{
    return (float)pcg32(state) * 2.3283064365386963e-10f;
}
__device__ __forceinline__ float ks_r_from_xyz(float x, float y, float z, float a2)
{
    float rho2 = fmaf(x, x, fmaf(y, y, z * z));
    float u = rho2 - a2;
    float s_sq = fmaf(u, u, 4.0f * a2 * y * y);
    float inv_s = rsqrtf(s_sq);
    float s = s_sq * inv_s;
    float r2 = 0.5f * (u + s);
    return __fsqrt_rn(r2);
}
__device__ __forceinline__ void ks_bl_from_xyz(float x, float y, float z, const KerrParams &p, float &r, float &sin_th,
                                               float &cos_th, float &phi)
{
    float a2 = p.aa;
    float rho2 = fmaf(x, x, fmaf(y, y, z * z));
    float u = rho2 - a2;
    float s_sq = fmaf(u, u, 4.0f * a2 * y * y);
    float inv_s = rsqrtf(s_sq);
    float s = s_sq * inv_s;
    float r2 = 0.5f * (u + s);
    float inv_r = rsqrtf(r2);
    r = r2 * inv_r;
    cos_th = y * inv_r;
    float cos2 = cos_th * cos_th;
    float sin_sq = 1.0f - cos2;
    float inv_sin = rsqrtf(sin_sq);
    sin_th = sin_sq * inv_sin;
    float denom = r2 + a2;
    float inv = inv_sin * __fdividef(1.0f, denom);
    float cos_phi = (r * z - p.a * x) * inv;
    float sin_phi = (r * x + p.a * z) * inv;
    phi = atan2f(sin_phi, cos_phi);
}
__device__ __forceinline__ RayDerivs get_derivs(const RayState &s, float pt)
{
    const KerrParams &p = c_params;
    float x = s.x;
    float y = s.y;
    float z = s.z;
    float a2 = p.aa;
    float rho2 = fmaf(x, x, fmaf(y, y, z * z));
    float u = rho2 - a2;
    float s_sq = fmaf(u, u, 4.0f * a2 * y * y);
    float inv_s = rsqrtf(s_sq);
    float r2 = 0.5f * (u + s_sq * inv_s);
    float inv_r = rsqrtf(r2);
    float r = r2 * inv_r;
    float dr2_dx = x * (1.0f + u * inv_s);
    float dr2_dy = y * (1.0f + (u + 2.0f * a2) * inv_s);
    float dr2_dz = z * (1.0f + u * inv_s);
    float inv_2r = 0.5f * inv_r;
    float drdx = dr2_dx * inv_2r;
    float drdy = dr2_dy * inv_2r;
    float drdz = dr2_dz * inv_2r;
    float denom = r2 + a2;
    float inv_denom = __fdividef(1.0f, denom);
    float inv_denom2 = inv_denom * inv_denom;
    float lx = (r * x + p.a * z) * inv_denom;
    float ly = y * inv_r;
    float lz = (r * z - p.a * x) * inv_denom;
    float r3 = r2 * r;
    float denomH = fmaf(r2, r2, a2 * y * y);
    float inv_denomH = __fdividef(1.0f, denomH);
    float H = p.M * r3 * inv_denomH;
    float lp = fmaf(lx, s.px, fmaf(ly, s.py, fmaf(lz, s.pz, pt)));
    RayDerivs d;
    float common_H_lp = 2.0f * H * lp;
    d.dx = s.px - common_H_lp * lx;
    d.dy = s.py - common_H_lp * ly;
    d.dz = s.pz - common_H_lp * lz;
    float dD_dx = 2.0f * r * drdx;
    float dD_dy = 2.0f * r * drdy;
    float dD_dz = 2.0f * r * drdz;
    float Nx = r * x + p.a * z;
    float Nz = r * z - p.a * x;
    float dNx_dx = drdx * x + r;
    float dNx_dy = drdy * x;
    float dNx_dz = drdz * x + p.a;
    float dNz_dx = drdx * z - p.a;
    float dNz_dy = drdy * z;
    float dNz_dz = drdz * z + r;
    float dlx_dx = (dNx_dx * denom - Nx * dD_dx) * inv_denom2;
    float dlx_dy = (dNx_dy * denom - Nx * dD_dy) * inv_denom2;
    float dlx_dz = (dNx_dz * denom - Nx * dD_dz) * inv_denom2;
    float dlz_dx = (dNz_dx * denom - Nz * dD_dx) * inv_denom2;
    float dlz_dy = (dNz_dy * denom - Nz * dD_dy) * inv_denom2;
    float dlz_dz = (dNz_dz * denom - Nz * dD_dz) * inv_denom2;
    float inv_r2 = inv_r * inv_r;
    float dly_dx = -y * drdx * inv_r2;
    float dly_dy = (r - y * drdy) * inv_r2;
    float dly_dz = -y * drdz * inv_r2;
    float dDenH_dx = 4.0f * r3 * drdx;
    float dDenH_dy = fmaf(4.0f * r3, drdy, 2.0f * a2 * y);
    float dDenH_dz = 4.0f * r3 * drdz;
    float inv_denomH2 = inv_denomH * inv_denomH;
    float factor_H = p.M * inv_denomH2;
    float dH_dx = factor_H * (3.0f * r2 * drdx * denomH - r3 * dDenH_dx);
    float dH_dy = factor_H * (3.0f * r2 * drdy * denomH - r3 * dDenH_dy);
    float dH_dz = factor_H * (3.0f * r2 * drdz * denomH - r3 * dDenH_dz);
    float dlp_dx = dlx_dx * s.px + dly_dx * s.py + dlz_dx * s.pz;
    float dlp_dy = dlx_dy * s.px + dly_dy * s.py + dlz_dy * s.pz;
    float dlp_dz = dlx_dz * s.px + dly_dz * s.py + dlz_dz * s.pz;
    float lp2 = lp * lp;
    d.dpx = dH_dx * lp2 + common_H_lp * dlp_dx;
    d.dpy = dH_dy * lp2 + common_H_lp * dlp_dy;
    d.dpz = dH_dz * lp2 + common_H_lp * dlp_dz;
    return d;
}
