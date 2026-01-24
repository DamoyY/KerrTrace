#ifndef KERR_MATH_CUH
#define KERR_MATH_CUH
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
__device__ __forceinline__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float length(float3 a) { return __fsqrt_rn(dot(a, a)); }
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
    return M * (3.0f + Z2 + sign * __fsqrt_rn(term_inside));
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
__device__ __forceinline__ void ks_bl_from_xyz(float x, float y, float z, const KerrParams &p, float &r, float &sin_th, float &cos_th, float &phi)
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
    float cos_phi = (r * z + p.a * x) * inv;
    float sin_phi = (r * x - p.a * z) * inv;
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
    float dr2_dy = y * (1.0f + (u + 4.0f * a2) * (0.5f * inv_s));
    float dr2_dz = z * (1.0f + u * inv_s);
    float inv_2r = 0.5f * inv_r;
    float drdx = dr2_dx * inv_2r;
    float drdy = dr2_dy * inv_2r;
    float drdz = dr2_dz * inv_2r;
    float denom = r2 + a2;
    float inv_denom = __fdividef(1.0f, denom);
    float inv_denom2 = inv_denom * inv_denom;
    float lx = (r * x - p.a * z) * inv_denom;
    float ly = y * inv_r;
    float lz = (r * z + p.a * x) * inv_denom;
    float r3 = r2 * r;
    float denomH = fmaf(r2, r2, a2 * y * y);
    float inv_denomH = __fdividef(1.0f, denomH);
    float H = p.M * r3 * inv_denomH;
    float lp = fmaf(lx, s.px, fmaf(ly, s.py, fmaf(lz, s.pz, -pt)));
    RayDerivs d;
    float common_H_lp = 2.0f * H * lp;
    d.dx = s.px - common_H_lp * lx;
    d.dy = s.py - common_H_lp * ly;
    d.dz = s.pz - common_H_lp * lz;
    float dD_dx = 2.0f * r * drdx;
    float dD_dy = 2.0f * r * drdy;
    float dD_dz = 2.0f * r * drdz;
    float Nx = r * x - p.a * z;
    float Nz = r * z + p.a * x;
    float dNx_dx = drdx * x + r;
    float dNx_dy = drdy * x;
    float dNx_dz = drdz * x - p.a;
    float dNz_dx = drdx * z + p.a;
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
__device__ __forceinline__ float3 get_sky_color(float3 dir, float transmittance)
{
    const float PI = 3.14159265f;
    float theta_sky = atan2f(dir.z, dir.x);
    float phi_lat_sky = asinf(dir.y);
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
__device__ __forceinline__ float fade(float t)
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}
__device__ __forceinline__ float lerp(float a, float b, float t)
{
    return fmaf(t, b - a, a);
}
__device__ __forceinline__ float lerp_hermite(float y0, float m0, float y1, float m1, float t)
{
    float t2 = t * t;
    float t3 = t2 * t;
    return (2.0f * t3 - 3.0f * t2 + 1.0f) * y0 + (t3 - 2.0f * t2 + t) * m0 + (-2.0f * t3 + 3.0f * t2) * y1 + (t3 - t2) * m1;
}
__device__ __forceinline__ float2 gradient_from_hash(unsigned int h)
{
    const float TWO_PI = 6.283185307f;
    float angle = (float)h * (TWO_PI * 2.3283064365386963e-10f);
    float s, c;
    __sincosf(angle, &s, &c);
    return make_float2(c, s);
}
__device__ __forceinline__ float2 gradient(int ix, int iy)
{
    unsigned int ux = (unsigned int)ix;
    unsigned int uy = (unsigned int)iy;
    unsigned long long state = ((unsigned long long)ux << 32) | (unsigned long long)uy;
    return gradient_from_hash(pcg32(state));
}
__device__ __forceinline__ float gradient_noise(float u, float v)
{
    int ix0 = __float2int_rd(u);
    int iy0 = __float2int_rd(v);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    float fx = u - (float)ix0;
    float fy = v - (float)iy0;
    float2 g00 = gradient(ix0, iy0);
    float2 g10 = gradient(ix1, iy0);
    float2 g01 = gradient(ix0, iy1);
    float2 g11 = gradient(ix1, iy1);
    float2 d00 = make_float2(fx, fy);
    float2 d10 = make_float2(fx - 1.0f, fy);
    float2 d01 = make_float2(fx, fy - 1.0f);
    float2 d11 = make_float2(fx - 1.0f, fy - 1.0f);
    float n00 = g00.x * d00.x + g00.y * d00.y;
    float n10 = g10.x * d10.x + g10.y * d10.y;
    float n01 = g01.x * d01.x + g01.y * d01.y;
    float n11 = g11.x * d11.x + g11.y * d11.y;
    float u_f = fade(fx);
    float v_f = fade(fy);
    float x1 = lerp(n00, n10, u_f);
    float x2 = lerp(n01, n11, u_f);
    return lerp(x1, x2, v_f);
}
__device__ __forceinline__ float fbm(float u, float v, int octaves)
{
    if (octaves <= 0)
    {
        return 0.5f;
    }
    float sum = 0.0f;
    float amp = 1.0f;
    float freq = 1.0f;
    float amp_sum = 0.0f;
    for (int i = 0; i < octaves; i++)
    {
        sum += gradient_noise(u * freq, v * freq) * amp;
        amp_sum += amp;
        amp *= 0.5f;
        freq *= 2.0f;
    }
    float val = amp_sum > 0.0f ? (sum / amp_sum) : 0.0f;
    val = val * 0.5f + 0.5f;
    return val;
}
#endif
