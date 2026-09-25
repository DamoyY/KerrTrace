#pragma once
#include "geometry.cuh"
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
