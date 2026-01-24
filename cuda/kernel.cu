__device__ __forceinline__ float3 fetch_color_from_lut(float T, cudaTextureObject_t lut_tex, int lut_size, float max_temp)
{
    float t_norm = fminf(fmaxf(T, 0.0f), max_temp) / max_temp;
    float pos = t_norm * (float)(lut_size - 1);
    float u = (pos + 0.5f) / (float)lut_size;
    float4 c = tex1D<float4>(lut_tex, u);
    return make_float3(c.x, c.y, c.z);
}
__device__ __forceinline__ float3 aces_tone_map(float3 x)
{
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    float3 res;
    res.x = (x.x * (a * x.x + b)) / (x.x * (c * x.x + d) + e);
    res.y = (x.y * (a * x.y + b)) / (x.y * (c * x.y + d) + e);
    res.z = (x.z * (a * x.z + b)) / (x.z * (c * x.z + d) + e);
    return res;
}
__device__ __forceinline__ float srgb_oetf(float x)
{
    x = fmaxf(x, 0.0f);
    if (x <= 0.0031308f)
        return 12.92f * x;
    return 1.055f * powf(x, 0.4166666666666667f) - 0.055f;
}
__device__ __forceinline__ float3 srgb_oetf(float3 x)
{
    float3 res;
    res.x = srgb_oetf(x.x);
    res.y = srgb_oetf(x.y);
    res.z = srgb_oetf(x.z);
    return res;
}
__device__ unsigned char float_to_byte(float val)
{
    return (unsigned char)__float2int_rn(__saturatef(val) * 255.0f);
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
    return (float)pcg32(state) * (1.0f / 4294967296.0f);
}
#include "kerr.cuh"
extern "C"
{
    __global__ __launch_bounds__(1024) void kernel(
        unsigned int *__restrict__ image_out,
        int width, int height,
        float cam_x, float cam_y, float cam_z,
        float fwd_x, float fwd_y, float fwd_z,
        float rgt_x, float rgt_y, float rgt_z,
        float up_x, float up_y, float up_z,
        cudaTextureObject_t lut_tex, int lut_size, float max_temp,
        cudaTextureObject_t disk_tex, float disk_inner, float disk_outer,
        float fov_scale)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height)
            return;
        const int SPP = CONFIG_SPP;
        const float INV_SPP = 1.0f / (float)SPP;
        const float inv_w_2 = 2.0f / (float)width;
        const float inv_h_2 = 2.0f / (float)height;
        const float aspect_ratio = (float)width / (float)height;
        const float3 cam_pos = make_float3(cam_x, cam_y, cam_z);
        float3 accumulated_color = make_float3(0.0f, 0.0f, 0.0f);
        unsigned int pixel_index = (unsigned int)(y * width + x);
#pragma unroll
        for (int s = 0; s < SPP; s++)
        {
            unsigned long long base =
                (unsigned long long)(pixel_index) << 32 | (unsigned long long)(unsigned int)s;
            float rx = rand01(base);
            float ry = rand01(base ^ 0xda3e39cb94b95bdbULL);
            float u = ((x + rx) * inv_w_2 - 1.0f) * aspect_ratio * fov_scale;
            float v = (1.0f - (y + ry) * inv_h_2) * fov_scale;
            float3 ray_dir;
            ray_dir.x = u * rgt_x + v * up_x + fwd_x;
            ray_dir.y = u * rgt_y + v * up_y + fwd_y;
            ray_dir.z = u * rgt_z + v * up_z + fwd_z;
            float3 sample_color = trace_ray(
                cam_pos,
                ray_dir,
                lut_tex,
                lut_size,
                max_temp,
                disk_tex,
                disk_inner,
                disk_outer);
            accumulated_color.x += sample_color.x;
            accumulated_color.y += sample_color.y;
            accumulated_color.z += sample_color.z;
        }
        const float final_scale = CONFIG_EXPOSURE_SCALE * INV_SPP;
        accumulated_color.x *= final_scale;
        accumulated_color.y *= final_scale;
        accumulated_color.z *= final_scale;
        float3 final_color = srgb_oetf(aces_tone_map(accumulated_color));
        int idx = y * width + x;
        unsigned int r = (unsigned int)float_to_byte(final_color.x);
        unsigned int g = (unsigned int)float_to_byte(final_color.y);
        unsigned int b = (unsigned int)float_to_byte(final_color.z);
        image_out[idx] = (r << 16) | (g << 8) | b;
    }
}
