__device__ __forceinline__ float3 fetch_color_from_lut(float T, cudaTextureObject_t lut_tex, int lut_size, float max_temp, unsigned int *error_flag)
{
    if (T > max_temp)
    {
        if (atomicCAS(error_flag, 0u, 1u) == 0u)
        {
            error_flag[1] = __float_as_uint(T);
        }
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    float t_norm = fmaxf(T, 0.0f) / max_temp;
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
__device__ __forceinline__ float gaussian_weight(int x, float sigma)
{
    float t = (float)x;
    float denom = 2.0f * sigma * sigma;
    return __expf(-(t * t) / denom);
}
__device__ __forceinline__ float3 gaussian_blur_axis(
    const float4 *__restrict__ buffer,
    int width, int height,
    int x, int y,
    int radius,
    float sigma,
    int step_x, int step_y)
{
    float3 accum = make_float3(0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    for (int i = -radius; i <= radius; i++)
    {
        int sx = x + i * step_x;
        int sy = y + i * step_y;
        if (sx < 0 || sx >= width || sy < 0 || sy >= height)
            continue;
        float w = gaussian_weight(i, sigma);
        float4 c = buffer[sy * width + sx];
        accum.x += c.x * w;
        accum.y += c.y * w;
        accum.z += c.z * w;
        weight_sum += w;
    }
    if (weight_sum > 0.0f)
    {
        accum.x /= weight_sum;
        accum.y /= weight_sum;
        accum.z /= weight_sum;
    }
    return accum;
}
#include "kerr_trace.cuh"
extern "C"
{
    __global__ __launch_bounds__(1024) void trace_kernel(
        float4 *__restrict__ accumulation_buffer,
        int width, int height,
        float cam_x, float cam_y, float cam_z,
        float fwd_x, float fwd_y, float fwd_z,
        float rgt_x, float rgt_y, float rgt_z,
        float up_x, float up_y, float up_z,
        cudaTextureObject_t lut_tex, int lut_size, float max_temp, unsigned int *error_flag,
        cudaTextureObject_t disk_tex, float disk_inner, float disk_outer,
        float fov_scale)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height)
            return;
        const KerrParams &p = c_params;
        const int SPP = CONFIG_SPP;
        const float INV_SPP = 1.0f / (float)SPP;
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
            float u = ((x + rx) * p.inv_w_2 - 1.0f) * p.aspect_ratio * fov_scale;
            float v = (1.0f - (y + ry) * p.inv_h_2) * fov_scale;
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
                disk_outer,
                error_flag);
            accumulated_color.x += sample_color.x;
            accumulated_color.y += sample_color.y;
            accumulated_color.z += sample_color.z;
        }
        const float final_scale = CONFIG_EXPOSURE_SCALE * INV_SPP;
        accumulated_color.x *= final_scale;
        accumulated_color.y *= final_scale;
        accumulated_color.z *= final_scale;
        int idx = y * width + x;
        accumulation_buffer[idx] = make_float4(accumulated_color.x, accumulated_color.y, accumulated_color.z, 1.0f);
    }

    __global__ __launch_bounds__(1024) void bloom_horizontal(
        const float4 *__restrict__ accumulation_buffer,
        float4 *__restrict__ bloom_buffer,
        int width, int height,
        int radius,
        float sigma,
        int enabled)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height)
            return;
        int idx = y * width + x;
        if (!enabled || radius <= 0)
        {
            bloom_buffer[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            return;
        }
        float3 accum = gaussian_blur_axis(
            accumulation_buffer,
            width,
            height,
            x,
            y,
            radius,
            sigma,
            1,
            0);
        bloom_buffer[idx] = make_float4(accum.x, accum.y, accum.z, 0.0f);
    }

    __global__ __launch_bounds__(1024) void post_process(
        const float4 *__restrict__ accumulation_buffer,
        const float4 *__restrict__ bloom_buffer,
        unsigned int *__restrict__ image_out,
        int width, int height,
        int radius,
        float sigma,
        float intensity,
        int bloom_enabled)
    {
        extern __shared__ float4 bloom_tile[];
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        bool in_bounds = x < width && y < height;
        if (bloom_enabled && radius > 0 && intensity > 0.0f)
        {
            int tile_w = (int)blockDim.x;
            int tile_h = (int)blockDim.y + radius * 2;
            int local_x = (int)threadIdx.x;
            int local_y = (int)threadIdx.y;
            for (int tile_y = local_y; tile_y < tile_h; tile_y += (int)blockDim.y)
            {
                int global_y = (int)blockIdx.y * (int)blockDim.y + tile_y - radius;
                int global_x = (int)blockIdx.x * (int)blockDim.x + local_x;
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (global_x < width && global_y >= 0 && global_y < height)
                {
                    v = bloom_buffer[global_y * width + global_x];
                }
                bloom_tile[tile_y * tile_w + local_x] = v;
            }
            __syncthreads();
            if (!in_bounds)
                return;
        }
        else
        {
            if (!in_bounds)
                return;
        }
        int idx = y * width + x;
        float4 base = accumulation_buffer[idx];
        float3 final_color = make_float3(base.x, base.y, base.z);
        if (bloom_enabled && radius > 0 && intensity > 0.0f)
        {
            float3 accum = make_float3(0.0f, 0.0f, 0.0f);
            float weight_sum = 0.0f;
            int tile_w = (int)blockDim.x;
            int base_tile_y = (int)threadIdx.y + radius;
            for (int i = -radius; i <= radius; i++)
            {
                int sy = y + i;
                if (sy < 0 || sy >= height)
                    continue;
                float w = gaussian_weight(i, sigma);
                float4 c = bloom_tile[(base_tile_y + i) * tile_w + (int)threadIdx.x];
                accum.x += c.x * w;
                accum.y += c.y * w;
                accum.z += c.z * w;
                weight_sum += w;
            }
            if (weight_sum > 0.0f)
            {
                accum.x /= weight_sum;
                accum.y /= weight_sum;
                accum.z /= weight_sum;
            }
            final_color.x += accum.x * intensity;
            final_color.y += accum.y * intensity;
            final_color.z += accum.z * intensity;
        }
        float3 mapped = srgb_oetf(aces_tone_map(final_color));
        unsigned int r = (unsigned int)float_to_byte(mapped.x);
        unsigned int g = (unsigned int)float_to_byte(mapped.y);
        unsigned int b = (unsigned int)float_to_byte(mapped.z);
        image_out[idx] = (r << 16) | (g << 8) | b;
    }
}
