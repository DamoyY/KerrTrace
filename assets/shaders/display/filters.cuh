#pragma once
#include "gaussian.cuh"
extern "C"
{
    __global__ __launch_bounds__(1024) void bloom_horizontal(const float4 *__restrict__ accumulation_buffer,
                                                             float4 *__restrict__ bloom_buffer, int width, int height,
                                                             int radius, float sigma, int enabled)
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
        float3 accum = gaussian_blur_axis(accumulation_buffer, width, height, x, y, radius, sigma, 1, 0);
        bloom_buffer[idx] = make_float4(accum.x, accum.y, accum.z, 0.0f);
    }

    __global__ __launch_bounds__(1024) void post_process(const float4 *__restrict__ accumulation_buffer,
                                                         const float4 *__restrict__ bloom_buffer,
                                                         unsigned int *__restrict__ image_out, int width, int height,
                                                         int radius, float sigma, float intensity, int bloom_enabled)
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
