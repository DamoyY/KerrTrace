__device__ __forceinline__ float3 fetch_color_from_lut(float T, const float *__restrict__ lut, int lut_size, float max_temp)
{
    float t_norm = T / max_temp;
    float pos = t_norm * (float)(lut_size - 1);
    int idx = (int)pos;
    float frac = pos - (float)idx;
    int base_idx = idx * 3;
    float3 c1 = make_float3(__ldg(&lut[base_idx]), __ldg(&lut[base_idx + 1]), __ldg(&lut[base_idx + 2]));
    float3 c2 = make_float3(__ldg(&lut[base_idx + 3]), __ldg(&lut[base_idx + 4]), __ldg(&lut[base_idx + 5]));
    float3 res;
    res.x = c1.x + frac * (c2.x - c1.x);
    res.y = c1.y + frac * (c2.y - c1.y);
    res.z = c1.z + frac * (c2.z - c1.z);
    return res;
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
__device__ unsigned char float_to_byte(float val)
{
    return (unsigned char)__float2int_rn(__saturatef(val) * 255.0f);
}
#include "kerr.cuh"
extern "C"
{
    __global__ void kernel(
        uchar4 *__restrict__ image_out,
        int width, int height,
        float cam_x, float cam_y, float cam_z,
        float fwd_x, float fwd_y, float fwd_z,
        float rgt_x, float rgt_y, float rgt_z,
        float up_x, float up_y, float up_z,
        float *__restrict__ lut, int lut_size, float max_temp,
        float fov_scale)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height)
            return;
        const int SSAA_SAMPLES = CONFIG_SSAA_SAMPLES;
        const float INV_SAMPLES = 1.0f / (float)SSAA_SAMPLES;
        const float inv_w_2 = 2.0f / (float)width;
        const float inv_h_2 = 2.0f / (float)height;
        const float aspect_ratio = (float)width / (float)height;
        const float3 cam_pos = make_float3(cam_x, cam_y, cam_z);
        float start_offset = 0.5f * INV_SAMPLES;
        float u_start = ((x + start_offset) * inv_w_2 - 1.0f) * aspect_ratio * fov_scale;
        float v_start = (1.0f - (y + start_offset) * inv_h_2) * fov_scale;
        float3 ray_base;
        ray_base.x = u_start * rgt_x + v_start * up_x + fwd_x;
        ray_base.y = u_start * rgt_y + v_start * up_y + fwd_y;
        ray_base.z = u_start * rgt_z + v_start * up_z + fwd_z;
        float du = (INV_SAMPLES * inv_w_2) * aspect_ratio * fov_scale;
        float dv = -(INV_SAMPLES * inv_h_2) * fov_scale;
        float3 step_x, step_y;
        step_x.x = rgt_x * du;
        step_x.y = rgt_y * du;
        step_x.z = rgt_z * du;
        step_y.x = up_x * dv;
        step_y.y = up_y * dv;
        step_y.z = up_z * dv;
        float3 accumulated_color = make_float3(0.0f, 0.0f, 0.0f);
        float3 row_ray = ray_base;
#pragma unroll
        for (int sy = 0; sy < SSAA_SAMPLES; sy++)
        {
            float3 current_ray = row_ray;
#pragma unroll
            for (int sx = 0; sx < SSAA_SAMPLES; sx++)
            {
                float3 sample_color = trace_ray(cam_pos, current_ray, lut, lut_size, max_temp);
                accumulated_color.x += sample_color.x;
                accumulated_color.y += sample_color.y;
                accumulated_color.z += sample_color.z;
                current_ray.x += step_x.x;
                current_ray.y += step_x.y;
                current_ray.z += step_x.z;
            }
            row_ray.x += step_y.x;
            row_ray.y += step_y.y;
            row_ray.z += step_y.z;
        }
        const float final_scale = CONFIG_EXPOSURE_SCALE * (INV_SAMPLES * INV_SAMPLES);
        accumulated_color.x *= final_scale;
        accumulated_color.y *= final_scale;
        accumulated_color.z *= final_scale;
        float3 final_color = aces_tone_map(accumulated_color);
        int idx = y * width + x;
        image_out[idx] = make_uchar4(
            float_to_byte(final_color.x),
            float_to_byte(final_color.y),
            float_to_byte(final_color.z),
            255);
    }
}
