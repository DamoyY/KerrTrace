#pragma once
__device__ __forceinline__ float gaussian_weight(int x, float sigma)
{
    float t = (float)x;
    float denom = 2.0f * sigma * sigma;
    return __expf(-(t * t) / denom);
}
__device__ __forceinline__ float3 gaussian_blur_axis(const float4 *__restrict__ buffer, int width, int height, int x,
                                                     int y, int radius, float sigma, int step_x, int step_y)
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
