#pragma once
__device__ __forceinline__ float dp_dense_output(float y0, float y1, float k1, float k3, float k4, float k5, float k6,
                                                 float k7, float h, float t)
{
    const float d1 = -12715105075.0f / 11282082432.0f;
    const float d3 = 87487479700.0f / 32700410799.0f;
    const float d4 = -10690763975.0f / 1880347072.0f;
    const float d5 = 701980252875.0f / 199316789632.0f;
    const float d6 = -1453857185.0f / 822651844.0f;
    const float d7 = 69997945.0f / 29380423.0f;
    float r1 = y0;
    float r2 = y1 - y0;
    float r3 = y0 + h * k1 - y1;
    float r4 = 2.0f * (y1 - y0) - h * (k1 + k7);
    float r5 = h * (d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6 + d7 * k7);
    float t2 = t * t;
    float t3 = t2 * t;
    float t4 = t2 * t2;
    return r1 + t * (r2 + r3) - t2 * (r3 - r4 - r5) - t3 * (r4 + 2.0f * r5) + t4 * r5;
}
__device__ __forceinline__ float dp_dense_output_derivative(float y0, float y1, float k1, float k3, float k4, float k5,
                                                            float k6, float k7, float h, float t)
{
    const float d1 = -12715105075.0f / 11282082432.0f;
    const float d3 = 87487479700.0f / 32700410799.0f;
    const float d4 = -10690763975.0f / 1880347072.0f;
    const float d5 = 701980252875.0f / 199316789632.0f;
    const float d6 = -1453857185.0f / 822651844.0f;
    const float d7 = 69997945.0f / 29380423.0f;
    float r2 = y1 - y0;
    float r3 = y0 + h * k1 - y1;
    float r4 = 2.0f * (y1 - y0) - h * (k1 + k7);
    float r5 = h * (d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6 + d7 * k7);
    float t2 = t * t;
    float t3 = t2 * t;
    return (r2 + r3) - 2.0f * t * (r3 - r4 - r5) - 3.0f * t2 * (r4 + 2.0f * r5) + 4.0f * t3 * r5;
}
