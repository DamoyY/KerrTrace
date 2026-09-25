extern "C" __global__ void formula_probe(float *output, unsigned int *errors, cudaTextureObject_t spectrum,
                                         cudaTextureObject_t disk)
{
    if (threadIdx.x != 0)
        return;
    RayState previous = {0.0f, 0.01f, sqrtf(25.0f + c_params.aa), 0.1f, -1.0f, 0.0f};
    RayState next = previous;
    next.y = -0.01f;
    RayDerivs tangent = {0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float transmission = 1.0f;
    accumulate_disk(previous, next, tangent, tangent, tangent, tangent, tangent, tangent, 0.02f, 1.0f, 1.0f, spectrum,
                    5, 1000.0f, disk, 4.0f, 8.0f, errors, color, transmission);
    output[0] = gradient_residual();
    output[1] = color.x;
    output[2] = transmission;
    RayState camera;
    float pt;
    initialize_camera_ray(make_float3(80.0f, 6.0f, 0.0f), make_float3(-1.0f, -0.1f, 0.2f), camera, pt);
    float r = ks_r_from_xyz(camera.x, camera.y, camera.z, c_params.aa);
    float r2 = r * r;
    float H = c_params.M * r2 * r / (r2 * r2 + c_params.aa * camera.y * camera.y);
    float lx = (r * camera.x + c_params.a * camera.z) / (r2 + c_params.aa);
    float ly = camera.y / r;
    float lz = (r * camera.z - c_params.a * camera.x) / (r2 + c_params.aa);
    output[3] = -pt - 2.0f * H * (pt + lx * camera.px + ly * camera.py + lz * camera.pz);
    output[4] = (float)reference_hamiltonian(camera.x, camera.y, camera.z, camera, pt);
    float2 orbit = circular_orbit(5.0f);
    float azimuthal_speed = orbit.x * previous.z * orbit.y;
    float null_contraction = -orbit.y + c_params.a / previous.z * azimuthal_speed;
    output[5] = -orbit.y * orbit.y + azimuthal_speed * azimuthal_speed +
                2.0f * c_params.M / 5.0f * null_contraction * null_contraction;
    output[6] = step_scale(0.0f);
    output[7] = step_scale(__int_as_float(0x7fffffff));
}
