#pragma once
#include "geometry.cuh"
__device__ bool finite_state(const RayState &s)
{
    return isfinite(s.x) && isfinite(s.y) && isfinite(s.z) && isfinite(s.px) && isfinite(s.py) && isfinite(s.pz);
}
__device__ float step_scale(float error)
{
    if (!isfinite(error))
        return 0.2f;
    if (error == 0.0f)
        return 5.0f;
    return fminf(5.0f, fmaxf(0.2f, 0.9f * __powf(error, -0.2f)));
}
__device__ RayState integrate_step(const RayState &s, float pt, float h_step, float tol, const RayDerivs &k1,
                                   RayDerivs &k3, RayDerivs &k4, RayDerivs &k5, RayDerivs &k6, RayDerivs &k7,
                                   float &error)
{
    RayDerivs k2;
    static const float a21 = 1.0f / 5.0f;
    static const float a31 = 3.0f / 40.0f, a32 = 9.0f / 40.0f;
    static const float a41 = 44.0f / 45.0f, a42 = -56.0f / 15.0f, a43 = 32.0f / 9.0f;
    static const float a51 = 19372.0f / 6561.0f, a52 = -25360.0f / 2187.0f, a53 = 64448.0f / 6561.0f,
                       a54 = -212.0f / 729.0f;
    static const float a61 = 9017.0f / 3168.0f, a62 = -355.0f / 33.0f, a63 = 46732.0f / 5247.0f, a64 = 49.0f / 176.0f,
                       a65 = -5103.0f / 18656.0f;
    static const float a71 = 35.0f / 384.0f, a73 = 500.0f / 1113.0f, a74 = 125.0f / 192.0f, a75 = -2187.0f / 6784.0f,
                       a76 = 11.0f / 84.0f;
    static const float b1 = 35.0f / 384.0f, b3 = 500.0f / 1113.0f, b4 = 125.0f / 192.0f, b5 = -2187.0f / 6784.0f,
                       b6 = 11.0f / 84.0f;
    static const float dc1 = 35.0f / 384.0f - 5179.0f / 57600.0f, dc3 = 500.0f / 1113.0f - 7571.0f / 16695.0f,
                       dc4 = 125.0f / 192.0f - 393.0f / 640.0f, dc5 = -2187.0f / 6784.0f + 92097.0f / 339200.0f,
                       dc6 = 11.0f / 84.0f - 187.0f / 2100.0f, dc7 = -1.0f / 40.0f;
    k2 = get_derivs({s.x + h_step * a21 * k1.dx, s.y + h_step * a21 * k1.dy, s.z + h_step * a21 * k1.dz,
                     s.px + h_step * a21 * k1.dpx, s.py + h_step * a21 * k1.dpy, s.pz + h_step * a21 * k1.dpz},
                    pt);
    k3 = get_derivs({s.x + h_step * (a31 * k1.dx + a32 * k2.dx), s.y + h_step * (a31 * k1.dy + a32 * k2.dy),
                     s.z + h_step * (a31 * k1.dz + a32 * k2.dz), s.px + h_step * (a31 * k1.dpx + a32 * k2.dpx),
                     s.py + h_step * (a31 * k1.dpy + a32 * k2.dpy), s.pz + h_step * (a31 * k1.dpz + a32 * k2.dpz)},
                    pt);
    k4 = get_derivs({s.x + h_step * (a41 * k1.dx + a42 * k2.dx + a43 * k3.dx),
                     s.y + h_step * (a41 * k1.dy + a42 * k2.dy + a43 * k3.dy),
                     s.z + h_step * (a41 * k1.dz + a42 * k2.dz + a43 * k3.dz),
                     s.px + h_step * (a41 * k1.dpx + a42 * k2.dpx + a43 * k3.dpx),
                     s.py + h_step * (a41 * k1.dpy + a42 * k2.dpy + a43 * k3.dpy),
                     s.pz + h_step * (a41 * k1.dpz + a42 * k2.dpz + a43 * k3.dpz)},
                    pt);
    k5 = get_derivs({s.x + h_step * (a51 * k1.dx + a52 * k2.dx + a53 * k3.dx + a54 * k4.dx),
                     s.y + h_step * (a51 * k1.dy + a52 * k2.dy + a53 * k3.dy + a54 * k4.dy),
                     s.z + h_step * (a51 * k1.dz + a52 * k2.dz + a53 * k3.dz + a54 * k4.dz),
                     s.px + h_step * (a51 * k1.dpx + a52 * k2.dpx + a53 * k3.dpx + a54 * k4.dpx),
                     s.py + h_step * (a51 * k1.dpy + a52 * k2.dpy + a53 * k3.dpy + a54 * k4.dpy),
                     s.pz + h_step * (a51 * k1.dpz + a52 * k2.dpz + a53 * k3.dpz + a54 * k4.dpz)},
                    pt);
    k6 = get_derivs({s.x + h_step * (a61 * k1.dx + a62 * k2.dx + a63 * k3.dx + a64 * k4.dx + a65 * k5.dx),
                     s.y + h_step * (a61 * k1.dy + a62 * k2.dy + a63 * k3.dy + a64 * k4.dy + a65 * k5.dy),
                     s.z + h_step * (a61 * k1.dz + a62 * k2.dz + a63 * k3.dz + a64 * k4.dz + a65 * k5.dz),
                     s.px + h_step * (a61 * k1.dpx + a62 * k2.dpx + a63 * k3.dpx + a64 * k4.dpx + a65 * k5.dpx),
                     s.py + h_step * (a61 * k1.dpy + a62 * k2.dpy + a63 * k3.dpy + a64 * k4.dpy + a65 * k5.dpy),
                     s.pz + h_step * (a61 * k1.dpz + a62 * k2.dpz + a63 * k3.dpz + a64 * k4.dpz + a65 * k5.dpz)},
                    pt);
    k7 = get_derivs({s.x + h_step * (a71 * k1.dx + a73 * k3.dx + a74 * k4.dx + a75 * k5.dx + a76 * k6.dx),
                     s.y + h_step * (a71 * k1.dy + a73 * k3.dy + a74 * k4.dy + a75 * k5.dy + a76 * k6.dy),
                     s.z + h_step * (a71 * k1.dz + a73 * k3.dz + a74 * k4.dz + a75 * k5.dz + a76 * k6.dz),
                     s.px + h_step * (a71 * k1.dpx + a73 * k3.dpx + a74 * k4.dpx + a75 * k5.dpx + a76 * k6.dpx),
                     s.py + h_step * (a71 * k1.dpy + a73 * k3.dpy + a74 * k4.dpy + a75 * k5.dpy + a76 * k6.dpy),
                     s.pz + h_step * (a71 * k1.dpz + a73 * k3.dpz + a74 * k4.dpz + a75 * k5.dpz + a76 * k6.dpz)},
                    pt);
    RayState next_s = {s.x + h_step * (b1 * k1.dx + b3 * k3.dx + b4 * k4.dx + b5 * k5.dx + b6 * k6.dx),
                       s.y + h_step * (b1 * k1.dy + b3 * k3.dy + b4 * k4.dy + b5 * k5.dy + b6 * k6.dy),
                       s.z + h_step * (b1 * k1.dz + b3 * k3.dz + b4 * k4.dz + b5 * k5.dz + b6 * k6.dz),
                       s.px + h_step * (b1 * k1.dpx + b3 * k3.dpx + b4 * k4.dpx + b5 * k5.dpx + b6 * k6.dpx),
                       s.py + h_step * (b1 * k1.dpy + b3 * k3.dpy + b4 * k4.dpy + b5 * k5.dpy + b6 * k6.dpy),
                       s.pz + h_step * (b1 * k1.dpz + b3 * k3.dpz + b4 * k4.dpz + b5 * k5.dpz + b6 * k6.dpz)};
    float scale_x = tol * fmaxf(fabsf(s.x), 1.0f);
    float scale_y = tol * fmaxf(fabsf(s.y), 1.0f);
    float scale_z = tol * fmaxf(fabsf(s.z), 1.0f);
    float scale_px = tol * fmaxf(fabsf(s.px), 1.0f);
    float scale_py = tol * fmaxf(fabsf(s.py), 1.0f);
    float scale_pz = tol * fmaxf(fabsf(s.pz), 1.0f);
    float err_x =
        fabsf(h_step * (dc1 * k1.dx + dc3 * k3.dx + dc4 * k4.dx + dc5 * k5.dx + dc6 * k6.dx + dc7 * k7.dx)) / scale_x;
    float err_y =
        fabsf(h_step * (dc1 * k1.dy + dc3 * k3.dy + dc4 * k4.dy + dc5 * k5.dy + dc6 * k6.dy + dc7 * k7.dy)) / scale_y;
    float err_z =
        fabsf(h_step * (dc1 * k1.dz + dc3 * k3.dz + dc4 * k4.dz + dc5 * k5.dz + dc6 * k6.dz + dc7 * k7.dz)) / scale_z;
    float err_px =
        fabsf(h_step * (dc1 * k1.dpx + dc3 * k3.dpx + dc4 * k4.dpx + dc5 * k5.dpx + dc6 * k6.dpx + dc7 * k7.dpx)) /
        scale_px;
    float err_py =
        fabsf(h_step * (dc1 * k1.dpy + dc3 * k3.dpy + dc4 * k4.dpy + dc5 * k5.dpy + dc6 * k6.dpy + dc7 * k7.dpy)) /
        scale_py;
    float err_pz =
        fabsf(h_step * (dc1 * k1.dpz + dc3 * k3.dpz + dc4 * k4.dpz + dc5 * k5.dpz + dc6 * k6.dpz + dc7 * k7.dpz)) /
        scale_pz;
    error = fmaxf(fmaxf(fmaxf(err_x, err_y), fmaxf(err_z, err_px)), fmaxf(err_py, err_pz));
    if (!finite_state(next_s) || !isfinite(err_x) || !isfinite(err_y) || !isfinite(err_z) || !isfinite(err_px) ||
        !isfinite(err_py) || !isfinite(err_pz))
        error = __int_as_float(0x7f800000);
    return next_s;
}
