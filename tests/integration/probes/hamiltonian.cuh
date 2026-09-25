__device__ double reference_hamiltonian(double x, double y, double z, const RayState &s, double pt)
{
    double a = c_params.a, mass = c_params.M;
    double u = x * x + y * y + z * z - a * a;
    double r2 = 0.5 * (u + sqrt(u * u + 4.0 * a * a * y * y));
    double r = sqrt(r2);
    double lx = (r * x + a * z) / (r2 + a * a);
    double ly = y / r;
    double lz = (r * z - a * x) / (r2 + a * a);
    double potential = mass * r2 * r / (r2 * r2 + a * a * y * y);
    double lp = pt + lx * s.px + ly * s.py + lz * s.pz;
    return 0.5 * (-pt * pt + (double)s.px * s.px + (double)s.py * s.py + (double)s.pz * s.pz) - potential * lp * lp;
}
__device__ float gradient_residual()
{
    RayState s = {2.0f, 1.3f, -0.7f, 0.4f, -0.6f, 0.8f};
    float pt = 1.2f;
    RayDerivs actual = get_derivs(s, pt);
    double delta = 1e-4;
    double dx =
        -(reference_hamiltonian(s.x + delta, s.y, s.z, s, pt) - reference_hamiltonian(s.x - delta, s.y, s.z, s, pt)) /
        (2.0 * delta);
    double dy =
        -(reference_hamiltonian(s.x, s.y + delta, s.z, s, pt) - reference_hamiltonian(s.x, s.y - delta, s.z, s, pt)) /
        (2.0 * delta);
    double dz =
        -(reference_hamiltonian(s.x, s.y, s.z + delta, s, pt) - reference_hamiltonian(s.x, s.y, s.z - delta, s, pt)) /
        (2.0 * delta);
    return (float)fmax(fmax(fabs(dx - actual.dpx), fabs(dy - actual.dpy)), fabs(dz - actual.dpz));
}
