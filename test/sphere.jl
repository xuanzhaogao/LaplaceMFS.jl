using LaplaceMFS
using Test
using LinearAlgebra

@testset "Single Sphere Polarization Under Ez" begin
    r = 1.0
    r_p = 0.8
    N = 201
    M = 243
    Ez = 1.0
    eps_r = 2.5

    B = LaplaceMFS.singlesphere_B(r, r_p, M, N, eps_r)
    rhs = LaplaceMFS.singlesphere_Ez_rhs(r, M, Ez, eps_r)
    x = B \ rhs

    pts = load_sphdes_N(M)
    z = pts[:, 3]
    projcoef(v) = dot(v, z) / dot(z, z)

    # Reference: refs/single_sphere.md
    # ΔΦ_ext = a^3 * ((eps_r - 1)/(eps_r + 2)) * Ez * z / ρ^3.
    # On ρ = a this gives u_ext = ((eps_r - 1)/(eps_r + 2)) * Ez * z.
    # Here z is the unit-sphere coordinate, so this is:
    # u_ext = ((eps_r - 1)/(eps_r + 2)) * Ez * r * z_unit.
    # With boundary relation u_ext + u_int = (1 - 1/eps_r) * Ez * r * z_unit,
    # we get u_int = 2/(eps_r + 2) * (1 - 1/eps_r) * Ez * r * z_unit.
    c = (1 - 1 / eps_r) * Ez * r
    u_ext_num = B[1:M, 1:N] * x[1:N]
    u_int_num = B[1:M, N+1:2N] * x[N+1:2N]

    @test projcoef(u_ext_num) ≈ c * eps_r / (eps_r + 2) atol = 3e-3
    @test projcoef(u_int_num) ≈ 2c / (eps_r + 2) atol = 3e-3
end

@testset "eval_exterior_pot matches single-sphere Ez exterior potential" begin
    r = 1.2
    r_p = 0.8
    N = 1059
    M = 1302
    Ez = 1.0
    eps_r = 2.5

    B = LaplaceMFS.singlesphere_B(r, r_p, M, N, eps_r)
    rhs = LaplaceMFS.singlesphere_Ez_rhs(r, M, Ez, eps_r)
    lambda = B \ rhs

    centers = reshape([0.0, 0.0, 0.0], 1, 3)
    targets = [
        0.0  0.0   0.0   0.0   0.0;
        0.0  0.3  -0.7   0.2  -0.4;
        1.5  1.3  -1.4   2.0   1.8
    ]

    u_ext_num = LaplaceMFS.eval_exterior_pot(centers, N, lambda, r_p, 1e-12, targets)

    rho = vec(sqrt.(sum(abs2, targets; dims = 1)))
    u_ext_th = ((eps_r - 1) / (eps_r + 2)) * Ez * r^3 .* vec(targets[3, :]) ./ (rho .^ 3)

    @test u_ext_num ≈ u_ext_th rtol = 1e-5
end
