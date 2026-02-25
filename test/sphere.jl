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

    B = LaplaceMFS.single_sphere_B(r, r_p, M, N)
    rhs = LaplaceMFS.single_sphere_Ez_rhs(r, M, Ez, eps_r)
    x = B \ rhs

    pts = load_sphdes_N(M)
    z = pts[:, 3]
    projcoef(v) = dot(v, z) / dot(z, z)

    # In this formulation: u_ext + u_int = c * z on r=1, with c = (1 - 1/eps_r)Ez r.
    # For l=1 mode and derivative continuity, the exact amplitudes are:
    # u_ext = (c/3) z, u_int = (2c/3) z.
    c = (1 - 1 / eps_r) * Ez * r
    u_ext_num = B[1:M, 1:N] * x[1:N]
    u_int_num = B[1:M, N+1:2N] * x[N+1:2N]

    @test projcoef(u_ext_num) ≈ c / 3 atol = 3e-3
    @test projcoef(u_int_num) ≈ 2c / 3 atol = 3e-3
end
