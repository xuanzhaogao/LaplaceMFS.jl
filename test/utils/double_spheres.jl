using LaplaceMFS
using Test
using LinearAlgebra

@testset "Double-sphere image dipole coefficients" begin
    r = 1.0
    eps_r = 2.5
    Ez = 1.0
    d = 6.0

    a1, a2 = LaplaceMFS.double_sphere_image_coefficients(r, d, Ez, eps_r)
    @test a1 ≈ a2 atol = 1e-14

    alpha = LaplaceMFS.single_sphere_alpha(eps_r) * r^3
    @test a1 ≈ alpha * Ez / (1 + alpha / d^3) rtol = 1e-12
end

@testset "Double-sphere image potential approaches MFS for separated spheres" begin
    r = 1.0
    r_p = 0.7
    eps_r = 2.5
    Ez = 1.0
    d = 6.0
    centers = [-d / 2 0.0 0.0; d / 2 0.0 0.0]
    M = 62
    N = 50

    G = LaplaceMFS.multispheres_G(r, r_p, M, N, centers)
    rhs = LaplaceMFS.multispheres_Ez_rhs(r, M, Ez, eps_r, centers)
    lambda = G \ rhs

    targets = [
        0.0   0.0   0.0   0.0   0.0;
        0.0   1.5  -1.2   0.8  -1.1;
        3.5   2.8  -3.2   4.0  -2.6
    ]

    u_mfs = LaplaceMFS.eval_exterior_pot(centers, N, lambda, r_p, 1e-12, targets)
    u_img = LaplaceMFS.double_sphere_image_potential(targets, centers, r, Ez, eps_r)

    c = dot(u_img, u_mfs) / dot(u_img, u_img)
    @test c > 0.0
    @test norm(c .* u_img - u_mfs) / norm(u_mfs) < 5e-2
end
