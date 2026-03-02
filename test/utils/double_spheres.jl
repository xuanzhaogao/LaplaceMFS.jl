using LaplaceMFS
using Test
using LinearAlgebra

@testset "double-sphere image coefficients recover axis eigenvalue limits" begin
    r = 1.0
    eps_r = 2.5
    Ez = 1.0
    alpha = (eps_r - 1) / (eps_r + 2)

    # Axis along z: Ez is parallel to center-to-center direction.
    centers_par = [0.0 0.0 -4.0; 0.0 0.0 4.0]
    coeffs_par = LaplaceMFS.double_sphere_image_coefficients(r, Ez, eps_r, centers_par)
    Rpar = norm(vec(centers_par[1, :] .- centers_par[2, :]))
    kpar = alpha * (r / Rpar)^3
    s_par_ref = alpha * r^3 * Ez / (1 - 2kpar)
    @test coeffs_par[3, 1] ≈ s_par_ref rtol = 1e-13
    @test coeffs_par[3, 2] ≈ s_par_ref rtol = 1e-13

    # Axis along x: Ez is perpendicular to center-to-center direction.
    centers_perp = [-4.0 0.0 0.0; 4.0 0.0 0.0]
    coeffs_perp = LaplaceMFS.double_sphere_image_coefficients(r, Ez, eps_r, centers_perp)
    Rperp = norm(vec(centers_perp[1, :] .- centers_perp[2, :]))
    kperp = alpha * (r / Rperp)^3
    s_perp_ref = alpha * r^3 * Ez / (1 + kperp)
    @test coeffs_perp[3, 1] ≈ s_perp_ref rtol = 1e-13
    @test coeffs_perp[3, 2] ≈ s_perp_ref rtol = 1e-13
end

@testset "double-sphere image potential is close to two-sphere MFS for separated spheres" begin
    r = 1.0
    r_p = 0.7
    N = 201
    M = 243
    Ez = 1.0
    eps_r = 2.5
    centers = [-4.0 0.0 0.0; 4.0 0.0 0.0]

    B = LaplaceMFS.doublespheres_B(r, r_p, M, N, eps_r, centers)
    rhs = LaplaceMFS.doublespheres_Ez_rhs(M, Ez, eps_r)
    x = B \ rhs

    p_coeffs = vcat(x[1:N], x[N+1:2N])

    targets = [
        0.0   0.0   0.0   1.0  -2.0   2.0;
        0.0   0.6  -0.8   0.4   1.2  -0.7;
        3.2   4.5   5.3   3.8   4.1   4.7
    ]

    u_mfs = LaplaceMFS.eval_exterior_pot(centers, N, p_coeffs, r_p, 1e-12, targets)
    u_img = LaplaceMFS.double_sphere_image_potential(targets, r, Ez, eps_r, centers)

    relerr = norm(u_mfs - u_img) / norm(u_mfs)
    @test relerr < 8e-2

    coeffs = LaplaceMFS.double_sphere_image_coefficients(r, Ez, eps_r, centers)
    u_img2 = LaplaceMFS.double_sphere_image_potential(targets, centers, coeffs)
    @test u_img2 ≈ u_img rtol = 1e-14 atol = 1e-14
end
