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

    p_coeffs = vcat(x[1:N], x[2N+1:3N])

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

@testset "single-sphere forward point+line images" begin
    center = [0.0, 0.0, 0.0]
    source = [0.0, 0.0, 1.5]

    out = LaplaceMFS.single_sphere_forward_point_line_images(
        center, 1.0, 0.4, 1.0, source; n_line = 8, cutoff = 0.0
    )
    @test length(out.q) == 9
    @test size(out.x) == (3, 9)
    @test out.q[1] ≈ -0.4 / 1.5 atol = 1e-12
    @test out.x[:, 1] ≈ [0.0, 0.0, 2 / 3] atol = 1e-12
    @test all(abs.(out.x[1, :]) .< 1e-14)
    @test all(abs.(out.x[2, :]) .< 1e-14)
    @test minimum(out.x[3, :]) >= -1e-14
    @test maximum(out.x[3, :]) <= (2 / 3 + 1e-14)

    out_point_only = LaplaceMFS.single_sphere_forward_point_line_images(
        center, 1.0, 1.0, 1.0, source; n_line = 8, cutoff = 0.0
    )
    @test length(out_point_only.q) == 1
    @test size(out_point_only.x) == (3, 1)

    out_cut = LaplaceMFS.single_sphere_forward_point_line_images(
        center, 1.0, 0.4, 1.0, source; n_line = 8, cutoff = 1.0
    )
    @test isempty(out_cut.q)
    @test size(out_cut.x) == (3, 0)
end

@testset "double-sphere forward reflections (point-only chain)" begin
    centers = [0.0 0.0 0.0; 0.0 0.0 3.0]
    radii = (1.0, 1.0)
    gammas = (0.5, 0.2)
    source = [0.0, 0.0, 1.5]

    out1 = LaplaceMFS.double_sphere_forward_point_line_images(
        centers, radii, gammas, 1.0, source; n_line = 0, n_reflections = 1, cutoff = 0.0
    )
    @test length(out1.q1) == 1
    @test length(out1.q2) == 1
    @test size(out1.x1) == (3, 1)
    @test size(out1.x2) == (3, 1)
    @test out1.q1[1] ≈ -gammas[1] * radii[1] / 1.5 atol = 1e-12
    @test out1.q2[1] ≈ -gammas[2] * radii[2] / 1.5 atol = 1e-12

    out3 = LaplaceMFS.double_sphere_forward_point_line_images(
        centers, radii, gammas, 1.0, source; n_line = 0, n_reflections = 3, cutoff = 0.0
    )
    @test length(out3.q1) == 3
    @test length(out3.q2) == 3
    @test size(out3.x1) == (3, 3)
    @test size(out3.x2) == (3, 3)

    @test_throws ArgumentError LaplaceMFS.double_sphere_forward_point_line_images(
        centers, radii, gammas, 1.0, source; n_line = 0, n_reflections = 0
    )
end

@testset "single-sphere forward dipole images reuse point-charge positions" begin
    center = [0.0, 0.0, 0.0]
    source = [0.0, 0.0, 1.5]
    p0 = [1.0, -2.0, 3.0]

    out_q = LaplaceMFS.single_sphere_forward_point_line_images(
        center, 1.0, 0.4, 1.0, source; n_line = 8, cutoff = 0.0
    )
    out_p = LaplaceMFS.single_sphere_forward_point_line_dipole_images(
        center, 1.0, 0.4, p0, source; n_line = 8, cutoff = 0.0
    )

    @test size(out_p.x) == size(out_q.x)
    @test out_p.x ≈ out_q.x atol = 1e-14 rtol = 1e-14
    @test size(out_p.p) == (3, length(out_q.q))

    for j in eachindex(out_q.q)
        @test out_p.p[:, j] ≈ out_q.q[j] .* p0 atol = 1e-14 rtol = 1e-14
    end
end

@testset "double-sphere forward dipole images reuse point-charge positions" begin
    centers = [0.0 0.0 0.0; 0.0 0.0 3.0]
    radii = (1.0, 1.0)
    gammas = (0.5, 0.2)
    source = [0.0, 0.0, 1.5]
    p0 = [0.0, 0.0, 2.0]

    out_q = LaplaceMFS.double_sphere_forward_point_line_images(
        centers, radii, gammas, 1.0, source; n_line = 0, n_reflections = 3, cutoff = 0.0
    )
    out_p = LaplaceMFS.double_sphere_forward_point_line_dipole_images(
        centers, radii, gammas, p0, source; n_line = 0, n_reflections = 3, cutoff = 0.0
    )

    @test size(out_p.x1) == size(out_q.x1)
    @test size(out_p.x2) == size(out_q.x2)
    @test out_p.x1 ≈ out_q.x1 atol = 1e-14 rtol = 1e-14
    @test out_p.x2 ≈ out_q.x2 atol = 1e-14 rtol = 1e-14
    @test size(out_p.p1) == (3, length(out_q.q1))
    @test size(out_p.p2) == (3, length(out_q.q2))

    for j in eachindex(out_q.q1)
        @test out_p.p1[:, j] ≈ out_q.q1[j] .* p0 atol = 1e-14 rtol = 1e-14
    end
    for j in eachindex(out_q.q2)
        @test out_p.p2[:, j] ≈ out_q.q2[j] .* p0 atol = 1e-14 rtol = 1e-14
    end
end
