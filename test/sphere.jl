using LaplaceMFS
using Test
using LinearAlgebra

@testset "Single sphere RHS matches single-layer formulation" begin
    r = 1.0
    M = 243
    Ez = 1.25
    eps_r = 2.5

    rhs = LaplaceMFS.singlesphere_Ez_rhs(r, M, Ez, eps_r)
    pts = load_sphdes_N(M)

    @test rhs[1:M] ≈ zeros(M) atol = 1e-14
    @test rhs[M+1:2M] ≈ ((eps_r - 1) * Ez) .* pts[:, 3] atol = 1e-14
end

@testset "Multisphere RHS repeats single-sphere normal term per sphere" begin
    r = 1.0
    M = 243
    Ez = 0.7
    eps_r = 3.1
    centers = [0.0 0.0 0.0; 2.0 0.5 -0.1; -1.3 1.7 0.2]

    rhs = LaplaceMFS.multispheres_Ez_rhs(r, M, Ez, eps_r, centers)
    pts = load_sphdes_N(M)
    rhs_loc = vcat(zeros(M), ((eps_r - 1) * Ez) .* pts[:, 3])

    for s in 1:size(centers, 1)
        loc = (s - 1) * 2M + 1 : s * 2M
        @test rhs[loc] ≈ rhs_loc atol = 1e-14
    end
end

@testset "Single Sphere Polarization Under Ez" begin
    r = 1.0
    r_p = 0.7
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

    # Reference: refs/note.pdf Eq. (3)-(6) and refs/single_sphere.md.
    # We solve for x = [p; -q], so interior potential is -S_qr * x_q.
    alpha = (eps_r - 1) / (eps_r + 2)
    u_ext_num = B[1:M, 1:N] * x[1:N]
    u_int_num = -B[1:M, N+1:2N] * x[N+1:2N]

    @test projcoef(u_ext_num) ≈ alpha * Ez * r atol = 3e-3
    @test projcoef(u_int_num) ≈ alpha * Ez * r atol = 3e-3
end

@testset "eval_exterior_pot matches single-sphere Ez exterior potential" begin
    r = 1.2
    r_p = 0.7
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

@testset "single-sphere analytic helpers" begin
    r = 1.2
    Ez = 1.0
    eps_r = 2.5
    alpha = (eps_r - 1) / (eps_r + 2)

    targets_out = [
        0.0  0.0   0.0   0.0;
        0.0  0.3  -0.7   0.2;
        1.5  1.3  -1.4   2.0
    ]
    rho = vec(sqrt.(sum(abs2, targets_out; dims = 1)))
    u_out_ref = alpha * Ez * r^3 .* vec(targets_out[3, :]) ./ (rho .^ 3)
    u_out = LaplaceMFS.single_sphere_scattered_exterior(targets_out, r, Ez, eps_r)
    @test u_out ≈ u_out_ref rtol = 1e-13

    targets_in = [
        0.0  0.1  -0.2;
        0.0  0.2   0.1;
        0.3 -0.4   0.8
    ]
    u_in_ref = alpha * Ez .* vec(targets_in[3, :])
    u_in = LaplaceMFS.single_sphere_scattered_interior(targets_in, Ez, eps_r)
    @test u_in ≈ u_in_ref rtol = 1e-13
end

@testset "Two-sphere matrix follows note Eq. (9)" begin
    r = 1.0
    r_p = 0.7
    M = 62
    N = 50
    eps_r = 2.5
    centers = [-3.0 0.0 0.0; 3.0 0.0 0.0]

    B = LaplaceMFS.doublespheres_B(r, r_p, M, N, eps_r, centers)
    @test size(B) == (4M, 4N)

    row_p1 = 1:M
    row_p2 = M + 1 : 2M
    row_dn1 = 2M + 1 : 3M
    row_dn2 = 3M + 1 : 4M
    col_p1 = 1:N
    col_p2 = N + 1 : 2N
    col_q1 = 2N + 1 : 3N
    col_q2 = 3N + 1 : 4N

    @test B[row_p1, col_p2] ≈ zeros(M, N) atol = 1e-14
    @test B[row_p1, col_q2] ≈ zeros(M, N) atol = 1e-14
    @test B[row_p2, col_p1] ≈ zeros(M, N) atol = 1e-14
    @test B[row_p2, col_q1] ≈ zeros(M, N) atol = 1e-14

    pts_M = load_sphdes_N(M)
    pts_N = load_sphdes_N(N)
    r_q = r * r / r_p
    i = 7
    j = 11

    trg1 = vec(centers[1, :]) .+ r .* vec(pts_M[i, :])
    trg2 = vec(centers[2, :]) .+ r .* vec(pts_M[i, :])
    n1 = vec(pts_M[i, :])
    n2 = vec(pts_M[i, :])
    src_p1 = vec(centers[1, :]) .+ r_p .* vec(pts_N[j, :])
    src_p2 = vec(centers[2, :]) .+ r_p .* vec(pts_N[j, :])
    src_q1 = vec(centers[1, :]) .+ r_q .* vec(pts_N[j, :])
    src_q2 = vec(centers[2, :]) .+ r_q .* vec(pts_N[j, :])

    @test B[i, j] ≈ laplace3d_pot(src_p1, trg1) atol = 1e-14
    @test B[i, 2N + j] ≈ laplace3d_pot(src_q1, trg1) atol = 1e-14
    @test B[M + i, N + j] ≈ laplace3d_pot(src_p2, trg2) atol = 1e-14
    @test B[M + i, 3N + j] ≈ laplace3d_pot(src_q2, trg2) atol = 1e-14

    @test B[2M + i, j] ≈ laplace3d_grad(src_p1, trg1, n1) atol = 1e-14
    @test B[2M + i, N + j] ≈ (1 - eps_r) * laplace3d_grad(src_p2, trg1, n1) atol = 1e-14
    @test B[2M + i, 2N + j] ≈ eps_r * laplace3d_grad(src_q1, trg1, n1) atol = 1e-14
    @test B[2M + i, 3N + j] ≈ 0.0 atol = 1e-14

    @test B[3M + i, j] ≈ (1 - eps_r) * laplace3d_grad(src_p1, trg2, n2) atol = 1e-14
    @test B[3M + i, N + j] ≈ laplace3d_grad(src_p2, trg2, n2) atol = 1e-14
    @test B[3M + i, 2N + j] ≈ 0.0 atol = 1e-14
    @test B[3M + i, 3N + j] ≈ eps_r * laplace3d_grad(src_q2, trg2, n2) atol = 1e-14
end

@testset "Two-sphere note RHS" begin
    M = 62
    Ez = 1.0
    eps_r = 2.5
    rhs = LaplaceMFS.doublespheres_Ez_rhs(M, Ez, eps_r)
    pts_M = load_sphdes_N(M)
    nz = pts_M[:, 3]
    @test rhs[1:2M] ≈ zeros(2M) atol = 1e-14
    @test rhs[2M+1:3M] ≈ (-(eps_r - 1) * Ez) .* nz atol = 1e-14
    @test rhs[3M+1:4M] ≈ (-(eps_r - 1) * Ez) .* nz atol = 1e-14
end
