using LaplaceMFS
using Test
using LinearAlgebra

@testset "Multisphere G Matrix" begin
    r = 1.0
    r_p = 0.8
    M = 62
    N = 42
    centers1 = zeros(1, 3)
    G1 = LaplaceMFS.multispheres_G(r, r_p, M, N, centers1)
    B1 = LaplaceMFS.singlesphere_B(r, r_p, M, N, 1.0)
    @test size(G1) == size(B1)
    @test G1 ≈ B1 atol = 1e-12

    centers2 = [0.0 0.0 0.0; 3.0 0.0 0.0]
    G2 = LaplaceMFS.multispheres_G(r, r_p, M, N, centers2)
    ns = size(centers2, 1)
    @test size(G2) == (4M, 4N)

    row_pot_1 = 1:M
    row_dn_1 = M + 1 : 2M
    col_p_1 = 1:N
    col_q_1 = N + 1 : 2N
    @test G2[row_pot_1, col_p_1] ≈ B1[1:M, 1:N] atol = 1e-12
    @test G2[row_pot_1, col_q_1] ≈ B1[1:M, N+1:2N] atol = 1e-12
    @test G2[row_dn_1, col_p_1] ≈ B1[M+1:2M, 1:N] atol = 1e-12
    @test G2[row_dn_1, col_q_1] ≈ B1[M+1:2M, N+1:2N] atol = 1e-12

    pts_M = load_sphdes_N(M)
    pts_N = load_sphdes_N(N)
    trg = [0.0, 0.0, 0.0] .+ r .* vec(pts_M[1, :])
    src = [3.0, 0.0, 0.0] .+ r_p .* vec(pts_N[1, :])
    expected = laplace3d_pot(src, trg)
    col_p_2_first = 2N + 1
    @test G2[1, col_p_2_first] ≈ expected atol = 1e-12

    centers3 = [0.0 0.0 0.0; 3.0 0.2 -0.1; -2.2 1.7 0.4]
    G3 = LaplaceMFS.multispheres_G(r, r_p, M, N, centers3)
    for s in 1:3
        rpot = 2 * (s - 1) * M + 1 : 2 * (s - 1) * M + M
        rdn = 2 * (s - 1) * M + M + 1 : 2 * s * M
        cp = 2 * (s - 1) * N + 1 : 2 * (s - 1) * N + N
        cq = 2 * (s - 1) * N + N + 1 : 2 * s * N
        @test G3[rpot, cp] ≈ B1[1:M, 1:N] atol = 1e-12
        @test G3[rpot, cq] ≈ B1[1:M, N+1:2N] atol = 1e-12
        @test G3[rdn, cp] ≈ B1[M+1:2M, 1:N] atol = 1e-12
        @test G3[rdn, cq] ≈ B1[M+1:2M, N+1:2N] atol = 1e-12
    end
end

@testset "Multisphere G FMM LinearMap" begin
    r = 1.0
    r_p = 0.8
    M = 62
    N = 42
    centers = [0.0 0.0 0.0; 2.8 0.1 -0.4; -0.3 2.9 0.2]
    mats = LaplaceMFS.SphereMats(r, r_p, M, N, 1e-12)

    Gdense = LaplaceMFS.multispheres_G(r, r_p, M, N, centers)
    Gfmm = LaplaceMFS.multispheres_G_fmm(mats, centers, 2.5, 1e-13)

    x = randn(2 * N * size(centers, 1))
    y_dense = Gdense * x
    y_fmm = Gfmm * x
    @test norm(y_fmm - y_dense) / norm(y_dense) < 1e-10

    Gfmm_c = LaplaceMFS.multispheres_G_fmm(mats, centers, 2.5 + 0.1im, 1e-13)
    xc = randn(2 * N * size(centers, 1)) .+ im * randn(2 * N * size(centers, 1))
    y_dense_c = Gdense * xc
    y_fmm_c = Gfmm_c * xc
    @test norm(y_fmm_c - y_dense_c) / norm(y_dense_c) < 1e-10
end

@testset "Multisphere Mu To Lambda" begin
    r = 1.0
    r_p = 0.8
    M = 62
    N = 42
    ns = 3
    mats = LaplaceMFS.SphereMats(r, r_p, M, N, 1e-12)

    Bplus = mats.Vt_B' * Diagonal(mats.S_B_inv) * mats.U_B'
    Bhat = zeros(Float64, 2 * N * ns, 2 * M * ns)
    for s in 1:ns
        src = 2 * (s - 1) * M + 1 : 2 * s * M
        trg = 2 * (s - 1) * N + 1 : 2 * s * N
        Bhat[trg, src] = Bplus
    end

    mu = randn(2 * M * ns)
    λ_ref = Bhat * mu
    λ = LaplaceMFS.multispheres_mu_to_lambda(mats, mu)
    @test norm(λ - λ_ref) / norm(λ_ref) < 1e-13

    muc = randn(2 * M * ns) .+ im * randn(2 * M * ns)
    λ_ref_c = ComplexF64.(Bhat) * muc
    λ_c = LaplaceMFS.multispheres_mu_to_lambda(mats, muc)
    @test norm(λ_c - λ_ref_c) / norm(λ_ref_c) < 1e-13

    λ_out = similar(λ_c)
    LaplaceMFS.multispheres_mu_to_lambda!(λ_out, mats, muc)
    @test norm(λ_out - λ_ref_c) / norm(λ_ref_c) < 1e-13
end

@testset "Multisphere Ghat Dense Matrix" begin
    r = 1.0
    r_p = 0.8
    M = 62
    N = 42
    centers = [0.0 0.0 0.0; 2.8 0.1 -0.4; -0.3 2.9 0.2]
    ns = size(centers, 1)
    mats = LaplaceMFS.SphereMats(r, r_p, M, N, 1e-12)
    Gdense = LaplaceMFS.multispheres_G(r, r_p, M, N, centers)

    function dense_ghat_matrix(eps_r)
        VT = promote_type(Float64, typeof(eps_r))
        Uadj = VT.(mats.U_B')
        Vadj = VT.(mats.Vt_B')
        Sinv = VT.(mats.S_B_inv)
        Bhat = zeros(VT, 2 * N * ns, 2 * M * ns)
        Bblkdiag = zeros(VT, 2 * M * ns, 2 * N * ns)
        for s in 1:ns
            src = 2 * (s - 1) * M + 1 : 2 * s * M
            trg = 2 * (s - 1) * N + 1 : 2 * s * N
            block = Vadj * (Diagonal(Sinv) * Uadj)
            Bhat[trg, src] = block
            Bblkdiag[src, trg] = VT.(mats.B)
        end
        return Matrix{VT}(I, 2 * M * ns, 2 * M * ns) + VT.(Gdense) * Bhat - Bblkdiag * Bhat
    end

    Ghat = LaplaceMFS.multispheres_Ghat(mats, centers, 2.5)
    Ghat_ref = dense_ghat_matrix(2.5)
    x = randn(2 * M * ns)
    y_ref = Ghat_ref * x
    y = Ghat * x
    @test norm(y - y_ref) / norm(y_ref) < 1e-13

    Ghat_c = LaplaceMFS.multispheres_Ghat(mats, centers, 2.5 + 0.1im)
    Ghat_ref_c = dense_ghat_matrix(2.5 + 0.1im)
    xc = randn(2 * M * ns) .+ im * randn(2 * M * ns)
    y_ref_c = Ghat_ref_c * xc
    y_c = Ghat_c * xc
    @test norm(y_c - y_ref_c) / norm(y_ref_c) < 1e-13
end

@testset "Multisphere Ghat FMM LinearMap" begin
    r = 1.0
    r_p = 0.7
    M = 62
    N = 42
    centers = [0.0 0.0 0.0; 2.8 0.1 -0.4; -0.3 2.9 0.2]
    mats = LaplaceMFS.SphereMats(r, r_p, M, N, 1e-12)

    Ghat = LaplaceMFS.multispheres_Ghat(mats, centers, 2.5)
    Ghat_fmm = LaplaceMFS.multispheres_Ghat_fmm(mats, centers, 2.5, 1e-13)
    x = randn(2 * M * size(centers, 1))
    y = Ghat * x
    y_fmm = Ghat_fmm * x
    @test norm(y_fmm - y) / norm(y) < 1e-10

    Ghat_c = LaplaceMFS.multispheres_Ghat(mats, centers, 2.5 + 0.1im)
    Ghat_fmm_c = LaplaceMFS.multispheres_Ghat_fmm(mats, centers, 2.5 + 0.1im, 1e-13)
    xc = randn(2 * M * size(centers, 1)) .+ im * randn(2 * M * size(centers, 1))
    y_c = Ghat_c * xc
    y_fmm_c = Ghat_fmm_c * xc
    @test norm(y_fmm_c - y_c) / norm(y_c) < 1e-10
end
