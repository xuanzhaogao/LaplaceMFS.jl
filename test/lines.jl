using LinearAlgebra, Krylov
import HybridSolve

function _lines_solve(sys, X, q; fmm = false)
    rhs = multispheres_pointcharge_rhs(sys, X, q)
    Gh = fmm ? multispheres_Ghat_fmm(sys, 1e-13) : multispheres_Ghat(sys)
    mu, stats = Krylov.gmres(Gh, rhs; rtol = 1e-13, atol = 1e-15, itmax = 500)
    @test stats.solved
    return multispheres_mu_to_lambda(sys, mu)
end

@testset "line geometry (Eqs. (23)-(27) of refs/stokesmfs.pdf)" begin
    for r in (1.0, 2.0), δ in (0.3, 0.01)
        # the accumulation point of each sphere is the reflection of the other one's
        R = line_accumulation_radius(r, δ)
        c1, c2 = zeros(3), [0.0, 0.0, 2r + δ]
        x1, x2 = c1 .+ R .* [0, 0, 1.0], c2 .- R .* [0, 0, 1.0]
        @test c1 .+ r^2 .* (x2 .- c1) ./ norm(x2 .- c1)^2 ≈ x1 rtol = 1e-12
    end
    @test line_accumulation_radius(1.0, line_critical_gap(1.0, 0.6)) ≈ 0.6
    @test line_accumulation_radius(2.0, line_critical_gap(2.0, 1.2)) ≈ 1.2
    @test line_num_nodes(1.0, line_accumulation_radius(1.0, 1e-3)) == ceil(Int, 8.72 * 3 - 6.15)
    L = line_nodes(zeros(3), [1.0, 0, 0], 0.5, 0.9, 5; Δ = 0.05)
    @test L[:, 1] ≈ [0.9, 0, 0]
    @test all(L[1, 2:end] .< L[1, 1:end-1]) && L[1, end] > 0.5 * 1.05
    @test norm(L[2:3, :]) == 0
end

@testset "dipole kernels" begin
    src, trg, p = [0.1, -0.2, 0.3], [1.2, 0.4, -0.7], [0.3, -0.5, 0.8]
    n, h = normalize([0.2, 1.0, -0.4]), 1e-6
    fd = sum(p[k] * (laplace3d_pot(src .+ h .* (1:3 .== k), trg) - laplace3d_pot(src .- h .* (1:3 .== k), trg)) / 2h for k in 1:3)
    @test laplace3d_dipole_pot(src, trg, p) ≈ fd rtol = 1e-8
    fdn = (laplace3d_dipole_pot(src, trg .+ h .* n, p) - laplace3d_dipole_pot(src, trg .- h .* n, p)) / 2h
    @test laplace3d_dipole_grad(src, trg, p, n) ≈ fdn rtol = 1e-8
end

@testset "MultiSphereLines without lines is the plain system" begin
    r, r_p, M, N, eps_r = 1.0, 0.5, 243, 201, 2.5
    C = [0.0 0.0 0.0; 0.0 0.0 3.0]
    X = [0.0 1.8; 0.0 0.0; -2.0 1.5]; q = [1.0, -0.5]
    sys = MultiSphereLines(r, r_p, M, N, C, eps_r, 1e-13; charge_pos = X)
    @test all(s -> size(s.pl_pos, 2) == 0 && size(s.targets, 2) == M, sys.spheres)
    @test multispheres_G(sys) ≈ LaplaceMFS.multispheres_G(r, r_p, M, N, C, eps_r) rtol = 1e-13
    @test multispheres_pointcharge_rhs(sys, X, q) ≈ multispheres_pointcharge_rhs(r, M, eps_r, C, X, q)
    @test multispheres_uniform_rhs(sys, [0.1, 0.0, 0.7]) ≈ multispheres_uniform_rhs(r, M, eps_r, C, [0.1, 0.0, 0.7])
end

@testset "line placement: close pair and nearby charge" begin
    r, r_p, M, N = 1.0, 0.5, 243, 201
    C = [0.0 0.0 0.0; 0.0 0.0 2.05; 5.0 0.0 0.0]
    X = reshape([5.0, 0.0, 1.1], 3, 1)          # 0.1 above sphere 3
    sys = MultiSphereLines(r, r_p, M, N, C, 2.5, 1e-13; charge_pos = X, n_line = 4)
    s1, s2, s3 = sys.spheres
    # spheres 1, 2: one line each towards the other, first node at the accumulation point
    @test size(s1.pl_pos, 2) == size(s2.pl_pos, 2) == 4
    @test s1.pl_pos[:, 1] ≈ [0, 0, line_accumulation_radius(r, 0.05)]
    @test s2.pl_pos[:, 1] ≈ [0, 0, 2.05 - line_accumulation_radius(r, 0.05)]
    # sphere 3: one line towards the charge, ending at its Kelvin image; the mirrored end is the charge
    @test s3.pl_pos[:, 1] ≈ [5.0, 0, r^2 / 1.1]
    @test s3.ql_pos[:, 1] ≈ X[:, 1]
    # (q, px, py, pz) per node on both lines, two caps of 6 · 4n points each
    @test LaplaceMFS._ncols(s1) == 2N + 2 * 4 * 4
    @test size(s1.targets, 2) == M + 2 * 6 * 16
    @test all(s1.normals[3, M+1:end] .> cos(π / 5) - 1e-12)
    @test all(s1.weights[1:M] .== 1) && all(s1.weights[M+1:end] .< 1)
end

@testset "sphere with a nearby charge vs exact series" begin
    r, r_p, M, N, eps_r = 1.0, 0.5, 614, 513, 2.5
    c = zeros(1, 3); q = [1.0]
    dirs = reduce(hcat, [normalize(v) for v in ([0.2, 0.3, 1.0], [0.25, 0.3, 1.0], [1.0, 0, 0], [0.0, -1, 0.2])])
    X = reshape(1.05 .* dirs[:, 1], 3, 1)       # 0.05 from the surface
    Tout, Tin = 1.02 .* dirs, 0.97 .* dirs
    ext_ref = HybridSolve.single_sphere_pointcharge_exterior(Tout, vec(c), r, eps_r, q[1], vec(X))
    int_ref = HybridSolve.single_sphere_pointcharge_interior(Tin, vec(c), r, eps_r, q[1], vec(X))
    err(sys, lam) = (norm(eval_exterior_pot(sys, lam, 1e-13, Tout) - ext_ref) / norm(ext_ref),
                     norm(eval_total_pot(sys, lam, 1e-13, Tin, X, q) - int_ref) / norm(int_ref))
    # measured without lines: 0.41 (exterior); with 17 nodes: 8e-13 (exterior), 2e-12 (interior)
    sys0 = MultiSphereLines(r, r_p, M, N, c, eps_r, 1e-13; charge_pos = X, n_line = 0)
    @test err(sys0, _lines_solve(sys0, X, q))[1] > 1e-2
    sys = MultiSphereLines(r, r_p, M, N, c, eps_r, 1e-13; charge_pos = X)
    eo, ei = err(sys, _lines_solve(sys, X, q))
    @test eo < 1e-10
    @test ei < 1e-10
end

@testset "close-to-touching spheres vs HybridSolve" begin
    r, r_p, M, N, eps_r = 1.0, 0.5, 614, 513, 2.5
    C = [0.0 0.0 0.0; 0.0 0.0 2.01]
    X = reshape([0.0, 0.0, -2.0], 3, 1); q = [1.0]
    T = [0.0 0.0 1.3 -1.4 0.9 0.05; 0.0 0.9 0.0 0.0 -1.1 0.0; -1.6 2.6 0.7 1.9 0.3 1.005]
    sol = HybridSolve.hybrid_solve(C, ones(2), fill(eps_r, 2), q, X; p = 60, im = 8, sph_tol = 1e3)
    ref = HybridSolve.eval_exterior_pot(sol, T)             # p = 60 vs 90: 2e-10
    relerr(sys, lam) = norm(eval_exterior_pot(sys, lam, 1e-13, T) - ref) / norm(ref)
    # measured: no lines 1.2e-4; 12 nodes per line 2.7e-6 (dense and FMM)
    sys0 = MultiSphereLines(r, r_p, M, N, C, eps_r, 1e-13; n_line = 0)
    @test relerr(sys0, _lines_solve(sys0, X, q)) > 5e-5
    sys = MultiSphereLines(r, r_p, M, N, C, eps_r, 1e-13)
    @test relerr(sys, _lines_solve(sys, X, q)) < 1e-5
    @test relerr(sys, _lines_solve(sys, X, q; fmm = true)) < 1e-5
end

@testset "Ghat dense vs FMM, real and complex eps_r" begin
    r, r_p, M, N = 1.0, 0.5, 243, 201
    C = [0.0 0.0 0.0; 0.0 0.0 2.02; 2.3 0.0 1.0]
    X = reshape([-1.1, 0.0, -0.2], 3, 1)
    for eps_r in (2.5, 2.5 + 0.7im)
        sys = MultiSphereLines(r, r_p, M, N, C, eps_r, 1e-13; charge_pos = X, n_line = 5)
        # a physical right-hand side: B⁺ of a random vector is huge and amplifies rounding
        x = multispheres_pointcharge_rhs(sys, X, [1.0]) .+ multispheres_uniform_rhs(sys, [0.3, -0.2, 1.0])
        @test multispheres_Ghat_fmm(sys, 1e-13) * x ≈ multispheres_Ghat(sys) * x rtol = 1e-10
        G = multispheres_G(sys)
        lam = multispheres_mu_to_lambda(sys, x)
        @test multispheres_Ghat(sys) * x ≈ x .+ G * lam .- reduce(vcat, [sys.B[s] * lam[LaplaceMFS._cols(sys, s)] for s in 1:3]) rtol = 1e-12
    end
end
