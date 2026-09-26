using LinearAlgebra
import HybridSolve

function _solve_dense(r, r_p, M, N, centers, eps_r, charge_pos, charges)
    G = LaplaceMFS.multispheres_G(r, r_p, M, N, centers, eps_r)
    return G \ multispheres_pointcharge_rhs(r, M, eps_r, centers, charge_pos, charges)
end

@testset "multispheres_pointcharge_rhs" begin
    r, M, eps_r = 1.0, 243, 2.5
    centers = [0.0 0.0 0.0; 0.0 0.0 3.0]
    X = [0.0 1.8; 0.0 0.0; -2.0 1.5]; q = [1.0, -0.5]
    rhs = multispheres_pointcharge_rhs(r, M, eps_r, centers, X, q)
    pts = load_sphdes_N(M)
    @test length(rhs) == 2M * 2
    @test all(iszero, rhs[1:M]) && all(iszero, rhs[2M+1:3M])      # potential rows
    i, s = 7, 2
    n = vec(pts[i, :]); y = vec(centers[s, :]) .+ r .* n
    @test rhs[2M + M + i] ≈ (eps_r - 1) * sum(q[k] * laplace3d_grad(X[:, k], y, n) for k in 1:2)
    @test_throws DimensionMismatch multispheres_pointcharge_rhs(r, M, eps_r, centers, X, [1.0])
end

@testset "eval_total_pot: single sphere vs exact series inside and outside" begin
    r, r_p, M, N, eps_r = 1.0, 0.5, 1302, 1059, 2.5
    c = zeros(1, 3); X = reshape([0.0, 0.3, 2.0], 3, 1); q = [1.0]
    lam = _solve_dense(r, r_p, M, N, c, eps_r, X, q)
    Tout = [0.0 1.2 -0.8; 0.0 0.5 0.9; 1.4 0.0 -1.1]
    Tin  = [0.0 0.4 -0.3 0.0; 0.0 0.2 0.5 0.0; 0.5 -0.1 0.2 0.0]
    inc(T) = [q[1] / (4π * norm(T[:, j] - X[:, 1])) for j in axes(T, 2)]
    ext_ref = HybridSolve.single_sphere_pointcharge_exterior(Tout, vec(c), r, eps_r, q[1], vec(X)) .+ inc(Tout)
    int_ref = HybridSolve.single_sphere_pointcharge_interior(Tin, vec(c), r, eps_r, q[1], vec(X))
    # measured 1e-13 (outside) and 4e-14 (inside) at M = 1302, N = 1059
    @test norm(eval_total_pot(c, r, N, lam, r_p, 1e-13, Tout, X, q) - ext_ref) / norm(ext_ref) < 1e-11
    @test norm(eval_total_pot(c, r, N, lam, r_p, 1e-13, Tin, X, q) - int_ref) / norm(int_ref) < 1e-11
end

@testset "eval_total_pot: two spheres, continuity across both surfaces" begin
    r, r_p, M, N, eps_r = 1.0, 0.5, 614, 513, 2.5
    C = [0.0 0.0 0.0; 0.0 0.0 3.0]; X = [0.0 1.8; 0.0 0.0; -2.0 1.5]; q = [1.0, -0.5]
    lam = _solve_dense(r, r_p, M, N, C, eps_r, X, q)
    dirs = reduce(hcat, [normalize(v) for v in ([1.0, 0, 0], [0.2, -0.5, 0.8], [-0.3, 0.1, -0.9])])
    # at M = 614 the potential jump between collocation points is the discretisation's own
    # continuity residual: measured 4.7e-8 (sphere 1) and 2.6e-8 (sphere 2) of max|φ| for h ≤ 1e-9
    h = 1e-10
    for s in 1:2
        out = vec(C[s, :]) .+ (r + h) .* dirs
        in_ = vec(C[s, :]) .+ (r - h) .* dirs
        φo = eval_total_pot(C, r, N, lam, r_p, 1e-13, out, X, q)
        φi = eval_total_pot(C, r, N, lam, r_p, 1e-13, in_, X, q)
        @test maximum(abs.(φo - φi)) < 2e-7 * maximum(abs.(φo))
    end
    # outside both spheres it is the incident field plus the scattered field
    T = [0.0 1.3; 0.0 0.0; -1.6 0.7]
    inc = [sum(q[k] / (4π * norm(T[:, j] - X[:, k])) for k in 1:2) for j in 1:2]
    @test eval_total_pot(C, r, N, lam, r_p, 1e-13, T, X, q) ≈
          eval_exterior_pot(C, N, lam, r_p, 1e-13, T) .+ inc rtol = 1e-12
    # incident = false drops exactly u_inc, inside and outside
    Tmix = hcat(T, [0.2 0.1; -0.3 0.0; 0.4 3.3])      # two exterior, two interior points
    incm = [sum(q[k] / (4π * norm(Tmix[:, j] - X[:, k])) for k in 1:2) for j in 1:4]
    @test eval_total_pot(C, r, N, lam, r_p, 1e-13, Tmix, X, q; incident = false) ≈
          eval_total_pot(C, r, N, lam, r_p, 1e-13, Tmix, X, q) .- incm rtol = 1e-12
    @test eval_total_pot(C, r, N, lam, r_p, 1e-13, T, X, q; incident = false) ≈
          eval_exterior_pot(C, N, lam, r_p, 1e-13, T) rtol = 1e-12
end
