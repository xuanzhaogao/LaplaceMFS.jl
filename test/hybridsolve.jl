# Cross-validation of the multi-sphere MFS solve against HybridSolve.jl, an independent
# image-charge + spherical-harmonic reference (HybridMD). Exterior point-charge excitation;
# all spheres share one radius and one eps_r, as LaplaceMFS requires.
#
# HybridSolve is imported, not `using`'d: both packages export `eval_exterior_pot`.
# `sph_tol = 1e3` is passed explicitly: HybridMD's cut-off drops the coupling of sphere
# pairs more than `sph_tol` radii apart, and HybridSolve versions before the `Inf` default
# used 4, which is off by 3.6e-3 for the cube below.

import HybridSolve
using Krylov, LinearAlgebra

# Point-charge RHS: rows M+1..2M of each sphere block hold (eps_r - 1) ∂ₙu_inc.
function _pointcharge_rhs(r, M, eps_r, centers, charge_pos, charges)
    ns = size(centers, 1)
    pts = load_sphdes_N(M)
    rhs = zeros(2M * ns)
    for s in 1:ns, i in 1:M
        n = vec(pts[i, :])
        trg = vec(centers[s, :]) .+ r .* n
        rhs[2(s - 1) * M + M + i] = (eps_r - 1) * sum(charges[k] * laplace3d_grad(charge_pos[:, k], trg, n)
                                                      for k in eachindex(charges))
    end
    return rhs
end

function _hybrid_reference(centers, eps_r, charges, charge_pos, targets; p)
    ns = size(centers, 1)
    sol = HybridSolve.hybrid_solve(centers, ones(ns), fill(eps_r, ns), charges, charge_pos;
                                   p = p, im = 8, sph_tol = 1e3)
    return HybridSolve.eval_exterior_pot(sol, targets)
end

function _mfs_ghat(centers, eps_r, charges, charge_pos, targets; M, N, r_p, fmm = false)
    mats = SphereMats(1.0, r_p, M, N, eps_r, 1e-13)
    rhs = _pointcharge_rhs(1.0, M, eps_r, centers, charge_pos, charges)
    Gh = fmm ? multispheres_Ghat_fmm(mats, centers, 1e-13) : multispheres_Ghat(mats, centers)
    mu, stats = Krylov.gmres(Gh, rhs; rtol = 1e-13, atol = 1e-15, itmax = 500)
    @test stats.solved
    lambda = multispheres_mu_to_lambda(mats, mu)
    return eval_exterior_pot(centers, N, lambda, r_p, 1e-13, targets)
end

function _mfs_dense_ls(centers, eps_r, charges, charge_pos, targets; M, N, r_p)
    G = LaplaceMFS.multispheres_G(1.0, r_p, M, N, centers, eps_r)
    rhs = _pointcharge_rhs(1.0, M, eps_r, centers, charge_pos, charges)
    return eval_exterior_pot(centers, N, G \ rhs, r_p, 1e-13, targets)
end

_relerr(u, ref) = norm(u - ref) / norm(ref)

@testset "two spheres, 1a gap, two charges vs HybridSolve" begin
    centers = [0.0 0.0 0.0; 0.0 0.0 3.0]
    charge_pos = [0.0 1.8; 0.0 0.0; -2.0 1.5]
    charges = [1.0, -0.5]
    targets = hcat(vcat(collect(range(1.2, 4.0; length = 6))', zeros(2, 6)),
                   [0.0 1.3 -1.4; 0.0 0.0 0.0; -1.6 0.7 4.5])
    ref = _hybrid_reference(centers, 2.5, charges, charge_pos, targets; p = 20)
    # measured 7.2e-9 for both solve paths at M = 614, N = 513, r_p = 0.5
    kw = (M = 614, N = 513, r_p = 0.5)
    @test _relerr(_mfs_ghat(centers, 2.5, charges, charge_pos, targets; kw...), ref) < 5e-8
    @test _relerr(_mfs_dense_ls(centers, 2.5, charges, charge_pos, targets; kw...), ref) < 5e-8
end

@testset "two spheres, 0.05a gap vs HybridSolve" begin
    # geometry of docs/figures/precond.jl; all five targets are exterior at R = 2.05
    centers = [0.0 0.0 0.0; 0.0 0.0 2.05]
    charge_pos = reshape([0.0, 0.0, -2.0], 3, 1)
    targets = [0.0 0.0 1.3 -1.4 0.9; 0.0 0.9 0.0 0.0 -1.1; -1.6 2.6 0.7 1.9 0.3]
    ref = _hybrid_reference(centers, 2.5, [1.0], charge_pos, targets; p = 40)
    # measured at M = 614, N = 513, r_p = 0.5: Ghat 1.1e-6, dense LS 2.7e-6 (cf. findings §3.7)
    kw = (M = 614, N = 513, r_p = 0.5)
    @test _relerr(_mfs_ghat(centers, 2.5, [1.0], charge_pos, targets; kw...), ref) < 1e-5
    @test _relerr(_mfs_dense_ls(centers, 2.5, [1.0], charge_pos, targets; kw...), ref) < 1e-5
end

@testset "8-sphere cube, FMM path vs HybridSolve" begin
    centers = reduce(vcat, [[x y z] for x in (-1.5, 1.5) for y in (-1.5, 1.5) for z in (-1.5, 1.5)])
    charge_pos = [0.0 0.2; 0.0 -0.1; 0.0 3.2]
    charges = [1.0, -0.4]
    targets = [3.5 -3.5 0.0 0.0 0.0 0.0 1.0;
               0.0 0.0 3.5 -3.5 0.0 0.0 0.3;
               0.0 0.0 0.0 0.0 3.5 -3.5 0.1]
    ref = _hybrid_reference(centers, 2.5, charges, charge_pos, targets; p = 20)
    # measured 2.8e-9 at M = 366, N = 314, r_p = 0.5 (4.9e-12 at M = 614, N = 513)
    u = _mfs_ghat(centers, 2.5, charges, charge_pos, targets; M = 366, N = 314, r_p = 0.5, fmm = true)
    @test _relerr(u, ref) < 3e-8
end
