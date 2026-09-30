using LinearAlgebra, Krylov
import HybridSolve
include("axisym_reference.jl")

fibonacci_sphere(n) = reduce(hcat, [begin z = 1 - 2(i + 0.5) / n; ρ = sqrt(1 - z^2); g = π * (3 - sqrt(5)) * i
                                     [ρ * cos(g), ρ * sin(g), z] end for i in 0:n-1])
outside_all(T, C) = T[:, [j for j in axes(T, 2) if all(norm(T[:, j] - C[s, :]) > 1.01 for s in axes(C, 1))]]
# targets on the shells |x - c_i| = 1.25 and inside every gap (a quarter gap off the mid-plane)
function chain_targets(zs, g)
    C = [zeros(length(zs)) zeros(length(zs)) zs]
    far = outside_all(reduce(hcat, [[0, 0, z] .+ 1.25 .* fibonacci_sphere(40) for z in zs]), C)
    gap = reduce(hcat, [[s, 0.0, zs[k] + 1 + g / 2 + h * g / 4] for k in 1:length(zs)-1 for s in (0.0, 0.05, 0.1) for h in (-1, 1)])
    return C, far, gap
end
maxrel(u, v) = maximum(abs, u - v) / maximum(abs, v)

@testset "axial multipole kernels" begin
    mp = AxialMultipole([0.1, -0.2, 0.3], normalize([0.3, 0.4, 0.8]), 0.7)
    x, n = [0.9, 0.5, -0.4], normalize([0.2, -1.0, 0.5])
    orders = 0:5
    pot, grad = LaplaceMFS._axial_multipole_pot_grad(x, mp, orders)
    # ψ̃_m = ℓ^(m+1)/m! (e·∇_y)^m G: m = 0, 1, 2 by finite differences in y
    G(y) = 1 / (4π * norm(x .- y)); h = 1e-4
    @test pot[1] ≈ mp.ℓ * G(mp.y) rtol = 1e-12
    @test pot[2] ≈ mp.ℓ^2 * (G(mp.y .+ h .* mp.e) - G(mp.y .- h .* mp.e)) / 2h rtol = 1e-7
    @test pot[3] ≈ mp.ℓ^3 * (G(mp.y .+ h .* mp.e) - 2G(mp.y) + G(mp.y .- h .* mp.e)) / h^2 / 2 rtol = 1e-5
    fd = (LaplaceMFS._axial_multipole_pot_grad(x .+ 1e-6 .* n, mp, orders)[1] .-
          LaplaceMFS._axial_multipole_pot_grad(x .- 1e-6 .* n, mp, orders)[1]) ./ 2e-6
    @test vec(n' * grad) ≈ fd rtol = 1e-7
end

@testset "multipoles and limit-set nodes: dense vs FMM operators" begin
    d = 2.02
    C = [0.0 0.0 0.0; d 0.0 0.0; d / 2 d*sqrt(3)/2 0.0]
    for eps in (2.5, -2.0 + 0.3im)
        sys = MultiSphereLines(1.0, 0.5, 243, 201, C, eps, 1e-13; n_line = 5, multipole_order = 5, limit_words = (4,))
        @test all(s -> length(s.pm) == 2 && length(s.qm) == 2 && s.morders == 2:5, sys.spheres)
        @test all(s -> size(s.pl_pos, 2) > 2 * 5, sys.spheres)          # limit-set nodes on top of the two lines
        x = multispheres_uniform_rhs(sys, [0.3, -0.2, 1.0])
        @test multispheres_Ghat_fmm(sys, 1e-13) * x ≈ multispheres_Ghat(sys) * x rtol = 1e-10
        lam = multispheres_mu_to_lambda(sys, x)
        @test multispheres_G_fmm(sys, 1e-13) * lam ≈ multispheres_G(sys) * lam rtol = 1e-10
        # exterior evaluation includes the multipoles: FMM path vs the dense source matrices
        T = [3.5 -1.2 1.0; 0.4 2.0 -1.6; 0.9 -0.7 0.2]
        ext = sum(LaplaceMFS._direct_pot(T, sp.p_pos, sp.pl_pos, lam[LaplaceMFS._cols(sys, s)][1:LaplaceMFS._next(sp)], sp.pm, sp.morders)
                  for (s, sp) in enumerate(sys.spheres))
        @test eval_exterior_pot(sys, lam, 1e-13, T) ≈ ext rtol = 1e-10
    end
end

@testset "plasmonic pair, gap 0.01: multipoles + dense solve" begin
    g = 0.01; zs = [0.0, 2 + g]
    C, far, gap = chain_targets(zs, g)
    T = hcat(far, gap); nf = size(far, 2)
    eps = -2.0 + 0.3im
    ref = axisym_reference(zs, eps, 1500, T)
    # measured: default lines + one-body GMRES 2e-2 in the gap; multipoles to order 8 + dense solve 7e-9
    sys = MultiSphereLines(1.0, 0.5, 614, 513, C, eps, 1e-13; multipole_order = 8)
    b = multispheres_uniform_rhs(sys, [0.0, 0.0, 1.0])
    lam, info = multispheres_dense_solve(sys, b; cutoff = 1e-15)
    u = eval_exterior_pot(sys, lam, 1e-13, T)
    @test maxrel(u[nf+1:end], ref[nf+1:end]) < 1e-7
    @test maxrel(u[1:nf], ref[1:nf]) < 1e-8
end

@testset "block Gauss-Seidel solve: chain of three, gap 0.01" begin
    g = 0.01; zs = [0.0, 2 + g, 4 + 2g]
    C, far, gap = chain_targets(zs, g)
    T = hcat(far, gap); nf = size(far, 2)
    for eps in (2.5, -2.0 + 1.0im)
        ref = axisym_reference(zs, eps, 1500, T)
        sys = MultiSphereLines(1.0, 0.5, 614, 513, C, eps, 1e-13; multipole_order = 8)
        b = multispheres_uniform_rhs(sys, [0.0, 0.0, 1.0])
        lg, sg = multispheres_solve(sys, b; preconditioner = :gauss_seidel)
        lo, so = multispheres_solve(sys, b; preconditioner = :onebody)
        @test sg.solved && so.solved
        @test sg.niter < so.niter                                  # measured 14 vs 23 (2.5), 48 vs 84 (-2+i)
        ug = eval_exterior_pot(sys, lg, 1e-13, T)
        @test maxrel(ug[nf+1:end], ref[nf+1:end]) < 1e-7           # measured 2.5e-9 (2.5), 1.2e-8 (-2+i)
        @test maxrel(ug[1:nf], ref[1:nf]) < 1e-8
    end
end

@testset "triangle: limit-set nodes vs HybridSolve" begin
    d = 2.01
    C = [0.0 0.0 0.0; d 0.0 0.0; d / 2 d*sqrt(3)/2 0.0]
    E = normalize([0.3, -0.2, 1.0])
    far = outside_all(reduce(hcat, [C[s, :] .+ 1.25 .* fibonacci_sphere(40) for s in 1:3]), C)
    ref = HybridSolve.eval_exterior_pot(HybridSolve.hybrid_solve(C, ones(3), fill(2.5, 3), Float64[], zeros(3, 0);
                                        p = 100, im = 16, efield = E), far)
    err(kw) = begin
        sys = MultiSphereLines(1.0, 0.5, 614, 513, C, 2.5, 1e-13; multipole_order = 8, kw...)
        lam, _ = multispheres_dense_solve(sys, multispheres_uniform_rhs(sys, E))
        maxrel(eval_exterior_pot(sys, lam, 1e-13, far), ref)
    end
    e0, e1 = err((;)), err((; limit_words = (4, 6)))
    @test e1 < e0 / 5                                              # measured 5e-7 -> 3e-8
    @test e1 < 1e-7
end
