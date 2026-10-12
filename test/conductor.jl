# Neutral formulation of MultiSphereLines: flux rows divided by ε (ε = Inf is the perfect conductor) and one row
# per sphere fixing the net charge of its exterior sources to zero.
using LinearAlgebra, Krylov
include("chain_helpers.jl")

net_charges(sys, lam) = [begin
                             sp = sys.spheres[s]; c = lam[LaplaceMFS._cols(sys, s)]
                             np, nl = size(sp.p_pos, 2), size(sp.pl_pos, 2)
                             sum(c[1:np]) + sum(c[np .+ 4 .* (0:nl-1) .+ 1])
                         end for s in 1:LaplaceMFS.nspheres(sys)]

@testset "neutral formulation: rows and right-hand side" begin
    C = [0.0 0.0 0.0; 0.0 0.0 2.05]
    X = reshape([0.0, 0.0, -1.5], 3, 1)
    for eps in (10.0, Inf)
        sys = MultiSphereLines(1.0, 0.5, 243, 201, C, eps, 1e-13; charge_pos = X, n_line = 4)
        G = multispheres_G(sys)
        b = multispheres_uniform_rhs(sys, [0.3, -0.2, 1.0]) .+ multispheres_pointcharge_rhs(sys, X, [1.0])
        for s in 1:2
            sp = sys.spheres[s]; m = size(sp.targets, 2)
            rows = LaplaceMFS._rows(sys, s)
            @test length(rows) == 2m + 1
            # last row: 1/r at the proxy charges and the q of every line node, nothing else
            np, nl = size(sp.p_pos, 2), size(sp.pl_pos, 2)
            w = zeros(size(G, 2)); w[LaplaceMFS._cols(sys, s)[[1:np; np .+ 4 .* (0:nl-1) .+ 1]]] .= 1
            @test G[rows[end], :] == w
            @test b[rows[end]] == 0
        end
    end
    @test_throws ArgumentError MultiSphereLines(1.0, 0.5, 243, 201, C, Inf, 1e-13; neutral = false)
end

@testset "neutral formulation agrees with the plain one at moderate ε" begin
    g = 0.01; zs = [0.0, 2 + g]
    C, far, gap = chain_targets(zs, g)
    T = hcat(far, gap)
    u(neutral) = begin
        sys = MultiSphereLines(1.0, 0.5, 614, 513, C, 2.5, 1e-13; neutral)
        lam, _ = multispheres_solve(sys, multispheres_uniform_rhs(sys, [0.0, 0.0, 1.0]))
        eval_exterior_pot(sys, lam, 1e-13, T)
    end
    @test maxrel(u(true), u(false)) < 1e-7
end

@testset "conductor limit: pair at gap 0.01 vs coaxial reference" begin
    g = 0.01; zs = [0.0, 2 + g]
    C, far, gap = chain_targets(zs, g)
    T = hcat(far, gap); nf = size(far, 2)
    E = [0.0, 0.0, 1.0]
    for eps in (1e8, Inf)
        ref = axisym_reference(zs, isfinite(eps) ? eps : 1e16, 1500, T)
        sys = MultiSphereLines(1.0, 0.5, 614, 513, C, eps, 1e-13)
        b = multispheres_uniform_rhs(sys, E)
        # measured (gap error): Gauss-Seidel 6e-9 in 15 iterations, one-body 2e-9 in 21, dense 1e-9
        for pc in (:gauss_seidel, :onebody)
            lam, st = multispheres_solve(sys, b; preconditioner = pc)
            @test st.solved && st.niter < 40
            v = eval_exterior_pot(sys, lam, 1e-13, T)
            @test maxrel(v[nf+1:end], ref[nf+1:end]) < 1e-7
            @test maxrel(v[1:nf], ref[1:nf]) < 1e-8
            @test maximum(abs, net_charges(sys, lam)) < 1e-10 * maximum(abs, lam)
        end
        lam, _ = multispheres_dense_solve(sys, b; cutoff = 1e-15)
        @test maxrel(eval_exterior_pot(sys, lam, 1e-13, T), ref) < 1e-7
    end
    # the plain formulation loses the net-charge condition at large ε: measured 1e-2 in the gap at ε = 1e8
    ref = axisym_reference(zs, 1e8, 1500, T)
    sys = MultiSphereLines(1.0, 0.5, 614, 513, C, 1e8, 1e-13; neutral = false)
    lam, _ = multispheres_dense_solve(sys, multispheres_uniform_rhs(sys, E); cutoff = 1e-15)
    @test maxrel(eval_exterior_pot(sys, lam, 1e-13, T), ref) > 1e-5
end

@testset "neutral formulation: dense vs FMM operators at ε = 10 and Inf" begin
    C = [0.0 0.0 0.0; 0.0 0.0 2.02; 2.3 0.0 1.0]
    for eps in (10.0, Inf)
        sys = MultiSphereLines(1.0, 0.5, 243, 201, C, eps, 1e-13; n_line = 5, multipole_order = 4)
        x = multispheres_uniform_rhs(sys, [0.3, -0.2, 1.0])
        @test multispheres_Ghat_fmm(sys, 1e-13) * x ≈ multispheres_Ghat(sys) * x rtol = 1e-10
        lam = multispheres_mu_to_lambda(sys, x)
        @test multispheres_G_fmm(sys, 1e-13) * lam ≈ multispheres_G(sys) * lam rtol = 1e-10
    end
end
