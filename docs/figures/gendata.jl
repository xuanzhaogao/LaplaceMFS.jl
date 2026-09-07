using LaplaceMFS, LinearAlgebra, Krylov, Printf

# ---------- exact single-sphere reaction potential ----------
function reac_analytic(targets, center, a, src, q, eps_r; nmax=8000, tol=1e-15)
    dv = src .- center; d = norm(dv); dh = dv ./ d
    out = zeros(size(targets, 2))
    for k in 1:size(targets, 2)
        rel = targets[:, k] .- center; rho = norm(rel); ct = dot(rel, dh) / rho
        Pm1, P0 = 0.0, 1.0; s = 0.0; small = 0
        for n in 1:nmax
            Pn = ((2n - 1) * ct * P0 - (n - 1) * Pm1) / n
            Pm1, P0 = P0, Pn
            Bn = -n * (eps_r - 1) / (n * (eps_r + 1) + 1)
            term = Bn * a^(2n+1) / (d^(n+1) * rho^(n+1)) * Pn
            s += term
            small = abs(term) <= tol * max(abs(s), 1e-300) ? small + 1 : 0
            small >= 8 && break
        end
        out[k] = q / (4pi) * s
    end
    return out
end

function ss_rhs(r, M, eps_r, src, q)
    pts = load_sphdes_N(M); rhs = zeros(2M)
    for i in 1:M
        n = vec(pts[i, :]); trg = r .* n
        rhs[M + i] = (eps_r - 1) * q * LaplaceMFS.laplace3d_grad(src, trg, n)
    end
    return rhs
end

function ms_rhs(r, M, eps_r, centers, src, q)
    ns = size(centers, 1); pts = load_sphdes_N(M); rhs = zeros(2 * M * ns)
    for s in 1:ns
        c = vec(centers[s, :])
        for i in 1:M
            n = vec(pts[i, :]); trg = c .+ r .* n
            rhs[2*(s-1)*M + M + i] = (eps_r - 1) * q * LaplaceMFS.laplace3d_grad(src, trg, n)
        end
    end
    return rhs
end

function lattice(n, L)
    C = Matrix{Float64}(undef, n^3, 3); k = 0
    for i in 0:n-1, j in 0:n-1, m in 0:n-1
        k += 1; C[k, :] = [i*L, j*L, m*L]
    end
    return C
end

const a = 1.0; const eps_r = 2.5; const q = 1.0
const OUT = (d = joinpath(@__DIR__, "data"); mkpath(d); d)

targets_local(a) = begin
    cols = Vector{Float64}[]
    for rho in (1.05a, 1.3a, 1.8a, 2.5a), (ct, st) in ((1.0,0.0), (0.4,sqrt(0.84)), (-0.6,0.8))
        push!(cols, [rho*st, 0.0, rho*ct])
    end
    T = Matrix{Float64}(undef, 3, length(cols)); for (j,c) in enumerate(cols); T[:,j]=c; end; T
end

# =========== 1. convergence vs proxy degree, per d/a ===========
open("$OUT/f1_convergence.csv", "w") do io
    println(io, "da,M,N,t,r_p,relerr,resid")
    tg = targets_local(a)
    pairs = [(243,201), (366,314), (546,451), (723,614), (926,801), (1202,1014), (1459,1251)]
    for da in (3.0, 2.0, 1.5, 1.2)
        src = [0.0, 0.0, -da * a]
        uref = reac_analytic(tg, [0.0,0.0,0.0], a, src, q, eps_r)
        for (M, N) in pairs
            rp = 0.7
            B = LaplaceMFS.singlesphere_B(a, rp, M, N, eps_r)
            rhs = ss_rhs(a, M, eps_r, src, q)
            x = B \ rhs
            resid = norm(B*x - rhs) / norm(rhs)
            u = LaplaceMFS.eval_exterior_pot(zeros(1,3), N, x, rp, 1e-13, tg)
            e = norm(u - uref) / norm(uref)
            @printf(io, "%.2f,%d,%d,%.1f,%.2f,%.6e,%.6e\n", da, M, N, sqrt(2N)-1, rp, e, resid)
            flush(io)
        end
        println("f1 done da=$da")
    end
end

# =========== 2. r_p sweep: error vs residual inversion ===========
open("$OUT/f2_rp.csv", "w") do io
    println(io, "r_p,relerr,resid,cond")
    tg = targets_local(a); da = 1.2
    src = [0.0, 0.0, -da * a]
    uref = reac_analytic(tg, [0.0,0.0,0.0], a, src, q, eps_r)
    M, N = 723, 614
    rhs = ss_rhs(a, M, eps_r, src, q)
    for rp in 0.35:0.05:0.95
        B = LaplaceMFS.singlesphere_B(a, rp, M, N, eps_r)
        sv = svdvals(B); cnd = sv[1] / sv[end]
        x = B \ rhs
        resid = norm(B*x - rhs) / norm(rhs)
        u = LaplaceMFS.eval_exterior_pot(zeros(1,3), N, x, rp, 1e-13, tg)
        e = norm(u - uref) / norm(uref)
        @printf(io, "%.2f,%.6e,%.6e,%.6e\n", rp, e, resid, cnd); flush(io)
    end
    println("f2 done")
end

# =========== 3. many-body far-field decay: O(L^-4) ===========
open("$OUT/f3_decay.csv", "w") do io
    println(io, "nspheres,L,relerr,iters")
    M, N, rp = 614, 513, 0.5
    mats = SphereMats(a, rp, M, N, eps_r, 1e-12)
    tg = targets_local(a)
    src = [0.0, 0.0, -2.0 * a]
    uref = reac_analytic(tg, [0.0,0.0,0.0], a, src, q, eps_r)
    for (nside, Ls) in ((1, (6.0,8.0,12.0,16.0,24.0,32.0,48.0,64.0,96.0)),
                        (3, (6.0,8.0,12.0,16.0,24.0,32.0,48.0,64.0,96.0)))
        for L in Ls
            C = nside == 1 ? [0.0 0.0 0.0; 0.0 0.0 L] : lattice(3, L)
            rhs = ms_rhs(a, M, eps_r, C, src, q)
            Gf = multispheres_Ghat_fmm(mats, C, 1e-14)
            mu, st = Krylov.gmres(Gf, rhs; rtol=1e-13, atol=1e-15, itmax=300)
            u = LaplaceMFS.eval_exterior_pot(C, N, multispheres_mu_to_lambda(mats, mu), rp, 1e-13, tg)
            ns = nside == 1 ? 2 : 27
            @printf(io, "%d,%.1f,%.6e,%d\n", ns, L, norm(u - uref)/norm(uref), st.niter); flush(io)
        end
        println("f3 done nside=$nside")
    end
end

# =========== 4. iteration count vs sphere count ===========
open("$OUT/f4_scaling.csv", "w") do io
    println(io, "nspheres,unknowns,iters,solve_s")
    M, N, rp = 243, 201, 0.5
    mats = SphereMats(a, rp, M, N, eps_r, 1e-12)
    src = [-2.0, 0.0, 0.0]
    for n in (2, 3, 4, 5, 6, 8)
        C = lattice(n, 3.0)
        rhs = ms_rhs(a, M, eps_r, C, src, q)
        Gf = multispheres_Ghat_fmm(mats, C, 1e-10)
        Krylov.gmres(Gf, rhs; rtol=1e-8, atol=1e-12, itmax=3)
        t0 = time()
        mu, st = Krylov.gmres(Gf, rhs; rtol=1e-10, atol=1e-13, itmax=300)
        dt = time() - t0
        @printf(io, "%d,%d,%d,%.3f\n", n^3, length(rhs), st.niter, dt); flush(io)
        println("f4 done ns=$(n^3) iters=$(st.niter)")
    end
end

println("ALL DONE")
