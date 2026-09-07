using LaplaceMFS, LinearAlgebra, Krylov, Printf

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

dense(L, n) = (A = zeros(size(L,1), n); e = zeros(n);
               for j in 1:n; fill!(e,0); e[j]=1; A[:,j] = L * e; end; A)

const a = 1.0; const eps_r = 2.5; const q = 1.0

# ---------------------------------------------------------------
# 1.  Is  Ghat == G*Bhat + (I - P)  exactly?
# ---------------------------------------------------------------
println("="^76)
println("1. Structural identity:  Ghat  vs  G*Bhat + (I - P_blk)")
println("="^76)
let M = 243, N = 201, rp = 0.5, R = 3.0
    mats = SphereMats(a, rp, M, N, eps_r, 1e-12)
    C = [0.0 0.0 0.0; 0.0 0.0 R]
    ns = 2; nr = 2*M*ns; nc = 2*N*ns
    G  = LaplaceMFS.multispheres_G(a, rp, M, N, C, eps_r)
    Bp = mats.Vt_B' * Diagonal(mats.S_B_inv) * mats.U_B'      # one-body pseudo-inverse
    Bhat = zeros(nc, nr); Bblk = zeros(nr, nc)
    for s in 1:ns
        rs = 2*(s-1)*M+1 : 2*s*M;  cs = 2*(s-1)*N+1 : 2*s*N
        Bhat[cs, rs] = Bp;  Bblk[rs, cs] = mats.B
    end
    P  = Bblk * Bhat
    Gh = dense(multispheres_Ghat(mats, C), nr)
    claim = G * Bhat + (I - P)
    @printf("  ||Ghat - (G*Bhat + I - P)||  / ||Ghat||  = %.3e\n",
            norm(Gh - claim) / norm(Gh))
    @printf("  ||I - P||  (size of the discarded complement) = %.3e\n", norm(I - P))
    @printf("  rank deficiency per sphere = 2M - 2N = %d\n", 2M - 2N)
end

# ---------------------------------------------------------------
# 2.  Does the preconditioned solve agree with direct least squares
#     as the spheres approach?
# ---------------------------------------------------------------
println()
println("="^76)
println("2. Ghat+GMRES vs direct least squares, closing the gap")
println("="^76)
let M = 614, N = 513, rp = 0.5
    mats = SphereMats(a, rp, M, N, eps_r, 1e-12)
    tg = [0.0 0.0 1.3 -1.4  0.9;
          0.0 0.9 0.0  0.0 -1.1;
         -1.6 2.6 0.7  1.9  0.3]
    @printf("%6s %6s %7s %13s %13s %12s %12s %6s\n",
            "R/a", "gap/a", "iters", "lambda_rel", "field_rel", "LSres_dense", "LSres_ghat", "ok")
    for R in (8.0, 4.0, 3.0, 2.5, 2.2, 2.05)
        C = [0.0 0.0 0.0; 0.0 0.0 R]
        src = [0.0, 0.0, -2.0]
        rhs = ms_rhs(a, M, eps_r, C, src, q)

        G = LaplaceMFS.multispheres_G(a, rp, M, N, C, eps_r)
        lam_ls = G \ rhs
        res_ls = norm(G*lam_ls - rhs) / norm(rhs)

        Gh = multispheres_Ghat(mats, C)
        mu, st = Krylov.gmres(Gh, rhs; rtol=1e-13, atol=1e-15, itmax=500)
        lam_gh = multispheres_mu_to_lambda(mats, mu)
        res_gh = norm(G*lam_gh - rhs) / norm(rhs)

        u_ls = LaplaceMFS.eval_exterior_pot(C, N, lam_ls, rp, 1e-13, tg)
        u_gh = LaplaceMFS.eval_exterior_pot(C, N, lam_gh, rp, 1e-13, tg)

        @printf("%6.2f %6.2f %7d %13.3e %13.3e %12.3e %12.3e %6s\n",
                R, R - 2a, st.niter,
                norm(lam_gh - lam_ls)/norm(lam_ls),
                norm(u_gh - u_ls)/norm(u_ls),
                res_ls, res_gh, st.solved)
        flush(stdout)
    end
end
