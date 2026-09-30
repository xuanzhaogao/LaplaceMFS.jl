# Test helper: exact reference for unit-radius spheres on the z axis in a uniform field along z, for any real
# or complex ε (coaxial multipole expansion, order m = 0). Outside sphere i its scattered field is
# Σ_n A[i,n] P_n(cos θ_i) / ρ_i^(n+1); the arriving regular field L[i,n] gives A = -n(ε-1)/(nε+n+1) L, and
# P_l/ρ^(l+1) about z_j translates to Σ_n s C(n+l, n) ρ^n P_n / |d|^(n+l+1) about z_i = z_j + d,
# s = (-1)^n for d > 0 and (-1)^l for d < 0. Converged to ~1e-15 with nmax = 1500 for gaps ≥ 0.01.
function axisym_reference(zs, eps, nmax, X)
    K = length(zs)
    lf = cumsum([0.0; log.(1:2nmax+2)])
    idx(i, n) = (i - 1) * nmax + n
    Dn = [-n * (eps - 1) / (n * eps + n + 1) for i in 1:K for n in 1:nmax]
    Tm = zeros(K * nmax, K * nmax)
    for i in 1:K, j in 1:K
        i == j && continue
        d = zs[i] - zs[j]; ad = abs(d)
        for n in 1:nmax, l in 1:nmax
            Tm[idx(i, n), idx(j, l)] = (d > 0 ? (-1)^n : (-1)^l) * exp(lf[n+l+1] - lf[n+1] - lf[l+1] - (n + l + 1) * log(ad))
        end
    end
    L = zeros(typeof(complex(eps)), K * nmax)
    for i in 1:K; L[idx(i, 1)] = -1; end
    A = (I - Dn .* Tm) \ (Dn .* L)
    u = zeros(typeof(complex(eps)), size(X, 2))
    for j in axes(X, 2), i in 1:K
        x, y, z = X[1, j], X[2, j], X[3, j] - zs[i]
        ρ = sqrt(x^2 + y^2 + z^2); c = z / ρ
        P0, P1 = 1.0, c; ρn = 1 / ρ^2
        acc = A[idx(i, 1)] * P1 * ρn
        for n in 2:nmax
            P0, P1 = P1, ((2n - 1) * c * P1 - (n - 1) * P0) / n
            ρn /= ρ
            acc += A[idx(i, n)] * P1 * ρn
        end
        u[j] += acc
    end
    return isreal(eps) ? real.(u) : u
end
