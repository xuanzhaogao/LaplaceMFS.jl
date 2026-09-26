# SphereMats stores the matrices for the single sphere case, together with the SVD of the B matrix for preconditioning.
struct SphereMats{RT, VT}
    r::RT
    r_p::RT
    eps_r::VT
    M::Int
    N::Int

    S_pr::Matrix{VT}
    S_qr::Matrix{VT}
    D_pr::Matrix{VT}
    D_qr::Matrix{VT}
    B::Matrix{VT}

    S_B::Vector{VT}
    U_B::Matrix{VT}
    Vt_B::Matrix{VT}

    S_B_inv::Vector{VT}
end

function SphereMats(r::RT, r_p::RT, M::Int, N::Int, eps_r::ET, tol::TT) where {RT, ET, TT}
    VT = promote_type(RT, ET)
    B = singlesphere_B(r, r_p, M, N, eps_r)

    res = svd(B)
    S_B = VT.(res.S)
    U_B = VT.(res.U)
    Vt_B = VT.(res.Vt)

    tolT = float(real(tol))
    S_B_inv = [abs(s / S_B[1]) > tolT ? inv(s) : zero(VT) for s in S_B]

    return SphereMats(r, r_p, VT(eps_r), M, N, VT.(B[1:M, 1:N]), VT.(B[1:M, N+1:2N]), VT.(B[M+1:2M, 1:N]), VT.(B[M+1:2M, N+1:2N]), VT.(B), S_B, U_B, Vt_B, S_B_inv)
end

# r is the sphere radius, r_p is the inner proxy-surface radius,
# M is number of surface points, N is number of proxy points.
function singlesphere_B(r::RT, r_p::RT, M::Int, N::Int, eps_r::ET) where {RT, ET}
    VT = promote_type(RT, ET)
    r_p > r && throw(ArgumentError("r_p must be <= r, got r_p=$r_p, r=$r"))
    r_q = r * r / r_p

    M <= N && throw(ArgumentError("M must be > N to make the system overdetermined, got M=$M, N=$N"))

    B = zeros(VT, 2 * M, 2 * N)
    pts_M = load_sphdes_N(M)  # surface points
    pts_N = load_sphdes_N(N)  # proxy points

    S_pr = zeros(VT, M, N)
    S_qr = zeros(VT, M, N)
    D_pr = zeros(VT, M, N)
    D_qr = zeros(VT, M, N)

    for i in 1:M
        for j in 1:N
            S_pr[i, j] = laplace3d_pot(r_p .* pts_N[j, :], r .* pts_M[i, :])
            S_qr[i, j] = laplace3d_pot(r_q .* pts_N[j, :], r .* pts_M[i, :])
            D_pr[i, j] = laplace3d_grad(r_p .* pts_N[j, :], r .* pts_M[i, :], pts_M[i, :])
            D_qr[i, j] = laplace3d_grad(r_q .* pts_N[j, :], r .* pts_M[i, :], pts_M[i, :]) * eps_r
        end
    end

    B[1:M, 1:N] = S_pr
    B[1:M, N+1:2N] = S_qr
    B[M+1:2M, 1:N] = D_pr
    B[M+1:2M, N+1:2N] = D_qr

    return B
end

function singlesphere_Ez_rhs(r::T, M::Int, Ez::T, eps_r::T) where T
    rhs = zeros(T, 2 * M)
    pts_M = load_sphdes_N(M)
    _ = r

    for i in 1:M
        rhs[M + i] = - Ez * (eps_r - 1) * pts_M[i, 3]
    end

    return rhs
end

function multispheres_Ez_rhs(r::T, M::Int, Ez::T, eps_r::VT, centers::Matrix{T}) where {T, VT}
    nspheres = size(centers, 1)
    rhs = zeros(T, 2 * M * nspheres)
    pts_M = load_sphdes_N(M)
    _ = r
    _ = centers
    for s in 1:nspheres
        for i in 1:M
            rhs[(s - 1) * 2M + M + i] = - Ez * (eps_r - 1) * pts_M[i, 3]
        end
    end
    return rhs
end

"""
    multispheres_uniform_rhs(r, M, eps_r, centers, E)

Right-hand side for a uniform incident field `E` (a 3-vector), i.e. `u_inc = -E·x`. Potential
rows are zero; flux rows hold `(eps_r - 1) ∂ₙu_inc = -(eps_r - 1) E·n`, in the interleaved
order used by `multispheres_G` and `multispheres_Ghat`. `E = (0, 0, Ez)` reproduces
`multispheres_Ez_rhs`.
"""
function multispheres_uniform_rhs(r::T, M::Int, eps_r, centers::Matrix{T}, E::AbstractVector{T}) where {T}
    length(E) == 3 || throw(DimensionMismatch("E must be a 3-vector"))
    _ = r
    nspheres = size(centers, 1)
    pts_M = load_sphdes_N(M)
    rhs = zeros(promote_type(T, typeof(eps_r)), 2 * M * nspheres)
    for s in 1:nspheres, i in 1:M
        rhs[(s - 1) * 2M + M + i] = -(eps_r - 1) * (E[1] * pts_M[i, 1] + E[2] * pts_M[i, 2] + E[3] * pts_M[i, 3])
    end
    return rhs
end

"""
    multispheres_pointcharge_rhs(r, M, eps_r, centers, charge_pos, charges)

Right-hand side of the multi-sphere system for point charges `charges` at the columns of
`charge_pos` (`3 × nq`), all outside every sphere. Potential rows are zero; flux rows of
sphere `s` hold `(eps_r - 1) ∂ₙu_inc` at its `M` collocation points, in the interleaved
`[pot_1; flux_1; pot_2; flux_2; …]` order used by `multispheres_G` and `multispheres_Ghat`.
"""
function multispheres_pointcharge_rhs(r::T, M::Int, eps_r, centers::Matrix{T},
                                      charge_pos::AbstractMatrix{T}, charges::AbstractVector{T}) where {T}
    size(charge_pos, 1) == 3 || throw(DimensionMismatch("charge_pos must be 3 × nq"))
    length(charges) == size(charge_pos, 2) ||
        throw(DimensionMismatch("length(charges) = $(length(charges)) but charge_pos has $(size(charge_pos, 2)) columns"))
    nspheres = size(centers, 1)
    pts_M = load_sphdes_N(M)
    rhs = zeros(promote_type(T, typeof(eps_r)), 2 * M * nspheres)
    for s in 1:nspheres
        c = vec(centers[s, :])
        for i in 1:M
            n = vec(pts_M[i, :])
            trg = c .+ r .* n
            dn = zero(T)
            for k in eachindex(charges)
                dn += charges[k] * laplace3d_grad(charge_pos[:, k], trg, n)
            end
            rhs[(s - 1) * 2M + M + i] = (eps_r - 1) * dn
        end
    end
    return rhs
end

"""
    doublespheres_B(r, r_p, M, N, eps_r, centers)

Construct the explicit 2-sphere overdetermined matrix in Eq. (10) of `refs/note.pdf`
for unknown ordering `[p1; -q1; p2; -q2]` and row ordering
`[pot on sphere 1; dn on sphere 1; pot on sphere 2; dn on sphere 2]`.
"""
function doublespheres_B(
    r::T,
    r_p::T,
    M::Int,
    N::Int,
    eps_r::T,
    centers::AbstractMatrix{T},
) where {T}
    size(centers) == (2, 3) || throw(DimensionMismatch("centers must be a 2x3 matrix"))
    r_p < r || throw(ArgumentError("r_p must be < r, got r_p=$r_p, r=$r"))
    M > N || throw(ArgumentError("M must be > N, got M=$M, N=$N"))

    r_q = r * r / r_p
    c1 = vec(centers[1, :])
    c2 = vec(centers[2, :])

    pts_M = load_sphdes_N(M)
    pts_N = load_sphdes_N(N)

    B = zeros(T, 4 * M, 4 * N)

    row_p1 = 1:M
    row_dn1 = M + 1:2M
    row_p2 = 2M + 1:3M
    row_dn2 = 3M + 1:4M
    col_p1 = 1:N
    col_q1 = N + 1:2N
    col_p2 = 2N + 1:3N
    col_q2 = 3N + 1:4N

    @inbounds for i in 1:M
        trg1 = c1 .+ r .* vec(pts_M[i, :])
        trg2 = c2 .+ r .* vec(pts_M[i, :])
        n1 = vec(pts_M[i, :])
        n2 = vec(pts_M[i, :])

        for j in 1:N
            src_p1 = c1 .+ r_p .* vec(pts_N[j, :])
            src_p2 = c2 .+ r_p .* vec(pts_N[j, :])
            src_q1 = c1 .+ r_q .* vec(pts_N[j, :])
            src_q2 = c2 .+ r_q .* vec(pts_N[j, :])

            B[row_p1[i], col_p1[j]] = laplace3d_pot(src_p1, trg1)
            B[row_p1[i], col_q1[j]] = laplace3d_pot(src_q1, trg1)
            B[row_p2[i], col_p2[j]] = laplace3d_pot(src_p2, trg2)
            B[row_p2[i], col_q2[j]] = laplace3d_pot(src_q2, trg2)

            B[row_dn1[i], col_p1[j]] = laplace3d_grad(src_p1, trg1, n1)
            B[row_dn1[i], col_p2[j]] = (one(T) - eps_r) * laplace3d_grad(src_p2, trg1, n1)
            B[row_dn1[i], col_q1[j]] = eps_r * laplace3d_grad(src_q1, trg1, n1)

            B[row_dn2[i], col_p1[j]] = (one(T) - eps_r) * laplace3d_grad(src_p1, trg2, n2)
            B[row_dn2[i], col_p2[j]] = laplace3d_grad(src_p2, trg2, n2)
            B[row_dn2[i], col_q2[j]] = eps_r * laplace3d_grad(src_q2, trg2, n2)
        end
    end
    return B
end

"""
    doublespheres_Ez_rhs(M, Ez, eps_r)

Right-hand side in Eq. (10) of `refs/note.pdf` for ordering compatible with
`doublespheres_B`.
"""
function doublespheres_Ez_rhs(M::Int, Ez::T, eps_r::T) where {T}
    rhs = zeros(T, 4 * M)
    pts_M = load_sphdes_N(M)
    nz = pts_M[:, 3]
    rhs[M+1:2M] .= -(eps_r - one(T)) * Ez .* nz
    rhs[3M+1:4M] .= -(eps_r - one(T)) * Ez .* nz
    return rhs
end
