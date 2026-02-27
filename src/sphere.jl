# SphereMats stores the matrices for the single sphere case, together with the SVD of the B matrix for preconditioning.
struct SphereMats{T}
    r::T
    r_p::T
    M::Int
    N::Int

    S_pr::Matrix{T}
    S_qr::Matrix{T}
    D_pr::Matrix{T}
    D_qr::Matrix{T}
    B::Matrix{T}

    S_B::Vector{T}
    U_B::Matrix{T}
    Vt_B::Matrix{T}

    S_B_inv::Vector{T}
end

function SphereMats(r::T, r_p::T, M::Int, N::Int, eps_r::T, tol::T) where T
    B = singlesphere_B(r, r_p, M, N, eps_r)

    res = svd(B)
    S_B = res.S
    U_B = res.U
    Vt_B = res.Vt

    S_B_inv = [s / S_B[1] > tol ? inv(s) : zero(T) for s in S_B]

    return SphereMats(r, r_p, M, N, B[1:M, 1:N], B[1:M, N+1:2N], B[M+1:2M, 1:N], B[M+1:2M, N+1:2N], B, S_B, U_B, Vt_B, S_B_inv)
end

function SphereMats(r::T, r_p::T, M::Int, N::Int, tol::T) where T
    return SphereMats(r, r_p, M, N, one(T), tol)
end

# r is the sphere radius, r_p is the inner proxy-surface radius,
# M is number of surface points, N is number of proxy points.
function singlesphere_B(r::T, r_p::T, M::Int, N::Int, eps_r::T) where T
    r_p > r && throw(ArgumentError("r_p must be <= r, got r_p=$r_p, r=$r"))
    r_q = r * r / r_p

    M <= N && throw(ArgumentError("M must be > N to make the system overdetermined, got M=$M, N=$N"))

    B = zeros(T, 2 * M, 2 * N)
    pts_M = load_sphdes_N(M)  # surface points
    pts_N = load_sphdes_N(N)  # proxy points

    S_pr = zeros(T, M, N)
    S_qr = zeros(T, M, N)
    D_pr = zeros(T, M, N)
    D_qr = zeros(T, M, N)

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

    for i in 1:M
        rhs[i] = Ez * (1 - 1 / eps_r) * r * pts_M[i, 3]
    end

    return rhs
end

function multispheres_Ez_rhs(r::T, M::Int, Ez::T, eps_r::VT, centers::Matrix{T}) where {T, VT}
    nspheres = size(centers, 1)
    rhs = zeros(T, 2 * M * nspheres)
    for s in 1:nspheres
        center_s = vec(centers[s, :])
        pts_M = load_sphdes_N(M) .+ center_s'
        for i in 1:M
            rhs[(s - 1) * 2M + i] = Ez * (1 - 1 / eps_r) * r * pts_M[i, 3]
        end
    end
    return rhs
end
