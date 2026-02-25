# r is the sphere radius, r_p is the inner proxy-surface radius,
# M is number of surface points, N is number of proxy points.
function single_sphere_B(r::T, r_p::T, M::Int, N::Int) where T
    r_p > r && throw(ArgumentError("r_p must be <= r, got r_p=$r_p, r=$r"))
    r_q = r * r / r_p

    M <= N && throw(ArgumentError("M must be > N to keep the system overdetermined, got M=$M, N=$N"))

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
            D_qr[i, j] = laplace3d_grad(r_q .* pts_N[j, :], r .* pts_M[i, :], pts_M[i, :])
        end
    end

    B[1:M, 1:N] = S_pr
    B[1:M, N+1:2N] = S_qr
    B[M+1:2M, 1:N] = D_pr
    B[M+1:2M, N+1:2N] = D_qr

    return B
end

function single_sphere_Ez_rhs(r::T, M::Int, Ez::T, eps_r::T) where T
    rhs = zeros(T, 2 * M)
    pts_M = load_sphdes_N(M)

    for i in 1:M
        rhs[i] = Ez * (1 - 1 / eps_r) * r * pts_M[i, 3]
    end

    return rhs
end
