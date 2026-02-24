# r is the radius of the sphere, r_p is the radius of the inner proxy surface, M is number of proxy points, N is number of target points on the sphere
function single_sphere_B(r::T, r_p::T, M::Int, N::Int) where T
    r_p > r && throw(ArgumentError("r_p must be <= r, got r_p=$r_p, r=$r"))
    r_q = r * r / r_p

    B = zeros(T, 2 * N, 2 * M)
    pts_M = load_sphdes_N(M)
    pts_N = load_sphdes_N(N)

    S_pr = zeros(T, N, M)
    S_qr = zeros(T, N, M)
    D_pr = zeros(T, N, M)
    D_qr = zeros(T, N, M)

    for i in 1:N
        for j in 1:M
            S_pr[i, j] = laplace3d_pot(r_p .* pts_M[j, :], r .* pts_N[i, :])
            S_qr[i, j] = laplace3d_pot(r_q .* pts_M[j, :], r .* pts_N[i, :])
            D_pr[i, j] = laplace3d_grad(r_p .* pts_M[j, :], r .* pts_N[i, :], pts_N[i, :])
            D_qr[i, j] = laplace3d_grad(r_q .* pts_M[j, :], r .* pts_N[i, :], pts_N[i, :])
        end
    end

    B[1:N, 1:M] = S_pr
    B[1:N, M+1:2M] = S_qr
    B[N+1:2N, 1:M] = D_pr
    B[N+1:2N, M+1:2M] = D_qr

    return B
end

function single_sphere_Ez_rhs(r::T, N::Int, Ez::T, eps_r::T) where T
    rhs = zeros(T, 2 * N)
    pts_N = load_sphdes_N(N)

    for i in 1:N
        rhs[i] = Ez * (1 - 1 / eps_r) * r * pts_N[i, 3]
    end

    return rhs
end