# construct the mfs operator for multiple spheres
# G * (p; -q) = (phi(r) * (1 - 1/eps_r); 0)
function multispheres_G(r::T, r_p::T, M::Int, N::Int, centers::Matrix{T}) where {T}
    @assert r_p < r "r_p must be less than r, got r_p=$r_p, r=$r"
    @assert M > N "M must be greater than N to make the system overdetermined, got M=$M, N=$N"

    r_q = r * r / r_p
    pts_M = load_sphdes_N(M)  # surface points
    pts_N = load_sphdes_N(N)  # proxy points

    nspheres = size(centers, 1)
    @assert size(centers, 2) == 3 "centers should be an nspheres x 3 matrix of sphere centers"

    G = zeros(T, 2 * M * nspheres, 2 * N * nspheres)
    for s_i in 1:nspheres
        center_i = vec(centers[s_i, :])
        for s_j in 1:nspheres
            center_j = vec(centers[s_j, :])
            for i in 1:M
                trg = center_i .+ r .* vec(pts_M[i, :])
                norm = vec(pts_M[i, :])
                for j in 1:N
                    src_p = center_j .+ r_p .* vec(pts_N[j, :])
                    src_q = center_j .+ r_q .* vec(pts_N[j, :])

                    row_pot = 2 * (s_i - 1) * M + i
                    row_dn = 2 * (s_i - 1) * M + M + i
                    col_p = 2 * (s_j - 1) * N + j
                    col_q = 2 * (s_j - 1) * N + N + j

                    G[row_pot, col_p] = laplace3d_pot(src_p, trg)
                    G[row_dn, col_p] = laplace3d_grad(src_p, trg, norm)

                    # q only contributes if it's the same sphere, since it's an exterior image
                    if s_i == s_j
                        G[row_pot, col_q] = laplace3d_pot(src_q, trg)
                        G[row_dn, col_q] = laplace3d_grad(src_q, trg, norm)
                    end
                end
            end
        end
    end
    return G
end

function multispheres_mu_to_lambda!(lambda::AbstractVector{VT}, mats::SphereMats{T}, mu::AbstractVector) where {VT, T}
    nrows_loc = 2 * mats.M
    ncols_loc = 2 * mats.N
    nspheres, rem = divrem(length(mu), nrows_loc)
    rem == 0 || throw(DimensionMismatch("mu has length $(length(mu)); expected multiple of $nrows_loc"))
    length(lambda) == ncols_loc * nspheres || throw(DimensionMismatch("lambda has length $(length(lambda)); expected $(ncols_loc * nspheres)"))

    Uadj = VT.(mats.U_B')
    Vadj = VT.(mats.Vt_B')
    Sinv = VT.(mats.S_B_inv)
    tmp_u = zeros(VT, ncols_loc)
    tmp_λ = zeros(VT, ncols_loc)

    @inbounds for s in 1:nspheres
        μloc = (s - 1) * nrows_loc + 1 : s * nrows_loc
        λloc = (s - 1) * ncols_loc + 1 : s * ncols_loc
        mul!(tmp_u, Uadj, view(mu, μloc))
        tmp_u .*= Sinv
        mul!(tmp_λ, Vadj, tmp_u)
        view(lambda, λloc) .= tmp_λ
    end
    return lambda
end

function multispheres_mu_to_lambda(mats::SphereMats{T}, mu::AbstractVector{CT}) where {T, CT}
    nrows_loc = 2 * mats.M
    ncols_loc = 2 * mats.N
    nspheres, rem = divrem(length(mu), nrows_loc)
    rem == 0 || throw(DimensionMismatch("mu has length $(length(mu)); expected multiple of $nrows_loc"))
    VT = promote_type(T, CT)
    lambda = zeros(VT, ncols_loc * nspheres)
    return multispheres_mu_to_lambda!(lambda, mats, VT.(mu))
end

function _fmm_left_blocks(
    sources::Matrix{Float64},
    targets::Matrix{Float64},
    normals::Matrix{Float64},
    p::Vector{Float64},
    fmm_tol::Float64,
)
    out = lfmm3d(fmm_tol, sources; charges = p, targets = targets, pgt = 2)
    inv4pi = 1.0 / (4π)
    pot = inv4pi .* out.pottarg
    grad = out.gradtarg
    ntrg = size(targets, 2)
    dn = Vector{Float64}(undef, ntrg)
    @inbounds for i in 1:ntrg
        dn[i] = inv4pi * (
            normals[1, i] * grad[1, i] +
            normals[2, i] * grad[2, i] +
            normals[3, i] * grad[3, i]
        )
    end
    return pot, dn
end

function _fmm_left_blocks(
    sources::Matrix{Float64},
    targets::Matrix{Float64},
    normals::Matrix{Float64},
    p::Vector{ComplexF64},
    fmm_tol::Float64,
)
    pot_re, dn_re = _fmm_left_blocks(sources, targets, normals, real.(p), fmm_tol)
    pot_im, dn_im = _fmm_left_blocks(sources, targets, normals, imag.(p), fmm_tol)
    return complex.(pot_re, pot_im), complex.(dn_re, dn_im)
end

function multispheres_G_fmm(
    mats::SphereMats{T},
    centers::Matrix{T},
    eps_r::CT,
    fmm_tol::Float64
) where {T, CT}
    @assert size(centers, 2) == 3 "centers should be an nspheres x 3 matrix of sphere centers"

    nspheres = size(centers, 1)
    pts_M = load_sphdes_N(mats.M)
    pts_N = load_sphdes_N(mats.N)

    nsrc = nspheres * mats.N
    ntrg = nspheres * mats.M
    sources = Matrix{Float64}(undef, 3, nsrc)
    targets = Matrix{Float64}(undef, 3, ntrg)
    normals = Matrix{Float64}(undef, 3, ntrg)

    @inbounds for s in 1:nspheres
        c = vec(centers[s, :])
        for j in 1:mats.N
            idx = (s - 1) * mats.N + j
            sources[:, idx] = c .+ mats.r_p .* vec(pts_N[j, :])
        end
        for i in 1:mats.M
            idx = (s - 1) * mats.M + i
            targets[:, idx] = c .+ mats.r .* vec(pts_M[i, :])
            normals[:, idx] = vec(pts_M[i, :])
        end
    end

    VT = promote_type(T, CT)
    nrows = 2 * mats.M * nspheres
    ncols = 2 * mats.N * nspheres
    _ = eps_r

    function _mul!(y, x)
        length(x) == ncols || throw(DimensionMismatch("x has length $(length(x)), expected $ncols"))
        fill!(y, zero(VT))

        xT = VT.(x)
        p = Vector{VT}(undef, nsrc)
        q = Vector{VT}(undef, nsrc)
        @inbounds for s in 1:nspheres
            src_local = 2 * (s - 1) * mats.N + 1 : 2 * s * mats.N
            p_src = first(src_local) : first(src_local) + mats.N - 1
            q_src = first(src_local) + mats.N : last(src_local)
            trg = (s - 1) * mats.N + 1 : s * mats.N
            p[trg] .= xT[p_src]
            q[trg] .= xT[q_src]
        end

        pot_left, dn_left = if VT <: Real
            _fmm_left_blocks(sources, targets, normals, Float64.(p), fmm_tol)
        else
            _fmm_left_blocks(sources, targets, normals, ComplexF64.(p), fmm_tol)
        end
        @inbounds for s in 1:nspheres
            pot_trg = 2 * (s - 1) * mats.M + 1 : 2 * (s - 1) * mats.M + mats.M
            dn_trg = 2 * (s - 1) * mats.M + mats.M + 1 : 2 * s * mats.M
            src = (s - 1) * mats.M + 1 : s * mats.M
            y[pot_trg] .= VT.(pot_left[src])
            y[dn_trg] .= VT.(dn_left[src])
        end

        @inbounds for s in 1:nspheres
            pot_trg = 2 * (s - 1) * mats.M + 1 : 2 * (s - 1) * mats.M + mats.M
            dn_trg = 2 * (s - 1) * mats.M + mats.M + 1 : 2 * s * mats.M
            src = (s - 1) * mats.N + 1 : s * mats.N
            y[pot_trg] .+= mats.S_qr * q[src]
            y[dn_trg] .+= mats.D_qr * q[src]
        end
        return y
    end

    return LinearMap{VT}(_mul!, nrows, ncols; ismutating = true)
end

function multispheres_Ghat(mats::SphereMats{T}, centers::Matrix{T}, eps_r::CT) where {T, CT}
    @assert size(centers, 2) == 3 "centers should be an nspheres x 3 matrix of sphere centers"
    G = multispheres_G(mats.r, mats.r_p, mats.M, mats.N, centers)
    nspheres = size(centers, 1)

    VT = promote_type(T, CT)
    nrows = 2 * mats.M * nspheres
    ncols = 2 * mats.N * nspheres
    _ = eps_r

    GVT = VT.(G)
    Bmat = VT.(mats.B)
    Uadj = VT.(mats.U_B')
    Vadj = VT.(mats.Vt_B')
    Sinv = VT.(mats.S_B_inv)
    tmp_λall = zeros(VT, ncols)
    tmp_uall = zeros(VT, nrows)
    tmp_bdiag = zeros(VT, nrows)
    tmp_u = zeros(VT, 2 * mats.N)
    tmp_λ = zeros(VT, 2 * mats.N)
    tmp_Bλ = zeros(VT, 2 * mats.M)

    function _mul!(y, x)
        length(x) == nrows || throw(DimensionMismatch("x has length $(length(x)), expected $nrows"))
        xT = VT.(x)
        @inbounds for s in 1:nspheres
            μloc = 2 * (s - 1) * mats.M + 1 : 2 * s * mats.M
            λloc = 2 * (s - 1) * mats.N + 1 : 2 * s * mats.N
            mul!(tmp_u, Uadj, view(xT, μloc))
            tmp_u .*= Sinv
            mul!(tmp_λ, Vadj, tmp_u)
            view(tmp_λall, λloc) .= tmp_λ
            mul!(tmp_Bλ, Bmat, tmp_λ)
            view(tmp_bdiag, μloc) .= tmp_Bλ
        end
        mul!(tmp_uall, GVT, tmp_λall)
        y .= xT .+ tmp_uall .- tmp_bdiag
        return y
    end

    return LinearMap{VT}(_mul!, nrows, nrows; ismutating = true)
end

function multispheres_Ghat_fmm(
    mats::SphereMats{T},
    centers::Matrix{T},
    eps_r::CT,
    fmm_tol::Float64,
) where {T, CT}
    @assert size(centers, 2) == 3 "centers should be an nspheres x 3 matrix of sphere centers"
    Gfmm = multispheres_G_fmm(mats, centers, eps_r, fmm_tol)
    nspheres = size(centers, 1)

    VT = promote_type(T, CT)
    nrows = 2 * mats.M * nspheres
    ncols = 2 * mats.N * nspheres

    Uadj = VT.(mats.U_B')
    Vadj = VT.(mats.Vt_B')
    Sinv = VT.(mats.S_B_inv)
    Bmat = VT.(mats.B)

    tmp_inter = zeros(VT, ncols)
    tmp_bdiag = zeros(VT, nrows)
    tmp_u = zeros(VT, 2 * mats.N)
    tmp_λ = zeros(VT, 2 * mats.N)
    tmp_Bλ = zeros(VT, 2 * mats.M)
    function _mul!(y, x)
        length(x) == nrows || throw(DimensionMismatch("x has length $(length(x)), expected $nrows"))
        xT = VT.(x)
        @inbounds for s in 1:nspheres
            μloc = 2 * (s - 1) * mats.M + 1 : 2 * s * mats.M
            λloc = 2 * (s - 1) * mats.N + 1 : 2 * s * mats.N
            mul!(tmp_u, Uadj, view(xT, μloc))
            tmp_u .*= Sinv
            mul!(tmp_λ, Vadj, tmp_u)
            view(tmp_inter, λloc) .= tmp_λ
            mul!(tmp_Bλ, Bmat, tmp_λ)
            view(tmp_bdiag, μloc) .= tmp_Bλ
        end
        y .= xT .+ (Gfmm * tmp_inter) .- tmp_bdiag
        return y
    end

    return LinearMap{VT}(_mul!, nrows, nrows; ismutating = true)
end
