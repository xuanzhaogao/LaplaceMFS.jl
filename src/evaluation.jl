function eval_exterior_pot(
    centers::Matrix{Float64},
    N::Int,
    lambda::AbstractVector,
    r_p::Float64,
    fmm_tol::Float64,
    targets::Matrix{Float64},
)
    ns = size(centers, 1)
    pts = load_sphdes_N(N)
    nsrc = ns * N
    src_p = Matrix{Float64}(undef, 3, nsrc)
    p = Vector{Float64}(undef, nsrc)

    for s in 1:ns
        c = vec(centers[s, :])
        λloc = 2 * (s - 1) * N + 1 : 2 * s * N
        p_src = first(λloc) : first(λloc) + N - 1
        idx_rng = (s - 1) * N + 1 : s * N
        p[idx_rng] .= lambda[p_src]
        for j in 1:N
            idx = (s - 1) * N + j
            u = vec(pts[j, :])
            src_p[:, idx] = c .+ r_p .* u
        end
    end

    out_p = lfmm3d(fmm_tol, src_p; charges = p, targets = targets, pgt = 1)
    return out_p.pottarg ./ (4π)
end