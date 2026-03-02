function eval_exterior_pot(
    centers::Matrix{Float64},
    N::Int,
    coeffs::AbstractVector,
    r_p::Float64,
    fmm_tol::Float64,
    targets::Matrix{Float64},
)
    ns = size(centers, 1)
    pts = load_sphdes_N(N)
    nsrc = ns * N
    src_p = Matrix{Float64}(undef, 3, nsrc)
    p = Vector{Float64}(undef, nsrc)
    ncoeff = length(coeffs)
    np = ns * N
    nfull = 2 * np

    if ncoeff == np
        # p-only layout: [p1; p2; ...; p_ns]
        p .= coeffs
    elseif ncoeff == nfull
        # full layout: [p1; -q1; p2; -q2; ...] (q is ignored for exterior)
        for s in 1:ns
            src = 2 * (s - 1) * N + 1 : 2 * (s - 1) * N + N
            trg = (s - 1) * N + 1 : s * N
            p[trg] .= coeffs[src]
        end
    else
        throw(DimensionMismatch("coeff length $ncoeff does not match p-only ($np) or full ($nfull) layout"))
    end

    for s in 1:ns
        c = vec(centers[s, :])
        for j in 1:N
            idx = (s - 1) * N + j
            u = vec(pts[j, :])
            src_p[:, idx] = c .+ r_p .* u
        end
    end

    out_p = lfmm3d(fmm_tol, src_p; charges = p, targets = targets, pgt = 1)
    return out_p.pottarg ./ (4π)
end
