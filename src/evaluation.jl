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

"""
    eval_total_pot(centers, r, N, coeffs, r_p, fmm_tol, targets, charge_pos, charges; incident = true)

Total potential at the columns of `targets` (`3 × ntrg`), inside or outside the spheres, for
point-charge excitation. With `incident = false` the incident field `u_inc` is left out, giving
the scattered (reaction) potential. `coeffs` is the full `[p1; -q1; p2; -q2; …]` solution vector.
Outside every sphere this is `u_inc + Σ_j u_ext_j`; inside sphere `i` it is
`u_inc + Σ_{j≠i} u_ext_j + u_int_i`, where `u_int_i` comes from sphere `i`'s `q` sources at
`r_q = r^2 / r_p`. Points on a surface are treated as exterior.
"""
function eval_total_pot(
    centers::Matrix{Float64},
    r::Float64,
    N::Int,
    coeffs::AbstractVector,
    r_p::Float64,
    fmm_tol::Float64,
    targets::Matrix{Float64},
    charge_pos::AbstractMatrix{Float64},
    charges::AbstractVector{Float64};
    incident::Bool = true,
)
    ns = size(centers, 1)
    length(coeffs) == 2 * ns * N ||
        throw(DimensionMismatch("coeffs must be the full [p; -q] layout of length $(2 * ns * N)"))
    size(targets, 1) == 3 || throw(DimensionMismatch("targets must be 3 × ntrg"))
    ntrg = size(targets, 2)
    # which sphere (if any) each target is strictly inside
    owner = zeros(Int, ntrg)
    for j in 1:ntrg
        for s in 1:ns
            if (targets[1, j] - centers[s, 1])^2 + (targets[2, j] - centers[s, 2])^2 +
               (targets[3, j] - centers[s, 3])^2 < r * r
                owner[j] = s
                break
            end
        end
    end
    phi = zeros(ntrg)
    ext = findall(iszero, owner)
    isempty(ext) || (phi[ext] .= eval_exterior_pot(centers, N, coeffs, r_p, fmm_tol, targets[:, ext]))
    pts = load_sphdes_N(N)
    r_q = r * r / r_p
    for s in 1:ns
        idx = findall(==(s), owner)
        isempty(idx) && continue
        T = targets[:, idx]
        # the other spheres' exterior fields (never this sphere's own p sources, which lie inside it)
        others = [t for t in 1:ns if t != s]
        if !isempty(others)
            cols = reduce(vcat, [2 * (t - 1) * N + 1 : 2 * t * N for t in others])
            phi[idx] .+= eval_exterior_pot(centers[others, :], N, coeffs[cols], r_p, fmm_tol, T)
        end
        # this sphere's interior field from its q sources at r_q (outside it); coeffs hold -q
        c = centers[s, :]
        qcol = 2 * (s - 1) * N + N
        for (jj, j) in enumerate(idx), m in 1:N
            src = c .+ r_q .* pts[m, :]
            phi[j] -= coeffs[qcol + m] * laplace3d_pot(src, T[:, jj])
        end
    end
    if incident
        for j in 1:ntrg, k in eachindex(charges)
            phi[j] += charges[k] * laplace3d_pot(charge_pos[:, k], targets[:, j])
        end
    end
    return phi
end

"""
    plot_surface_potential(centers, r, r_p, N, coeffs, charge_pos, charges; kwargs...)

Draw the total potential on every sphere surface in 3D. Requires a Makie backend
(`using CairoMakie` or `GLMakie`); implemented in the `LaplaceMFSMakieExt` extension.
"""
function plot_surface_potential end

"""
    plot_plane_potential(centers, r, r_p, N, coeffs, charge_pos, charges; kwargs...)

Draw the total potential on a plane cutting the system, with the sphere cross-sections
outlined. Requires a Makie backend; implemented in the `LaplaceMFSMakieExt` extension.
"""
function plot_plane_potential end

"""
    plot_plane_error(centers, r, r_p, N, coeffs, charge_pos, charges, reference; kwargs...)

Map of `log10(|u_MFS − u_ref| / max|u_ref|)` for the scattered potential on a cutting plane,
outside the spheres. `reference(targets)` must return the reference scattered potential at the
columns of a `3 × n` matrix. Requires a Makie backend; implemented in `LaplaceMFSMakieExt`.
"""
function plot_plane_error end
