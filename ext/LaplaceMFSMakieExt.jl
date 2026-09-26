module LaplaceMFSMakieExt

using LaplaceMFS, Makie
import LaplaceMFS: plot_surface_potential, plot_plane_potential, plot_plane_error

# Symmetric colour range from a high quantile of |φ|, so the singular values next to a point
# charge do not wash out the rest of the picture.
function _symmetric_range(vals; quantile = 0.99)
    a = sort!(abs.(filter(isfinite, vec(vals))))
    isempty(a) && return (-1.0, 1.0)
    m = a[clamp(ceil(Int, quantile * length(a)), 1, length(a))]
    m > 0 || (m = 1.0)
    return (-m, m)
end

function _check_field(field)
    field in (:total, :scattered) || throw(ArgumentError("field must be :total or :scattered"))
    return field === :total
end

_field_label(field) = field === :total ? "Total potential" : "Scattered potential"

function _draw_charges!(ax, charge_pos, charges; markersize)
    pos = [Point3f(charge_pos[:, k]...) for k in eachindex(charges)]
    col = [q > 0 ? :firebrick : :royalblue4 for q in charges]
    scatter!(ax, pos; color = col, markersize = markersize, strokewidth = 1, strokecolor = :white)
end

"""
    plot_surface_potential(centers, r, r_p, N, coeffs, charge_pos, charges;
                           field = :total, ntheta = 48, nphi = 96, fmm_tol = 1e-12,
                           colormap = :balance, colorrange = nothing, title = ...,
                           figure = (; size = (820, 700)))

3D view of the potential on every sphere surface for the MFS solution `coeffs` (full
`[p1; -q1; …]` layout). `field = :scattered` leaves out the incident field. Point charges are
drawn red (positive) and blue (negative). `colorrange` defaults to ±(99th percentile of |φ|).
Returns the `Figure`.
"""
function plot_surface_potential(centers::Matrix{Float64}, r::Float64, r_p::Float64, N::Int,
                                coeffs::AbstractVector, charge_pos::AbstractMatrix{Float64},
                                charges::AbstractVector{Float64};
                                field::Symbol = :total, ntheta::Int = 48, nphi::Int = 96,
                                fmm_tol::Float64 = 1e-12, colormap = :balance, colorrange = nothing,
                                title = "$(_field_label(field)) on the sphere surfaces",
                                figure = (; size = (820, 700)))
    incident = _check_field(field)
    ns = size(centers, 1)
    θ = range(0, π; length = ntheta)
    φ = range(0, 2π; length = nphi)
    npts = ntheta * nphi
    targets = Matrix{Float64}(undef, 3, ns * npts)
    for s in 1:ns, (j, p) in enumerate(φ), (i, t) in enumerate(θ)
        k = (s - 1) * npts + (j - 1) * ntheta + i
        targets[:, k] .= centers[s, :] .+ r .* (sin(t) * cos(p), sin(t) * sin(p), cos(t))
    end
    vals = eval_total_pot(centers, r, N, coeffs, r_p, fmm_tol, targets, charge_pos, charges;
                          incident = incident)
    crange = colorrange === nothing ? _symmetric_range(vals) : colorrange

    fig = Figure(; figure...)
    ax = Axis3(fig[1, 1]; aspect = :data, title = title, xlabel = "x", ylabel = "y", zlabel = "z")
    for s in 1:ns
        blk = (s - 1) * npts + 1 : s * npts
        X = reshape(targets[1, blk], ntheta, nphi)
        Y = reshape(targets[2, blk], ntheta, nphi)
        Z = reshape(targets[3, blk], ntheta, nphi)
        surface!(ax, X, Y, Z; color = reshape(vals[blk], ntheta, nphi), colormap = colormap,
                 colorrange = crange, shading = NoShading)
    end
    _draw_charges!(ax, charge_pos, charges; markersize = 14)
    Colorbar(fig[1, 2]; colormap = colormap, limits = crange, label = "φ")
    resize_to_layout!(fig)
    return fig
end

const _PLANE_AXES = Dict(:x => (1, 2, 3), :y => (2, 1, 3), :z => (3, 1, 2))   # normal, horizontal, vertical
const _LABELS = ("x", "y", "z")

# Grid on the plane x_normal = offset: returns (us, vs, targets, (kn, ku, kv)).
function _plane_grid(centers, r, charge_pos, normal, offset, extent, npts)
    haskey(_PLANE_AXES, normal) || throw(ArgumentError("normal must be :x, :y or :z"))
    kn, ku, kv = _PLANE_AXES[normal]
    if extent === nothing
        pts = hcat(permutedims(centers), charge_pos)
        extent = ((minimum(pts[ku, :]) - 2r, maximum(pts[ku, :]) + 2r),
                  (minimum(pts[kv, :]) - 2r, maximum(pts[kv, :]) + 2r))
    end
    (umin, umax), (vmin, vmax) = extent
    span_u, span_v = umax - umin, vmax - vmin
    nu = span_u >= span_v ? npts : max(2, round(Int, npts * span_u / span_v))
    nv = span_v >= span_u ? npts : max(2, round(Int, npts * span_v / span_u))
    us = range(umin, umax; length = nu)
    vs = range(vmin, vmax; length = nv)
    targets = zeros(3, nu * nv)
    for (j, v) in enumerate(vs), (i, u) in enumerate(us)
        k = (j - 1) * nu + i
        targets[kn, k] = offset
        targets[ku, k] = u
        targets[kv, k] = v
    end
    return us, vs, targets, (kn, ku, kv)
end

function _plane_decorations!(ax, centers, r, charge_pos, charges, offset, axes_idx, us, vs)
    kn, ku, kv = axes_idx
    for s in axes(centers, 1)
        dn = centers[s, kn] - offset
        abs(dn) < r || continue
        ρ = sqrt(r^2 - dn^2)
        t = range(0, 2π; length = 181)
        lines!(ax, centers[s, ku] .+ ρ .* cos.(t), centers[s, kv] .+ ρ .* sin.(t);
               color = :black, linewidth = 1.5)
    end
    near = [k for k in eachindex(charges) if abs(charge_pos[kn, k] - offset) < r]
    if !isempty(near)
        scatter!(ax, charge_pos[ku, near], charge_pos[kv, near];
                 color = [charges[k] > 0 ? :firebrick : :royalblue4 for k in near],
                 markersize = 12, strokewidth = 1, strokecolor = :white)
    end
    limits!(ax, first(us), last(us), first(vs), last(vs))
end

"""
    plot_plane_potential(centers, r, r_p, N, coeffs, charge_pos, charges;
                         field = :total, style = :both, normal = :y, offset = 0.0,
                         extent = nothing, npts = 300, fmm_tol = 1e-12, colormap = :balance,
                         colorrange = nothing, contours = 15, title = ...,
                         figure = (; size = (820, 700)))

Potential on the plane `x_normal = offset` (`normal ∈ (:x, :y, :z)`), inside and outside the
spheres, with the sphere cross-sections outlined and charges within one radius of the plane
marked. `field = :scattered` leaves out the incident field. `style` is `:heatmap` (smooth
colour map), `:contour` (filled contour bands with lines) or `:both` (heatmap with contour
lines). `extent = ((umin, umax), (vmin, vmax))` in the plane's two in-plane coordinates; by
default the spheres and charges plus a margin of `2r`. `colorrange` defaults to ±(99th
percentile of |φ|); `contours` is the number of contour levels. Returns the `Figure`.
"""
function plot_plane_potential(centers::Matrix{Float64}, r::Float64, r_p::Float64, N::Int,
                              coeffs::AbstractVector, charge_pos::AbstractMatrix{Float64},
                              charges::AbstractVector{Float64};
                              field::Symbol = :total, style::Symbol = :both,
                              normal::Symbol = :y, offset::Real = 0.0, extent = nothing,
                              npts::Int = 300, fmm_tol::Float64 = 1e-12, colormap = :balance,
                              colorrange = nothing, contours::Int = 15,
                              title = "$(_field_label(field)) on the plane $(normal) = $(offset)",
                              figure = (; size = (820, 700)))
    incident = _check_field(field)
    style in (:heatmap, :contour, :both) || throw(ArgumentError("style must be :heatmap, :contour or :both"))
    us, vs, targets, idx = _plane_grid(centers, r, charge_pos, normal, offset, extent, npts)
    vals = eval_total_pot(centers, r, N, coeffs, r_p, fmm_tol, targets, charge_pos, charges;
                          incident = incident)
    Φ = reshape(map(x -> isfinite(x) ? x : NaN, vals), length(us), length(vs))
    crange = colorrange === nothing ? _symmetric_range(Φ) : colorrange
    levels = range(crange...; length = contours + 2)[2:end-1]

    fig = Figure(; figure...)
    ax = Axis(fig[1, 1]; aspect = DataAspect(), title = title,
              xlabel = _LABELS[idx[2]], ylabel = _LABELS[idx[3]])
    Φc = clamp.(Φ, crange...)
    if style === :contour
        contourf!(ax, us, vs, Φc; levels = range(crange...; length = contours + 1), colormap = colormap)
        contour!(ax, us, vs, Φc; levels = levels, color = (:black, 0.35), linewidth = 0.8)
    else
        heatmap!(ax, us, vs, Φ; colormap = colormap, colorrange = crange)
        style === :both && contour!(ax, us, vs, Φc; levels = levels, color = (:black, 0.35), linewidth = 0.8)
    end
    _plane_decorations!(ax, centers, r, charge_pos, charges, offset, idx, us, vs)
    Colorbar(fig[1, 2]; colormap = colormap, limits = crange, label = "φ")
    resize_to_layout!(fig)
    return fig
end

"""
    plot_plane_error(centers, r, r_p, N, coeffs, charge_pos, charges, reference;
                     normal = :y, offset = 0.0, extent = nothing, npts = 200,
                     fmm_tol = 1e-12, colorrange = nothing, colormap = :viridis, title = ...,
                     figure = (; size = (820, 700)))

Heatmap of `log10(|u_MFS − u_ref| / max|u_ref|)` for the *scattered* potential on the plane
`x_normal = offset`, at grid points outside every sphere (interiors are left blank).
`reference(targets)` returns the reference scattered potential at the columns of a `3 × n`
matrix (for example `t -> HybridSolve.eval_exterior_pot(sol, t)`). The title reports the
maximum relative error on the plane. Returns the `Figure`.
"""
function plot_plane_error(centers::Matrix{Float64}, r::Float64, r_p::Float64, N::Int,
                          coeffs::AbstractVector, charge_pos::AbstractMatrix{Float64},
                          charges::AbstractVector{Float64}, reference;
                          normal::Symbol = :y, offset::Real = 0.0, extent = nothing,
                          npts::Int = 200, fmm_tol::Float64 = 1e-12, colorrange = nothing,
                          colormap = :viridis, title = nothing, figure = (; size = (820, 700)))
    us, vs, targets, idx = _plane_grid(centers, r, charge_pos, normal, offset, extent, npts)
    # strictly outside every sphere, with a small margin so the reference is well defined
    outside = [all(sum(abs2, targets[:, j] .- centers[s, :]) > (r * (1 + 1e-9))^2 for s in axes(centers, 1))
               for j in axes(targets, 2)]
    ext = findall(outside)
    u = eval_total_pot(centers, r, N, coeffs, r_p, fmm_tol, targets[:, ext], charge_pos, charges;
                       incident = false)
    uref = reference(targets[:, ext])
    scale = maximum(abs, uref)
    err = fill(NaN, size(targets, 2))
    err[ext] .= log10.(max.(abs.(u .- uref) ./ scale, eps()))
    E = reshape(err, length(us), length(vs))
    finite = filter(isfinite, err)
    crange = colorrange === nothing ? (floor(minimum(finite)), ceil(maximum(finite))) : colorrange
    ttl = title === nothing ?
        "log₁₀ relative error of the scattered potential, $(normal) = $(offset)\n" *
        "max = $(round(10.0^maximum(finite); sigdigits = 2)) (relative to max|u_ref|)" : title

    fig = Figure(; figure...)
    ax = Axis(fig[1, 1]; aspect = DataAspect(), title = ttl,
              xlabel = _LABELS[idx[2]], ylabel = _LABELS[idx[3]])
    heatmap!(ax, us, vs, E; colormap = colormap, colorrange = crange, nan_color = :white)
    _plane_decorations!(ax, centers, r, charge_pos, charges, offset, idx, us, vs)
    Colorbar(fig[1, 2]; colormap = colormap, limits = crange, label = "log₁₀ |Δu| / max|u_ref|")
    resize_to_layout!(fig)
    return fig
end

end
