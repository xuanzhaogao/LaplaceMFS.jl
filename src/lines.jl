# Line image sources for close-to-touching spheres and for point charges near a sphere,
# placed as in Broms, Barnett & Tornberg (refs/stokesmfs.pdf, Eqs. (23)-(27), (36)-(38)).
#
# Every line node carries a charge and a dipole, i.e. four unknowns (q, px, py, pz). A line
# inside sphere i runs from the proxy sphere towards
#   - the image accumulation point of a close neighbour j (Eq. (24)), or
#   - the Kelvin image r^2 / d of a point charge a distance d from the center,
# whenever that end point lies outside the proxy sphere. With `mirror = true` the line is
# also reflected in the sphere (t -> r^2 / t), giving the matching nodes outside the sphere
# for the interior field, just as the q proxy sphere at r_q = r^2 / r_p mirrors the p one.

const _GOLDEN_RATIO = (1 + sqrt(5)) / 2

"""
    line_accumulation_radius(r, δ)

Distance from a sphere center to the image accumulation point for two spheres of radius `r`
a gap `δ` apart, Eq. (24) of `refs/stokesmfs.pdf` scaled to radius `r`.
"""
function line_accumulation_radius(r::Real, δ::Real)
    δ > 0 || throw(ArgumentError("gap must be positive, got δ=$δ"))
    d = δ / r
    return r * (1 + d / 2 - sqrt(d + d^2 / 4))
end

"""
    line_critical_gap(r, r_p)

Gap `δ*` at which the accumulation point reaches the proxy sphere, Eq. (25):
`δ* = r (1 - ρ)^2 / ρ` with `ρ = r_p / r`. Pairs closer than `δ*` get image lines.
"""
line_critical_gap(r::Real, r_p::Real) = (ρ = r_p / r; r * (1 - ρ)^2 / ρ)

"""
    line_num_nodes(r, t_end)

Default number of nodes on a line ending a distance `t_end` from the center: the adaptive
rule Eq. (39), `⌈-8.72 log10(δ) - 6.15⌉` (at least 2), evaluated at the equivalent gap
`δ = (1 - ρ)^2 / ρ`, `ρ = t_end / r`, whose accumulation point sits at `t_end`.
"""
function line_num_nodes(r::Real, t_end::Real)
    ρ = t_end / r
    d = (1 - ρ)^2 / ρ
    return max(2, ceil(Int, -8.72 * log10(d) - 6.15))
end

"""
    line_nodes(center, dir, r_p, t_end, n; Δ = 0.05)

The `n` nodes of Eq. (27) on the segment `center + t dir`, `t ∈ (r_p, t_end]`: half a set of
Chebyshev nodes clustered towards (and including) `t_end`, the last one at `r_p (1 + Δ)`.
Returns a `3 × n` matrix, first column at `t_end`.
"""
function line_nodes(center::AbstractVector{T}, dir::AbstractVector{T}, r_p::T, t_end::T, n::Int; Δ::Real = 0.05) where {T}
    n >= 1 || throw(ArgumentError("n must be >= 1"))
    t0 = r_p * (1 + Δ)
    X = Matrix{T}(undef, 3, n)
    for j in 1:n
        t = t0 + (t_end - t0) * cospi((j - 1) / (2n))
        X[:, j] .= center .+ t .* dir
    end
    return X
end

# orthonormal frame (e1, e2, e3 = d)
function _frame(d::AbstractVector{T}) where {T}
    a = abs(d[1]) < 0.9 ? T[1, 0, 0] : T[0, 1, 0]
    e1 = normalize(a .- dot(a, d) .* d)
    e2 = cross(d, e1)
    return e1, e2
end

# `m` Fibonacci-cap unit vectors within angle β of `d`, Eq. (37)
function _cap_directions(d::AbstractVector{T}, β::Real, m::Int) where {T}
    e1, e2 = _frame(d)
    U = Matrix{T}(undef, 3, m)
    for j in 1:m
        θ = acos(1 - j * (1 - cos(β)) / m)
        φ = mod(2π * j / _GOLDEN_RATIO, 2π)
        U[:, j] .= sin(θ) * cos(φ) .* e1 .+ sin(θ) * sin(φ) .* e2 .+ cos(θ) .* d
    end
    return U
end

# potential and normal derivative at `trg` of a unit charge and of unit dipoles along x, y, z
# at `src`; dipole potential is p·(trg - src) / (4π |trg - src|^3)
@inline function _node_kernels(src, trg, n)
    Rx, Ry, Rz = trg[1] - src[1], trg[2] - src[2], trg[3] - src[3]
    r2 = Rx^2 + Ry^2 + Rz^2
    ir = inv(sqrt(r2))
    ir3 = ir^3 / 4π
    ir5 = 3 * ir3 / r2
    nR = n[1] * Rx + n[2] * Ry + n[3] * Rz
    pot = (ir / 4π, Rx * ir3, Ry * ir3, Rz * ir3)
    dn = (-nR * ir3, n[1] * ir3 - Rx * nR * ir5, n[2] * ir3 - Ry * nR * ir5, n[3] * ir3 - Rz * nR * ir5)
    return pot, dn
end

"""
    laplace3d_dipole_pot(src, trg, p)

Potential at `trg` of the dipole `p` at `src`: `p·(trg - src) / (4π |trg - src|^3)`.
"""
laplace3d_dipole_pot(src, trg, p) = (k = _node_kernels(src, trg, p)[1]; p[1] * k[2] + p[2] * k[3] + p[3] * k[4])

"""
    laplace3d_dipole_grad(src, trg, p, n)

Normal derivative `n·∇_trg` of `laplace3d_dipole_pot(src, trg, p)`.
"""
laplace3d_dipole_grad(src, trg, p, n) = (k = _node_kernels(src, trg, n)[2]; p[1] * k[2] + p[2] * k[3] + p[3] * k[4])

"""
    LineSphere

Discretization of one sphere: collocation points `targets` (with outward `normals` and row
`weights`), exterior-field sources inside the sphere (proxy charges `p_pos`, line nodes
`pl_pos`) and interior-field sources outside it (proxy charges `q_pos`, line nodes `ql_pos`).
Its coefficient block is `[p; pl; q; ql]`, where each line node holds `(q, px, py, pz)`;
as elsewhere in the package the interior block stores the negated strengths.
"""
struct LineSphere{T}
    center::Vector{T}
    targets::Matrix{T}
    normals::Matrix{T}
    weights::Vector{T}
    p_pos::Matrix{T}
    pl_pos::Matrix{T}
    q_pos::Matrix{T}
    ql_pos::Matrix{T}
end

_next(s::LineSphere) = size(s.p_pos, 2) + 4 * size(s.pl_pos, 2)
_nint(s::LineSphere) = size(s.q_pos, 2) + 4 * size(s.ql_pos, 2)
_ncols(s::LineSphere) = _next(s) + _nint(s)
_nrows(s::LineSphere) = 2 * size(s.targets, 2)

# M × (nc + 4 nn) potential and normal-derivative matrices from charges at `chg` and
# (q, px, py, pz) nodes at `nodes`
function _source_matrices(targets, normals, chg, nodes)
    T = eltype(targets)
    m, nc, nn = size(targets, 2), size(chg, 2), size(nodes, 2)
    S = zeros(T, m, nc + 4nn)
    D = zeros(T, m, nc + 4nn)
    @inbounds for i in 1:m
        x = view(targets, :, i)
        n = view(normals, :, i)
        for j in 1:nc
            pot, dn = _node_kernels(view(chg, :, j), x, n)
            S[i, j] = pot[1]
            D[i, j] = dn[1]
        end
        for j in 1:nn
            pot, dn = _node_kernels(view(nodes, :, j), x, n)
            c = nc + 4 * (j - 1)
            for k in 1:4
                S[i, c + k] = pot[k]
                D[i, c + k] = dn[k]
            end
        end
    end
    return S, D
end

"""
    MultiSphereLines(r, r_p, M, N, centers, eps_r, tol; charge_pos = zeros(3, 0), kwargs...)

Multi-sphere MFS discretization augmented with line image sources. Starting from the usual
`N` proxy points at `r_p` and `r_q = r^2 / r_p` and `M` collocation points per sphere, every
sphere gets

- a line towards each neighbour closer than `line_critical_gap(r, r_p)`, ending at the
  accumulation point `line_accumulation_radius(r, δ)`;
- a line towards each point charge (columns of `charge_pos`) whose Kelvin image `r^2 / d`
  lies outside the proxy sphere, ending at the Kelvin image;

each with `n_line` nodes of Eq. (27) carrying `(q, px, py, pz)`, and two Fibonacci caps of
extra collocation points around the line direction (Eqs. (36)-(37)). Row `weights` are the
square roots of the local areas relative to the base grid (Remark 5), so a system without
lines is exactly the one of `multispheres_G`. The per-sphere self blocks and their truncated
pseudo-inverses (relative cut-off `tol`) are stored for the one-body preconditioner.

Keywords: `n_line` (`nothing` for `line_num_nodes`, an `Int`, or a function of `t_end`),
`Δ = 0.05`, `cap_angles = (π/5, π/60)`, `cap_factors = (6, 6)` (points per cap per unknown
line node), `mirror = true` (also place the reflected line outside the sphere),
`weighted = true`.
"""
struct MultiSphereLines{T, VT}
    r::T
    r_p::T
    eps_r::VT
    spheres::Vector{LineSphere{T}}
    row_offsets::Vector{Int}
    col_offsets::Vector{Int}
    B::Vector{Matrix{VT}}
    U_B::Vector{Matrix{VT}}
    Vt_B::Vector{Matrix{VT}}
    S_B_inv::Vector{Vector{VT}}
end

function MultiSphereLines(r::T, r_p::T, M::Int, N::Int, centers::Matrix{T}, eps_r, tol::Real;
                          charge_pos::AbstractMatrix{T} = zeros(T, 3, 0),
                          n_line = nothing, Δ::Real = 0.05,
                          cap_angles = (π / 5, π / 60), cap_factors = (6, 6),
                          mirror::Bool = true, weighted::Bool = true) where {T}
    r_p < r || throw(ArgumentError("r_p must be < r, got r_p=$r_p, r=$r"))
    M > N || throw(ArgumentError("M must be > N, got M=$M, N=$N"))
    size(centers, 2) == 3 || throw(DimensionMismatch("centers must be nspheres × 3"))
    size(charge_pos, 1) == 3 || throw(DimensionMismatch("charge_pos must be 3 × nq"))
    nodes_for(t_end) = n_line === nothing ? line_num_nodes(r, t_end) :
                       n_line isa Integer ? Int(n_line) : Int(n_line(t_end))

    ns = size(centers, 1)
    pts_M = Matrix(load_sphdes_N(M)')
    pts_N = Matrix(load_sphdes_N(N)')
    r_q = r^2 / r_p
    δstar = line_critical_gap(r, r_p)
    w_base = 4π * r^2 / M

    spheres = LineSphere{T}[]
    for s in 1:ns
        c = vec(centers[s, :])
        # (direction, end radius) of every line of this sphere
        lines = Tuple{Vector{T}, T}[]
        for t in 1:ns
            t == s && continue
            v = vec(centers[t, :]) .- c
            δ = norm(v) - 2r
            δ > 0 || throw(ArgumentError("spheres $s and $t overlap or touch (gap $δ)"))
            δ < δstar && push!(lines, (v ./ norm(v), line_accumulation_radius(r, δ)))
        end
        for k in axes(charge_pos, 2)
            v = charge_pos[:, k] .- c
            d = norm(v)
            d > r || throw(ArgumentError("charge $k lies inside sphere $s"))
            r^2 / d > r_p && push!(lines, (v ./ d, r^2 / d))
        end

        targets = [c .+ r .* pts_M]
        normals = [pts_M]
        weights = [ones(T, M)]
        pl = Matrix{T}[zeros(T, 3, 0)]
        ql = Matrix{T}[zeros(T, 3, 0)]
        for (d, t_end) in lines
            n = nodes_for(t_end)
            n >= 1 || continue
            L = line_nodes(c, d, r_p, t_end, n; Δ = Δ)
            push!(pl, L)
            if mirror
                tt = vec(sqrt.(sum(abs2, L .- c; dims = 1)))
                push!(ql, c .+ d .* (r^2 ./ tt'))
            end
            nunk = 4n
            for (β, α) in zip(cap_angles, cap_factors)
                m = ceil(Int, α * nunk)
                U = _cap_directions(d, β, m)
                push!(targets, c .+ r .* U)
                push!(normals, U)
                w = 2π * r^2 * (1 - cos(β)) / m
                push!(weights, fill(weighted ? sqrt(w / w_base) : one(T), m))
            end
        end
        push!(spheres, LineSphere(c, reduce(hcat, targets), reduce(hcat, normals), reduce(vcat, weights),
                                  c .+ r_p .* pts_N, reduce(hcat, pl), c .+ r_q .* pts_N, reduce(hcat, ql)))
    end

    VT = promote_type(T, typeof(eps_r))
    eps = VT(eps_r)
    row_offsets = cumsum([0; [_nrows(s) for s in spheres]])
    col_offsets = cumsum([0; [_ncols(s) for s in spheres]])
    Bs, Us, Vts, Sinvs = Matrix{VT}[], Matrix{VT}[], Matrix{VT}[], Vector{VT}[]
    tolT = float(real(tol))
    for s in spheres
        B = _self_block(s, eps)
        F = svd(B)
        push!(Bs, B)
        push!(Us, F.U)
        push!(Vts, F.Vt)
        push!(Sinvs, [abs(σ / F.S[1]) > tolT ? inv(VT(σ)) : zero(VT) for σ in F.S])
    end
    return MultiSphereLines(r, r_p, eps, spheres, row_offsets, col_offsets, Bs, Us, Vts, Sinvs)
end

# [W S_ext  W S_int; W D_ext  eps W D_int] for one sphere
function _self_block(s::LineSphere{T}, eps::VT) where {T, VT}
    m = size(s.targets, 2)
    Se, De = _source_matrices(s.targets, s.normals, s.p_pos, s.pl_pos)
    Si, Di = _source_matrices(s.targets, s.normals, s.q_pos, s.ql_pos)
    B = zeros(VT, 2m, _ncols(s))
    ne = _next(s)
    B[1:m, 1:ne] .= s.weights .* Se
    B[1:m, ne+1:end] .= s.weights .* Si
    B[m+1:2m, 1:ne] .= s.weights .* De
    B[m+1:2m, ne+1:end] .= eps .* s.weights .* Di
    return B
end

nspheres(sys::MultiSphereLines) = length(sys.spheres)
_rows(sys::MultiSphereLines, s) = sys.row_offsets[s]+1:sys.row_offsets[s+1]
_cols(sys::MultiSphereLines, s) = sys.col_offsets[s]+1:sys.col_offsets[s+1]

"""
    multispheres_G(sys::MultiSphereLines)

Dense weighted target-from-source matrix: rows `[pot_1; dn_1; pot_2; dn_2; …]`, columns
`[p_1; pl_1; q_1; ql_1; p_2; …]`. The self blocks are `sys.B`; sphere `j`'s exterior sources
enter the flux rows of sphere `i ≠ j` with the factor `1 - eps_r`.
"""
function multispheres_G(sys::MultiSphereLines{T, VT}) where {T, VT}
    G = zeros(VT, sys.row_offsets[end], sys.col_offsets[end])
    one_minus_eps = one(VT) - sys.eps_r
    for i in 1:nspheres(sys)
        si = sys.spheres[i]
        m = size(si.targets, 2)
        rows = _rows(sys, i)
        G[rows, _cols(sys, i)] .= sys.B[i]
        for j in 1:nspheres(sys)
            j == i && continue
            sj = sys.spheres[j]
            _, D = _source_matrices(si.targets, si.normals, sj.p_pos, sj.pl_pos)
            G[rows[m+1:2m], sys.col_offsets[j] .+ (1:_next(sj))] .= one_minus_eps .* si.weights .* D
        end
    end
    return G
end

function _apply_pinv!(lambda, sys::MultiSphereLines, mu)
    for s in 1:nspheres(sys)
        u = sys.U_B[s]' * view(mu, _rows(sys, s))
        u .*= sys.S_B_inv[s]
        view(lambda, _cols(sys, s)) .= sys.Vt_B[s]' * u
    end
    return lambda
end

"""
    multispheres_mu_to_lambda(sys::MultiSphereLines, mu)

Apply the block-diagonal truncated pseudo-inverse of `sys.B` to `mu`.
"""
function multispheres_mu_to_lambda(sys::MultiSphereLines{T, VT}, mu::AbstractVector) where {T, VT}
    length(mu) == sys.row_offsets[end] || throw(DimensionMismatch("mu has length $(length(mu)), expected $(sys.row_offsets[end])"))
    CT = promote_type(VT, eltype(mu))
    return _apply_pinv!(zeros(CT, sys.col_offsets[end]), sys, CT.(mu))
end

"""
    multispheres_Ghat(sys::MultiSphereLines)

One-body right-preconditioned operator `mu ↦ mu + (G - blkdiag(B)) B⁺ mu` with a dense `G`.
"""
function multispheres_Ghat(sys::MultiSphereLines{T, VT}) where {T, VT}
    G = multispheres_G(sys)
    for s in 1:nspheres(sys)
        G[_rows(sys, s), _cols(sys, s)] .= zero(VT)
    end
    n = sys.row_offsets[end]
    lambda = zeros(VT, sys.col_offsets[end])
    function _mul!(y, x)
        _apply_pinv!(lambda, sys, VT.(x))
        y .= x .+ G * lambda
        return y
    end
    return LinearMap{VT}(_mul!, n, n; ismutating = true)
end

# all exterior-field sources: charges at `chg`, dipoles at `dip`, and a map from the
# coefficient vector to (charges, dipvecs)
function _exterior_sources(sys::MultiSphereLines{T}) where {T}
    chg = reduce(hcat, [hcat(s.p_pos, s.pl_pos) for s in sys.spheres])
    dip = reduce(hcat, [s.pl_pos for s in sys.spheres])
    return chg, dip
end

function _exterior_strengths(sys::MultiSphereLines, coeffs::AbstractVector{CT}) where {CT}
    charges = CT[]
    dipvecs = CT[]
    for s in 1:nspheres(sys)
        sp = sys.spheres[s]
        c = view(coeffs, _cols(sys, s))
        np, nl = size(sp.p_pos, 2), size(sp.pl_pos, 2)
        append!(charges, view(c, 1:np))
        L = reshape(view(c, np+1:np+4nl), 4, nl)
        append!(charges, view(L, 1, :))
        append!(dipvecs, vec(L[2:4, :]))
    end
    return charges, reshape(dipvecs, 3, :)
end

# potential and gradient of all exterior sources at `targets` via FMM, 1/(4π) included
function _fmm_exterior(sys::MultiSphereLines, coeffs::AbstractVector{<:Real}, fmm_tol, targets; pgt = 1)
    chg, dip = _exterior_sources(sys)
    charges, dipvecs = _exterior_strengths(sys, coeffs)
    src = hcat(chg, dip)
    q = vcat(charges, zeros(size(dip, 2)))
    dv = hcat(zeros(3, size(chg, 2)), dipvecs)
    out = lfmm3d(fmm_tol, src; charges = Float64.(q), dipvecs = Float64.(dv), targets = targets, pgt = pgt)
    return out.pottarg ./ 4π, pgt == 2 ? out.gradtarg ./ 4π : nothing
end

function _fmm_exterior(sys::MultiSphereLines, coeffs::AbstractVector{<:Complex}, fmm_tol, targets; pgt = 1)
    pr, gr = _fmm_exterior(sys, real.(coeffs), fmm_tol, targets; pgt = pgt)
    pi_, gi = _fmm_exterior(sys, imag.(coeffs), fmm_tol, targets; pgt = pgt)
    return complex.(pr, pi_), pgt == 2 ? complex.(gr, gi) : nothing
end

"""
    multispheres_Ghat_fmm(sys::MultiSphereLines, fmm_tol)

Same operator as `multispheres_Ghat(sys)`, with the sphere-sphere interactions evaluated by
one FMM call (charges and dipoles) per application.
"""
function multispheres_Ghat_fmm(sys::MultiSphereLines{T, VT}, fmm_tol::Float64) where {T, VT}
    targets = reduce(hcat, [s.targets for s in sys.spheres])
    normals = reduce(hcat, [s.normals for s in sys.spheres])
    weights = reduce(vcat, [s.weights for s in sys.spheres])
    toff = cumsum([0; [size(s.targets, 2) for s in sys.spheres]])
    n = sys.row_offsets[end]
    one_minus_eps = one(VT) - sys.eps_r
    lambda = zeros(VT, sys.col_offsets[end])
    function _mul!(y, x)
        xT = VT.(x)
        _apply_pinv!(lambda, sys, xT)
        _, grad = _fmm_exterior(sys, lambda, fmm_tol, targets; pgt = 2)
        y .= xT
        for s in 1:nspheres(sys)
            sp = sys.spheres[s]
            m = size(sp.targets, 2)
            idx = toff[s]+1:toff[s+1]
            dn = vec(sum(normals[:, idx] .* grad[:, idx]; dims = 1)) .* weights[idx]
            # remove the sphere's own exterior sources, already in the self block
            ne = _next(sp)
            self_dn = view(sys.B[s], m+1:2m, 1:ne) * view(lambda, sys.col_offsets[s] .+ (1:ne))
            view(y, sys.row_offsets[s] + m .+ (1:m)) .+= one_minus_eps .* (dn .- self_dn)
        end
        return y
    end
    return LinearMap{VT}(_mul!, n, n; ismutating = true)
end

"""
    multispheres_pointcharge_rhs(sys::MultiSphereLines, charge_pos, charges)

Weighted right-hand side for exterior point charges; flux rows hold `(eps_r - 1) ∂ₙu_inc`.
"""
function multispheres_pointcharge_rhs(sys::MultiSphereLines{T, VT}, charge_pos::AbstractMatrix{T},
                                      charges::AbstractVector{T}) where {T, VT}
    size(charge_pos, 1) == 3 || throw(DimensionMismatch("charge_pos must be 3 × nq"))
    length(charges) == size(charge_pos, 2) || throw(DimensionMismatch("one charge per column of charge_pos"))
    rhs = zeros(VT, sys.row_offsets[end])
    for s in 1:nspheres(sys)
        sp = sys.spheres[s]
        m = size(sp.targets, 2)
        _, D = _source_matrices(sp.targets, sp.normals, charge_pos, zeros(T, 3, 0))
        view(rhs, sys.row_offsets[s] + m .+ (1:m)) .= (sys.eps_r - 1) .* sp.weights .* (D * charges)
    end
    return rhs
end

"""
    multispheres_uniform_rhs(sys::MultiSphereLines, E)

Weighted right-hand side for the uniform field `E`, `u_inc = -E·x`.
"""
function multispheres_uniform_rhs(sys::MultiSphereLines{T, VT}, E::AbstractVector{T}) where {T, VT}
    length(E) == 3 || throw(DimensionMismatch("E must be a 3-vector"))
    rhs = zeros(VT, sys.row_offsets[end])
    for s in 1:nspheres(sys)
        sp = sys.spheres[s]
        m = size(sp.targets, 2)
        view(rhs, sys.row_offsets[s] + m .+ (1:m)) .= -(sys.eps_r - 1) .* sp.weights .* vec(E' * sp.normals)
    end
    return rhs
end

"""
    eval_exterior_pot(sys::MultiSphereLines, coeffs, fmm_tol, targets)

Scattered potential outside the spheres: all proxy charges and line charges/dipoles inside
the spheres, evaluated by FMM at the columns of `targets` (`3 × ntrg`).
"""
function eval_exterior_pot(sys::MultiSphereLines, coeffs::AbstractVector, fmm_tol::Float64, targets::Matrix{Float64})
    length(coeffs) == sys.col_offsets[end] || throw(DimensionMismatch("coeffs must have length $(sys.col_offsets[end])"))
    return _fmm_exterior(sys, coeffs, fmm_tol, targets)[1]
end

# potential at `targets` of the charges `chg` and (q, p) nodes `nodes` with coefficients `c`
function _direct_pot(targets, chg, nodes, c)
    S, _ = _source_matrices(targets, zeros(size(targets)), chg, nodes)
    return S * c
end

"""
    eval_total_pot(sys::MultiSphereLines, coeffs, fmm_tol, targets, charge_pos, charges; incident = true)

Total potential anywhere, as `eval_total_pot` for the plain discretization: outside the
spheres `u_inc + Σ_j u_ext_j`, inside sphere `i` `u_inc + Σ_{j≠i} u_ext_j + u_int_i`, with
`u_int_i` from sphere `i`'s interior block (proxy charges at `r_q` and mirrored line nodes).
"""
function eval_total_pot(sys::MultiSphereLines{T}, coeffs::AbstractVector, fmm_tol::Float64, targets::Matrix{Float64},
                        charge_pos::AbstractMatrix{Float64}, charges::AbstractVector{Float64};
                        incident::Bool = true) where {T}
    length(coeffs) == sys.col_offsets[end] || throw(DimensionMismatch("coeffs must have length $(sys.col_offsets[end])"))
    size(targets, 1) == 3 || throw(DimensionMismatch("targets must be 3 × ntrg"))
    ns, r = nspheres(sys), sys.r
    owner = [something(findfirst(s -> sum(abs2, targets[:, j] .- sys.spheres[s].center) < r^2, 1:ns), 0)
             for j in axes(targets, 2)]
    phi = zeros(eltype(coeffs), size(targets, 2))
    ext = findall(iszero, owner)
    isempty(ext) || (phi[ext] .= eval_exterior_pot(sys, coeffs, fmm_tol, targets[:, ext]))
    for s in 1:ns
        idx = findall(==(s), owner)
        isempty(idx) && continue
        X = targets[:, idx]
        for t in 1:ns
            sp = sys.spheres[t]
            c = view(coeffs, _cols(sys, t))
            ne = _next(sp)
            if t == s
                # the interior block holds the negated strengths
                phi[idx] .-= _direct_pot(X, sp.q_pos, sp.ql_pos, c[ne+1:end])
            else
                phi[idx] .+= _direct_pot(X, sp.p_pos, sp.pl_pos, c[1:ne])
            end
        end
    end
    if incident
        phi .+= _direct_pot(targets, charge_pos, zeros(3, 0), charges)
    end
    return phi
end
