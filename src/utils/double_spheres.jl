function _stack_columns(cols::Vector{Vector{T}}) where {T}
    X = Matrix{T}(undef, 3, length(cols))
    @inbounds for j in eachindex(cols)
        col = cols[j]
        length(col) == 3 || throw(DimensionMismatch("each column must have length 3"))
        X[1, j] = col[1]
        X[2, j] = col[2]
        X[3, j] = col[3]
    end
    return X
end

function _scale_image_vectors(v::AbstractVector, coeffs::AbstractVector)
    length(v) == 3 || throw(DimensionMismatch("vector source must have length 3"))
    T = promote_type(eltype(v), eltype(coeffs))
    vT = T.(v)
    out = Matrix{T}(undef, 3, length(coeffs))
    @inbounds for j in eachindex(coeffs)
        cj = T(coeffs[j])
        out[1, j] = cj * vT[1]
        out[2, j] = cj * vT[2]
        out[3, j] = cj * vT[3]
    end
    return out
end

function _point_along_center_source_line(center::AbstractVector{T}, source::AbstractVector{T}, r_s::T, dist::T) where {T<:Real}
    t = dist / r_s
    return [
        center[1] + t * (source[1] - center[1]),
        center[2] + t * (source[2] - center[2]),
        center[3] + t * (source[3] - center[3]),
    ]
end

function _jacobi_roots_weights(n::Int, α::T, β::T) where {T<:Real}
    n > 0 || throw(ArgumentError("n must be positive"))
    α > -one(T) || throw(ArgumentError("alpha must be > -1"))
    β > -one(T) || throw(ArgumentError("beta must be > -1"))

    diag = Vector{T}(undef, n)
    offdiag = Vector{T}(undef, max(n - 1, 0))

    diag[1] = (β - α) / (α + β + T(2))
    @inbounds for i in 2:n
        k = T(i - 1)
        abi = T(2) * k + α + β
        diag[i] = (β^2 - α^2) / (abi * (abi + T(2)))
    end

    @inbounds for i in 1:(n - 1)
        k = T(i)
        abi = T(2) * k + α + β
        num = T(4) * k * (k + α) * (k + β) * (k + α + β)
        den = abi^2 * (abi + one(T)) * (abi - one(T))
        offdiag[i] = sqrt(num / den)
    end

    F = eigen(SymTridiagonal(diag, offdiag))
    # This utility only needs α = 0 (from the point+line image model), where the
    # Jacobi normalization constant simplifies to 2^(β+1)/(β+1).
    iszero(α) || throw(ArgumentError("_jacobi_roots_weights currently supports alpha = 0 only"))
    prefactor = T(2)^(β + one(T)) / (β + one(T))
    weights = prefactor .* (F.vectors[1, :] .^ 2)

    return F.values, collect(weights)
end

"""
    single_sphere_forward_point_line_images(center, radius, gamma, q, source; n_line=32, cutoff=0.0)

Generate forward image sources for a single sphere using the point+line image model
used by the HBDMM reference implementation. This is a geometry-only utility and does
not solve any linear system.

Returns a named tuple `(q, x)` where `q` is a vector of image strengths and `x` is a
`3 x N` matrix of image positions.
"""
function single_sphere_forward_point_line_images(
    center::AbstractVector{<:Real},
    radius::Real,
    gamma::Real,
    q::Number,
    source::AbstractVector{<:Real};
    n_line::Int = 32,
    cutoff::Real = 0.0,
)
    length(center) == 3 || throw(DimensionMismatch("center must have length 3"))
    length(source) == 3 || throw(DimensionMismatch("source must have length 3"))
    radius > 0 || throw(ArgumentError("radius must be positive"))
    n_line >= 0 || throw(ArgumentError("n_line must be nonnegative"))
    cutoff >= 0 || throw(ArgumentError("cutoff must be nonnegative"))
    gamma <= 1 || throw(ArgumentError("gamma must be <= 1"))

    GT = promote_type(eltype(center), eltype(source), typeof(radius), typeof(gamma))
    GT <: Real || throw(ArgumentError("center/source/radius/gamma must promote to a real type"))
    QT = promote_type(typeof(q), GT)

    c = GT.(center)
    s = GT.(source)
    r_s = norm(s .- c)
    iszero(r_s) && throw(ArgumentError("source cannot coincide with center"))

    q_images = QT[]
    x_cols = Vector{Vector{GT}}()

    # Kelvin point image.
    qK = -QT(gamma) * QT(radius) * QT(q) / QT(r_s)
    rK = GT(radius)^2 / r_s
    if abs(qK) > cutoff
        push!(q_images, qK)
        push!(x_cols, _point_along_center_source_line(c, s, r_s, rK))
    end

    # Line image quadrature.
    λ = (one(GT) - GT(gamma)) / GT(2)
    if n_line > 0 && λ > zero(GT)
        α = zero(GT)
        β = λ - one(GT)
        nodes, weights = _jacobi_roots_weights(n_line, α, β)
        base = (rK / GT(2))^(α + β + one(GT)) * rK^(one(GT) - λ)
        pref = QT(GT(gamma) * λ * base / GT(radius)) * QT(q)

        @inbounds for i in eachindex(nodes)
            x_i = nodes[i] * rK / GT(2) + rK / GT(2)
            q_i = QT(weights[i]) * pref
            if abs(q_i) > cutoff
                push!(q_images, q_i)
                push!(x_cols, _point_along_center_source_line(c, s, r_s, x_i))
            end
        end
    end

    return (; q = q_images, x = _stack_columns(x_cols))
end

"""
    double_sphere_forward_point_line_images(centers, radii, gammas, q0, source;
                                            n_line=32, n_reflections=1, cutoff=0.0)

Generate forward point+line image sources for two spheres by repeatedly reflecting
images between sphere 1 and sphere 2. This routine is geometry-only and does not
solve for boundary-matching coefficients.

`centers` is a `2x3` matrix. `radii` and `gammas` must each provide two values.
Returns `(q1, x1, q2, x2)`, where `q1/x1` are images in sphere 1 and `q2/x2` in
sphere 2, with `x1` and `x2` stored as `3xN` matrices.
"""
function double_sphere_forward_point_line_images(
    centers::AbstractMatrix{<:Real},
    radii,
    gammas,
    q0::Number,
    source::AbstractVector{<:Real};
    n_line::Int = 32,
    n_reflections::Int = 1,
    cutoff::Real = 0.0,
)
    size(centers) == (2, 3) || throw(DimensionMismatch("centers must be a 2x3 matrix"))
    length(radii) == 2 || throw(DimensionMismatch("radii must contain two entries"))
    length(gammas) == 2 || throw(DimensionMismatch("gammas must contain two entries"))
    n_reflections >= 1 || throw(ArgumentError("n_reflections must be >= 1"))

    c1 = vec(centers[1, :])
    c2 = vec(centers[2, :])
    r1, r2 = radii[1], radii[2]
    γ1, γ2 = gammas[1], gammas[2]

    first1 = single_sphere_forward_point_line_images(c1, r1, γ1, q0, source; n_line = n_line, cutoff = cutoff)
    first2 = single_sphere_forward_point_line_images(c2, r2, γ2, q0, source; n_line = n_line, cutoff = cutoff)

    q1_all = copy(first1.q)
    x1_all = copy(first1.x)
    q2_all = copy(first2.q)
    x2_all = copy(first2.x)

    q1_prev = first1.q
    x1_prev = first1.x
    q2_prev = first2.q
    x2_prev = first2.x

    for _ in 1:(n_reflections - 1)
        q1_next = eltype(q1_all)[]
        x1_next_cols = Vector{Vector{eltype(x1_all)}}()
        for j in eachindex(q2_prev)
            out = single_sphere_forward_point_line_images(c1, r1, γ1, q2_prev[j], x2_prev[:, j]; n_line = n_line, cutoff = cutoff)
            append!(q1_next, out.q)
            for col in eachcol(out.x)
                push!(x1_next_cols, copy(col))
            end
        end
        x1_next = _stack_columns(x1_next_cols)

        q2_next = eltype(q2_all)[]
        x2_next_cols = Vector{Vector{eltype(x2_all)}}()
        for j in eachindex(q1_prev)
            out = single_sphere_forward_point_line_images(c2, r2, γ2, q1_prev[j], x1_prev[:, j]; n_line = n_line, cutoff = cutoff)
            append!(q2_next, out.q)
            for col in eachcol(out.x)
                push!(x2_next_cols, copy(col))
            end
        end
        x2_next = _stack_columns(x2_next_cols)

        append!(q1_all, q1_next)
        x1_all = hcat(x1_all, x1_next)
        append!(q2_all, q2_next)
        x2_all = hcat(x2_all, x2_next)

        q1_prev = q1_next
        x1_prev = x1_next
        q2_prev = q2_next
        x2_prev = x2_next
    end

    return (; q1 = q1_all, x1 = x1_all, q2 = q2_all, x2 = x2_all)
end

"""
    single_sphere_forward_point_line_dipole_images(center, radius, gamma, p, source; n_line=32, cutoff=0.0)

Generate forward image dipoles for a single sphere using the same image positions
and scalar reflection coefficients as `single_sphere_forward_point_line_images`.

Returns a named tuple `(p, x)` where `p` is a `3 x N` matrix of dipole vectors
and `x` is a `3 x N` matrix of image positions.
"""
function single_sphere_forward_point_line_dipole_images(
    center::AbstractVector{<:Real},
    radius::Real,
    gamma::Real,
    p::AbstractVector,
    source::AbstractVector{<:Real};
    n_line::Int = 32,
    cutoff::Real = 0.0,
)
    length(p) == 3 || throw(DimensionMismatch("dipole source must have length 3"))
    PT = promote_type(eltype(p), eltype(center), eltype(source), typeof(radius), typeof(gamma))
    unit_source = one(PT)
    out = single_sphere_forward_point_line_images(
        center, radius, gamma, unit_source, source; n_line = n_line, cutoff = cutoff
    )
    return (; p = _scale_image_vectors(p, out.q), x = out.x)
end

"""
    double_sphere_forward_point_line_dipole_images(centers, radii, gammas, p0, source;
                                                   n_line=32, n_reflections=1, cutoff=0.0)

Generate forward dipole images for two spheres using the same image positions and
scalar reflection coefficients as `double_sphere_forward_point_line_images`.

Returns `(p1, x1, p2, x2)`, where `p1/x1` are the dipole vectors and image
positions in sphere 1 and `p2/x2` are the corresponding outputs in sphere 2.
"""
function double_sphere_forward_point_line_dipole_images(
    centers::AbstractMatrix{<:Real},
    radii,
    gammas,
    p0::AbstractVector,
    source::AbstractVector{<:Real};
    n_line::Int = 32,
    n_reflections::Int = 1,
    cutoff::Real = 0.0,
)
    length(p0) == 3 || throw(DimensionMismatch("dipole source must have length 3"))
    PT = promote_type(eltype(p0), eltype(centers), eltype(source), eltype(radii), eltype(gammas))
    unit_source = one(PT)
    out = double_sphere_forward_point_line_images(
        centers, radii, gammas, unit_source, source;
        n_line = n_line, n_reflections = n_reflections, cutoff = cutoff,
    )
    return (
        ;
        p1 = _scale_image_vectors(p0, out.q1),
        x1 = out.x1,
        p2 = _scale_image_vectors(p0, out.q2),
        x2 = out.x2,
    )
end

"""
    double_sphere_image_coefficients(r, Ez, eps_r, centers; method=:closed_form, max_reflections=200, tol=1e-13)

Compute effective dipole coefficients for the two-sphere image approximation in
`refs/double_sphere_image_derivation.md`.

Returns a `3 x 2` matrix whose columns are the reduced dipoles `s1, s2` such that
the scattered potential is approximated by

`u(x) = dot(s1, x-c1) / |x-c1|^3 + dot(s2, x-c2) / |x-c2|^3`.

`method=:closed_form` solves the 6x6 coupled system exactly.
`method=:series` sums the reflection series up to `max_reflections` or until `tol`.
"""
function double_sphere_image_coefficients(
    r::Real,
    Ez,
    eps_r,
    centers::AbstractMatrix{<:Real};
    method::Symbol = :closed_form,
    max_reflections::Int = 200,
    tol::Real = 1e-13,
)
    size(centers) == (2, 3) || throw(DimensionMismatch("centers must be a 2x3 matrix"))
    max_reflections >= 0 || throw(ArgumentError("max_reflections must be nonnegative"))
    tol > 0 || throw(ArgumentError("tol must be positive"))

    CT = promote_type(typeof(r), typeof(Ez), typeof(eps_r), eltype(centers))
    rT = CT(r)
    alpha = (CT(eps_r) - one(CT)) / (CT(eps_r) + CT(2))

    c1 = CT.(vec(centers[1, :]))
    c2 = CT.(vec(centers[2, :]))
    d = c1 .- c2
    R = norm(d)
    iszero(R) && throw(ArgumentError("sphere centers must be distinct"))
    n = d ./ R

    I3 = Matrix{CT}(I, 3, 3)
    G = (CT(3) .* (n * transpose(n)) .- I3) ./ (R^3)
    T = alpha * rT^3 .* G

    E0 = zeros(CT, 3)
    E0[3] = CT(Ez)
    s0 = alpha * rT^3 .* E0

    if method == :closed_form
        A = zeros(CT, 6, 6)
        A[1:3, 1:3] .= I3
        A[1:3, 4:6] .= -T
        A[4:6, 1:3] .= -T
        A[4:6, 4:6] .= I3

        b = vcat(s0, s0)
        sol = A \ b
        return hcat(sol[1:3], sol[4:6])
    elseif method == :series
        s1 = copy(s0)
        s2 = copy(s0)
        prev1 = copy(s0)
        prev2 = copy(s0)

        for _ in 1:max_reflections
            next1 = T * prev2
            next2 = T * prev1
            s1 .+= next1
            s2 .+= next2

            scale = max(norm(s1), norm(s2), one(real(CT)))
            if max(norm(next1), norm(next2)) <= tol * scale
                break
            end
            prev1 = next1
            prev2 = next2
        end
        return hcat(s1, s2)
    else
        throw(ArgumentError("method must be :closed_form or :series"))
    end
end

"""
    double_sphere_image_potential(targets, centers, coeffs)

Evaluate the dipole image approximation at `targets` using precomputed
`coeffs = double_sphere_image_coefficients(...)`.
"""
function double_sphere_image_potential(
    targets::AbstractMatrix{<:Real},
    centers::AbstractMatrix{<:Real},
    coeffs::AbstractMatrix,
)
    size(targets, 1) == 3 || throw(DimensionMismatch("targets must be a 3xN matrix"))
    size(centers) == (2, 3) || throw(DimensionMismatch("centers must be a 2x3 matrix"))
    size(coeffs) == (3, 2) || throw(DimensionMismatch("coeffs must be a 3x2 matrix"))

    CT = promote_type(eltype(targets), eltype(centers), eltype(coeffs))
    nt = size(targets, 2)
    out = zeros(CT, nt)

    @inbounds for k in 1:nt
        x = CT(targets[1, k])
        y = CT(targets[2, k])
        z = CT(targets[3, k])
        uk = zero(CT)
        for s in 1:2
            dx = x - CT(centers[s, 1])
            dy = y - CT(centers[s, 2])
            dz = z - CT(centers[s, 3])
            rho2 = dx * dx + dy * dy + dz * dz
            iszero(rho2) && throw(ArgumentError("targets cannot coincide with sphere centers"))
            inv_rho3 = inv(rho2 * sqrt(rho2))
            uk += (coeffs[1, s] * dx + coeffs[2, s] * dy + coeffs[3, s] * dz) * inv_rho3
        end
        out[k] = uk
    end

    return out
end

"""
    double_sphere_image_potential(targets, r, Ez, eps_r, centers; kwargs...)

Convenience wrapper that computes image coefficients and evaluates the potential.
`kwargs...` are passed to `double_sphere_image_coefficients`.
"""
function double_sphere_image_potential(
    targets::AbstractMatrix{<:Real},
    r::Real,
    Ez,
    eps_r,
    centers::AbstractMatrix{<:Real};
    kwargs...,
)
    coeffs = double_sphere_image_coefficients(r, Ez, eps_r, centers; kwargs...)
    return double_sphere_image_potential(targets, centers, coeffs)
end
