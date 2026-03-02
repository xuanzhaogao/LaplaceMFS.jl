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
