function single_sphere_alpha(eps_r::T) where {T}
    return (eps_r - one(T)) / (eps_r + T(2))
end

function single_sphere_scattered_exterior(targets::AbstractMatrix{T}, r::T, Ez::T, eps_r::T) where {T}
    size(targets, 1) == 3 || throw(DimensionMismatch("targets must be a 3xN matrix"))

    rho = vec(sqrt.(sum(abs2, targets; dims = 1)))
    any(iszero, rho) && throw(ArgumentError("targets for exterior potential cannot include the origin"))

    alpha = single_sphere_alpha(eps_r)
    return vec(alpha * Ez * r^3 .* targets[3, :] ./ (rho .^ 3))
end

function single_sphere_scattered_interior(targets::AbstractMatrix{T}, Ez::T, eps_r::T) where {T}
    size(targets, 1) == 3 || throw(DimensionMismatch("targets must be a 3xN matrix"))

    alpha = single_sphere_alpha(eps_r)
    return vec(alpha * Ez .* targets[3, :])
end
