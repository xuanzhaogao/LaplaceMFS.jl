function _double_sphere_validate_centers(centers::AbstractMatrix{T}) where {T}
    size(centers) == (2, 3) || throw(DimensionMismatch("centers must be a 2x3 matrix"))
    abs(centers[1, 2] - centers[2, 2]) <= eps(T) || throw(ArgumentError("first-case image solver requires equal y-centers"))
    abs(centers[1, 3] - centers[2, 3]) <= eps(T) || throw(ArgumentError("first-case image solver requires equal z-centers"))
    d = abs(centers[2, 1] - centers[1, 1])
    d > zero(T) || throw(ArgumentError("sphere centers must be distinct"))
    return d
end

"""
    double_sphere_image_coefficients(r, d, Ez, eps_r; tol=1e-14, maxiter=200)

Image-reflection (method of reflections) dipole model for two identical dielectric
spheres of radius `r`, center distance `d` along x, in a uniform field `Ez * ẑ`.

Returns `(a1, a2)`, where each sphere contributes the scattered potential
`u_s(x) = a_s * z_local / |x-c_s|^3`.
"""
function double_sphere_image_coefficients(
    r::T,
    d::T,
    Ez::CT,
    eps_r::CT;
    tol::Float64 = 1e-14,
    maxiter::Int = 200,
) where {T, CT}
    d > 2r || throw(ArgumentError("spheres overlap or touch: d=$d, r=$r"))

    VT = promote_type(T, CT)
    alpha = VT(single_sphere_alpha(VT(eps_r))) * VT(r)^3
    a1 = alpha * VT(Ez)
    a2 = alpha * VT(Ez)
    invd3 = inv(VT(d)^3)

    for _ in 1:maxiter
        a1_new = alpha * (VT(Ez) - a2 * invd3)
        a2_new = alpha * (VT(Ez) - a1 * invd3)
        err = max(abs(a1_new - a1), abs(a2_new - a2))
        scale = max(one(VT), abs(a1_new), abs(a2_new))
        a1, a2 = a1_new, a2_new
        err <= tol * scale && return a1, a2
    end
    throw(ErrorException("double_sphere_image_coefficients did not converge in $maxiter iterations"))
end

function double_sphere_image_coefficients(
    centers::AbstractMatrix{T},
    r::T,
    Ez::CT,
    eps_r::CT;
    tol::Float64 = 1e-14,
    maxiter::Int = 200,
) where {T, CT}
    d = _double_sphere_validate_centers(centers)
    return double_sphere_image_coefficients(r, d, Ez, eps_r; tol = tol, maxiter = maxiter)
end

"""
    double_sphere_image_potential(targets, centers, r, Ez, eps_r; tol=1e-14, maxiter=200)

Scattered exterior potential for two identical spheres using the image-reflection
dipole model from `double_sphere_image_coefficients`.
"""
function double_sphere_image_potential(
    targets::AbstractMatrix{T},
    centers::AbstractMatrix{T},
    r::T,
    Ez::CT,
    eps_r::CT;
    tol::Float64 = 1e-14,
    maxiter::Int = 200,
) where {T, CT}
    size(targets, 1) == 3 || throw(DimensionMismatch("targets must be a 3xN matrix"))
    d = _double_sphere_validate_centers(centers)
    a1, a2 = double_sphere_image_coefficients(r, d, Ez, eps_r; tol = tol, maxiter = maxiter)

    VT = promote_type(T, CT)
    ntrg = size(targets, 2)
    out = Vector{VT}(undef, ntrg)
    c1 = VT.(vec(centers[1, :]))
    c2 = VT.(vec(centers[2, :]))

    @inbounds for i in 1:ntrg
        x = VT.(vec(targets[:, i]))
        R1 = x .- c1
        R2 = x .- c2
        r1 = norm(R1)
        r2 = norm(R2)
        (r1 > VT(r) && r2 > VT(r)) || throw(ArgumentError("targets must be outside both spheres"))
        out[i] = a1 * R1[3] / (r1^3) + a2 * R2[3] / (r2^3)
    end
    return out
end
