struct SperatedSpheresMFS{T, CT}
    r::T
    r_p::T
    eps_r::CT # relative permittivity of the inner sphere, can be real or complex

    M::Int
    N::Int

    num::Int
    centers::Matrix{T}
end

