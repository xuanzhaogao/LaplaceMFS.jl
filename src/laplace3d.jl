function laplace3d_pot(src::Vector{T}, trg::Vector{T}) where T
    r2 = sum((src .- trg).^2)
    r = sqrt(r2)
    inv_r = one(T) / r

    return inv_r / 4π
end

function laplace3d_grad(src::Vector{T}, trg::Vector{T}, norm::Vector{T}) where T
    r2 = sum((src .- trg).^2)
    r = sqrt(r2)
    inv_r = one(T) / r

    return - dot(norm, inv_r^3 .* (trg .- src))  / 4π
end