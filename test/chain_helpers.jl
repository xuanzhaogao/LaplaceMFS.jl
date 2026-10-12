# Shared by plasmonic.jl and conductor.jl: exact coaxial reference, targets around chains of unit spheres
include("axisym_reference.jl")

fibonacci_sphere(n) = reduce(hcat, [begin z = 1 - 2(i + 0.5) / n; ρ = sqrt(1 - z^2); g = π * (3 - sqrt(5)) * i
                                     [ρ * cos(g), ρ * sin(g), z] end for i in 0:n-1])
outside_all(T, C) = T[:, [j for j in axes(T, 2) if all(norm(T[:, j] - C[s, :]) > 1.01 for s in axes(C, 1))]]
# targets on the shells |x - c_i| = 1.25 and inside every gap (a quarter gap off the mid-plane)
function chain_targets(zs, g)
    C = [zeros(length(zs)) zeros(length(zs)) zs]
    far = outside_all(reduce(hcat, [[0, 0, z] .+ 1.25 .* fibonacci_sphere(40) for z in zs]), C)
    gap = reduce(hcat, [[s, 0.0, zs[k] + 1 + g / 2 + h * g / 4] for k in 1:length(zs)-1 for s in (0.0, 0.05, 0.1) for h in (-1, 1)])
    return C, far, gap
end
maxrel(u, v) = maximum(abs, u - v) / maximum(abs, v)
