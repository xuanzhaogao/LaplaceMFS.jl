using CairoMakie
using LaplaceMFS
using LinearAlgebra
using Statistics

# Optional CLI args:
# M is the number of surface points, N is the number of proxy points, M > N so that gives a overdetermined system.
M = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1302 
N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1059 
Ez = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0
eps_r = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 2.5
r = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1.0
r_p = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 0.7

# Solve B * [p; -q] = rhs
B = LaplaceMFS.single_sphere_B(r, r_p, M, N)
rhs = LaplaceMFS.single_sphere_Ez_rhs(r, M, Ez, eps_r)
x = B \ rhs
p = x[1:N]
qneg = x[N+1:2N]

pts_N = LaplaceMFS.load_sphdes_N(N)
r_q = r * r / r_p

function potential_from_sources(coeffs, src_scale, src_pts, y, z)
    trg = [0.0, y, z]
    u = 0.0
    @inbounds for j in eachindex(coeffs)
        src = src_scale .* vec(src_pts[j, :])
        u += coeffs[j] * LaplaceMFS.laplace3d_pot(src, trg)
    end
    return u
end

# Evaluate on x=0 plane
L = 2.0 * r
ngrid = 251
ys = range(-L, L; length=ngrid)
zs = range(-L, L; length=ngrid)

u_ext_num = Matrix{Float64}(undef, ngrid, ngrid)
u_int_num = Matrix{Float64}(undef, ngrid, ngrid)
rho = Matrix{Float64}(undef, ngrid, ngrid)

for iz in eachindex(zs), iy in eachindex(ys)
    y = ys[iy]
    z = zs[iz]
    u_ext_num[iz, iy] = potential_from_sources(p, r_p, pts_N, y, z)
    u_int_num[iz, iy] = potential_from_sources(qneg, r_q, pts_N, y, z)
    rho[iz, iy] = hypot(y, z)
end

# Theoretical mode amplitudes for this formulation:
# c = (1 - 1/eps_r) * Ez * r,
# u_ext^(th) = (c*r^3/3) * z/rho^3 (outside),
# u_int^(th) = (2c/3) * z (inside).
c = (1.0 - 1.0 / eps_r) * Ez * r
u_ext_th = Matrix{Float64}(undef, ngrid, ngrid)
u_int_th = Matrix{Float64}(undef, ngrid, ngrid)
for iz in eachindex(zs), iy in eachindex(ys)
    z = zs[iz]
    rr = rho[iz, iy]
    u_ext_th[iz, iy] = (c * r^3 / 3.0) * z / rr^3
    u_int_th[iz, iy] = (2.0 * c / 3.0) * z
end

inside = rho .<= r

ext_rel_err = abs.(u_ext_num .- u_ext_th) ./ abs.(u_ext_th)
int_rel_err = abs.(u_int_num .- u_int_th) ./ abs.(u_int_th)

# Keep only physically relevant regions for each mode.
ext_rel_err[inside] .= NaN
int_rel_err[.!inside] .= NaN

finite_ext = filter(isfinite, ext_rel_err)
finite_int = filter(isfinite, int_rel_err)
println("Exterior mode: max relative error = ", maximum(finite_ext), ", rms = ", sqrt(mean(finite_ext .^ 2)))
println("Interior mode: max relative error = ", maximum(finite_int), ", rms = ", sqrt(mean(finite_int .^ 2)))

# Avoid zeros for logarithmic color scale.
tiny = eps(Float64)
ext_rel_plot = map(x -> isfinite(x) ? max(x, tiny) : NaN, ext_rel_err)
int_rel_plot = map(x -> isfinite(x) ? max(x, tiny) : NaN, int_rel_err)

combined_rel_plot = similar(ext_rel_plot)
combined_rel_plot[inside] .= int_rel_plot[inside]
combined_rel_plot[.!inside] .= ext_rel_plot[.!inside]

fig = Figure(size = (760, 620))
ax = Axis(
    fig[1, 1],
    title = "Relative Error on x=0 Plane (inside: interior mode, outside: exterior mode)",
    xlabel = "y",
    ylabel = "z",
    aspect = DataAspect(),
)
hm = heatmap!(ax, ys, zs, combined_rel_plot; colormap = :viridis, colorscale = log10)
Colorbar(fig[1, 2], hm)

output = joinpath(@__DIR__, "single_sphere_Ez_plane_error.png")
save(output, fig)
println("Saved figure to ", output)
