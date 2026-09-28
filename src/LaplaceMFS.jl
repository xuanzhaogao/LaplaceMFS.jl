module LaplaceMFS

using LinearAlgebra, Krylov, SparseArrays
using LinearMaps, FMM3D
using Artifacts, DelimitedFiles, Printf

export effsphdes_path, load_sphdes_t, load_sphdes_N, sphdes_num_points
export SphereMats
export laplace3d_pot, laplace3d_grad
export multispheres_mu_to_lambda, multispheres_mu_to_lambda!
export multispheres_G, multispheres_G_fmm, multispheres_Ghat, multispheres_Ghat_fmm
export eval_exterior_pot, eval_total_pot
export plot_surface_potential, plot_plane_potential, plot_plane_error
export single_sphere_alpha, single_sphere_scattered_exterior, single_sphere_scattered_interior
export double_sphere_image_coefficients, double_sphere_image_potential
export single_sphere_forward_point_line_images, double_sphere_forward_point_line_images
export single_sphere_forward_point_line_dipole_images, double_sphere_forward_point_line_dipole_images
export doublespheres_B, doublespheres_Ez_rhs
export multispheres_pointcharge_rhs, multispheres_uniform_rhs
export MultiSphereLines, LineSphere
export line_accumulation_radius, line_critical_gap, line_num_nodes, line_nodes
export laplace3d_dipole_pot, laplace3d_dipole_grad

include("core.jl")

include("effsphdes.jl")
include("laplace3d.jl")

include("sphere.jl")

include("operators.jl")
include("evaluation.jl")
include("lines.jl")

include("utils/single_sphere.jl")
include("utils/double_spheres.jl")

end
