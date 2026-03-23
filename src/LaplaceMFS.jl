module LaplaceMFS

using LinearAlgebra, Krylov, SparseArrays
using LinearMaps, FMM3D
using Artifacts, DelimitedFiles, Printf

export effsphdes_path, load_sphdes_t, load_sphdes_N, sphdes_num_points
export SphereMats
export laplace3d_pot, laplace3d_grad
export multispheres_mu_to_lambda, multispheres_mu_to_lambda!
export multispheres_G, multispheres_G_fmm, multispheres_Ghat, multispheres_Ghat_fmm
export eval_exterior_pot
export single_sphere_alpha, single_sphere_scattered_exterior, single_sphere_scattered_interior
export double_sphere_image_coefficients, double_sphere_image_potential
export single_sphere_forward_point_line_images, double_sphere_forward_point_line_images
export doublespheres_B, doublespheres_Ez_rhs

include("core.jl")

include("effsphdes.jl")
include("laplace3d.jl")

include("sphere.jl")

include("operators.jl")
include("evaluation.jl")

include("utils/single_sphere.jl")
include("utils/double_spheres.jl")

end
