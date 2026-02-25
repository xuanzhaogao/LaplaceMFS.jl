module LaplaceMFS

using LinearAlgebra, KrylovKit, SparseArrays
using LinearMaps, FMM3D
using Artifacts, DelimitedFiles, Printf

export effsphdes_path, load_sphdes_t, load_sphdes_N, sphdes_num_points
export SphereMats
export laplace3d_pot, laplace3d_grad
export multispheres_mu_to_lambda, multispheres_mu_to_lambda!
export multispheres_G, multispheres_G_fmm, multispheres_Ghat, multispheres_Ghat_fmm

include("core.jl")

include("effsphdes.jl")
include("laplace3d.jl")

include("sphere.jl")

include("operators.jl")

end
