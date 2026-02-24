module LaplaceMFS

using LinearAlgebra, KrylovKit
using FMM3D
using Artifacts
using DelimitedFiles
using Printf

export effsphdes_path, load_sphdes_t, load_sphdes_N, sphdes_num_points
export laplace3d_pot, laplace3d_grad

include("effsphdes.jl")
include("laplace3d.jl")

include("sphere.jl")

end
