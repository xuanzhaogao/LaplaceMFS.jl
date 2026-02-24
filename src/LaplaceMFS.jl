module LaplaceMFS

using LinearAlgebra, KrylovKit
using FMM3D
using Artifacts

export effsphdes_path

effsphdes_path() = joinpath(artifact"EffSphDes")


end
