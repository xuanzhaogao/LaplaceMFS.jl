module LaplaceMFS

using LinearAlgebra, KrylovKit
using FMM3D
using LazyArtifacts

export effsphdes_path

effsphdes_path() = joinpath(artifact"EffSphDes", "EffSphDes")


end
