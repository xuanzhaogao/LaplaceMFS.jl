# LaplaceMFS

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ArrogantGao.github.io/LaplaceMFS.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ArrogantGao.github.io/LaplaceMFS.jl/dev/)
[![Build Status](https://github.com/ArrogantGao/LaplaceMFS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ArrogantGao/LaplaceMFS.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ArrogantGao/LaplaceMFS.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ArrogantGao/LaplaceMFS.jl)

## Current Operator Formulas

For `ns` spheres, vectors are stored in interleaved sphere-local order:

- `lambda = [p1; q1; p2; q2; ...; pns; qns]`, each `ps, qs in C^N`
- `mu = [mus,1; mus,2; ...; mus,ns]`, each `mus in C^(2M)`

The per-sphere pseudoinverse application is done stably (without explicitly forming `Bplus`):

- `lambda_s = V_B' * Diagonal(S_B_inv) * (U_B' * mu_s)`

where `(U_B, S_B, V_B)` come from `svd(B)` for the single-sphere matrix `B`.

The dense multi-sphere map `G` is assembled in the same interleaved ordering and acts directly on `lambda`.

The current `Ghat` operator follows Eq. (13) in `refs/stokesmfs.pdf`:

- `Ghat(mu) = mu + u_all - B_blkdiag * lambda`
- `u_all = G * lambda`

with `B_blkdiag = blkdiag(B, ..., B)`.

`multispheres_Ghat_fmm` uses the same formula and ordering, but computes `u_all` via `multispheres_G_fmm` (a `LinearMap`) instead of a dense `G`.
