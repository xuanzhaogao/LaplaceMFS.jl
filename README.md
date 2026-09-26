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

## Point-charge excitation, total potential and plotting

- `multispheres_pointcharge_rhs(r, M, eps_r, centers, charge_pos, charges)` builds the
  right-hand side for exterior point charges (`charge_pos` is `3 × nq`).
- `eval_total_pot(centers, r, N, coeffs, r_p, fmm_tol, targets, charge_pos, charges)` returns
  the total potential anywhere: `u_inc + Σ_j u_ext_j` outside the spheres and
  `u_inc + Σ_{j≠i} u_ext_j + u_int_i` inside sphere `i`.

With a Makie backend loaded (`using CairoMakie` or `GLMakie`), the `LaplaceMFSMakieExt`
extension provides

- `plot_surface_potential(centers, r, r_p, N, coeffs, charge_pos, charges; ...)`: 3D view of the
  total potential on every sphere surface;
- `plot_plane_potential(centers, r, r_p, N, coeffs, charge_pos, charges; normal = :y, offset = 0.0, ...)`:
  the total potential on a cutting plane, inside and outside the spheres, with the sphere
  cross-sections outlined and nearby charges marked.

```julia
using LaplaceMFS, CairoMakie, Krylov
r, r_p, M, N, eps_r = 1.0, 0.5, 614, 513, 2.5
centers = [0.0 0.0 0.0; 0.0 0.0 3.0]
charge_pos = [0.0 1.8; 0.0 0.0; -2.0 1.5]; charges = [1.0, -0.5]
mats = SphereMats(r, r_p, M, N, eps_r, 1e-13)
rhs = multispheres_pointcharge_rhs(r, M, eps_r, centers, charge_pos, charges)
mu, _ = Krylov.gmres(multispheres_Ghat_fmm(mats, centers, 1e-13), rhs; rtol = 1e-12)
lambda = multispheres_mu_to_lambda(mats, mu)
save("surface.png", plot_surface_potential(centers, r, r_p, N, lambda, charge_pos, charges))
save("plane.png", plot_plane_potential(centers, r, r_p, N, lambda, charge_pos, charges; normal = :y))
```
