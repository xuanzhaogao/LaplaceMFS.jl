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
  `u_inc + Σ_{j≠i} u_ext_j + u_int_i` inside sphere `i`. With `incident = false` it returns the
  scattered potential (the same quantity without `u_inc`).

With a Makie backend loaded (`using CairoMakie` or `GLMakie`), the `LaplaceMFSMakieExt`
extension provides

- `plot_surface_potential(centers, r, r_p, N, coeffs, charge_pos, charges; ...)`: 3D view of the
  total potential on every sphere surface;
- `plot_plane_potential(centers, r, r_p, N, coeffs, charge_pos, charges; normal = :y, offset = 0.0, ...)`:
  the potential on a cutting plane, inside and outside the spheres, with the sphere
  cross-sections outlined and nearby charges marked. `style = :heatmap`, `:contour` (filled
  contour bands) or `:both` (default);
- `plot_plane_error(centers, r, r_p, N, coeffs, charge_pos, charges, reference; ...)`: a map of
  `log10(|u − u_ref| / max|u_ref|)` for the scattered potential outside the spheres, where
  `reference(targets)` returns a reference scattered potential at the columns of `targets`
  (for example `t -> HybridSolve.eval_exterior_pot(sol, t)`).

Both potential plots take `field = :total` (default) or `field = :scattered`. All three plotting functions take
`rasterize` (default `2`): heatmaps, filled contours and surfaces are rasterized at 2× resolution
when saving to PDF or SVG, keeping the files small; `rasterize = false` gives pure vector output.

```julia
using LaplaceMFS, CairoMakie, Krylov
r, r_p, M, N, eps_r = 1.0, 0.5, 614, 513, 2.5
centers = [0.0 0.0 0.0; 0.0 0.0 3.0]
charge_pos = [0.0 1.8; 0.0 0.0; -2.0 1.5]; charges = [1.0, -0.5]
mats = SphereMats(r, r_p, M, N, eps_r, 1e-13)
rhs = multispheres_pointcharge_rhs(r, M, eps_r, centers, charge_pos, charges)
mu, _ = Krylov.gmres(multispheres_Ghat_fmm(mats, centers, 1e-13), rhs; rtol = 1e-12)
lambda = multispheres_mu_to_lambda(mats, mu)
save("surface.pdf", plot_surface_potential(centers, r, r_p, N, lambda, charge_pos, charges))
save("plane.pdf", plot_plane_potential(centers, r, r_p, N, lambda, charge_pos, charges; normal = :y))
```

## Line image sources for close spheres and nearby charges

`MultiSphereLines(r, r_p, M, N, centers, eps_r, tol; charge_pos)` augments the proxy spheres
with lines of image nodes placed as in `refs/stokesmfs.pdf` (Eqs. (23)–(27)): inside each
sphere, a line from the proxy sphere to the image accumulation point of every neighbour
closer than `line_critical_gap(r, r_p)`, and to the Kelvin image `r²/d` of every charge whose
image lies outside the proxy sphere. Each node carries `(q, px, py, pz)`; the line is also
reflected outside the sphere (`mirror = true`) for the interior field, and two caps of extra
collocation points (Eqs. (36)–(37), row-weighted as in Remark 5) sit above each line. The
number of nodes follows Eq. (39) by default (`n_line` overrides it).

The same functions take the system in place of the plain arguments:
`multispheres_G(sys)`, `multispheres_Ghat(sys)`, `multispheres_Ghat_fmm(sys, fmm_tol)`,
`multispheres_mu_to_lambda(sys, mu)`, `multispheres_pointcharge_rhs(sys, charge_pos, charges)`,
`multispheres_uniform_rhs(sys, E)`, `eval_exterior_pot(sys, lambda, fmm_tol, targets)` and
`eval_total_pot(sys, lambda, fmm_tol, targets, charge_pos, charges)`. Without any line the
system is exactly the plain one.

```julia
centers = [0.0 0.0 0.0; 0.0 0.0 2.01]                    # gap 0.01
charge_pos = reshape([0.0, 0.0, -2.0], 3, 1); charges = [1.0]
sys = MultiSphereLines(1.0, 0.5, 614, 513, centers, 2.5, 1e-13; charge_pos)
rhs = multispheres_pointcharge_rhs(sys, charge_pos, charges)
mu, _ = Krylov.gmres(multispheres_Ghat_fmm(sys, 1e-13), rhs; rtol = 1e-13)
lambda = multispheres_mu_to_lambda(sys, mu)
eval_exterior_pot(sys, lambda, 1e-13, targets)          # rel. error 2.7e-6 vs 1.2e-4 without lines
```
