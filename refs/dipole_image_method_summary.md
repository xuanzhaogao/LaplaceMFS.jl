# Dipole Image Method for Two Dielectric Spheres — Summary

## Problem Setup

Two identical dielectric spheres of radius $a$, relative permittivity $\varepsilon_r$, centered at $\mathbf{c}_1$ and $\mathbf{c}_2$ (separation $R = |\mathbf{c}_1 - \mathbf{c}_2| > 2a$), in a uniform external field $\mathbf{E}_0 = E_0 \hat{\mathbf{z}}$. Background permittivity $\varepsilon_0 = 1$.

The scattered potential $u(x)$ satisfies:
- $\Delta u = 0$ everywhere,
- Continuity of potential across each sphere boundary,
- Jump condition on normal derivative: $(1 - \varepsilon_r) \partial_n u^+ = -(\varepsilon_r - 1) E_0 n_z$ on each $\partial\Omega_i$,
- $u \to 0$ as $|x| \to \infty$.

## Current Implementation (Point Dipole Only)

In `src/utils/double_spheres.jl`, the existing `double_sphere_image_coefficients` uses a **point-dipole-only** approximation (l=1 truncation). Each sphere responds to the external field by producing a point dipole at its center:

$$\mathbf{s}_0 = \alpha a^3 \mathbf{E}_0, \quad \alpha = \frac{\varepsilon_r - 1}{\varepsilon_r + 2}$$

These dipoles interact via the dipole field tensor:

$$\mathbf{G} = \frac{1}{R^3}(3\hat{\mathbf{R}}\hat{\mathbf{R}}^T - \mathbf{I})$$

The reflection series $\mathbf{T} = \alpha a^3 \mathbf{G}$ gives:

$$\mathbf{s}_1^{(n+1)} = \mathbf{T} \cdot \mathbf{s}_2^{(n)}, \quad \mathbf{s}_2^{(n+1)} = \mathbf{T} \cdot \mathbf{s}_1^{(n)}$$

Closed-form solution: $(\mathbf{I} - \mathbf{T}^2)^{-1}(\mathbf{s}_0 + \mathbf{T}\mathbf{s}_0)$ for each sphere (by symmetry).

**Limitation:** This only captures the $l=1$ (dipole) channel. The actual image of a dipole in a dielectric sphere generates **higher-order multipoles**, which become significant when spheres are close.

## Proposed Improvement: Exact Kelvin Image Method

### Key Physics

When a **point dipole** $\mathbf{p}$ is located at distance $d$ from the center of a dielectric sphere of radius $a$ (with $d > a$), the **exact electrostatic image** inside the sphere consists of:

1. **A point dipole** at the Kelvin image point $\mathbf{x}' = \frac{a^2}{d^2}\mathbf{x}_{\text{src}}$ (relative to sphere center), with strength depending on $\varepsilon_r$, $a/d$.

2. **A line charge (monopole) distribution** along the segment from the sphere center to the Kelvin image point.

This is the classical result for a dielectric sphere (as opposed to a conducting sphere where the image is simpler).

### Iterative Procedure

1. **Initialization:** The external field $E_0 \hat{z}$ induces an equivalent dipole $\mathbf{p}_i^{(0)} = 4\pi \alpha a^3 E_0 \hat{z}$ at the center of each sphere $i$.

2. **Reflection step:** The dipole $\mathbf{p}_j^{(n)}$ at sphere $j$'s center acts as an external source for sphere $i$ ($i \neq j$). Its image inside sphere $i$ is:
   - A **point dipole** at the Kelvin image point inside sphere $i$,
   - A **line charge distribution** from sphere $i$'s center to the image point.

3. **Discretization:** The line charge distribution is discretized using numerical quadrature (e.g., Gauss-Legendre) into a set of point charges along the line segment.

4. **Next reflection:** Each of these image sources (point dipole + discretized line charges) at sphere $i$ now acts as external sources for sphere $j$. For each such source at distance $d_k$ from sphere $j$'s center, we compute the Kelvin image (new point dipole + line charges inside sphere $j$).

5. **Repeat** until the strength of newly generated images falls below a tolerance.

6. **Evaluation:** The scattered potential at any exterior point is the sum of all image sources (point dipoles and point charges from discretized line distributions) across all reflection levels and both spheres.

### Image Formulas for a Dielectric Sphere

For a point charge $q$ at distance $d > a$ from the center of a sphere of radius $a$, permittivity $\varepsilon_r$ (background $\varepsilon_0 = 1$):

- **Image point charge** at $d' = a^2/d$:
  $$q' = -\frac{\varepsilon_r - 1}{\varepsilon_r + 1} \cdot \frac{a}{d} \cdot q$$
  (This is approximate; the exact expression involves both a point image and a line image.)

- **Line charge** from center to $d' = a^2/d$: density $\lambda(t)$ for $t \in [0, a^2/d]$ depends on the specific formulation.

For a point dipole, the image is obtained by differentiating the point-charge image with respect to the source position, which yields:
- A **point dipole** at $d' = a^2/d$,
- A **point charge** at $d' = a^2/d$ (from differentiating the line charge endpoint),
- A **line dipole distribution** from center to $d'$,
- A **line charge distribution** from center to $d'$.

The exact formulas need to be specialized for the axial and transverse components of the dipole relative to the center-to-center axis.

## Codebase Structure

| File | Purpose |
|------|---------|
| `src/LaplaceMFS.jl` | Module definition, includes, exports |
| `src/laplace3d.jl` | Point-source potential and gradient kernels |
| `src/sphere.jl` | `SphereMats` struct, single/multi-sphere B matrices, RHS |
| `src/operators.jl` | `multispheres_G`, `multispheres_Ghat`, FMM variants |
| `src/evaluation.jl` | `eval_exterior_pot` — evaluate MFS potential at targets |
| `src/utils/single_sphere.jl` | Analytical single-sphere scattered potential |
| `src/utils/double_spheres.jl` | Point-dipole image method (current, l=1 only) |
| `test/utils/double_spheres.jl` | Tests: eigenvalue limits + comparison with MFS |
| `refs/note.pdf` | Full derivation of the MFS formulation |
| `refs/double_sphere_image_derivation.md` | Derivation of the dipole-only reflection series |

## Design Decisions from User

- Use the **standard Kelvin image** (point dipole + line charge/dipole).
- **Equal radii only** (both spheres have the same radius $a$).
- Discretize line images with **quadrature**.
- Iterate reflections until **convergence**.

## MFS Reference Solution

The MFS approach (`doublespheres_B`) solves the full boundary-value problem with $N$ proxy point sources inside and outside each sphere, matched at $M$ surface collocation points. This gives a high-accuracy reference for validating the image method. The existing test shows the point-dipole image agrees to ~8% relative error for well-separated spheres ($R/a = 8$).
