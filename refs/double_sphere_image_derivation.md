# Point+Line Image Generators for One and Two Spheres

This note documents the formulas implemented in:

- `single_sphere_forward_point_line_images`
- `double_sphere_forward_point_line_images`

in [`src/utils/double_spheres.jl`](../src/utils/double_spheres.jl).

The goal of these routines is different from the older
`double_sphere_image_coefficients` / `double_sphere_image_potential` path.
Those older functions build a dipole-only (`l = 1`) approximation and solve a
small coupled linear system. The new point+line routines do not solve any
boundary-matching problem. Instead, they take a given exterior point source and
explicitly generate the forward image sources produced by one sphere or by
multiple reflections between two spheres.

## 1. Scope and Conventions

We consider an exterior point source with strength `q` located at `x_s`, outside
a sphere of radius `a` centered at `c`.

The Laplace kernel used elsewhere in this repository is

\[
G(x,y) = \frac{1}{4\pi |x-y|},
\]

so a discrete image source with charge `q_j` at `x_j` contributes

\[
u_j(y) = q_j G(x_j, y) = \frac{q_j}{4\pi |y-x_j|}.
\]

The point+line image generators only return charges and positions. They do not
evaluate the potential themselves.

The current implementation takes a scalar parameter `gamma` as input. This note
documents the formulas for a given `gamma`; the code does not derive `gamma`
from `eps_r`, and any constitutive relation between material parameters and
`gamma` is left to the caller.

## 2. Single-Sphere Geometry

For one sphere:

- center: `c`
- radius: `a`
- source point: `x_s`
- source strength: `q`

Define

\[
r_s = |x_s - c|, \qquad \hat d = \frac{x_s - c}{r_s},
\]

with the requirement `r_s > 0`. In intended use, the source is also outside the
sphere, so typically `r_s > a`.

Any image source placed on the center-to-source line can be written as

\[
x(\tau) = c + \tau \hat d,
\]

where `tau` is the distance from the center measured along that ray.

The Kelvin image location used by the code is

\[
r_K = \frac{a^2}{r_s}.
\]

So the Kelvin point image lies at

\[
x_K = c + r_K \hat d.
\]

This is implemented by `_point_along_center_source_line(center, source, r_s, dist)`.

## 3. Single-Sphere Point Image

The point part of the image system is the Kelvin image

\[
q_K = -\gamma \frac{a}{r_s} q,
\qquad
x_K = c + \frac{a^2}{r_s} \hat d.
\]

This is exactly what the code computes:

```julia
qK = -gamma * radius * q / r_s
rK = radius^2 / r_s
```

If `abs(qK) > cutoff`, this point image is appended as the first entry of the
returned arrays.

## 4. Single-Sphere Line Image

The line part of the image system is represented on the segment from the center
to the Kelvin point:

\[
0 \le \tau \le r_K,
\qquad
x(\tau) = c + \tau \hat d.
\]

The code uses the parameter

\[
\lambda = \frac{1-\gamma}{2}.
\]

When `lambda <= 0`, the line part is skipped. In particular, `gamma = 1`
produces a point-only image system.

The continuous line image represented by the code has density

\[
\rho(\tau)
= \frac{\gamma \lambda q}{a} \, r_K^{1-\lambda} \tau^{\lambda-1},
\qquad 0 < \tau < r_K.
\]

So, for any linear evaluation functional `F(x)`, the line contribution is

\[
\int_0^{r_K} \rho(\tau) F(x(\tau)) \, d\tau
=
\frac{\gamma \lambda q}{a} r_K^{1-\lambda}
\int_0^{r_K} \tau^{\lambda-1} F(x(\tau)) \, d\tau.
\]

This is the quantity approximated by quadrature in the implementation.

## 5. Why Jacobi Quadrature Appears

The factor `tau^(lambda-1)` is integrably singular at `tau = 0` when
`0 < lambda < 1`. That is exactly why the code uses Jacobi quadrature rather
than an ordinary Gauss-Legendre rule.

The implementation chooses Jacobi parameters

\[
\alpha = 0, \qquad \beta = \lambda - 1,
\]

so the reference quadrature integrates functions on `[-1,1]` with weight

\[
(1-t)^\alpha (1+t)^\beta = (1+t)^{\lambda-1}.
\]

Let `(t_i, w_i)` be the Jacobi nodes and weights for this weight function. The
code computes them in `_jacobi_roots_weights` by forming the Jacobi
tridiagonal matrix and applying an eigenvalue solve, which is the standard
Golub-Welsch construction.

Now map `[-1,1]` to `[0,r_K]` by

\[
\tau = \frac{r_K}{2}(1+t),
\qquad
t = \frac{2\tau}{r_K} - 1.
\]

Then

\[
\int_0^{r_K} \tau^{\lambda-1} F(x(\tau)) \, d\tau
\approx
\left(\frac{r_K}{2}\right)^\lambda
\sum_{i=1}^{n_{\rm line}} w_i F(x(\tau_i)),
\qquad
\tau_i = \frac{r_K}{2}(1+t_i).
\]

Multiplying by the prefactor from the continuous line density gives

\[
\int_0^{r_K} \rho(\tau) F(x(\tau)) \, d\tau
\approx
\sum_{i=1}^{n_{\rm line}} q_i F(x_i),
\]

with

\[
x_i = c + \tau_i \hat d,
\]

and discrete quadrature charges

\[
q_i
=
w_i \frac{\gamma \lambda q}{a}
\left(\frac{r_K}{2}\right)^\lambda r_K^{1-\lambda}.
\]

This is the exact formula implemented in the code. Since

\[
\left(\frac{r_K}{2}\right)^\lambda r_K^{1-\lambda}
= \frac{r_K}{2^\lambda},
\]

an equivalent simplified form is

\[
q_i = w_i \frac{\gamma \lambda q}{a} \frac{r_K}{2^\lambda}.
\]

The code keeps the unsimplified form because it drops directly out of the
quadrature mapping:

```julia
lambda = (1 - gamma) / 2
alpha = 0
beta = lambda - 1
nodes, weights = _jacobi_roots_weights(n_line, alpha, beta)
base = (rK / 2)^(alpha + beta + 1) * rK^(1 - lambda)
pref = gamma * lambda * base / radius * q
q_i = weights[i] * pref
x_i = _point_along_center_source_line(..., dist = rK * (1 + nodes[i]) / 2)
```

## 6. What `single_sphere_forward_point_line_images` Returns

The function returns

```julia
(; q = q_images, x = x_positions)
```

where:

- `q` is a vector of image strengths
- `x` is a `3 x N` matrix of image positions
- `x[:, j]` is the position associated with `q[j]`

Ordering is:

1. Kelvin point image first, if it survives `cutoff`
2. then the line quadrature images in Jacobi-node order

Special cases:

- `n_line == 0`: the line part is disabled
- `gamma == 1`: `lambda = 0`, so only the Kelvin point image remains
- `cutoff > 0`: any image with `abs(q_image) <= cutoff` is discarded
- if all images are discarded, the function returns `q = []` and `x` with size `3 x 0`

## 7. Two-Sphere Forward Reflection Algorithm

Now consider two spheres:

- centers `c1`, `c2`
- radii `a1`, `a2`
- image parameters `gamma1`, `gamma2`

and one initial exterior source `(q0, x0)`.

Define the one-step reflection operator

\[
\mathcal R_i(q, x)
=
\text{all point+line images of source } (q,x)
\text{ inside sphere } i.
\]

In code, this is just one call to
`single_sphere_forward_point_line_images` with the geometry of sphere `i`.

The two-sphere routine performs the following forward recursion.

### Level 1

Reflect the original source into each sphere independently:

\[
L_1^{(1)} = \mathcal R_1(q_0, x_0),
\qquad
L_2^{(1)} = \mathcal R_2(q_0, x_0).
\]

### Higher levels

For `m >= 2`, generate new images in sphere 1 from the previous images in
sphere 2, and vice versa:

\[
L_1^{(m)}
=
\bigcup_{(q,x)\in L_2^{(m-1)}} \mathcal R_1(q, x),
\]

\[
L_2^{(m)}
=
\bigcup_{(q,x)\in L_1^{(m-1)}} \mathcal R_2(q, x).
\]

This is exactly what the loops in
`double_sphere_forward_point_line_images` do:

- iterate over each previous source on the opposite sphere
- call the single-sphere generator
- append all newly generated child images

### Accumulated output

After `n_reflections` levels, the function returns all images generated up to
that depth:

\[
Q_1 = \bigcup_{m=1}^{n_{\rm reflections}} L_1^{(m)},
\qquad
Q_2 = \bigcup_{m=1}^{n_{\rm reflections}} L_2^{(m)}.
\]

In Julia this comes back as

```julia
(; q1 = q1_all, x1 = x1_all, q2 = q2_all, x2 = x2_all)
```

with `x1[:,j]` paired with `q1[j]`, and similarly for sphere 2.

## 8. Growth of the Number of Images

Without `cutoff`, each reflected source produces:

- `1` child if only the Kelvin point image is kept
- `1 + n_line` children if the line part is active

So the number of sources can grow quickly with `n_line` and
`n_reflections`. This routine is best viewed as a forward image generator for
inspection, experimentation, or downstream evaluation, not as a globally
compressed representation.

## 9. How to Use the Single-Sphere Generator

Example:

```julia
using LaplaceMFS

center = [0.0, 0.0, 0.0]
source = [0.0, 0.0, 1.5]

out = LaplaceMFS.single_sphere_forward_point_line_images(
    center, 1.0, 0.4, 1.0, source;
    n_line = 8,
    cutoff = 0.0,
)

length(out.q)     # number of image sources
out.q[1]          # Kelvin point-image strength
out.x[:, 1]       # Kelvin point-image position
```

If you want to evaluate the generated images at an exterior target using the
repository kernel, you can sum:

```julia
target = [0.2, 0.1, 2.0]
u = sum(
    out.q[j] * LaplaceMFS.laplace3d_pot(vec(out.x[:, j]), target)
    for j in eachindex(out.q)
)
```

Because `laplace3d_pot` already includes the factor `1 / (4*pi)`, this `u` is
the potential associated with the generated discrete image charges.

## 10. How to Use the Two-Sphere Generator

Example:

```julia
using LaplaceMFS

centers = [
    0.0 0.0 0.0;
    0.0 0.0 3.0
]

out = LaplaceMFS.double_sphere_forward_point_line_images(
    centers,
    (1.0, 1.0),
    (0.5, 0.2),
    1.0,
    [0.0, 0.0, 1.5];
    n_line = 8,
    n_reflections = 3,
    cutoff = 1e-12,
)

length(out.q1)
length(out.q2)
out.x1[:, 1]
out.x2[:, 1]
```

To evaluate the full generated image system at a target, sum the contributions
from both spheres:

```julia
target = [0.1, 0.0, 2.5]

u = sum(
        out.q1[j] * LaplaceMFS.laplace3d_pot(vec(out.x1[:, j]), target)
        for j in eachindex(out.q1)
    ) +
    sum(
        out.q2[j] * LaplaceMFS.laplace3d_pot(vec(out.x2[:, j]), target)
        for j in eachindex(out.q2)
    )
```

## 11. Dipole Images with the Same Positions

The code also provides:

- `single_sphere_forward_point_line_dipole_images`
- `double_sphere_forward_point_line_dipole_images`

These routines reuse the exact same image positions, ordering, and recursion tree
as the point-charge generators above. The only change is the source type.

If the point-charge generator produces scalar image coefficients

\[
\{q_j, x_j\}_{j=1}^N
\]

for a unit source charge, then the dipole generator uses the same positions
\(\{x_j\}\) and attaches dipole vectors

\[
p_j = q_j p_0,
\]

where \(p_0 \in \mathbb{R}^3\) or \(\mathbb{C}^3\) is the input dipole moment.

So the current dipole implementation should be understood as:

- same geometry as the point-charge image construction,
- same scalar reflection weights as the point-charge image construction,
- a vector dipole moment carried at each of those positions.

This matches the intended "same positions as point charges" construction used to
build dipole image sets, but it is not a fresh derivation of the exact
dielectric-sphere image of a dipole.

## 12. Important Limitations

These routines are intentionally limited:

1. They are forward generators, not boundary-value solvers.
2. They do not solve for self-consistent coefficients.
3. They do not automatically sum an infinite reflection series to convergence.
4. They do not merge or compress repeated image sources.
5. They are separate from `double_sphere_image_potential`, which still uses the
   older dipole-only two-sphere approximation.

So the mental model should be:

- `single_sphere_forward_point_line_images`: "Given one exterior point source,
  what image charges does one sphere generate?"
- `single_sphere_forward_point_line_dipole_images`: "Use the same image locations,
  but attach dipole vectors scaled by the same scalar image coefficients."
- `double_sphere_forward_point_line_images`: "If those images are repeatedly
  reflected between two spheres for a fixed number of levels, what discrete
  image charges do I get?"
- `double_sphere_forward_point_line_dipole_images`: "Use the same two-sphere
  reflection tree, but carry dipole vectors instead of scalar charges."
- `double_sphere_image_potential`: "A different API using a reduced dipole-only
  approximation, not the point+line reflection generator documented here."
