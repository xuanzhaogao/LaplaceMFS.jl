# Double-Sphere Image-Reflection Derivation (Uniform `E0 ẑ`)

This note summarizes the first-case model implemented in `src/utils/double_spheres.jl`:

- two **identical** dielectric spheres,
- radius `r`, relative permittivity `eps_r`,
- centers on the x-axis separated by distance `d`,
- incident field is uniform along z: `E0 ẑ`.

## 1. Single-sphere dipole response

For one sphere in a uniform field, the scattered exterior potential has dipole form

\[
u_{\text{sc}}(\mathbf x)=\alpha E_0 \frac{z}{|\mathbf x|^3},
\quad
\alpha=r^3\frac{\varepsilon_r-1}{\varepsilon_r+2}.
\]

So `alpha` is the scalar polarizability used in code:

```julia
alpha = r^3 * (eps_r - 1) / (eps_r + 2)
```

## 2. Two-sphere reflection model

Let `a1`, `a2` be dipole amplitudes for spheres 1 and 2:

\[
u_{\text{sc}}(\mathbf x)
= a_1 \frac{z_1}{|\mathbf x-\mathbf c_1|^3}
  a_2 \frac{z_2}{|\mathbf x-\mathbf c_2|^3}.
\]

Each sphere polarizes in the local field = incident field + field induced by the other sphere.
Along the axis connecting centers, the leading-order coupling contributes a term proportional to
\(a_j/d^3\). Using a reflection iteration:

\[
a_1^{(k+1)}=\alpha\left(E_0-\frac{a_2^{(k)}}{d^3}\right),\qquad
a_2^{(k+1)}=\alpha\left(E_0-\frac{a_1^{(k)}}{d^3}\right).
\]

Initialization uses isolated-sphere values:

\[
a_1^{(0)}=a_2^{(0)}=\alpha E_0.
\]

For symmetric geometry, fixed point satisfies:

\[
a_1=a_2=a,\qquad
a=\frac{\alpha E_0}{1+\alpha/d^3}.
\]

## 3. Exterior potential evaluation

After convergence, evaluate at targets outside both spheres:

\[
u_{\text{sc}}(\mathbf x)=
a_1 \frac{(\mathbf x-\mathbf c_1)_z}{|\mathbf x-\mathbf c_1|^3}
a_2 \frac{(\mathbf x-\mathbf c_2)_z}{|\mathbf x-\mathbf c_2|^3}.
\]

This is implemented by:

- `double_sphere_image_coefficients(...)`
- `double_sphere_image_potential(...)`

## 4. Model scope

This is a first-case, first-order reflection dipole model:

- supports only two identical spheres with x-axis centers,
- uniform field along z,
- most accurate for separated spheres (`d > 2r`, better as `d/r` grows),
- captures interaction trend and spatial pattern, but may need amplitude correction vs full MFS for moderate separations.
