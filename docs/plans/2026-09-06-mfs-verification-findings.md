# MFS Verification: Findings and Plan

**Date:** 2026-09-06
**Scope:** Verifying `LaplaceMFS.jl` against the `HybridMD` reference implementation
(Gan, Jiang, Luijten & Xu), plus everything learned about both codes along the way.

---

## 0. Executive summary

1. **LaplaceMFS's formulation is correct.** The right-hand-side sign/normalization
   convention was confirmed analytically and numerically on the first attempt, and a
   single dielectric sphere driven by an exterior point charge converges to the exact
   Legendre-series answer at **2.6e-13** relative error (`d/a = 2`, `M = 1302`,
   `N = 1059`). There is no sign, normalization, or kernel defect anywhere in the
   single-sphere path.

2. **The multi-sphere path now passes its first physics test — in the
   well-separated regime only.** Every test in `test/operators.jl` had been a
   *self-consistency* check (FMM against dense, `Ghat` against the same formula
   rebuilt inline), and `Ghat` had never actually solved anything: it was only ever
   applied to `randn` vectors. §3.5 closes that for two well-separated spheres —
   the assembled operator converges to the exact single-sphere answer at the
   theoretically correct `O(R⁻⁴)` rate, and `Ghat` + GMRES + FMM reproduces the
   direct least-squares to 8e-14. The **near-touching and many-sphere regimes
   remain untested**, and are what the rest of this plan exists to cover.

3. **The near field converges, but only with the right `r_p`, and it is expensive.**
   At `d/a = 1.2` the method reaches **4.8e-07** with `N = 5001` and `r_p = 0.7`,
   converging geometrically. With `r_p = 0.5` the same sequence *stalls* at ~2.5e-04
   because `cond(B)` reaches 1.8e+12 and roundoff destroys the high-order content.
   The cost law is `N_proxy ~ n_max²` with `n_max ~ 1/ln(d/a)` — the regime the
   image-charge machinery in `src/utils/double_spheres.jl` exists to handle.

4. **The boundary residual is not a proxy for field error.** In the under-resolved
   regime, minimizing `‖Bx − rhs‖` can *increase* the field error by 40×. Any
   convergence criterion built on the residual is unsound.

5. **`HybridMD` works and can serve as an oracle**, after two build fixes — but its
   *force* path is broken and its shipped default precision silently disables the
   image charges. Both are documented below.

---

## 1. The mathematics

### 1.1 The transmission problem

A sphere `Ω` of radius `a`, interior relative permittivity `ε_r`, embedded in an
exterior medium of relative permittivity 1. Write the total potential as

```
φ = u_inc + u
```

where `u_inc` is the (smooth, source-free in Ω̄) incident potential and `u` is the
scattered/reaction field. On `∂Ω`, with `n` the outward unit normal:

```
φ⁺ = φ⁻                     (continuity of potential)
∂ₙφ⁺ = ε_r ∂ₙφ⁻             (continuity of normal flux)
```

### 1.2 The MFS ansatz

Two concentric proxy shells per sphere. With `r_p < a` and the Kelvin conjugate
`r_q = a²/r_p > a`, and `G(x,y) = 1/(4π|x−y|)`:

```
u_ext(x) = Σⱼ pⱼ G(x, c + r_p ûⱼ)      represents the field OUTSIDE  (sources inside)
u_int(x) = Σⱼ qⱼ G(x, c + r_q ûⱼ)      represents the field INSIDE   (sources outside)
```

Each source shell is singular only in the region where its field is *not* used, so
both representations are harmonic where they are needed. Collocating the two
interface conditions at `M > N` surface points gives the overdetermined system
`B x = rhs` with unknown `x = [p; −q]`:

```
B = ⎡ S_pr    S_qr  ⎤        rows 1..M    : potential
    ⎣ D_pr  ε_r·D_qr⎦        rows M+1..2M : flux
```

so that the two row blocks read

```
u_ext − u_int              = rhs_pot
∂ₙu_ext − ε_r ∂ₙu_int      = rhs_flux
```

(`src/sphere.jl:39-70`; `eps_r` is folded into `D_qr` at `sphere.jl:60`.)

### 1.3 The right-hand-side rule — CONFIRMED

Because `u_inc` is continuous and smooth across `∂Ω`, substituting `φ = u_inc + u`
into the interface conditions gives:

```
u_ext − u_int = 0
∂ₙ(u_inc + u_ext) = ε_r ∂ₙ(u_inc + u_int)
    ⟹  ∂ₙu_ext − ε_r ∂ₙu_int = (ε_r − 1) ∂ₙu_inc
```

**The rule:**

```
rhs_pot  [rows 1..M]      = 0
rhs_flux [rows M+1..2M]   = +(ε_r − 1) · ∂ₙu_inc(yᵢ)
                          = −(ε_r − 1) · E_inc(yᵢ) · nᵢ
```

Note the sign: it is **plus** `(ε_r − 1) ∂ₙu_inc`. The minus sign visible in
`singlesphere_Ez_rhs` belongs to `u_inc`, not to the rule.

Verified two ways:

- Rebuilding the existing uniform-field RHS from the rule reproduces
  `singlesphere_Ez_rhs` to `2.220e-16`.
- Negating the RHS produces exactly `−1.000×` the reference field, pinning the sign
  unambiguously.

**Consequence — what `Ez` means.** `sphere.jl:78` sets
`rhs[M+i] = −Ez(ε_r−1)n_z`, so `∂ₙu_inc = −Ez·n_z`, hence

```
u_inc = −Ez·z        E_inc = −∇u_inc = +Ez ẑ
```

`Ez` is the **field** amplitude, not a potential slope. This matches `refs/note.pdf`
Eq. (2) and is confirmed by the test at `test/sphere.jl:35-59`, which asserts
`proj(u_ext) ≈ +α·Ez·r` — a result that holds only for `u_inc = −E₀z`.

### 1.4 Exterior point-charge excitation

For charges `q_k` at positions `X_k`, all strictly outside every sphere:

```
u_inc(x) = Σₖ qₖ / (4π|x − Xₖ|)
∂ₙu_inc(yᵢ) = Σₖ qₖ · laplace3d_grad(Xₖ, yᵢ, nᵢ)
```

`laplace3d_grad(src, trg, n)` is exactly `∂/∂n` at the **target** of
`laplace3d_pot` (`src/laplace3d.jl:1-15`), so the RHS assembles directly. Unlike the
uniform-field case — where `∂ₙ(−Ez z) = −Ez n_z` is center-independent, which is why
`multispheres_Ez_rhs` legitimately discards `centers` (`sphere.jl:88-89`) — a point
charge genuinely depends on each sphere's center.

### 1.5 The analytic reference

Sphere radius `a` at the origin, interior `ε_r`, exterior 1; point charge `q` on the
`+z` axis at distance `d > a`. In the `1/(4πR)` normalization the package uses, the
**exterior reaction potential**, valid for all `ρ > a`, is

```
                q     ∞          a^(2n+1)
u_reac(ρ,θ) = ───── · Σ   Bₙ · ───────────── · Pₙ(cos θ)
               4π    n=1        d^(n+1) ρ^(n+1)

                    n(ε_r − 1)
with       Bₙ = − ───────────────
                   n(ε_r + 1) + 1
```

where `θ` is measured from the charge direction. The interior total field uses
`Cₙ = 1 + Bₙ = (2n+1)/(n(ε_r+1)+1)`.

**Sanity checks, all satisfied:**

- `B₀ = 0` — a neutral dielectric has no monopole reaction.
- `B₁ = −(ε_r−1)/(ε_r+2) = −α`, the standard polarizability.
- As `d → ∞`, `u_inc ≈ const + (q/4πd²)z`, i.e. `Ez = −q/(4πd²)` in the package's
  convention, and the series reduces to `α·Ez·a³·z/ρ³` — exactly
  `single_sphere_scattered_exterior` in `src/utils/single_sphere.jl`.
- Cross-validated against the package's own Lindell point+line image utility
  `single_sphere_forward_point_line_images` (with `γ = (ε_r−1)/(ε_r+1)`,
  `n_line = 600`): agreement to **1e-15 … 1e-14** relative.

**Convergence of the series.** Terms scale as `(a/d)ⁿ(a/ρ)^(n+1)`, and `Bₙ` is
bounded, so convergence is geometric with worst-case ratio `a/d` on the surface.
Terms needed for 1e-12:

| `d/a` | 1.1 | 1.2 | 1.5 | 3.0 |
|---|---|---|---|---|
| terms | 290 | 152 | 69 | 26 |

> **Implementation trap.** A truncation test of the form `|term| < tol·|sum|` breaks
> prematurely at equatorial targets (`cos θ = 0`), where every odd `Pₙ` vanishes.
> Require several *consecutive* small terms.

### 1.6 The `HybridMD` formulation and its units

`HybridMD` combines analytic image charges with a spherical-harmonic method of
moments, solved by GMRES and accelerated by FMM.

**Images.** For each (charge `i`, sphere `j`) pair close enough to the interface, a
Kelvin point image plus `im−1` line images on the radial segment `[0, a_j²/d]`:

```
Kelvin:  q_K = −γ_j a_j q_i / d      at   (a_j²/d²)(r_i − o_j) + o_j
line:    Gauss–Jacobi nodes/weights on [0, a_j²/d]
with     λ_j = ε_s/(ε_s + ε_j),   β_j = λ_j − 1,   γ_j = (ε_j − ε_s)/(ε_s + ε_j)
```

**Multipoles.** Each sphere carries `Bknm[s]`, with the potential
`Σ_{n,m} Bknm·Y_nm(θ,φ)/r^(n+1)` in the `sht` convention where `Y_nm` **includes**
`√(2n+1)`. FMM3D uses a different convention: `mpole_FMM = √(2n+1)·Bknm`
(`laprouts3d.f:265`). Both evaluation routes were run and agree to all printed digits.

**Units — this is the subtle part.** Every term in `HybridMD` uses a **bare `q/r`
kernel**: no `4π`, and — critically — **no division by `ε_s` anywhere**. The solvent
permittivity enters *only* through the boundary-condition coefficients (RHS
`(ε_s − ε_i)`, matvec diagonal `((k+1)ε_s/k + ε_i)`, off-diagonal `(ε_i − ε_s)`, and
the image coefficients `λ, β, γ`). Therefore the code's incident field is the
unscreened `Σq/r`, and uniformly

```
φ_HybridMD = ε_s · φ_physical
```

giving the conversion to LaplaceMFS's `1/(4πr)` convention:

```
φ_LaplaceMFS = φ_HybridMD / (4π·ε_s)
E_LaplaceMFS = E_HybridMD / (4π·ε_s)
```

Verified by re-running at `Solvent_Dielectric 2.0` and recovering the analytic free
term with **no** `1/ε_s` factor, to 7e-6.

**The printed energy** is the standard total
`E = Σ_{i<j} qᵢqⱼ/r + ½Σᵢ qᵢφ_pol(rᵢ)`, carrying the same `ε_s` factor. An
independently computed `0.5·q·φ_reaction = −0.135304721973` matched the printed
`total electrostatic energy` bit-for-bit.

---

## 2. Implementation findings

### 2.1 `LaplaceMFS.jl`

| Area | Finding |
|---|---|
| **Artifact** | The `EffSphDes` artifact URL in `Artifacts.toml` had gone stale (404, gist moved owner). **Fixed** — `Pkg.instantiate()` now installs 26.8 MiB and `load_sphdes_N` works. |
| **Valid `N`** | Only 180 specific counts exist: `3, 6, 8, 14, 18, 26, 32, 42, 50, 62, 72, 86, 98, 114, 128, 146, 163, 182, 201, 222, 243, 266, 289, 314, …, 614, …, 1059, …, 1302, …, 2114, …, 16382`. Values such as 605 or 500 are **not** valid and throw. |
| **Test coverage** | `test/operators.jl` is entirely self-consistency. The only physics tests are the single sphere under uniform `Ez` (dense `B \ rhs`) and a two-sphere test asserting only `relerr < 8e-2` against a leading-order dipole approximation at `R/r = 8`. |
| **No solve driver** | The package exposes operators only. There is no `solve` entry point; tests build and factor matrices inline. |
| **`eval_exterior_pot`** | Returns the **scattered field only** — the caller must add `u_inc` for the total. It **ignores the `q`-block entirely** (`evaluation.jl:21-27`); perturbing `x[N+1:2N]` with noise changes the output by exactly 0. |
| **No interior evaluator** | Correct and necessary — the `q` sources sit at `r_q > a`, i.e. *outside* the sphere, so including them in an exterior evaluation would place spurious singularities in the domain. But it means `eval_exterior_pot` **cannot detect a corrupted `q` block**. |
| **Signature traps** | `centers` is `nspheres×3`, `targets` is `3×ntrg` (transposed relative to each other); strictly `Float64`; `N` and `r_p` are passed separately from the `SphereMats` used to build `B` and must match, or sources land in the wrong places silently. |

### 2.2 `HybridMD`

**Build fixes required** (both upstream defects):

1. `Coulomb_accelerations_Hybrid.cpp:1227` contains a CJK ideographic comma `、`
   where a line-continuation `\` belongs. Hard compile error.
2. The 2010-era `FMM3dlib` Fortran fails gfortran 11's type checking; needs
   `-std=legacy -fallow-argument-mismatch`.

The committed `MD` binary is a **macOS Mach-O** and is unusable on Linux.

**Post-solve field representation.** The potential anywhere in the solvent is the
superposition of exactly three species, with no exclusions:

1. all `N` point charges (ions *and* colloids),
2. all `imcount` image charges,
3. one multipole expansion per sphere.

The first two can go through the FMM the solver leaves configured; **the multipoles
must be evaluated separately** (`ssheval_`, or `l3dmpeval_` after the `√(2n+1)`
rescale with `rscale = 1.0`). Validity requires `|t − o_s| > orad[s]` strictly, and
truncation at order `p` means accuracy degrades sharply near a surface.

**Defects and traps found:**

| # | Issue | Impact |
|---|---|---|
| 1 | `:692-700` indexes `srcDenarr` with hardcoded stride **4**, but the true stride is `(p+1)²`. Correct only when `p == 1` — leftover from `Coulomb_accelerations_Hybrid_static.cpp`. Measured: at `p = 6` it reads zeros where the true ℓ=1 coefficients are ≈3e-01. | `dipstr`, `dipvec`, `mpdipstr`, `mpdipvec` are **identically zero for any `p > 1`**; the induced-dipole↔multipole force term at `:1900-1948` is a silent no-op. **Forces only** — energies and the field representation are unaffected. |
| 2 | `:1925` passes `BknmCopy` to `l3dmpevalhessd_trunc_` **without** the `√(2n+1)` rescale that `matvec` correctly applies at `:2491-2494`. | A second, independent normalization error in the same force term. |
| 3 | At the shipped `precision_digits 6`, `Precisionsetting.h` sets `source_tol = 1.0`, making the image-generation guard `dr² < sourcetol²a² ∧ dr² > a²` **empty**. | **Zero image charges are ever created**; the "hybrid" method degenerates to pure order-6 MoM. Near-surface error ≈1e-2 at prec 6 versus ≈1e-4 at prec 4 — the default is *less* accurate where it matters most. |
| 4 | `imcount` is a function-local `int` (`:214`), not a global. | Must be recovered from the global `ionind` table. Verified exact. |
| 5 | `read_para` **aborts unless total charge is neutral**. | An electrolyte convention with no bearing on free-space Poisson. It blocks the single-charge case — the only one with an exact analytic answer. |
| 6 | `orad[i] = r[i] − c_lj/2`, and `c_lj` is **doubled** on read (`read_para.cpp:141`). | The *dielectric* radius is not the input radius. `Colloid_Radius 1.2` with `clj/2 0.2` gives a sphere of radius **0.8**. The easiest way to silently compare two different geometries. |
| 7 | Only two colloid species and two ion species, each with a single charge/radius/ε. | At most two distinct `(radius, ε)` pairs per run. |
| 8 | `Coulomb_accelerations_Hybrid_static.cpp` exists, is in no Makefile, and uses a **different** (ε_s-divided) normalization. | Do not mix the two when comparing. |

**Electrostatic domain.** `Rshell`, `QM`, `RM` and `epsi_M` appear **zero times** in
the solver. `Rshell` is purely mechanical (LJ wall, cell binning, RDF, dump bounds);
`QM`/`RM` feed only `Central_sph_force`, whose every call site is commented out. The
electrostatic problem is **unbounded free space of permittivity `ε_s` containing only
the dielectric spheres** — directly comparable to LaplaceMFS.

**Minimum driver.** A standalone one-shot driver needs only
`read_para(); allocate_arrays(); set_precision(prec);` then `x, y, z, q, r` set by
hand. `initial_conditions`, `mass_initialize` and `Cell_list` all drop out.
`set_precision` is **not** droppable — it calls `allocate_dynamic()` and sets
`p, im, imm, sourcetol, sphtol, gmrestol, fmmtol`, without which nothing is
allocated. Also required: `iter_indicator = 0` (it gates the only assignment of
`ε_s`/`ε_i`) and an explicit `force_compute = 0; energy_compute = 1`, since the
`MDpara.cpp` defaults are the opposite.

---

## 3. Measured results

### 3.1 Both codes hit the analytic answer

`HybridMD`, one sphere + one point charge, versus the exact Neumann series
(`a = 1`, `ε_i = 10`, `ε_s = 1`, `q = +1` at `z = 1.5`):

| target | charges | images | multipole | total | analytic | rel. err |
|---|---|---|---|---|---|---|
| (0,0,3) | 0.666667 | 0 | −0.048110 | 0.618556 | 0.618547 | 9.8e-06 |
| (0,0,8) | 0.153846 | 0 | −0.005703 | 0.148143 | 0.148143 | 2.7e-07 |
| (3,4,0) | 0.191565 | 0 | 0.000915 | 0.192480 | 0.192480 | 9.3e-08 |
| (0,0,1.2) | 3.333333 | 0 | −0.520265 | 2.813069 | 2.796366 | 1.7e-02 |

(The zero image column is defect #3 above — this is `precision 6`.)

### 3.2 LaplaceMFS versus the analytic series

Genuine Womersley designs, `a = 1`, `ε_r = 2.5`, `r_p = 0.5`, dense `B \ rhs`,
targets spanning `ρ/a ∈ [1.05, 3]` at four polar angles:

| `d/a` | M | N | rel. err | BC resid |
|---|---|---|---|---|
| 3.0 | 243 | 201 | 1.42e-06 | 1.07e-05 |
| 3.0 | 614 | 513 | 6.88e-10 | 1.54e-08 |
| 2.0 | 614 | 513 | 4.71e-10 | 1.64e-08 |
| 2.0 | 1302 | 1059 | **2.57e-13** | 2.15e-11 |
| 1.5 | 1302 | 1059 | 6.94e-07 | 2.51e-05 |
| 1.5 | 2666 | 2114 | 5.92e-08 | 9.34e-07 |
| 1.2 | 1302 | 1059 | 2.38e-04 | 1.68e-02 |
| 1.2 | 2666 | 2114 | 2.66e-04 | 3.63e-03 |

Mid-field reaches machine precision. **At `r_p = 0.5` the near field stalls**:
doubling the points at `d/a = 1.2` moved the error 2.38e-04 → 2.66e-04 while the
residual fell 4.6×. §3.3 shows this is a conditioning artifact of that particular
`r_p`, not a limitation of the method.

### 3.3 The near-field cost law — and the conditioning trap

Repeating the `d/a = 1.2` sequence at `r_p = 0.7`, where `cond(B)` stays moderate:

| M | N | `t ≈ √(2N)` | `r_p` | rel. err | BC resid |
|---|---|---|---|---|---|
| 1302 | 1059 | 46 | 0.7 | 1.09e-03 | 1.06e-02 |
| 2666 | 2114 | 65 | 0.7 | 9.63e-05 | 1.82e-03 |
| 6162 | 5001 | 100 | 0.7 | **4.81e-07** | 3.85e-05 |
| 6162 | 5001 | 100 | 0.8 | 5.35e-07 | 2.03e-05 |

Clean geometric convergence — roughly two decades per `√2` increase in `t`. So the
method is **not** limited at `d/a = 1.2`; the `r_p = 0.5` stall in §3.2 is roundoff,
with `cond(B) = 1.8e+12` at `N = 1059` destroying exactly the high-order content the
near field depends on. **`r_p` must be raised as `N` grows**, trading resolution
against conditioning.

With that resolved, the underlying resolution requirement still sets the cost.

The stall is not a bug. A proxy set of `N` points carries spherical-harmonic content
only up to degree

```
t ≈ √(2N) − 1
```

while resolving a charge at `d/a` to 1e-12 requires degree

```
n_max ≈ 12·ln10 / ln(d/a)
```

Every measurement is consistent with this:

| `d/a` | `n_max` needed | `t` available | outcome |
|---|---|---|---|
| 2.0 | ~40 | 46 (N=1059) | converged, 2.6e-13 |
| 1.5 | ~69 | 65 (N=2114) | just short, 5.9e-08 |
| 1.2 | ~152 | 65 (N=2114) | short, 9.6e-05 |
| 1.2 | ~152 | 100 (N=5001) | closer, 4.8e-07 |

Combining the two gives the **near-field cost law**:

```
N_proxy ≈ n_max² / 2 ,      n_max ~ 1 / ln(d/a)
```

so `d/a = 1.2` needs `N ≈ 1.2e4` proxy points **per sphere**, and `d/a = 1.1` needs
`N ≈ 4.2e4`. Since assembly is `O(MN)` and a dense solve `O(MN²)`, this is the wall
that makes pure MFS impractical for close-packed configurations — and it is precisely
what the image-charge machinery in `src/utils/double_spheres.jl` was built to avoid.

### 3.4 The residual is not a proxy for the error

Proxy-radius sweep at `d/a = 1.2`:

| M | N | `r_p` | rel. err | BC resid | cond(B) |
|---|---|---|---|---|---|
| 614 | 513 | 0.50 | 6.38e-03 | 6.09e-02 | 8.8e+08 |
| 614 | 513 | 0.70 | 4.66e-03 | 2.68e-02 | 3.1e+05 |
| 614 | 513 | 0.80 | 6.18e-03 | 4.70e-03 | 1.7e+04 |
| 614 | 513 | **0.90** | **9.30e-02** | **2.14e-03** | 1.4e+03 |
| 614 | 513 | 0.95 | 5.51e-01 | 8.31e-03 | 6.4e+02 |
| 1302 | 1059 | 0.50 | 2.38e-04 | 1.68e-02 | 1.8e+12 |
| 1302 | 1059 | 0.80 | 3.35e-04 | 3.16e-03 | 2.7e+05 |
| 1302 | 1059 | 0.90 | 8.63e-03 | 6.51e-03 | 4.4e+03 |

At `M = 614`, `r_p = 0.90` gives the **best** boundary residual and nearly the
**worst** field error — a 40× inversion. Two consequences:

- **Any convergence criterion built on `‖Bx − rhs‖` is unsound** in the
  under-resolved regime. An independent oracle is mandatory.
- The optimal `r_p` depends on the **spectral content of the excitation**, not on
  geometry alone. `r_p ≈ 0.5` beating `r_p ≈ 0.7` holds for uniform-field
  excitation, where all content sits at `n = 1`; under a nearby point charge the
  optimum shifts and the curve is non-monotonic.

### 3.5 First multi-sphere physics test

Two spheres (`a = 1`, `ε_r = 2.5`, `r_p = 0.5`, `M = 614`, `N = 513`), one exterior
point charge at `d/a = 2` below sphere 1, with sphere 2 receding to `+R`. Targets sit
near sphere 1. Three independent solve paths are compared, and the assembled operator
is checked against the **exact single-sphere Legendre series**, which the two-sphere
answer must approach as `R → ∞`.

| `R/a` | dense vs. exact 1-sphere | `Ghat`+GMRES vs. dense | `Ghat_fmm` vs. dense | GMRES iters |
|---|---|---|---|---|
| 8 | 3.66e-03 | 8.97e-14 | 8.32e-14 | 4 |
| 16 | 2.37e-04 | 7.24e-14 | 8.05e-14 | 3 |
| 32 | 1.57e-05 | 7.30e-14 | 7.62e-14 | 3 |
| 64 | 1.02e-06 | 7.97e-14 | 8.83e-14 | 3 |

**The decay rate is the result.** Each doubling of `R` reduces the discrepancy by
15.4× ≈ 2⁴, i.e. `O(R⁻⁴)` — exactly what theory requires. Sphere 2 sees a field
`~q/R²`, acquires an induced dipole `~α a³ q/R²`, whose potential back at sphere 1
scales as `q a³/R⁴`. The multi-sphere operator is therefore validated against physics,
not merely against itself.

Two further results:

- **`Ghat` + GMRES reproduces the direct dense least-squares to ~8e-14**, and so does
  `Ghat_fmm`. The second-kind reformulation, the SVD pseudo-inverse `μ → λ` mapping,
  and the Krylov solve all agree with the direct solve to machine precision. This is
  the first time any of them has solved a problem rather than been applied to a
  random vector.
- **GMRES converges in 3–4 iterations at every separation**, confirming the
  second-kind formulation is as well-conditioned as intended.

**Scope.** Two spheres, all at `R/a ≥ 8`. This says nothing about the near-touching
regime where §3.3's cost law bites, nor about many-sphere FMM at scale. Those remain
rungs 2–4 and still require `HybridSolve`.

### 3.6 Many spheres and scale

Three checks beyond the two-body case. Spheres are placed on a cubic lattice with
sphere 1 at the origin and a single exterior point charge at `d/a = 2` from it.

**Close-packed FMM agreement.** 8 spheres, lattice spacing `3a` (surface gap `1a`),
`M = 243`, `N = 201`. Dense `Ghat` and `Ghat_fmm` both converge in 8 iterations, and
their evaluated fields agree to

```
|u_fmm − u_dense| / |u_dense| = 2.54e-15
```

**The O(L⁻⁴) law survives many bodies.** 27 spheres on a widening lattice,
`M = 614`, `N = 513`, compared against the exact single-sphere Legendre series:

| spacing | rel. vs. 1-sphere | iters |
|---|---|---|
| 12 | 1.81e-03 | 4 |
| 24 | 1.21e-04 | 4 |
| 48 | 7.79e-06 | 3 |
| 96 | 4.95e-07 | 3 |

Ratios per doubling are 15.0, 15.5, 15.7 — the same `O(L⁻⁴)` decay found in §3.5,
now with 26 neighbours rather than one.

**Iteration count is flat with problem size.** Cubic lattice at spacing `3a`
(surface gap `1a`), `M = 243`, `N = 201`, `Ghat_fmm` + GMRES to `rtol = 1e-10`:

| spheres | unknowns | iters | solve (s) | µs/unknown/iter |
|---|---|---|---|---|
| 8 | 3,888 | 7 | 3.31 | 121.5 |
| 27 | 13,122 | 7 | 6.24 | 68.0 |
| 64 | 31,104 | 7 | 6.81 | 31.3 |
| 125 | 60,750 | 8 | 11.52 | 23.7 |
| 216 | 104,976 | 8 | 9.48 | 11.3 |
| 512 | 248,832 | 8 | 6.57 | 3.3 |

**7 → 8 iterations across a 64× growth in problem size**, at `1a` surface gaps. That
is the entire promise of the second-kind reformulation, and it holds.

> **Do not over-read the timings.** Per-unknown cost *falls* 37× across the sweep and
> total solve time is flat, which means every one of these problems is dominated by
> fixed overheads — FMM tree setup, per-sphere dense work, allocation inside the
> `LinearMap` — not by the far-field kernel. This establishes that the solver handles
> 512 spheres comfortably and is **not worse** than `O(N)` up to 250k unknowns; it
> does **not** measure the asymptotic scaling. The small-`ns` end also looks worth
> profiling: 59 ms per sphere per iteration at `ns = 8` is a lot for a 486×402 matvec.

**Scope.** At `M = 243`, `N = 201` the proxy set carries content only to degree ≈20
while a charge at `d/a = 2` needs ≈40, so the scaling runs are deliberately
under-resolved — they measure iteration count and time, not accuracy. The
near-touching regime (gaps well below `1a`) is still untested.

---

## 4. High-level plan

Two sub-projects, sequential. **A** builds the oracle; **B** uses it.

### Part A — `HybridSolve`: a standalone reference solver

A new repository at `~/project/laplacemfs/HybridSolve`, seeded from `HybridMD`
`1f87baf`, containing only the solver: `Coulomb_accelerations_Hybrid.cpp`,
`MDpara.cpp`, `allocate.cpp`, the four vendored libraries, and a new one-shot driver.
Everything MD — Verlet, Langevin, LJ, cell list, RDF, trajectories — is dropped.

Four deliberate departures from upstream's contract, each fixing something that would
otherwise corrupt a comparison:

1. **No charge-neutrality check.** Irrelevant to free-space Poisson, and it blocks
   the one case with an exact analytic answer.
2. **Radius means radius.** The input gives the dielectric radius directly; the
   driver sets `c_lj = 0` so `orad = r`.
3. **Per-sphere radius and ε**, lifting the two-species restriction.
4. **Potential at arbitrary target points**, emitted in both unit conventions.

Upstream source stays verbatim apart from a numbered, documented patch list, with a
`make verify-upstream` target that diffs against a pristine checkout. Only two
patches are needed: the `、` compile fix, and per-sphere ε. `imcount` is recovered
from `ionind` rather than patched.

**The force bugs are documented but not fixed** — they touch only the force path, and
repairing them would break traceability to upstream.

Part A is gated by its own regression test against the analytic Legendre series at
`precision` 3/4/5/6, which also pins down defect #3.

### Part B — MFS verification

| Task | What |
|---|---|
| **B0** | ~~Fix the `EffSphDes` artifact.~~ **Done.** |
| **B1** | Add `multispheres_pointcharge_rhs` to `src/sphere.jl`, with tests. |
| **B2** | Add `single_sphere_pointcharge_exterior` (the Legendre series) to `src/utils/single_sphere.jl` as a package utility. |
| **B3** | **An end-to-end `Ghat` + GMRES test against a physical answer.** Demonstrated in §3.5 for two well-separated spheres; still to be promoted into `test/` and extended to the near-touching and many-sphere regimes. |
| **B4** | A comparison harness: emit a `HybridSolve` config, run it, read targets back, run LaplaceMFS on the same geometry, report error against `N_proxy` and against gap. |

**B3 is the deliverable.** B1, B2 and B4 are scaffolding for it. The
`Ghat` + Krylov + FMM path is the product, and it has never been checked against
anything but itself.

Per §3.4, B4 must use the analytic reference (rung 1) or `HybridSolve` (rungs 2–4) as
the arbiter, reporting `‖Bx − rhs‖` only as diagnostic colour.

### The verification ladder

| Rung | Case | Oracle |
|---|---|---|
| 1 | 1 sphere, 1 charge, `d/a ∈ {3, 2, 1.5, 1.2}` | **Analytic** — pins units and conventions in *both* codes independently |
| 2 | 2 spheres, 1 charge, `R/a ∈ {4, 3, 2.5, 2.2, 2.05}` | `HybridSolve` |
| 3 | 8 and 27 spheres + several charges | `HybridSolve` — exercises the FMM path in both |
| 4 | near-touching pair, gap/a ∈ {0.5, 0.2, 0.1}, sweeping `precision` 3–6 | `HybridSolve`, cross-checking prec 4 against prec 6 |

Rung 1 is what makes the rest trustworthy: both codes must hit an exact answer before
they are ever compared to each other. Agreement between `HybridSolve`'s prec-4 and
prec-6 runs then serves as an ongoing oracle-health check on rungs 2–4.

### Beyond verification

The near-field cost law in §3.3 is the quantitative case for the work already
scaffolded in `src/utils/double_spheres.jl`: coupling the point+line image
generators into the MFS solve so that the images carry the high-order content and the
proxy sources handle only the smooth remainder. That is a separate design, but this
verification work produces exactly the error-versus-gap data needed to justify and
then measure it.

---

## 5. Open items

1. ~~`N = 5001` confirmation run at `d/a = 1.2`.~~ **Done** — 4.81e-07, confirming
   §3.3. Follow-up worth doing: an `(N, r_p)` joint sweep to characterise the
   resolution-versus-conditioning trade-off, since the optimal `r_p` clearly drifts
   upward with `N` and nobody has mapped it.
2. **Energy comparison** is nearly free (`HybridSolve` prints it, and unlike forces
   that path is correct), but LaplaceMFS has no energy functional. Currently out of
   scope.
3. **Forces** would require both a force evaluator in LaplaceMFS *and* repairing
   `HybridMD` defects #1 and #2. Out of scope.

## 6. Reproduction

```bash
# LaplaceMFS
julia --project=/mnt/home/xgao1/project/laplacemfs/LaplaceMFS.jl -e 'using Pkg; Pkg.instantiate()'

# HybridMD (after the two build fixes)
sed -i '1227s/、/\\/' Coulomb_accelerations_Hybrid.cpp
sed -i 's|^FFLAGS  =|FFLAGS  = -std=legacy -fallow-argument-mismatch -w|' Makefile.gnu
mkdir -p objects && make -f Makefile.gnu
```

`config.txt` is `index x y z` per line, **ions first, then colloids**; charges and
radii come from `para.txt` by species, not from the config file.
