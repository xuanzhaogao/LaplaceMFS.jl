# Double-Sphere Forward Point+Line Images Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a forward image generator for two dielectric spheres that produces point-image + line-image quadrature sources without solving any linear system.

**Architecture:** Add a geometry-only utility in `src/utils/double_spheres.jl` that mirrors the `image_by_sphere` and reflection recursion in the HBDMM reference. Implement one-sphere generation first (Kelvin point image + Jacobi-quadrature line images), then build two-sphere reflection chaining on top. Keep output structured so callers can inspect per-sphere image strengths and positions directly.

**Tech Stack:** Julia (`LinearAlgebra`), existing LaplaceMFS module structure, `Test` stdlib.

### Task 1: Define API and add failing tests for one-sphere forward images

**Files:**
- Modify: `test/utils/double_spheres.jl`
- Read: `HBDMM/previous/LCM_functions/Bisphere_reflect.py`

**Step 1: Write the failing test**

```julia
@testset "single-sphere forward point+line images" begin
    center = [0.0, 0.0, 0.0]
    source = [0.0, 0.0, 1.5]
    charges, positions = LaplaceMFS.single_sphere_forward_point_line_images(
        center, 1.0, 0.4, 1.0, source; n_line=8, cutoff=0.0
    )

    @test length(charges) == 9
    @test size(positions) == (3, 9)
    @test charges[1] ≈ -0.4 / 1.5 atol=1e-12
    @test positions[:, 1] ≈ [0.0, 0.0, 2/3] atol=1e-12
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project -e 'using Pkg; Pkg.test(; test_args=["utils/double_spheres"])'`
Expected: FAIL with `UndefVarError` for new API.

**Step 3: Commit checkpoint**

```bash
git add test/utils/double_spheres.jl
git commit -m "test: add failing tests for one-sphere point+line image generation"
```

### Task 2: Implement one-sphere point+line image generation

**Files:**
- Modify: `src/utils/double_spheres.jl`
- Modify: `src/LaplaceMFS.jl`
- Test: `test/utils/double_spheres.jl`

**Step 1: Write minimal implementation**

```julia
# internal helper
_jacobi_roots_weights(n::Int, α::Real, β::Real)

# public API
single_sphere_forward_point_line_images(center, radius, gamma, q, source;
    n_line::Int=32, cutoff::Real=0.0)
```

Implementation details:
- Compute `r_s = norm(source - center)`.
- Kelvin image: `qK = -gamma * radius * q / r_s`, `rK = radius^2 / r_s`, position on center→source line.
- Line images: `lambda = (1-gamma)/2`, `alpha=0`, `beta=lambda-1`, nodes/weights from Jacobi quadrature mapped to `[0, rK]`.
- Quadrature charge formula matches reference:
  `q_i = w_i * (rK/2)^(alpha+beta+1) * gamma * lambda * q / radius * rK^(1-lambda)`.
- Apply `cutoff` to both point and line image strengths.
- Return `charges::Vector` and `positions::Matrix{T}` with size `(3, N)`.

**Step 2: Run test to verify it passes**

Run: `julia --project -e 'using Pkg; Pkg.test(; test_args=["utils/double_spheres"])'`
Expected: PASS for the new one-sphere test.

**Step 3: Commit checkpoint**

```bash
git add src/utils/double_spheres.jl src/LaplaceMFS.jl test/utils/double_spheres.jl
git commit -m "feat: add single-sphere forward point+line image generator"
```

### Task 3: Add two-sphere reflection chaining tests (forward-only, no solve)

**Files:**
- Modify: `test/utils/double_spheres.jl`

**Step 1: Write failing tests**

```julia
@testset "double-sphere forward point+line reflections" begin
    centers = [0.0 0.0 0.0; 0.0 0.0 3.0]
    radii = (1.0, 1.0)
    gammas = (0.5, 0.2)

    out1 = LaplaceMFS.double_sphere_forward_point_line_images(
        centers, radii, gammas, 1.0, [0.0, 0.0, 1.5];
        n_line=6, n_reflections=1, cutoff=0.0
    )
    out2 = LaplaceMFS.double_sphere_forward_point_line_images(
        centers, radii, gammas, 1.0, [0.0, 0.0, 1.5];
        n_line=6, n_reflections=3, cutoff=0.0
    )

    @test length(out2.q1) > length(out1.q1)
    @test length(out2.q2) > length(out1.q2)
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project -e 'using Pkg; Pkg.test(; test_args=["utils/double_spheres"])'`
Expected: FAIL with missing function for two-sphere API.

**Step 3: Commit checkpoint**

```bash
git add test/utils/double_spheres.jl
git commit -m "test: add failing tests for two-sphere forward reflection chaining"
```

### Task 4: Implement two-sphere forward reflection chaining

**Files:**
- Modify: `src/utils/double_spheres.jl`
- Modify: `src/LaplaceMFS.jl`
- Test: `test/utils/double_spheres.jl`

**Step 1: Write implementation**

```julia
double_sphere_forward_point_line_images(
    centers, radii, gammas, q0, source;
    n_line::Int=32, n_reflections::Int=1, cutoff::Real=0.0,
)
```

Behavior:
- Reflection level 1: generate images of `(q0, source)` in each sphere.
- For each additional reflection level, map previous sphere-1 images into sphere 2, and previous sphere-2 images into sphere 1.
- Accumulate all levels and return a named tuple:
  `(q1=..., x1=..., q2=..., x2=...)` where `x1/x2` are `3xN` matrices.
- Keep return outside the loop to avoid early termination bug in reference Python.

**Step 2: Run tests**

Run:
- `julia --project -e 'using Pkg; Pkg.test(; test_args=["utils/double_spheres"])'`
- `julia --project -e 'using Pkg; Pkg.test()'`

Expected: both pass.

**Step 3: Commit checkpoint**

```bash
git add src/utils/double_spheres.jl src/LaplaceMFS.jl test/utils/double_spheres.jl
git commit -m "feat: add two-sphere forward point+line reflection image generator"
```

### Task 5: Optional docs note for discoverability

**Files:**
- Modify: `docs/src/index.md`

**Step 1: Add a short utility section**

```markdown
- `double_sphere_forward_point_line_images`: generates forward image charges for two spheres (point + line quadrature), no linear solve.
```

**Step 2: Build docs (optional local verification)**

Run: `julia --project=docs docs/make.jl`
Expected: docs build succeeds.

**Step 3: Commit checkpoint**

```bash
git add docs/src/index.md
git commit -m "docs: mention forward two-sphere image utility"
```
