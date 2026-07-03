# Dipole Forward Image Generators Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add forward single- and double-sphere dipole image generators that reuse the exact image positions from the existing point+line charge generators.

**Architecture:** Keep the current point-charge image code as the geometry source of truth. Build new dipole APIs on top by reusing the same scalar reflection coefficients and image positions, but carrying a 3-vector dipole moment at each image location. Preserve the existing recursive reflection structure for the two-sphere case.

**Tech Stack:** Julia, existing `LaplaceMFS` module structure, `Test` stdlib.

---

### Task 1: Add failing tests for same-position dipole image generation

**Files:**
- Modify: `test/utils/double_spheres.jl`

**Step 1: Write failing tests**

Add tests that:
- compare `single_sphere_forward_point_line_dipole_images` against `single_sphere_forward_point_line_images`
- verify image positions are identical
- verify dipole columns equal scalar image weights times the input dipole vector
- compare `double_sphere_forward_point_line_dipole_images` against the two-sphere charge generator in the same way

**Step 2: Run focused tests to verify failure**

Run: `julia --project -e 'using Pkg; Pkg.test(; test_args=["utils/double_spheres"])'`

Expected: fail because the new dipole image APIs are not defined yet.

### Task 2: Implement single- and double-sphere dipole image generators

**Files:**
- Modify: `src/utils/double_spheres.jl`
- Modify: `src/LaplaceMFS.jl`

**Step 1: Add a helper for stacking 3-vector dipole columns**

Return a `3 x N` matrix so `p[:, j]` matches `x[:, j]`.

**Step 2: Implement `single_sphere_forward_point_line_dipole_images`**

Behavior:
- input dipole is a length-3 vector
- generated image positions match the point-charge generator exactly
- each image dipole is the scalar point-charge image coefficient multiplied by the input dipole vector
- `cutoff` acts on dipole norm

**Step 3: Implement `double_sphere_forward_point_line_dipole_images`**

Behavior:
- mirror the existing reflection recursion
- generated image positions match the point-charge generator exactly at every level
- dipole vectors propagate by the same scalar reflection factors

**Step 4: Export the new public APIs**

Add exports in `src/LaplaceMFS.jl`.

### Task 3: Verify and document

**Files:**
- Modify: `refs/double_sphere_image_derivation.md`

**Step 1: Run focused tests**

Run: `julia --project -e 'using Pkg; Pkg.test(; test_args=["utils/double_spheres"])'`

Expected: pass.

**Step 2: Run full test suite**

Run: `julia --project -e 'using Pkg; Pkg.test()'`

Expected: pass.

**Step 3: Update derivation note**

Add a short note that the dipole image APIs reuse the exact point-charge image positions and scale dipole vectors by the same scalar image coefficients.
