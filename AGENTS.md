# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the package code. Entry point is `src/LaplaceMFS.jl`, which includes focused modules such as `sphere.jl`, `operators.jl`, `evaluation.jl`, and `laplace3d.jl`.
- `test/` holds unit and regression tests, orchestrated by `test/runtests.jl`.
- `docs/` contains Documenter.jl sources (`docs/src/index.md`) and the docs build script (`docs/make.jl`).
- `example/` includes runnable examples (for example, `example/single_sphere_Ez_plane_error.jl`).
- `refs/` stores derivations and references used by implementations and tests.

## Build, Test, and Development Commands
- `julia --project -e 'using Pkg; Pkg.instantiate()'`: install package dependencies.
- `julia --project -e 'using Pkg; Pkg.test()'`: run the full test suite in `test/`.
- `julia --project=docs docs/make.jl`: build documentation locally.
- `julia --project=example example/single_sphere_Ez_plane_error.jl`: run a representative example script.
- CI runs on Julia `1.12` (`.github/workflows/CI.yml`); validate locally on a matching version when possible.

## Coding Style & Naming Conventions
- Follow existing Julia style: 4-space indentation, no tabs, and concise function-level doc/comments only where needed.
- Use `snake_case` for functions/variables (`multispheres_mu_to_lambda!`), `CamelCase` for concrete types (`SphereMats`), and descriptive testset names.
- Keep exported API declarations in `src/LaplaceMFS.jl`; place implementation details in the most relevant source file.

## Testing Guidelines
- Add tests in `test/*.jl` and include new files from `test/runtests.jl`.
- Prefer deterministic numeric checks with explicit tolerances (`atol`/`rtol`) and relative-error assertions.
- Cover both real and complex parameter paths where applicable (as done in `test/operators.jl`).

## Commit & Pull Request Guidelines
- Recent history favors short, imperative commit messages (for example, `fix ci`, `add refs`, `update png`). Use a concise summary and scope when useful.
- For pull requests, include: purpose, key numerical/algorithmic changes, test updates, and docs/example updates if behavior or API changed.
- Link related issues and attach plots/images when output visuals are affected.

## Reference Materials
- `refs/` contains derivations and notes that informed the implementation. These are not necessarily polished for public consumption but provide insight into the mathematical and algorithmic decisions made. They can be useful for future contributors to understand the rationale behind certain approaches or to verify the correctness of implementations.