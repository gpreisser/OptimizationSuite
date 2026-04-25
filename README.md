# OptimizationSuite

`OptimizationSuite` is a thin Julia wrapper for running reproducible MaxCut experiments across three related variational backends:

- `:qiils`
- `:qiigs`
- `:qiils_itensor`

The package standardizes instance loading, backend invocation, result saving, and example runner scripts, while leaving the solver logic in the backend packages unchanged.

This repo includes only Gset G12 for a minimal runnable example.

## Relation To The Paper

This package is based on and inspired by:

“Variational (matrix) product states for combinatorial optimization”  
arXiv:2512.20613

The paper studies product-state (PS) and matrix-product-state (MPS) variational ansatzes for MaxCut, using energy minimization with a quantum annealing Hamiltonian together with iterated-local-search-style randomness and benchmarking on standard graph sets.

`OptimizationSuite` is intended as a practical experiment layer around that workflow.

## Backend Overview

- `:qiils`
  Product-state / ILS backend. This is the PS-style iterative local-search backend built around repeated local optimization plus mixing.

- `:qiigs`
  Product-state / continuous angle optimizer. This is a PS backend formulated directly in continuous angle space, with gradient-based or L-BFGS-based inner optimization.

- `:qiils_itensor`
  MPS / ITensor backend. This is the matrix-product-state backend implemented on top of ITensors / ITensorMPS.

Conceptually:

- `:qiils` and `:qiigs` correspond to PS-style methods
- `:qiils_itensor` corresponds to the MPS-style method

## Installation And Setup

The repository vendors the three backend packages under `backends/`, so a collaborator can clone a single repository and run it directly.

Typical setup:

```bash
git clone https://github.com/gpreisser/OptimizationSuite.git
cd OptimizationSuite
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Current data assumptions:

- Gset graphs are expected in the local backend data layout
- some backend packages still assume local graph / solution roots

So for now this package is best used in the same workspace layout in which the backend packages were developed.

## Data Configuration

The package and backends support environment variables for overriding default data paths.
These variables take precedence over built-in fallback paths.

- `OPTIMIZATIONSUITE_GSET_ROOT`
  Controls where `OptimizationSuite` looks for Gset graph files.

- `QIILS_GRAPHS_ROOT`
  Controls the default graph root used by `QiILS`.

- `QIILS_ITENSOR_GRAPHS_ROOT`
  Controls the default graph root used by `QiILS_ITensor`.

- `QIIGS_SOLUTIONS_ROOT`
  Controls where `QiIGS` looks for stored exact / reference solution JSON files.

Example:

```bash
export OPTIMIZATIONSUITE_GSET_ROOT=/path/to/gset
```

## Quickstart

The main entry point is:

```julia
solve_instance(; backend, instance_type, kwargs...)
```

Currently the main supported instance type is `:gset`.

For user-provided graphs, the package also provides:

```julia
solve_maxcut(wg; backend=:qiils, kwargs...)
```

Example:

```julia
using OptimizationSuite
using Graphs
using SimpleWeightedGraphs

wg = SimpleWeightedGraph(4)
add_edge!(wg, 1, 2, 1.0)
add_edge!(wg, 2, 3, 1.0)
add_edge!(wg, 3, 4, 1.0)
add_edge!(wg, 4, 1, 1.0)

result = solve_maxcut(
    wg;
    backend = :qiils,
    lambda_sweep = 0.29,
    attempts = 100,
    sweeps_per_attempt = 100,
    percentage = 0.2,
    seed = 1,
    angle_conv = 0.1,
)

println(result["best_cut"])
println(result["best_configuration"])
```

### QiILS Example

```julia
using Pkg
Pkg.activate("/path/to/OptimizationSuite")

using OptimizationSuite

result = solve_instance(
    backend = :qiils,
    instance_type = :gset,
    gset = 12,
    lambda_sweep = 0.29,
    attempts = 100,
    sweeps_per_attempt = 80,
    percentage = 0.2,
    seed = 2,
    angle_conv = 0.1,
)
```

### QiIGS Example

```julia
using Pkg
Pkg.activate("/path/to/OptimizationSuite")

using OptimizationSuite

result = solve_instance(
    backend = :qiigs,
    instance_type = :gset,
    gset = 12,
    solver = :grad,  # or :lbfgs
    lambda = 0.5,
    attempts = 20,
    percentage = 0.2,
    iterations = 1000,
    inner_iterations = 100,
    tao = 0.1,
    angle_conv = 0.1,
    seed = 2,
    init_mode = :updown,
    mix_strategy = :best,
    save_params = true,
)
```

### QiILS_ITensor Example

```julia
using Pkg
Pkg.activate("/path/to/OptimizationSuite")

using OptimizationSuite

result = solve_instance(
    backend = :qiils_itensor,
    instance_type = :gset,
    gset = 12,
    lambda_sweep = 0.5,
    attempts = 20,
    sweeps_per_attempt = 10,
    maxdim = 20,
    percentage = 0.2,
    sample_mode = :entangled,
    seed = 2,
)
```

## Runner Scripts

The repository also includes ready-to-run examples:

- `scripts/run_qiils_gset.jl`
- `scripts/run_qiigs_gset.jl`
- `scripts/run_qiils_itensor_gset.jl`

These scripts:

- define solver parameters at the top
- print a standardized run header
- save JSON outputs with parameter-tagged filenames
- print the final output path

Run them with:

```bash
julia scripts/run_qiigs_gset.jl
```

or the corresponding `qiils` / `qiils_itensor` script.

## Result JSON Format

Results are saved to parameter-tagged JSON files under:

```text
results/<backend>/gset/G<id>/
```

Typical top-level fields include:

- `"backend"`
- `"instance_type"`
- `"instance"`
- `"optimal_cut"`
- `"best_cut"`
- `"approximation_ratio"`
- `"result"`

The runner scripts also save normalized plotting-friendly fields such as:

- `"best_history"`
- `"cut_history"`
- `"best_configuration"`
- `"best_theta"`
- `"energy_history"`
- `"grad_norm_history"`
- `"metadata"`

JSON serialization is made safe for research outputs:

- `Symbol` values are stringified
- `NaN`, `Inf`, and `-Inf` are converted to `null`

## Parameter Glossary

- `MaxCut`
  The optimization problem studied in this package.

- `PS`
  Product-state ansatz. Used for `:qiils` and `:qiigs`.

- `MPS`
  Matrix-product-state ansatz. Used for `:qiils_itensor`.

- `ILS`
  Iterated local search. Here this refers to the repeated-attempt workflow with mixing / perturbation between attempts.

- `annealing parameter λ`
  The interpolation parameter in the annealing Hamiltonian.
  Depending on backend, this appears as `lambda` or `lambda_sweep`.

- `attempts`
  Number of outer restarts / ILS attempts.

- `percentage`
  Fraction of variables perturbed during mixing.

- `sweeps_per_attempt`
  Number of local update sweeps per attempt for sweep-based backends.

- `iterations`, `inner_iterations`
  Inner optimization controls used by `:qiigs`.

- `maxdim`
  MPS bond-dimension control in the ITensor backend.

- `sample_mode`
  Sampling mode for the MPS backend.

- `mix_strategy`
  Strategy for generating the next attempt from the current or best solution.

## Reproducibility Notes

- Always record the `seed` used for a run.
- Output filenames are parameter-tagged to avoid accidental overwrite.
- Backend-specific kwargs are passed through from `solve_instance`.
- Different backends report different internal diagnostics; not all metadata fields are present for all solvers.
- `QiIGS` distinguishes continuous optimization diagnostics from discrete rounded MaxCut quantities, so energy-related fields should be interpreted carefully.

## Current Limitations / Research Status

This is currently a research package, not a polished public library.

Current limitations:

- local-path assumptions still exist in the wrapper and backend packages
- Gset and solution data locations are not yet fully configurable
- the package is currently most reliable in the original sibling-repo workspace layout
- test coverage is still minimal
- the API is intended for experiments and may still evolve

In short: this package is suitable for collaborators working on the same research codebase, but it should still be treated as an actively developing experiment layer.

## Citation

If you use this codebase in research, please cite the paper:

```text
Variational (matrix) product states for combinatorial optimization
arXiv:2512.20613
```

If relevant for your workflow, also cite the associated backend implementations:

- `QiILS`
- `QiIGS`
- `QiILS_ITensor`
