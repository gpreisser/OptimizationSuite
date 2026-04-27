using OptimizationSuite

ENV["OPTIMIZATIONSUITE_GSET_ROOT"] = joinpath(@__DIR__, "..", "backends", "QiILS", "graphs", "gset")

const GSET = 12
const LAMBDA = 0.5
const ATTEMPTS = 20
const ITERATIONS = 10
const PERCENTAGE = 0.2
const SEED = 2
const TAO = 0.1
const ANGLE_CONV = 0.1
const INIT_MODE = :uniform
const MIX_STRATEGY = :best
const LBFGS_G_TOL = 0.1
const INNER_ITERATIONS_SWEEP = [25, 50, 100, 200]

function _tag_value(value)
    if value isa Symbol
        return String(value)
    elseif value isa AbstractFloat
        return replace(string(value), "." => "p")
    else
        return string(value)
    end
end

function _output_path(; solver, inner_iterations)
    filename = join([
        "budget",
        "solver$(_tag_value(solver))",
        "lam$(_tag_value(LAMBDA))",
        "att$(_tag_value(ATTEMPTS))",
        "it$(_tag_value(ITERATIONS))",
        "inner$(_tag_value(inner_iterations))",
        "pct$(_tag_value(PERCENTAGE))",
        "seed$(_tag_value(SEED))",
        "init$(_tag_value(INIT_MODE))",
        "mix$(_tag_value(MIX_STRATEGY))",
    ], "_")

    if solver == :lbfgs
        filename *= "_gtol$(_tag_value(LBFGS_G_TOL))"
    end

    return joinpath(
        dirname(@__DIR__),
        "results",
        "qiigs",
        "gset",
        "G$(GSET)",
        filename * ".json",
    )
end

function _run_solver(; solver, inner_iterations)
    output_path = _output_path(solver=solver, inner_iterations=inner_iterations)
    kwargs = Dict{Symbol, Any}(
        :backend => :qiigs,
        :instance_type => :gset,
        :gset => GSET,
        :solver => solver,
        :lambda => LAMBDA,
        :attempts => ATTEMPTS,
        :iterations => ITERATIONS,
        :inner_iterations => inner_iterations,
        :percentage => PERCENTAGE,
        :seed => SEED,
        :tao => TAO,
        :angle_conv => ANGLE_CONV,
        :init_mode => INIT_MODE,
        :mix_strategy => MIX_STRATEGY,
        :output_path => output_path,
    )
    if solver == :lbfgs
        kwargs[:g_tol] = LBFGS_G_TOL
    end

    result = solve_instance(;
        kwargs...,
    )

    return (
        inner_iterations=inner_iterations,
        solver=String(solver),
        best_cut=result["best_cut"],
        ratio=result["approximation_ratio"],
        runtime=get(result["result"]["metadata"], :runtime, nothing),
        output_path=result["output_path"],
    )
end

rows = NamedTuple[]

println("====================================================")
println("QiIGS Budget Sweep")
println("====================================================")

for inner_iterations in INNER_ITERATIONS_SWEEP
    push!(rows, _run_solver(solver=:grad, inner_iterations=inner_iterations))
    push!(rows, _run_solver(solver=:lbfgs, inner_iterations=inner_iterations))
end

println()
println("inner_iterations | solver | best_cut | ratio | total_runtime")
for row in rows
    println(
        "$(row.inner_iterations) | $(row.solver) | $(row.best_cut) | $(row.ratio) | $(row.runtime)",
    )
end

println()
println("Saved files:")
for row in rows
    println(row.output_path)
end
