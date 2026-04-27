using OptimizationSuite
using Statistics

ENV["OPTIMIZATIONSUITE_GSET_ROOT"] = joinpath(@__DIR__, "..", "backends", "QiILS", "graphs", "gset")

const GSET = 12
const LAMBDA = 0.5
const ATTEMPTS = 20
const ITERATIONS = 10
const PERCENTAGE = 0.2
const TAO = 0.1
const ANGLE_CONV = 0.1
const INIT_MODE = :uniform
const MIX_STRATEGY = :best
const LBFGS_G_TOL = 0.1
const INNER_ITERATIONS_SWEEP = [25, 50, 100]
const SEEDS = 1:5

function _tag_value(value)
    if value isa Symbol
        return String(value)
    elseif value isa AbstractFloat
        return replace(string(value), "." => "p")
    else
        return string(value)
    end
end

function _output_path(; solver, inner_iterations, seed)
    filename = join([
        "budgetseed",
        "solver$(_tag_value(solver))",
        "lam$(_tag_value(LAMBDA))",
        "att$(_tag_value(ATTEMPTS))",
        "it$(_tag_value(ITERATIONS))",
        "inner$(_tag_value(inner_iterations))",
        "pct$(_tag_value(PERCENTAGE))",
        "seed$(_tag_value(seed))",
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

function _run_solver(; solver, inner_iterations, seed)
    output_path = _output_path(solver=solver, inner_iterations=inner_iterations, seed=seed)
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
        :seed => seed,
        :tao => TAO,
        :angle_conv => ANGLE_CONV,
        :init_mode => INIT_MODE,
        :mix_strategy => MIX_STRATEGY,
        :output_path => output_path,
    )
    if solver == :lbfgs
        kwargs[:g_tol] = LBFGS_G_TOL
    end

    result = solve_instance(; kwargs...)
    return (
        best_cut=result["best_cut"],
        runtime=get(result["result"]["metadata"], :runtime, nothing),
    )
end

println("====================================================")
println("QiIGS Budget Sweep Across Seeds")
println("====================================================")

rows = NamedTuple[]

for inner_iterations in INNER_ITERATIONS_SWEEP
    for solver in (:grad, :lbfgs)
        cuts = Float64[]
        runtimes = Float64[]
        for seed in SEEDS
            result = _run_solver(solver=solver, inner_iterations=inner_iterations, seed=seed)
            push!(cuts, Float64(result.best_cut))
            if result.runtime !== nothing
                push!(runtimes, Float64(result.runtime))
            end
        end
        push!(rows, (
            inner_iterations=inner_iterations,
            solver=String(solver),
            mean_best_cut=mean(cuts),
            max_best_cut=maximum(cuts),
            mean_runtime=mean(runtimes),
        ))
    end
end

println()
println("inner_iterations | solver | mean best_cut | max best_cut | mean runtime")
for row in rows
    println(
        "$(row.inner_iterations) | $(row.solver) | $(row.mean_best_cut) | $(row.max_best_cut) | $(row.mean_runtime)",
    )
end
