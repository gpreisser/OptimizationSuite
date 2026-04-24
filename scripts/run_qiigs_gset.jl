using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JSON
using OptimizationSuite

function _json_ready(value)
    if value isa Dict
        return Dict(string(k) => _json_ready(v) for (k, v) in value)
    elseif value isa AbstractVector
        return [_json_ready(v) for v in value]
    elseif value isa Tuple
        return [_json_ready(v) for v in value]
    elseif value isa Symbol
        return String(value)
    elseif value isa AbstractFloat
        return isfinite(value) ? value : nothing
    else
        return value
    end
end

function _tag_value(value)
    if value isa Symbol
        return String(value)
    elseif value isa AbstractFloat
        return replace(string(value), "." => "p")
    else
        return string(value)
    end
end

function _build_output_path(; backend, solver, gset, lambda, attempts, percentage, seed, mix_strategy, g_tol=nothing)
    parts = [
        "solver$(_tag_value(solver))",
        "lam$(_tag_value(lambda))",
        "att$(_tag_value(attempts))",
        "pct$(_tag_value(percentage))",
        "seed$(_tag_value(seed))",
        "mix$(_tag_value(mix_strategy))",
    ]
    if solver == :lbfgs && g_tol !== nothing
        push!(parts, "gtol$(_tag_value(g_tol))")
    end
    filename = join(parts, "_") * ".json"

    return joinpath(
        dirname(@__DIR__),
        "results",
        String(backend),
        "gset",
        "G$(gset)",
        filename,
    )
end

function _print_header(title::AbstractString, instance_label::AbstractString, params::Vector{String}, optimal_value)
    println("====================================================")
    println(title)
    println("====================================================")
    println("Instance: $(instance_label)")
    println("Parameters:")
    for param in params
        println("  - $(param)")
    end
    println("Optimal value: $(optimal_value === nothing ? "none" : optimal_value)")
    println("====================================================")
end

function _print_results(best_value, approx_ratio)
    println("====================================================")
    println("Results")
    println("====================================================")
    println("Best value: $(best_value)")
    println("Approximation ratio: $(approx_ratio === nothing ? "none" : approx_ratio)")
    println("====================================================")
end

backend = :qiigs
instance_type = :gset
gset = 12

# Default solver:
# :grad is more robust and works out-of-the-box.
# :lbfgs can be faster if well-tuned (g_tol, scaling), but may fail to converge.
solver = :grad
attempts = 20
percentage = 0.2
seed = 2
lambda = 0.5
# max_steps:
# maximum number of optimizer steps per attempt
iterations = 1000
inner_iterations = 100
tao = 0.1
angle_conv = 0.1
init_mode = :updown
mix_strategy = :best
save_params = true
g_tol = 1e-3

optimal_cut = load_known_optimal_cut(instance_type=:gset, gset=gset)
output_path = _build_output_path(
    backend=backend,
    solver=solver,
    gset=gset,
    lambda=lambda,
    attempts=attempts,
    percentage=percentage,
    seed=seed,
    mix_strategy=mix_strategy,
    g_tol=solver == :lbfgs ? g_tol : nothing,
)

params = [
    "solver=$(solver)",
    "lambda=$(lambda)",
    "attempts=$(attempts)",
    "max_steps=$(iterations)",
    "inner_iterations=$(inner_iterations)",
    "percentage=$(percentage)",
    "seed=$(seed)",
    "tao=$(tao)",
    "angle_conv=$(angle_conv)",
    "init_mode=$(init_mode)",
    "mix_strategy=$(mix_strategy)",
]
if solver == :lbfgs
    push!(params, "g_tol=$(g_tol)")
end

_print_header(
    "QiIGS Runner",
    "Gset G$(gset)",
    params,
    optimal_cut,
)

solve_kwargs = Dict{Symbol, Any}(
    :backend => backend,
    :instance_type => instance_type,
    :gset => gset,
    :solver => solver,
    :attempts => attempts,
    :percentage => percentage,
    :seed => seed,
    :lambda => lambda,
    :iterations => iterations,
    :inner_iterations => inner_iterations,
    :tao => tao,
    :angle_conv => angle_conv,
    :init_mode => init_mode,
    :mix_strategy => mix_strategy,
    :save_params => save_params,
    :output_path => output_path,
)
if solver == :lbfgs
    solve_kwargs[:g_tol] = g_tol
end

result = redirect_stdout(devnull) do
    solve_instance(; solve_kwargs...)
end

backend_result = result["result"]
best_value = result["best_cut"]
approx_ratio = result["approximation_ratio"]

_print_results(best_value, approx_ratio)

save_data = Dict(
    "solver" => backend,
    "instance_type" => instance_type,
    "gset" => gset,
    "optimal_cut" => optimal_cut,
    "best_value" => best_value,
    "approximation_ratio" => approx_ratio,
    "best_history" => get(backend_result, "best_history", nothing),
    "cut_history" => get(backend_result, "cut_history", nothing),
    "best_configuration" => get(backend_result, "best_configuration", nothing),
    "best_theta" => get(backend_result, "best_theta", nothing),
    "energy_history" => get(backend_result, "energy_history", nothing),
    "grad_norm_history" => get(backend_result, "grad_norm_history", nothing),
    "metadata" => Dict(
        "solver" => solver,
        "lambda" => lambda,
        "attempts" => attempts,
        "iterations" => iterations,
        "inner_iterations" => inner_iterations,
        "percentage" => percentage,
        "seed" => seed,
        "tao" => tao,
        "angle_conv" => angle_conv,
        "init_mode" => init_mode,
        "mix_strategy" => mix_strategy,
        "backend_result" => backend_result,
    ),
)
if solver == :lbfgs
    save_data["metadata"]["g_tol"] = g_tol
end

mkpath(dirname(output_path))
open(output_path, "w") do io
    JSON.print(io, _json_ready(save_data))
end

println("Saved results to: $(output_path)")
