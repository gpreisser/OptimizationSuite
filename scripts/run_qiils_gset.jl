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

function _build_output_path(; backend, solver, gset, lambda_sweep, attempts, percentage, seed)
    filename = join((
        "solver$(_tag_value(solver))",
        "lamsw$(_tag_value(lambda_sweep))",
        "att$(_tag_value(attempts))",
        "pct$(_tag_value(percentage))",
        "seed$(_tag_value(seed))",
    ), "_") * ".json"

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

backend = :qiils
solver = :qiils
instance_type = :gset
gset = 12

lambda_sweep = 0.29
attempts = 4000
sweeps_per_attempt = 100
percentage = 0.2
seed = 2
angle_conv = 0.1

optimal_cut = load_known_optimal_cut(instance_type=:gset, gset=gset)
output_path = _build_output_path(
    backend=backend,
    solver=backend,
    gset=gset,
    lambda_sweep=lambda_sweep,
    attempts=attempts,
    percentage=percentage,
    seed=seed,
)

_print_header(
    "QiILS Runner",
    "Gset G$(gset)",
    [
        "solver=$(solver)",
        "lambda_sweep=$(lambda_sweep)",
        "attempts=$(attempts)",
        "sweeps_per_attempt=$(sweeps_per_attempt)",
        "percentage=$(percentage)",
        "seed=$(seed)",
        "angle_conv=$(angle_conv)",
    ],
    optimal_cut,
)

result = redirect_stdout(devnull) do
    solve_instance(
        backend=backend,
        instance_type=instance_type,
        gset=gset,
        lambda_sweep=lambda_sweep,
        attempts=attempts,
        sweeps_per_attempt=sweeps_per_attempt,
        percentage=percentage,
        seed=seed,
        angle_conv=angle_conv,
        output_path=output_path,
    )
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
    "best_theta" => get(backend_result, "best_angles", nothing),
    "energy_history" => get(backend_result, "energy_history", nothing),
    "grad_norm_history" => get(backend_result, "grad_norm_history", nothing),
    "metadata" => Dict(
        "lambda_sweep" => lambda_sweep,
        "attempts" => attempts,
        "sweeps_per_attempt" => sweeps_per_attempt,
        "percentage" => percentage,
        "seed" => seed,
        "angle_conv" => angle_conv,
        "backend_result" => backend_result,
    ),
)

mkpath(dirname(output_path))
open(output_path, "w") do io
    JSON.print(io, _json_ready(save_data))
end

println("Saved results to: $(output_path)")
