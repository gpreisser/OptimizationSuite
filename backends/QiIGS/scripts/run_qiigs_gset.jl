using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using JSON
using QiIGS
using SparseArrays

function load_gset_weight_matrix(path::AbstractString)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    open(path, "r") do io
        first = split(strip(readline(io)))
        N = parse(Int, first[1])

        for line in eachline(io)
            parts = split(strip(line))
            isempty(parts) && continue

            if length(parts) == 2
                u = parse(Int, parts[1])
                v = parse(Int, parts[2])
                w = 1.0
            elseif length(parts) == 3
                u = parse(Int, parts[1])
                v = parse(Int, parts[2])
                w = parse(Float64, parts[3])
            else
                error("Invalid Gset line: $line")
            end

            push!(rows, u); push!(cols, v); push!(vals, w)
            push!(rows, v); push!(cols, u); push!(vals, w)
        end

        return sparse(rows, cols, vals, N, N), N
    end
end

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

const GSET_OPT = Dict(
    1 => 11624.0, 2 => 11620.0, 3 => 11622.0, 4 => 11646.0, 5 => 11631.0,
    6 => 2178.0, 7 => 2006.0, 8 => 2005.0, 9 => 2054.0, 10 => 2000.0,
    11 => 564.0, 12 => 556.0,
)

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
gset = 12
solver = :lbfgs

attempts = 20
percentage = 0.2
lambda = 0.5
iterations = 1000
inner_iterations = 100
tao = 0.1
angle_conv = 0.1
seed = 2
init_mode = :updown
mix_strategy = :best
save_params = true

gset_path = joinpath(@__DIR__, "..", "..", "QiILS", "graphs", "gset", "G$(gset)")

# ---------------------------------------------------------
# Load graph
# ---------------------------------------------------------
W, N = load_gset_weight_matrix(gset_path)
optimal_cut = get(GSET_OPT, gset, nothing)

println("====================================================")
println("                 QiIGS Gset Runner                  ")
println("====================================================")
println("Gset              = G$(gset)")
println("Graph path        = $gset_path")
println("N                 = $N")
println("solver            = $solver")
println("attempts          = $attempts")
println("percentage        = $percentage")
println("lambda            = $lambda")
println("iterations        = $iterations")
println("inner_iterations  = $inner_iterations")
println("tao               = $tao")
println("angle_conv        = $angle_conv")
println("seed              = $seed")
println("init_mode         = $init_mode")
println("mix_strategy      = $mix_strategy")
println("save_params       = $save_params")
if optimal_cut !== nothing
    println("optimal_cut       = $optimal_cut")
else
    println("optimal_cut       = none")
end
println("====================================================")

# ---------------------------------------------------------
# Run solver
# ---------------------------------------------------------
best_history, cut_history, best_configuration, best_theta, energy_history, grad_norm_history, metadata =
    QiIGS.qiigs_solve(
        W,
        N;
        solver=solver,
        attempts=attempts,
        percentage=percentage,
        lambda=lambda,
        iterations=iterations,
        inner_iterations=inner_iterations,
        tao=tao,
        angle_conv=angle_conv,
        seed=seed,
        init_mode=init_mode,
        mix_strategy=mix_strategy,
        save_params=save_params,
    )

best_cut = best_history[end]
ratio = optimal_cut === nothing ? nothing : best_cut / optimal_cut
total_runtime = get(metadata, :runtime, nothing)
attempt_metadata = get(metadata, :attempt_metadata, Dict{Symbol, Any}[])

lbfgs_iters = Float64[]
lbfgs_fcalls = Float64[]
lbfgs_gcalls = Float64[]
for md in attempt_metadata
    it = get(md, :optim_iterations, nothing)
    fc = get(md, :f_calls, nothing)
    gc = get(md, :g_calls, nothing)
    it !== nothing && push!(lbfgs_iters, float(it))
    fc !== nothing && push!(lbfgs_fcalls, float(fc))
    gc !== nothing && push!(lbfgs_gcalls, float(gc))
end

diagnostics = Dict{String, Any}(
    "total_runtime" => total_runtime,
)
if !isempty(lbfgs_iters)
    diagnostics["lbfgs_iters_total"] = sum(lbfgs_iters)
    diagnostics["lbfgs_iters_mean"] = sum(lbfgs_iters) / length(lbfgs_iters)
end
if !isempty(lbfgs_fcalls)
    diagnostics["lbfgs_f_calls_total"] = sum(lbfgs_fcalls)
    diagnostics["lbfgs_f_calls_mean"] = sum(lbfgs_fcalls) / length(lbfgs_fcalls)
end
if !isempty(lbfgs_gcalls)
    diagnostics["lbfgs_g_calls_total"] = sum(lbfgs_gcalls)
    diagnostics["lbfgs_g_calls_mean"] = sum(lbfgs_gcalls) / length(lbfgs_gcalls)
end

println("\n====================================================")
println("                 QiIGS Results                      ")
println("====================================================")
println("Best cut          = $best_cut")
if ratio !== nothing
    println("Approx ratio      = $ratio")
end
if total_runtime !== nothing
    println("Total runtime     = $total_runtime")
end
if solver == :lbfgs
    if haskey(diagnostics, "lbfgs_iters_total")
        println("LBFGS iters total = $(diagnostics["lbfgs_iters_total"])")
        println("LBFGS iters mean  = $(diagnostics["lbfgs_iters_mean"])")
    end
    if haskey(diagnostics, "lbfgs_f_calls_total")
        println("LBFGS f_calls total = $(diagnostics["lbfgs_f_calls_total"])")
        println("LBFGS f_calls mean  = $(diagnostics["lbfgs_f_calls_mean"])")
    end
    if haskey(diagnostics, "lbfgs_g_calls_total")
        println("LBFGS g_calls total = $(diagnostics["lbfgs_g_calls_total"])")
        println("LBFGS g_calls mean  = $(diagnostics["lbfgs_g_calls_mean"])")
    end
end
println("====================================================")

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------
save_path = joinpath(
    @__DIR__,
    "..",
    "results",
    "Gset$(gset)",
    "solver_$(solver)_lam$(replace(string(lambda), "." => "p"))_att$(attempts)_pct$(replace(string(percentage), "." => "p"))_seed$(seed).json",
)

mkpath(dirname(save_path))

save_data = Dict(
    "gset" => gset,
    "graph_path" => gset_path,
    "N" => N,
    "solver" => solver,
    "attempts" => attempts,
    "percentage" => percentage,
    "lambda" => lambda,
    "iterations" => iterations,
    "inner_iterations" => inner_iterations,
    "tao" => tao,
    "angle_conv" => angle_conv,
    "seed" => seed,
    "init_mode" => init_mode,
    "mix_strategy" => mix_strategy,
    "save_params" => save_params,
    "best_cut" => best_cut,
    "approx_ratio" => ratio,
    "best_history" => best_history,
    "cut_history" => cut_history,
    "best_configuration" => best_configuration,
    "best_theta" => best_theta,
    "energy_history" => energy_history,
    "grad_norm_history" => grad_norm_history,
    "metadata" => metadata,
    "diagnostics" => diagnostics,
    "optimal_cut" => optimal_cut,
)

open(save_path, "w") do io
    JSON.print(io, _json_ready(save_data))
end

println("Saved results to: $save_path")
