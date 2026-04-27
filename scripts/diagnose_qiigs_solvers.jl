using JSON
using Statistics

const DEFAULT_GRAD_PATH = joinpath(
    @__DIR__,
    "..",
    "results",
    "qiigs",
    "gset",
    "G12",
    "solvergrad_lam0p5_att20_pct0p2_seed2_mixbest.json",
)

const DEFAULT_LBFGS_PATH = joinpath(
    @__DIR__,
    "..",
    "results",
    "qiigs",
    "gset",
    "G12",
    "solverlbfgs_lam0p5_att20_pct0p2_seed2_mixbest_gtol0p001.json",
)

function _fmt(x)
    x === nothing && return "n/a"
    x isa Bool && return string(x)
    x isa Integer && return string(x)
    x isa AbstractFloat && return string(round(x; digits=6))
    return string(x)
end

function _extract_attempt_metadata(data)
    meta = get(data, "metadata", Dict{String, Any}())
    backend_result = get(meta, "backend_result", Dict{String, Any}())
    backend_meta = get(backend_result, "metadata", Dict{String, Any}())
    return get(backend_meta, "attempt_metadata", nothing)
end

function _extract_total_runtime(data)
    diagnostics = get(data, "diagnostics", Dict{String, Any}())
    diagnostics_runtime = get(diagnostics, "total_runtime", nothing)
    diagnostics_runtime !== nothing && return diagnostics_runtime

    meta = get(data, "metadata", Dict{String, Any}())
    backend_result = get(meta, "backend_result", Dict{String, Any}())
    backend_meta = get(backend_result, "metadata", Dict{String, Any}())
    return get(backend_meta, "runtime", nothing)
end

function _numeric_values(attempt_metadata, key::String)
    attempt_metadata isa AbstractVector || return Float64[]
    values = Float64[]
    for md in attempt_metadata
        value = get(md, key, nothing)
        value isa Number || continue
        push!(values, Float64(value))
    end
    return values
end

function _bool_values(attempt_metadata, key::String)
    attempt_metadata isa AbstractVector || return Bool[]
    values = Bool[]
    for md in attempt_metadata
        value = get(md, key, nothing)
        value isa Bool || continue
        push!(values, value)
    end
    return values
end

function _summary(values)
    isempty(values) && return nothing
    return (
        minimum(values),
        mean(values),
        maximum(values),
    )
end

function _print_summary(label, values)
    stats = _summary(values)
    if stats === nothing
        println("  $(rpad(label, 30)) n/a")
    else
        lo, mid, hi = stats
        println(
            "  $(rpad(label, 30)) min=$( _fmt(lo) )  mean=$( _fmt(mid) )  max=$( _fmt(hi) )",
        )
    end
end

function _solver_name(data, fallback)
    meta = get(data, "metadata", Dict{String, Any}())
    solver = get(meta, "solver", nothing)
    solver === nothing && return fallback
    return string(solver)
end

function _final_best_history(data)
    best_history = get(data, "best_history", nothing)
    best_history isa AbstractVector || return nothing
    isempty(best_history) && return nothing
    return best_history[end]
end

function _diagnose(path, fallback_name)
    data = JSON.parsefile(path)
    attempt_metadata = _extract_attempt_metadata(data)

    name = _solver_name(data, fallback_name)
    best_value = get(data, "best_value", get(data, "best_cut", nothing))
    ratio = get(data, "approximation_ratio", nothing)
    total_runtime = _extract_total_runtime(data)
    final_best = _final_best_history(data)

    runtimes = _numeric_values(attempt_metadata, "runtime")
    optim_iterations = _numeric_values(attempt_metadata, "optim_iterations")
    f_calls = _numeric_values(attempt_metadata, "f_calls")
    g_calls = _numeric_values(attempt_metadata, "g_calls")
    continuous_grad_norm_final = _numeric_values(attempt_metadata, "continuous_grad_norm_final")
    gn_final = _numeric_values(attempt_metadata, "gn_final")
    converged = _bool_values(attempt_metadata, "optim_converged")

    println("====================================================")
    println("Solver: $(name)")
    println("====================================================")
    println("  $(rpad("file", 30)) $(path)")
    println("  $(rpad("best_value", 30)) $(_fmt(best_value))")
    println("  $(rpad("approximation_ratio", 30)) $(_fmt(ratio))")
    println("  $(rpad("total runtime", 30)) $(_fmt(total_runtime))")
    _print_summary("attempt runtime", runtimes)
    _print_summary("optim_iterations", optim_iterations)
    _print_summary("f_calls", f_calls)
    _print_summary("g_calls", g_calls)
    _print_summary("continuous_grad_norm_final", continuous_grad_norm_final)
    if isempty(converged)
        println("  $(rpad("optim_converged count", 30)) n/a")
    else
        println("  $(rpad("optim_converged count", 30)) $(count(identity, converged))/$(length(converged))")
    end
    _print_summary("gn_final", gn_final)
    println("  $(rpad("final best_history value", 30)) $(_fmt(final_best))")

    return Dict(
        "name" => name,
        "best_value" => best_value,
        "approximation_ratio" => ratio,
        "total_runtime" => total_runtime,
    )
end

function _interpretation(grad_diag, lbfgs_diag)
    grad_best = get(grad_diag, "best_value", nothing)
    lbfgs_best = get(lbfgs_diag, "best_value", nothing)
    grad_runtime = get(grad_diag, "total_runtime", nothing)
    lbfgs_runtime = get(lbfgs_diag, "total_runtime", nothing)

    println("====================================================")
    println("Interpretation")
    println("====================================================")

    if grad_best isa Number && lbfgs_best isa Number
        delta_cut = lbfgs_best - grad_best
        if delta_cut > 0
            println("L-BFGS improved the cut by $(_fmt(delta_cut)).")
        elseif delta_cut < 0
            println("L-BFGS produced a worse cut by $(_fmt(-delta_cut)).")
        else
            println("L-BFGS did not improve the cut.")
        end
    else
        println("Cut improvement could not be determined.")
    end

    if grad_runtime isa Number && lbfgs_runtime isa Number && grad_runtime != 0
        runtime_ratio = lbfgs_runtime / grad_runtime
        runtime_delta = lbfgs_runtime - grad_runtime
        println("L-BFGS used $(_fmt(runtime_delta)) more total runtime ($( _fmt(runtime_ratio) )x of grad).")

        if grad_best isa Number && lbfgs_best isa Number
            delta_cut = lbfgs_best - grad_best
            if delta_cut > 0 && runtime_ratio <= 2
                println("The improvement looks reasonably worth it for this run.")
            elseif delta_cut > 0
                println("The improvement is real, but it comes at a substantial runtime cost.")
            elseif delta_cut == 0
                println("The extra runtime does not look worth it here.")
            else
                println("The extra runtime does not look worth it here.")
            end
        else
            println("Worth-it judgment is inconclusive because the cut comparison is incomplete.")
        end
    else
        println("Runtime comparison could not be determined.")
    end
end

function main()
    grad_path, lbfgs_path = if length(ARGS) == 2
        ARGS[1], ARGS[2]
    else
        DEFAULT_GRAD_PATH, DEFAULT_LBFGS_PATH
    end

    grad_diag = _diagnose(grad_path, "grad")
    println()
    lbfgs_diag = _diagnose(lbfgs_path, "lbfgs")
    println()
    _interpretation(grad_diag, lbfgs_diag)
end

main()
