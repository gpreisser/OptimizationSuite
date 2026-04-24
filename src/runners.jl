function _tag_value(value)
    if value isa Symbol
        return String(value)
    elseif value isa AbstractFloat
        return replace(string(value), "." => "p")
    else
        return string(value)
    end
end

function _filename_tags(kwargs::Base.Pairs)
    ordered_keys = (
        :lambda,
        :lambda_sweep,
        :λ_sweep,
        :attempts,
        :sweeps_per_attempt,
        :iterations,
        :inner_iterations,
        :percentage,
        :maxdim,
        :seed,
        :mix_strategy,
        :sample_mode,
        :angle_conv,
    )

    short_names = Dict(
        :lambda => "lam",
        :lambda_sweep => "lamsw",
        :λ_sweep => "lamsw",
        :attempts => "att",
        :sweeps_per_attempt => "sw",
        :iterations => "it",
        :inner_iterations => "inner",
        :percentage => "pct",
        :maxdim => "maxd",
        :seed => "seed",
        :mix_strategy => "mix",
        :sample_mode => "sample",
        :angle_conv => "conv",
    )

    parts = String[]
    for key in ordered_keys
        if haskey(kwargs, key)
            push!(parts, "$(short_names[key])$(_tag_value(kwargs[key]))")
        end
    end

    return isempty(parts) ? "run" : join(parts, "_")
end

function _default_output_path(backend::Symbol, instance_type::Symbol, meta::Dict, kwargs::Base.Pairs)
    if instance_type === :gset
        return joinpath(
            dirname(@__DIR__),
            "results",
            String(backend),
            "gset",
            "G$(meta["gset"])",
            _filename_tags(kwargs) * ".json",
        )
    elseif instance_type === :random_regular
        return joinpath(
            dirname(@__DIR__),
            "results",
            String(backend),
            "random_regular",
            "N$(meta["N"])_k$(meta["k"])_seed$(meta["seed"])",
            _filename_tags(kwargs) * ".json",
        )
    else
        error("Unsupported instance_type=$instance_type")
    end
end

function _drop_keys(kwargs::Base.Pairs, keys_to_drop)
    filtered = Dict{Symbol, Any}()
    for (k, v) in kwargs
        if !(k in keys_to_drop)
            filtered[k] = v
        end
    end
    return filtered
end

function _run_qiils(wg; kwargs...)
    lambda_sweep = get(kwargs, :lambda_sweep, get(kwargs, :λ_sweep, nothing))
    lambda_sweep === nothing && error("QiILS backend requires lambda_sweep=... or λ_sweep=...")

    gvec = get(kwargs, :gvec, zeros(Float64, nv(wg)))
    attempts = get(kwargs, :attempts, 20)
    sweeps_per_attempt = get(kwargs, :sweeps_per_attempt, 80)
    percentage = get(kwargs, :percentage, 0.3)
    seed = get(kwargs, :seed, 1)
    theta0 = get(kwargs, :theta0, nothing)
    angle_conv = get(kwargs, :angle_conv, 1e-6)
    use_scaled_convergence = get(kwargs, :use_scaled_convergence, true)

    best_history, best_angles, sweeps_cumsum = QiILS.qiils_solve(
        wg,
        lambda_sweep,
        gvec,
        attempts,
        sweeps_per_attempt,
        percentage,
        seed,
        theta0,
        angle_conv,
        use_scaled_convergence,
    )

    return Dict(
        "best_cut" => best_history[end],
        "best_history" => best_history,
        "best_angles" => best_angles,
        "sweeps_cumsum" => sweeps_cumsum,
    )
end

function _run_qiils_itensor(wg; kwargs...)
    solver_kwargs = _drop_keys(kwargs, (:gset, :N, :k, :output_path))
    result = QiILS_ITensor.qiils_itensor_solver(wg; solver_kwargs...)
    best_history, cut_history, best_spins, energy_history = result

    return Dict(
        "best_cut" => best_history[end],
        "best_history" => best_history,
        "cut_history" => cut_history,
        "best_spins" => best_spins,
        "energy_history" => energy_history,
    )
end

function _total_edge_weight(wg)
    total = 0.0
    for e in edges(wg)
        total += Float64(wg.weights[src(e), dst(e)])
    end
    return total
end

function _run_qiigs(wg; kwargs...)
    W = graph_to_sparse_matrix(wg)
    N = nv(wg)
    solver_kwargs = _drop_keys(kwargs, (:gset, :N, :k, :output_path, :weighted))
    best_history, cut_history, best_configuration, best_theta,
    energy_history, grad_norm_history, metadata = QiIGS.qiigs_solve(W, N; solver_kwargs...)

    return Dict(
        "best_cut" => best_history[end],
        "best_history" => best_history,
        "cut_history" => cut_history,
        "best_configuration" => best_configuration,
        "best_theta" => best_theta,
        "energy_history" => energy_history,
        "grad_norm_history" => grad_norm_history,
        "metadata" => metadata,
    )
end

function solve_instance(; backend::Symbol, instance_type::Symbol, kwargs...)
    graph_kwargs = Dict{Symbol, Any}(kwargs)
    wg, meta = load_instance_graph(; instance_type=instance_type, graph_kwargs...)
    optimal_cut = load_known_optimal_cut(; instance_type=instance_type, graph_kwargs...)

    backend_result = if backend === :qiils
        _run_qiils(wg; kwargs...)
    elseif backend === :qiils_itensor
        _run_qiils_itensor(wg; kwargs...)
    elseif backend === :qiigs
        _run_qiigs(wg; kwargs...)
    else
        error("Unsupported backend=$backend. Use :qiils, :qiils_itensor, or :qiigs.")
    end

    best_cut = get(backend_result, "best_cut", nothing)
    approx_ratio = (best_cut !== nothing && optimal_cut !== nothing) ? (best_cut / optimal_cut) : nothing

    if best_cut !== nothing
        println("Best cut: $(best_cut)")
    end
    if approx_ratio !== nothing
        println("Approximation ratio: $(approx_ratio)")
    end

    result = Dict(
        "backend" => String(backend),
        "instance_type" => String(instance_type),
        "instance" => meta,
        "optimal_cut" => optimal_cut,
        "best_cut" => best_cut,
        "approximation_ratio" => approx_ratio,
        "result" => backend_result,
    )

    output_path = get(kwargs, :output_path, _default_output_path(backend, instance_type, meta, kwargs))
    save_result_json(output_path, result)
    result["output_path"] = output_path
    return result
end
