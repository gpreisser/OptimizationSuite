# scripts/debug_qiigs_angle_jump.jl
#
# Debug angle-solution jump around λ ∈ [0.20, 0.30]
# For one fixed graph seed, run many inits and save:
#   - converged angles θc
#   - hashed angle key
#   - rounded spin key
#   - rounded configuration
#   - energy / ratio
#
# Saves under ROOT/results/debug_angles/...

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using QiIGS
using JSON
using Printf
using Dates
using SparseArrays

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

@inline wrap_pi(x::Float64) = mod(x, pi)

@inline function quantize_angle(x::Float64, δ::Float64)
    return Int32(floor(x / δ + 0.5))
end

@inline function fnv1a64_combine(h::UInt64, x::UInt64)
    h ⊻= x
    return h * 0x100000001b3
end

function angle_key(θ::AbstractVector{<:AbstractFloat}; δ::Float64=1e-2, flip_equiv::Bool=true)
    function key_for(shift::Float64)
        h = UInt64(0xcbf29ce484222325)
        @inbounds for i in eachindex(θ)
            x = wrap_pi(Float64(θ[i]) + shift)
            q = quantize_angle(x, δ)
            h = fnv1a64_combine(h, reinterpret(UInt64, Int64(q)))
        end
        return h
    end
    h0 = key_for(0.0)
    if !flip_equiv
        return h0
    end
    h1 = key_for(pi / 2)
    return min(h0, h1)
end

function float_or_none(x)
    x === nothing && return "none"
    return Float64(x)
end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

function main()
    ROOT = normpath(joinpath(@__DIR__, ".."))
    RESULTS_DIR = joinpath(ROOT, "results")
    DEBUG_DIR = joinpath(RESULTS_DIR, "debug_angles")
    mkpath(DEBUG_DIR)

    ROOT_QIILS = "/Users/guillermo.preisser/Projects/QiILS_ITensor"
    GRAPHS_ROOT = joinpath(ROOT_QIILS, "graphs")
    SOLUTIONS_ROOT = joinpath(ROOT_QIILS, "solutions")

    # -----------------------
    # Debug parameters
    # -----------------------
    N = 50
    k = 3
    weighted = false

    graph_seed = 3
    λs = collect(0.25:0.0025:0.28)

    n_inits = 30

    iterations = 1
    inner_iterations = 5000
    tao = 0.1
    angle_conv = 0.01
    init_mode = :uniform
    save_params = true

    success_thr = 0.99

    angle_bin = 1e-2
    angle_flip_equiv = true
    seed_salt = 0

    # -----------------------
    # Load graph / optimum
    # -----------------------
    gpath = QiIGS.graph_path(N, k, graph_seed; weighted=weighted, graphs_root=GRAPHS_ROOT)
    @assert isfile(gpath) "Graph file not found: $gpath"
    W = QiIGS.load_weight_matrix(gpath)

    spath = QiIGS.akmax_solution_path(N, k, graph_seed; weighted=weighted, solutions_root=SOLUTIONS_ROOT)
    opt = QiIGS.load_optimal_cut(spath)

    all_results = Vector{Dict}(undef, length(λs))

    for (iλ, λ) in enumerate(λs)
        println("\nλ = ", λ)

        seen_angle_keys = Set{UInt64}()
        seen_spin_keys = Set{UInt64}()

        runs = Vector{Dict}(undef, n_inits)

        for r in 1:n_inits
            run_seed = graph_seed * 1_000_000 + r * 10_000 + Int(round(λ * 1000)) + seed_salt * 1_000_000_000

            sol = QiIGS.solve!(
                W, N;
                solver = :grad,
                seed = run_seed,
                lambda = λ,
                iterations = iterations,
                inner_iterations = inner_iterations,
                tao = tao,
                angle_conv = angle_conv,
                init_mode = init_mode,
                save_params = save_params,
                progressbar = false,
            )

            θc = get(sol.metadata, :theta_converged, Float64[])
            isempty(θc) && error("Missing :theta_converged in metadata")

            akey = angle_key(θc; δ=angle_bin, flip_equiv=angle_flip_equiv)
            skey = QiIGS.spin_config_key(sol.configuration)

            push!(seen_angle_keys, akey)
            push!(seen_spin_keys, skey)

            ratio = nothing
            if opt !== nothing
                _, ratio = QiIGS.cut_hat_and_ratio(W, sol.energy, opt)
            end

            runs[r] = Dict(
                "init_index" => r,
                "run_seed" => run_seed,
                "angle_key" => string(akey),
                "spin_key" => string(skey),
                "theta_converged" => Float64.(θc),
                "theta_first10" => Float64.(θc[1:min(10, end)]),
                "configuration" => Int.(sol.configuration),
                "energy" => Float64(sol.energy),
                "ratio" => float_or_none(ratio),
                "grad_norm_final" => Float64(sol.grad_norm),
                "gn_init" => Float64(get(sol.metadata, :gn_init, NaN)),
                "gn_max_inner" => Float64(get(sol.metadata, :gn_max_inner, NaN)),
                "gn_mean_inner" => Float64(get(sol.metadata, :gn_mean_inner, NaN)),
                "inner_sweeps_used" => Int(get(sol.metadata, :inner_sweeps_used, -1)),
            )
        end

        println("  unique angles = ", length(seen_angle_keys))
        println("  unique spins  = ", length(seen_spin_keys))

        all_results[iλ] = Dict(
            "lambda" => λ,
            "unique_angle_count" => length(seen_angle_keys),
            "unique_spin_count" => length(seen_spin_keys),
            "runs" => runs,
        )
    end

    abin_tag = replace(@sprintf("%.3g", angle_bin), "." => "p")
    out_path = joinpath(
        DEBUG_DIR,
        "debug_angle_jump_seed$(graph_seed)_lam0p20_to_0p30_d0p0125_abin$(abin_tag)_ninit$(n_inits).json"
    )

    save_data = Dict(
        "experiment" => "debug_qiigs_angle_jump",
        "N" => N,
        "k" => k,
        "weighted_flag" => weighted,
        "graph_seed" => graph_seed,
        "lambdas" => λs,
        "n_inits" => n_inits,
        "solver" => Dict(
            "iterations" => iterations,
            "inner_iterations" => inner_iterations,
            "tao" => tao,
            "angle_conv" => angle_conv,
            "init_mode" => String(init_mode),
            "save_params" => save_params,
        ),
        "angle_uniqueness" => Dict(
            "angle_bin" => angle_bin,
            "flip_equiv" => angle_flip_equiv,
        ),
        "results_per_lambda" => all_results,
        "timestamp_utc" => Dates.format(Dates.now(Dates.UTC), Dates.ISODateTimeFormat),
    )

    open(out_path, "w") do io
        JSON.print(io, save_data)
    end

    println("\nSaved debug file to:")
    println(out_path)
end

main()