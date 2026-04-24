# scripts/qiigs_lambda_sweep_unique_minima.jl
#
# Adds:
#   - unique_angle_count_raw:     distinct optimized angle minima BEFORE rounding
#                                 with no global-flip reduction
#   - unique_angle_count_reduced: distinct optimized angle minima BEFORE rounding
#                                 reduced by θ ~ θ + π/2
#
# NOTE:
#   - Requires save_params=true so solver stores theta_converged in metadata.
#   - Angles are NOT saved to JSON; only counts are recorded.
#
# IMPORTANT:
#   - angle_conv is now included in the FINAL output filename too
#   - angle_conv is also stored at top level in save_data

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using QiIGS
using JSON
using Printf
using Dates
using Statistics
using SparseArrays

# ---------------------------
# Helpers
# ---------------------------

@inline function mean_and_stderr(sumx, sumx2, n::Int)
    μ = sumx / n
    if n ≤ 1
        return μ, 0.0
    end
    var = (sumx2 - n * μ^2) / (n - 1)
    var = max(var, 0.0)
    σ = sqrt(var)
    return μ, σ / sqrt(n)
end

function atomic_json_write(path::AbstractString, data)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        JSON.print(io, data)
    end
    mv(tmp, path; force=true)
    return nothing
end

# ---------------------------
# Angle-key hashing (pre-rounding uniqueness)
# ---------------------------

function angle_key(θ::AbstractVector{<:AbstractFloat}; δ::Float64=1e-3, flip_equiv::Bool=true)
    function qvec_for(shift::Float64)
        q = Vector{Int32}(undef, length(θ))
        @inbounds for i in eachindex(θ)
            x = mod(Float64(θ[i]) + shift, pi)
            q[i] = Int32(floor(x / δ + 0.5))
        end
        return Tuple(q)
    end

    q0 = qvec_for(0.0)
    if !flip_equiv
        return q0
    end

    q1 = qvec_for(pi / 2)
    return min(q0, q1)   # lexicographic canonical representative
end

# ---------------------------
# Main
# ---------------------------

function main()
    println("===============================================================")
    println("  QiIGS λ-sweep: uniques + ratio(mean/best) + success + grad   ")
    println("  Inputs: graphs/solutions from QiILS_ITensor                 ")
    println("  Outputs: QiIGS/results/... (with rolling checkpoint)        ")
    println("===============================================================")

    # -----------------------
    # Experiment parameters
    # -----------------------
    N = 50
    k = 3
    weighted = false

    graph_seeds = collect(1:1)
    λs = collect(0.0:0.025:1.0)
    nλ = length(λs)

    n_inits = 10000

    iterations = 1
    inner_iterations = 5000
    tao = 0.1
    angle_conv = 1e-2
    init_mode = :uniform

    save_params = true

    success_thr = 0.999

    angle_bin = 2e-2
    angle_flip_equiv = true

    ckpt_namespace = "v4"

    init_sig = "ns=$(ckpt_namespace)_init=$(String(init_mode))_tao=$(tao)_conv=$(angle_conv)_save=$(save_params)_abin=$(angle_bin)_aflip=$(angle_flip_equiv)"
    seed_salt = 0

    keyT = typeof(angle_key(zeros(Float64, N); δ=angle_bin, flip_equiv=false))

    # -----------------------
    # Paths
    # -----------------------
    ROOT = normpath(joinpath(@__DIR__, ".."))
    RESULTS_DIR = joinpath(ROOT, "results")
    mkpath(RESULTS_DIR)

    ROOT_QIILS = "/Users/guillermo.preisser/Projects/QiILS_ITensor"
    GRAPHS_ROOT = joinpath(ROOT_QIILS, "graphs")
    SOLUTIONS_ROOT = joinpath(ROOT_QIILS, "solutions")

    wtag = weighted ? "weighted" : "unweighted"

    save_dir = joinpath(
        RESULTS_DIR,
        "qiigs_unique_minima_N$(N)_k$(k)_graphs$(length(graph_seeds))_$(wtag)"
    )
    mkpath(save_dir)

    λmin = first(λs)
    λmax = last(λs)
    dλ = (length(λs) > 1 ? (λs[2] - λs[1]) : 0.0)
    tag = "lam$(replace(@sprintf("%.3f", λmin), "."=>"p"))_to_$(replace(@sprintf("%.3f", λmax), "."=>"p"))_d$(replace(@sprintf("%.3f", dλ), "."=>"p"))"

    ckpt_path = joinpath(
        save_dir,
        "checkpoint_unique_ratio_meanbest_succ_grad_$(tag)_$(init_sig)_ninit$(n_inits)_outer$(iterations)_inner$(inner_iterations)_thr$(replace(@sprintf("%.3f", success_thr), "."=>"p")).json"
    )

    @info "QiIGS ROOT" ROOT
    @info "Saving under" save_dir
    @info "Reusing graphs from" GRAPHS_ROOT
    @info "Reusing solutions from" SOLUTIONS_ROOT
    @info "Checkpoint" ckpt_path

    # -----------------------
    # Accumulators
    # -----------------------
    sum_unique  = zeros(Float64, nλ)
    sum_unique2 = zeros(Float64, nλ)

    sum_unique_angle_raw  = zeros(Float64, nλ)
    sum_unique_angle_raw2 = zeros(Float64, nλ)

    sum_unique_angle_reduced  = zeros(Float64, nλ)
    sum_unique_angle_reduced2 = zeros(Float64, nλ)

    sum_best_ratio  = zeros(Float64, nλ)
    sum_best_ratio2 = zeros(Float64, nλ)

    sum_mean_ratio  = zeros(Float64, nλ)
    sum_mean_ratio2 = zeros(Float64, nλ)

    n_ratio = zeros(Int, nλ)

    sum_succ  = zeros(Float64, nλ)
    sum_succ2 = zeros(Float64, nλ)
    n_succ    = zeros(Int, nλ)

    sum_gn_final_mean  = zeros(Float64, nλ)
    sum_gn_final_mean2 = zeros(Float64, nλ)
    sum_gn_final_min   = zeros(Float64, nλ)
    sum_gn_final_min2  = zeros(Float64, nλ)

    sum_gn_init_mean  = zeros(Float64, nλ)
    sum_gn_init_mean2 = zeros(Float64, nλ)

    sum_gn_maxinner_mean  = zeros(Float64, nλ)
    sum_gn_maxinner_mean2 = zeros(Float64, nλ)

    sum_gn_meaninner_mean  = zeros(Float64, nλ)
    sum_gn_meaninner_mean2 = zeros(Float64, nλ)

    sum_inner_sweeps_mean  = zeros(Float64, nλ)
    sum_inner_sweeps_mean2 = zeros(Float64, nλ)

    sum_rt_mean  = zeros(Float64, nλ)
    sum_rt_mean2 = zeros(Float64, nλ)

    n_graphs = 0
    start_index = 1

    # -----------------------
    # Resume
    # -----------------------
    if isfile(ckpt_path)
        ck = JSON.parsefile(ckpt_path)

        if ck["N"] == N && ck["k"] == k && ck["weighted_flag"] == weighted &&
           length(ck["λs"]) == length(λs) &&
           ck["n_inits"] == n_inits &&
           ck["iterations"] == iterations &&
           ck["inner_iterations"] == inner_iterations &&
           ck["success_thr"] == success_thr &&
           get(ck, "init_sig", "") == init_sig

            sum_unique  .= Float64.(ck["sum_unique"])
            sum_unique2 .= Float64.(ck["sum_unique2"])

            if haskey(ck, "sum_unique_angle_raw")
                sum_unique_angle_raw  .= Float64.(ck["sum_unique_angle_raw"])
                sum_unique_angle_raw2 .= Float64.(ck["sum_unique_angle_raw2"])
            end

            if haskey(ck, "sum_unique_angle_reduced")
                sum_unique_angle_reduced  .= Float64.(ck["sum_unique_angle_reduced"])
                sum_unique_angle_reduced2 .= Float64.(ck["sum_unique_angle_reduced2"])
            end

            sum_best_ratio  .= Float64.(ck["sum_best_ratio"])
            sum_best_ratio2 .= Float64.(ck["sum_best_ratio2"])
            sum_mean_ratio  .= Float64.(ck["sum_mean_ratio"])
            sum_mean_ratio2 .= Float64.(ck["sum_mean_ratio2"])
            n_ratio .= Int.(ck["n_ratio"])

            sum_succ  .= Float64.(ck["sum_succ"])
            sum_succ2 .= Float64.(ck["sum_succ2"])
            n_succ    .= Int.(ck["n_succ"])

            sum_gn_final_mean  .= Float64.(ck["sum_gn_final_mean"])
            sum_gn_final_mean2 .= Float64.(ck["sum_gn_final_mean2"])
            sum_gn_final_min   .= Float64.(ck["sum_gn_final_min"])
            sum_gn_final_min2  .= Float64.(ck["sum_gn_final_min2"])

            sum_gn_init_mean  .= Float64.(ck["sum_gn_init_mean"])
            sum_gn_init_mean2 .= Float64.(ck["sum_gn_init_mean2"])
            sum_gn_maxinner_mean  .= Float64.(ck["sum_gn_maxinner_mean"])
            sum_gn_maxinner_mean2 .= Float64.(ck["sum_gn_maxinner_mean2"])
            sum_gn_meaninner_mean  .= Float64.(ck["sum_gn_meaninner_mean"])
            sum_gn_meaninner_mean2 .= Float64.(ck["sum_gn_meaninner_mean2"])

            if haskey(ck, "sum_inner_sweeps_mean")
                sum_inner_sweeps_mean  .= Float64.(ck["sum_inner_sweeps_mean"])
                sum_inner_sweeps_mean2 .= Float64.(ck["sum_inner_sweeps_mean2"])
            end

            sum_rt_mean  .= Float64.(ck["sum_rt_mean"])
            sum_rt_mean2 .= Float64.(ck["sum_rt_mean2"])

            n_graphs = Int(ck["n_graphs"])
            start_index = Int(ck["next_seed_index"])
            println("↻ Resuming: n_graphs=$n_graphs, next_seed_index=$start_index")
        else
            println("⚠ Found checkpoint but parameters differ; ignoring it.")
        end
    end

    # -----------------------
    # Outer loop
    # -----------------------
    for idx in start_index:length(graph_seeds)
        gs = graph_seeds[idx]
        println("\n► graph seed = $gs  (graph $idx / $(length(graph_seeds)))")

        gpath = QiIGS.graph_path(N, k, gs; weighted=weighted, graphs_root=GRAPHS_ROOT)
        @assert isfile(gpath) "Graph file not found: $gpath"
        W = QiIGS.load_weight_matrix(gpath)

        spath = QiIGS.akmax_solution_path(N, k, gs; weighted=weighted, solutions_root=SOLUTIONS_ROOT)
        opt = QiIGS.load_optimal_cut(spath)
        if opt === nothing
            println("    ⚠ No optimal cut found: $spath")
        else
            println("    ✔ optimal_cut = $opt")
        end

        unique_count = zeros(Float64, nλ)
        unique_angle_count_raw = zeros(Float64, nλ)
        unique_angle_count_reduced = zeros(Float64, nλ)

        best_ratio = Vector{Union{Nothing,Float64}}(undef, nλ); fill!(best_ratio, nothing)
        mean_ratio = Vector{Union{Nothing,Float64}}(undef, nλ); fill!(mean_ratio, nothing)
        success_rate = zeros(Float64, nλ)

        gn_final_mean = zeros(Float64, nλ)
        gn_final_min  = zeros(Float64, nλ)

        gn_init_mean      = zeros(Float64, nλ)
        gn_maxinner_mean  = zeros(Float64, nλ)
        gn_meaninner_mean = zeros(Float64, nλ)

        inner_sweeps_used_mean = zeros(Float64, nλ)
        rt_mean = zeros(Float64, nλ)

        for (iλ, λ) in enumerate(λs)
            seen_spins = Set{UInt64}()
            seen_angles_raw = Set{keyT}()
            seen_angles_reduced = Set{keyT}()

            sum_gn_final = 0.0
            min_gn_final = Inf

            sum_gn_init = 0.0
            sum_gn_maxinner = 0.0
            sum_gn_meaninner = 0.0

            sum_inner_sweeps_used = 0.0
            sum_rt = 0.0

            best_r = -Inf
            sum_r = 0.0
            succ = 0

            for r in 1:n_inits
                run_seed = gs * 1_000_000 + r * 10_000 + Int(round(λ * 1000)) + seed_salt * 1_000_000_000

                t0 = time()
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
                dt = time() - t0

                push!(seen_spins, QiIGS.spin_config_key(sol.configuration))

                θc = get(sol.metadata, :theta_converged, Float64[])
                isempty(θc) && error("Missing :theta_converged in sol.metadata")

                push!(seen_angles_raw, angle_key(θc; δ=angle_bin, flip_equiv=false))
                push!(seen_angles_reduced, angle_key(θc; δ=angle_bin, flip_equiv=angle_flip_equiv))

                sum_gn_final += sol.grad_norm
                min_gn_final = min(min_gn_final, sol.grad_norm)

                gni   = get(sol.metadata, :gn_init, NaN)
                gnm   = get(sol.metadata, :gn_max_inner, NaN)
                gmean = get(sol.metadata, :gn_mean_inner, NaN)

                sum_gn_init += gni
                sum_gn_maxinner += gnm
                sum_gn_meaninner += gmean

                sused = get(sol.metadata, :inner_sweeps_used, -1)
                sum_inner_sweeps_used += sused

                sum_rt += dt

                if opt !== nothing
                    _, ratio = QiIGS.cut_hat_and_ratio(W, sol.energy, opt)
                    best_r = max(best_r, ratio)
                    sum_r += ratio
                    if ratio >= success_thr
                        succ += 1
                    end
                end
            end

            unique_count[iλ] = length(seen_spins)
            unique_angle_count_raw[iλ] = length(seen_angles_raw)
            unique_angle_count_reduced[iλ] = length(seen_angles_reduced)

            gn_final_mean[iλ] = sum_gn_final / n_inits
            gn_final_min[iλ] = min_gn_final

            gn_init_mean[iλ] = sum_gn_init / n_inits
            gn_maxinner_mean[iλ] = sum_gn_maxinner / n_inits
            gn_meaninner_mean[iλ] = sum_gn_meaninner / n_inits

            inner_sweeps_used_mean[iλ] = sum_inner_sweeps_used / n_inits
            rt_mean[iλ] = sum_rt / n_inits

            if opt === nothing
                best_ratio[iλ] = nothing
                mean_ratio[iλ] = nothing
                success_rate[iλ] = 0.0
            else
                best_ratio[iλ] = best_r
                mean_ratio[iλ] = sum_r / n_inits
                success_rate[iλ] = succ / n_inits
            end

            @printf("    λ=%.4f  uniques(spin)=%d  angles_raw=%d  angles_reduced=%d\n",
                λ, Int(unique_count[iλ]), Int(unique_angle_count_raw[iλ]), Int(unique_angle_count_reduced[iλ]))
        end

        for iλ in 1:nλ
            u = unique_count[iλ]
            sum_unique[iλ]  += u
            sum_unique2[iλ] += u^2

            uar = unique_angle_count_raw[iλ]
            sum_unique_angle_raw[iλ]  += uar
            sum_unique_angle_raw2[iλ] += uar^2

            uared = unique_angle_count_reduced[iλ]
            sum_unique_angle_reduced[iλ]  += uared
            sum_unique_angle_reduced2[iλ] += uared^2

            gf = gn_final_mean[iλ]
            sum_gn_final_mean[iλ]  += gf
            sum_gn_final_mean2[iλ] += gf^2

            gfm = gn_final_min[iλ]
            sum_gn_final_min[iλ]  += gfm
            sum_gn_final_min2[iλ] += gfm^2

            gni = gn_init_mean[iλ]
            sum_gn_init_mean[iλ]  += gni
            sum_gn_init_mean2[iλ] += gni^2

            gnm = gn_maxinner_mean[iλ]
            sum_gn_maxinner_mean[iλ]  += gnm
            sum_gn_maxinner_mean2[iλ] += gnm^2

            gmean = gn_meaninner_mean[iλ]
            sum_gn_meaninner_mean[iλ]  += gmean
            sum_gn_meaninner_mean2[iλ] += gmean^2

            isw = inner_sweeps_used_mean[iλ]
            sum_inner_sweeps_mean[iλ]  += isw
            sum_inner_sweeps_mean2[iλ] += isw^2

            rm = rt_mean[iλ]
            sum_rt_mean[iλ]  += rm
            sum_rt_mean2[iλ] += rm^2

            if best_ratio[iλ] !== nothing
                br = Float64(best_ratio[iλ])
                mr = Float64(mean_ratio[iλ])

                sum_best_ratio[iλ]  += br
                sum_best_ratio2[iλ] += br^2

                sum_mean_ratio[iλ]  += mr
                sum_mean_ratio2[iλ] += mr^2

                n_ratio[iλ] += 1

                sr = success_rate[iλ]
                sum_succ[iλ]  += sr
                sum_succ2[iλ] += sr^2
                n_succ[iλ] += 1
            end
        end

        n_graphs += 1

        ckpt = Dict(
            "N" => N,
            "k" => k,
            "weighted_flag" => weighted,
            "graph_seeds" => graph_seeds,
            "λs" => λs,
            "n_inits" => n_inits,
            "iterations" => iterations,
            "inner_iterations" => inner_iterations,
            "tao" => tao,
            "angle_conv" => angle_conv,
            "init_mode" => String(init_mode),
            "init_sig" => init_sig,
            "seed_salt" => seed_salt,
            "success_thr" => success_thr,
            "angle_bin" => angle_bin,
            "angle_flip_equiv" => angle_flip_equiv,
            "n_graphs" => n_graphs,
            "next_seed_index" => idx + 1,
            "sum_unique" => sum_unique,
            "sum_unique2" => sum_unique2,
            "sum_unique_angle_raw" => sum_unique_angle_raw,
            "sum_unique_angle_raw2" => sum_unique_angle_raw2,
            "sum_unique_angle_reduced" => sum_unique_angle_reduced,
            "sum_unique_angle_reduced2" => sum_unique_angle_reduced2,
            "sum_best_ratio" => sum_best_ratio,
            "sum_best_ratio2" => sum_best_ratio2,
            "sum_mean_ratio" => sum_mean_ratio,
            "sum_mean_ratio2" => sum_mean_ratio2,
            "n_ratio" => n_ratio,
            "sum_succ" => sum_succ,
            "sum_succ2" => sum_succ2,
            "n_succ" => n_succ,
            "sum_gn_final_mean" => sum_gn_final_mean,
            "sum_gn_final_mean2" => sum_gn_final_mean2,
            "sum_gn_final_min" => sum_gn_final_min,
            "sum_gn_final_min2" => sum_gn_final_min2,
            "sum_gn_init_mean" => sum_gn_init_mean,
            "sum_gn_init_mean2" => sum_gn_init_mean2,
            "sum_gn_maxinner_mean" => sum_gn_maxinner_mean,
            "sum_gn_maxinner_mean2" => sum_gn_maxinner_mean2,
            "sum_gn_meaninner_mean" => sum_gn_meaninner_mean,
            "sum_gn_meaninner_mean2" => sum_gn_meaninner_mean2,
            "sum_inner_sweeps_mean" => sum_inner_sweeps_mean,
            "sum_inner_sweeps_mean2" => sum_inner_sweeps_mean2,
            "sum_rt_mean" => sum_rt_mean,
            "sum_rt_mean2" => sum_rt_mean2,
            "timestamp_utc" => Dates.format(Dates.now(Dates.UTC), Dates.ISODateTimeFormat),
        )
        atomic_json_write(ckpt_path, ckpt)
    end

    # -----------------------
    # Final aggregation
    # -----------------------
    results = Vector{Dict}(undef, nλ)

    for (iλ, λ) in enumerate(λs)
        μu, eu = mean_and_stderr(sum_unique[iλ], sum_unique2[iλ], n_graphs)
        μuar, euar = mean_and_stderr(sum_unique_angle_raw[iλ], sum_unique_angle_raw2[iλ], n_graphs)
        μuared, euared = mean_and_stderr(sum_unique_angle_reduced[iλ], sum_unique_angle_reduced2[iλ], n_graphs)

        μgf, egf = mean_and_stderr(sum_gn_final_mean[iλ], sum_gn_final_mean2[iλ], n_graphs)
        μgfm, egfm = mean_and_stderr(sum_gn_final_min[iλ], sum_gn_final_min2[iλ], n_graphs)

        μgni, egni = mean_and_stderr(sum_gn_init_mean[iλ], sum_gn_init_mean2[iλ], n_graphs)
        μgnm, egnm = mean_and_stderr(sum_gn_maxinner_mean[iλ], sum_gn_maxinner_mean2[iλ], n_graphs)
        μgmean, egmean = mean_and_stderr(sum_gn_meaninner_mean[iλ], sum_gn_meaninner_mean2[iλ], n_graphs)

        μisw, eisw = mean_and_stderr(sum_inner_sweeps_mean[iλ], sum_inner_sweeps_mean2[iλ], n_graphs)
        μrt, ert = mean_and_stderr(sum_rt_mean[iλ], sum_rt_mean2[iλ], n_graphs)

        best_ratio_mean = "none"
        best_ratio_stderr = "none"
        mean_ratio_mean = "none"
        mean_ratio_stderr = "none"
        if n_ratio[iλ] > 0
            μbr, ebr = mean_and_stderr(sum_best_ratio[iλ], sum_best_ratio2[iλ], n_ratio[iλ])
            μmr, emr = mean_and_stderr(sum_mean_ratio[iλ], sum_mean_ratio2[iλ], n_ratio[iλ])
            best_ratio_mean = μbr
            best_ratio_stderr = ebr
            mean_ratio_mean = μmr
            mean_ratio_stderr = emr
        end

        succ_mean = "none"
        succ_stderr = "none"
        if n_succ[iλ] > 0
            μs, es = mean_and_stderr(sum_succ[iλ], sum_succ2[iλ], n_succ[iλ])
            succ_mean = μs
            succ_stderr = es
        end

        results[iλ] = Dict(
            "λ" => λ,
            "n_graphs" => n_graphs,

            "unique_count_mean" => μu,
            "unique_count_stderr" => eu,

            "unique_angle_count_raw_mean" => μuar,
            "unique_angle_count_raw_stderr" => euar,

            "unique_angle_count_reduced_mean" => μuared,
            "unique_angle_count_reduced_stderr" => euared,

            # backward-compatible alias:
            "unique_angle_count_mean" => μuared,
            "unique_angle_count_stderr" => euared,

            "angle_bin" => angle_bin,
            "angle_flip_equiv" => angle_flip_equiv,

            "best_ratio_mean" => best_ratio_mean,
            "best_ratio_stderr" => best_ratio_stderr,
            "mean_ratio_mean" => mean_ratio_mean,
            "mean_ratio_stderr" => mean_ratio_stderr,

            "n_ratio" => n_ratio[iλ],

            "success_thr" => success_thr,
            "success_rate_mean" => succ_mean,
            "success_rate_stderr" => succ_stderr,
            "n_succ" => n_succ[iλ],

            "grad_norm_final_mean" => μgf,
            "grad_norm_final_stderr" => egf,

            "grad_norm_final_min_mean" => μgfm,
            "grad_norm_final_min_stderr" => egfm,

            "grad_norm_init_mean" => μgni,
            "grad_norm_init_stderr" => egni,

            "grad_norm_maxinner_mean" => μgnm,
            "grad_norm_maxinner_stderr" => egnm,

            "grad_norm_meaninner_mean" => μgmean,
            "grad_norm_meaninner_stderr" => egmean,

            "inner_sweeps_used_mean" => μisw,
            "inner_sweeps_used_stderr" => eisw,

            "runtime_mean_s" => μrt,
            "runtime_mean_stderr_s" => ert,
        )
    end

    save_data = Dict(
        "experiment" => "qiigs_lambda_sweep_unique_minima",
        "N" => N,
        "k" => k,
        "weighted_flag" => weighted,
        "graph_seeds" => graph_seeds,
        "λs" => λs,
        "n_inits" => n_inits,
        "angle_conv" => angle_conv,

        "solver" => Dict(
            "name" => "grad",
            "iterations" => iterations,
            "inner_iterations" => inner_iterations,
            "tao" => tao,
            "angle_conv" => angle_conv,
            "init_mode" => String(init_mode),
            "save_params" => save_params,
            "init_sig" => init_sig,
            "seed_salt" => seed_salt,
            "outputs" => ["gn_final", "gn_init", "gn_max_inner", "gn_mean_inner", "inner_sweeps_used"],
        ),

        "angle_uniqueness" => Dict(
            "angle_bin" => angle_bin,
            "flip_equiv" => angle_flip_equiv,
            "wrap" => "mod_pi",
            "source" => "sol.metadata[:theta_converged]",
            "reported_counts" => ["raw", "reduced"],
        ),

        "success_thr" => success_thr,

        "inputs" => Dict(
            "qiils_root" => ROOT_QIILS,
            "graphs_root" => GRAPHS_ROOT,
            "solutions_root" => SOLUTIONS_ROOT,
            "solution_key" => "maxcut_value",
        ),

        "results_per_lambda" => results,
        "timestamp_utc" => Dates.format(Dates.now(Dates.UTC), Dates.ISODateTimeFormat),
    )

    abin_tag = replace(@sprintf("%.3g", angle_bin), "."=>"p")
    conv_tag = replace(@sprintf("%.3g", angle_conv), "."=>"p")

    out_path = joinpath(
        save_dir,
        "qiigs_unique_ratio_meanbest_succ_grad_$(tag)_conv$(conv_tag)_abin$(abin_tag)_ngraphs$(n_graphs)_ninit$(n_inits)_outer$(iterations)_inner$(inner_iterations)_thr$(replace(@sprintf("%.3f", success_thr), "."=>"p")).json"
    )

    open(out_path, "w") do io
        JSON.print(io, save_data)
    end

    println("\n✔ Aggregated results saved to: $out_path")
    println("✔ Checkpoint saved to: $ckpt_path")
    println("Done.")
end

main()