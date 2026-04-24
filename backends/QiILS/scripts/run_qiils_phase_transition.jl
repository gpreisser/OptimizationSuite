using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using QiILS
using JSON
using Printf
using Graphs

function main()
    println("====================================================")
    println("  QiILS Runner (graph outer loop, λ inner loop)     ")
    println("  Experiment: start θ=π/4, minimize H_λ, measure cut")
    println("====================================================")

    # ---------------------------------------------------------
    # Graph distribution parameters
    # ---------------------------------------------------------
    N = 50
    k = 3
    weighted = true

    # graph instances (outer loop)
    graph_seeds = 1:1   # <-- set back to 1:100 when ready

    # ---------------------------------------------------------
    # Experiment parameters (λ sweep)
    # ---------------------------------------------------------
    λs = collect(0.0:0.1:1.0)
    nλ = length(λs)

    max_sweeps = 80
    angle_conv = 1e-20
    use_scaled_convergence = true

    # ---------------------------------------------------------
    # Per-λ accumulators (over graphs)
    # ---------------------------------------------------------
    sum_best  = zeros(Float64, nλ)
    sum_best2 = zeros(Float64, nλ)

    sum_sweeps  = zeros(Float64, nλ)
    sum_sweeps2 = zeros(Float64, nλ)

    sum_dev  = zeros(Float64, nλ)
    sum_dev2 = zeros(Float64, nλ)

    sum_ratio  = zeros(Float64, nλ)
    sum_ratio2 = zeros(Float64, nλ)
    n_ratio    = zeros(Int, nλ)

    n_graphs = 0

    # ---------------------------------------------------------
    # Output folder
    # ---------------------------------------------------------
    weight_tag = weighted ? "weighted" : "unweighted"
    save_dir = joinpath("results", "random_regular_N$(N)_k$(k)_graphs$(length(graph_seeds))_$(weight_tag)")
    mkpath(save_dir)

    println("▶ Starting sweep: $(length(graph_seeds)) graphs × $(nλ) λ-values (total runs = $(length(graph_seeds)*nλ))")

    # ---------------------------------------------------------
    # Graph file helpers (match YOUR create_and_save_graph_QiILS naming)
    # ---------------------------------------------------------
    graphs_base = joinpath(@__DIR__, "..", "graphs")

    @inline function graph_path_qiils(N::Int, k::Int, seed::Int)
        dir_path = joinpath(graphs_base, string(N), string(k))
        filename = "graph_N$(N)_k$(k)_seed$(seed)_seedb$(seed).txt"
        return joinpath(dir_path, filename)
    end

    # ---------------------------------------------------------
    # Main: outer loop over graphs
    # ---------------------------------------------------------
    for gs in graph_seeds
        println("\n► graph seed = $gs  (graph $((gs - first(graph_seeds)) + 1) / $(length(graph_seeds)))")

        # ensure graph exists on disk
        
        println("    nv(wg) = ", nv(wg), ", ne(wg) = ", ne(wg))

W = wg.weights
ws = Float64[]
for e in edges(wg)
    push!(ws, W[src(e), dst(e)])
end

println("    weight sample (first 10): ", ws[1:min(end,10)])
println("    weight stats: min=", minimum(ws), " max=", maximum(ws), " mean=", sum(ws)/length(ws))
println("    all_integer_weights? ", all(w -> isapprox(w, round(w); atol=1e-12), ws))
        # optional: load optimal
        optimal_cut = nothing
        try
            optimal_cut = load_optimal_cut(N, k, gs)
        catch
            optimal_cut = nothing
        end
        if optimal_cut !== nothing
            println("    ✔ loaded optimal_cut = $optimal_cut")
        else
            println("    ⚠ no stored optimal_cut for this graph (ratio will be skipped for it)")
        end

        gvec = zeros(Float64, nv(wg))

        for (iλ, λ_sweep) in enumerate(λs)
            cut_val, dev_meanabs, total_sweeps = qiils_minimize_then_measure(
                wg,
                λ_sweep,
                gvec;
                θ0 = nothing,
                sweeps = max_sweeps,
                angle_conv = angle_conv,
                use_scaled_convergence = use_scaled_convergence,
            )

            best_cut = float(cut_val)
            total_sweeps = float(total_sweeps)

            sum_best[iλ]  += best_cut
            sum_best2[iλ] += best_cut^2

            sum_sweeps[iλ]  += total_sweeps
            sum_sweeps2[iλ] += total_sweeps^2

            sum_dev[iλ]  += dev_meanabs
            sum_dev2[iλ] += dev_meanabs^2

            if optimal_cut !== nothing
                r = best_cut / optimal_cut
                n_ratio[iλ] += 1
                sum_ratio[iλ]  += r
                sum_ratio2[iλ] += r^2
            end
        end

        n_graphs += 1
    end

    # ---------------------------------------------------------
    # Compute mean and stderr per λ
    # ---------------------------------------------------------
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

    results = Vector{Dict}(undef, nλ)

    for (iλ, λ_sweep) in enumerate(λs)
        mean_best, err_best     = mean_and_stderr(sum_best[iλ],   sum_best2[iλ],   n_graphs)
        mean_sweeps, err_sweeps = mean_and_stderr(sum_sweeps[iλ], sum_sweeps2[iλ], n_graphs)
        mean_dev, err_dev       = mean_and_stderr(sum_dev[iλ],    sum_dev2[iλ],    n_graphs)

        ratio_mean = "none"
        ratio_stderr = "none"
        if n_ratio[iλ] > 0
            μr, er = mean_and_stderr(sum_ratio[iλ], sum_ratio2[iλ], n_ratio[iλ])
            ratio_mean = μr
            ratio_stderr = er
        end

        println("\nλ=$(round(λ_sweep, digits=4)) → " *
                "best_cut = $(round(mean_best, digits=6)) ± $(round(err_best, digits=6)), " *
                "dev = $(round(mean_dev, digits=6)) ± $(round(err_dev, digits=6)), " *
                "sweeps = $(round(mean_sweeps, digits=3)) ± $(round(err_sweeps, digits=3))")

        results[iλ] = Dict(
            "λ_sweep" => λ_sweep,
            "ngraphs" => n_graphs,
            "best_cut_mean" => mean_best,
            "best_cut_stderr" => err_best,
            "dev_meanabs_mean" => mean_dev,
            "dev_meanabs_stderr" => err_dev,
            "total_sweeps_mean" => mean_sweeps,
            "total_sweeps_stderr" => err_sweeps,
            "approx_ratio_mean" => ratio_mean,
            "approx_ratio_stderr" => ratio_stderr,
            "n_ratio" => n_ratio[iλ],
        )
    end

    # ---------------------------------------------------------
    # Save aggregated JSON
    # ---------------------------------------------------------
    save_data = Dict(
        "experiment_solver" => "qiils_minimize_then_measure",
        "init_theta" => "pi_over_4",
        "graph_type" => "random_regular",
        "N" => N,
        "k" => k,
        "weighted_flag" => weighted,
        "graph_seeds" => collect(graph_seeds),
        "max_sweeps" => max_sweeps,
        "angle_conv" => angle_conv,
        "use_scaled_convergence" => use_scaled_convergence,
        "λs" => λs,
        "results_per_lambda" => results,
    )

    λmin = first(λs); λmax = last(λs); dλ = λs[2] - λs[1]
    tag = "lam$(replace(@sprintf("%.3f", λmin), "."=>"p"))_to_$(replace(@sprintf("%.3f", λmax), "."=>"p"))_d$(replace(@sprintf("%.3f", dλ), "."=>"p"))"
    out_path = joinpath(save_dir, "qiils_lambda_sweep_graphavg_transition_$(tag)_ngraphs$(n_graphs).json")

    open(out_path, "w") do io
        JSON.print(io, save_data)
    end

    println("\n✔ Aggregated results saved to: $out_path")
    println("Done.")
end

main()