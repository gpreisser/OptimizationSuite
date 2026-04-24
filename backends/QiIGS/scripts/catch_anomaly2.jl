# scripts/inspect_basin_dominance_single_lambda.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using QiIGS
using Printf
using SparseArrays

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")
mkpath(RESULTS_DIR)

function main()
    # ------------------------------------------------------------
    # Experiment settings: edit these freely
    # ------------------------------------------------------------
    N = 50
    k = 3
    weighted = false
    graph_seed = 1

    λ = 0.3
    convs = [1e-2, 1e-8]

    n_inits = 10_000
    iterations = 1
    inner_iterations = 5000
    tao = 0.1
    init_mode = :uniform
    save_params = true

    success_thr = 0.999

    compute_hessian = true
    hessian_tol = 1e-8

    seed_salt = 0
    topk = 12

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------
    ROOT_QIILS = "/Users/guillermo.preisser/Projects/QiILS_ITensor"
    GRAPHS_ROOT = joinpath(ROOT_QIILS, "graphs")
    SOLUTIONS_ROOT = joinpath(ROOT_QIILS, "solutions")

    gpath = QiIGS.graph_path(N, k, graph_seed; weighted=weighted, graphs_root=GRAPHS_ROOT)
    @assert isfile(gpath) "Graph file not found: $gpath"
    W = QiIGS.load_weight_matrix(gpath)

    spath = QiIGS.akmax_solution_path(N, k, graph_seed; weighted=weighted, solutions_root=SOLUTIONS_ROOT)
    opt = QiIGS.load_optimal_cut(spath)

    println("==========================================================")
    println(" Basin dominance inspection")
    println("==========================================================")
    println("graph_seed   = $graph_seed")
    println("N, k         = $N, $k")
    println("weighted     = $weighted")
    println("λ            = $λ")
    println("n_inits      = $n_inits")
    println("iterations   = $iterations")
    println("inner_iters  = $inner_iterations")
    println("tao          = $tao")
    println("init_mode    = $init_mode")
    println("optimum cut  = $(opt === nothing ? "none" : string(opt))")
    println()

    for angle_conv in convs
        println("----------------------------------------------------------")
        @printf("angle_conv = %.1e\n", angle_conv)
        println("----------------------------------------------------------")

        counts = Dict{UInt64, Int}()
        rep_conf = Dict{UInt64, Vector{Int8}}()
        rep_energy = Dict{UInt64, Float64}()
        rep_ratio = Dict{UInt64, Float64}()
        rep_is_min = Dict{UInt64, Bool}()

        n_min = 0
        n_saddle = 0
        n_max = 0
        n_deg = 0

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
                compute_hessian = compute_hessian,
                hessian_tol = hessian_tol,
            )

            key = QiIGS.spin_config_key(sol.configuration)
            counts[key] = get(counts, key, 0) + 1

            if !haskey(rep_conf, key)
                rep_conf[key] = copy(sol.configuration)
                rep_energy[key] = Float64(sol.energy)

                if opt === nothing
                    rep_ratio[key] = NaN
                else
                    _, ratio = QiIGS.cut_hat_and_ratio(W, sol.energy, opt)
                    rep_ratio[key] = Float64(ratio)
                end

                rep_is_min[key] = Bool(get(sol.metadata, :hess_is_minimum, false))
            end

            if compute_hessian
                if get(sol.metadata, :hess_is_minimum, false)
                    n_min += 1
                elseif get(sol.metadata, :hess_is_saddle, false)
                    n_saddle += 1
                elseif get(sol.metadata, :hess_is_maximum, false)
                    n_max += 1
                else
                    n_deg += 1
                end
            end
        end

        pairs_sorted = sort(collect(counts); by = x -> x[2], rev = true)
        nunique = length(pairs_sorted)

        println("unique rounded spin configurations = $nunique")
        @printf("fraction minima     = %.6f\n", n_min / n_inits)
        @printf("fraction saddle     = %.6f\n", n_saddle / n_inits)
        @printf("fraction maximum    = %.6f\n", n_max / n_inits)
        @printf("fraction degenerate = %.6f\n", n_deg / n_inits)
        println()

        topn = min(topk, nunique)

        println("Top configurations:")
        println("rank   count     frac       ratio        energy        success>=thr   hess_min?")
        println(repeat("-", 82))

        cumulative = 0
        for (rank, (key, cnt)) in enumerate(pairs_sorted[1:topn])
            frac = cnt / n_inits
            ratio = rep_ratio[key]
            energy = rep_energy[key]
            success = isfinite(ratio) ? (ratio >= success_thr) : false
            cumulative += cnt

            @printf("%4d   %6d   %8.5f   %10.6f   %10.4f   %11s   %9s\n",
                rank,
                cnt,
                frac,
                ratio,
                energy,
                string(success),
                string(rep_is_min[key]),
            )
        end

        println(repeat("-", 82))
        @printf("Top-1 fraction  = %.6f\n", pairs_sorted[1][2] / n_inits)
        @printf("Top-3 fraction  = %.6f\n", sum(p[2] for p in pairs_sorted[1:min(3, nunique)]) / n_inits)
        @printf("Top-5 fraction  = %.6f\n", sum(p[2] for p in pairs_sorted[1:min(5, nunique)]) / n_inits)
        @printf("Top-%d fraction = %.6f\n", topn, cumulative / n_inits)

        if opt !== nothing
            println()
            println("Optimal / success-threshold summary:")

            success_pairs = [
                (key, cnt) for (key, cnt) in pairs_sorted
                if isfinite(rep_ratio[key]) && rep_ratio[key] >= success_thr
            ]

            n_opt_configs = length(success_pairs)
            opt_mass = isempty(success_pairs) ? 0.0 : sum(cnt for (_, cnt) in success_pairs) / n_inits

            println("number of distinct success configs = $n_opt_configs")
            @printf("total probability mass on success configs = %.6f\n", opt_mass)
        end

        println()
    end
end

main()