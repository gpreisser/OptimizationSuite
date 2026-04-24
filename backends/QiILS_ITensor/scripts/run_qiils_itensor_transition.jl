# scripts/qiils_itensor_lambda_sweep_minimize_then_measure.jl
#
# Adds:
#   - single rolling checkpoint after each seed (resume-safe)
#   - atomic checkpoint writes (tmp -> rename)
#
# IMPORTANT:
#   - checkpoint stores accumulators, n_graphs done, and next_seed_index
#   - if interrupted, rerun: it resumes automatically
#
# CRITICAL PATH RULE:
#   - ALL outputs saved under project ROOT/results/... (never relative to pwd()).

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using JSON
using Printf
using Graphs
using Statistics
using Random
using Dates

using QiILS_ITensor
using ITensors
using ITensorMPS

# ---------------------------
# Helpers
# ---------------------------

function minus_product_mps(sites::Vector{<:Index})
    N = length(sites)
    for lab in ("-", "Minus", "minus")
        try
            return productMPS(sites, fill(lab, N))
        catch
        end
    end
    error("Could not build |-> product MPS. Please confirm state labels for siteinds(\"Qubit\", N).")
end

function local_expect_Z(ψ::MPS, sites::Vector{<:Index})
    try
        return ITensorMPS.expect(ψ, "Z")
    catch
    end
    N = length(sites)
    mz = Vector{Float64}(undef, N)
    for i in 1:N
        Zi  = op("Z", sites, i)
        ψZi = apply(Zi, ψ)
        mz[i] = real(inner(ψ, ψZi))
    end
    return mz
end

function local_expect_X(ψ::MPS, sites::Vector{<:Index})
    try
        return ITensorMPS.expect(ψ, "X")
    catch
    end
    N = length(sites)
    mx = Vector{Float64}(undef, N)
    for i in 1:N
        Xi  = op("X", sites, i)
        ψXi = apply(Xi, ψ)
        mx[i] = real(inner(ψ, ψXi))
    end
    return mx
end

function total_edge_weight(wg)
    W = 0.0
    for e in edges(wg)
        i, j = src(e), dst(e)
        W += wg.weights[i, j]
    end
    return W
end

function energy_expectation(H::MPO, ψ::MPS)
    return real(inner(ψ, Apply(H, ψ)))
end

function itensor_minimize_then_measure(
    wg,
    λ::Float64;
    nsweeps::Int = 80,
    maxdim::Int = 64,
    weighted::Bool = false,
    do_debug_lambda0::Bool = false,
)
    N = nv(wg)
    sites = siteinds("Qubit", N)

    Hλ = QiILS_ITensor.build_H_mpo(wg, sites, Float64(λ); weighted=weighted)
    H1 = QiILS_ITensor.build_H_mpo(wg, sites, 1.0; weighted=weighted)

    ψ0 = minus_product_mps(sites)

    if do_debug_lambda0 && λ == 0.0
        x1 = real(inner(ψ0, apply(op("X", sites, 1), ψ0)))
        z1 = real(inner(ψ0, apply(op("Z", sites, 1), ψ0)))
        @info "INIT diagnostics (λ=0)" x1 z1
    end

    Eλ, ψ = dmrg(Hλ, ψ0; nsweeps=nsweeps, maxdim=maxdim, outputlevel=0)

    if do_debug_lambda0 && λ == 0.0
        mx = local_expect_X(ψ, sites)
        @info "POST-DMRG (λ=0)" energy=Eλ meanX=mean(mx) sumX=sum(mx) minX=minimum(mx) maxX=maximum(mx)
    end

    E1 = energy_expectation(H1, ψ)

    mz = local_expect_Z(ψ, sites)
    abs_mz = abs.(mz)

    devZ_abs = mean(abs_mz)
    θeff = 0.5 .* acos.(clamp.(abs_mz, 0.0, 1.0))
    devTheta_abs = mean(abs.(θeff .- (π/4)))

    Wtot = total_edge_weight(wg)
    cut_hat = (Wtot - E1) / 2

    return Eλ, E1, cut_hat, devZ_abs, devTheta_abs
end

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

# atomic JSON write: write tmp then rename
function atomic_json_write(path::AbstractString, data)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        JSON.print(io, data)
    end
    mv(tmp, path; force=true)
    return nothing
end

# ---------------------------
# Main
# ---------------------------

function main()
    println("====================================================")
    println("  QiILS_ITensor λ-sweep (graph outer, λ inner loop) ")
    println("  Experiment: start |->^N, DMRG minimize H(λ),      ")
    println("              measure E(λ) + E(1) + dev-from-X       ")
    println("              also compute cut_hat + approx ratio     ")
    println("              (with rolling checkpoint per seed)      ")
    println("====================================================")

    # -----------------------
    # Graph distribution
    # -----------------------
    N = 50
    k = 3
    weighted = false
    graph_seeds = collect(1:500)

    # -----------------------
    # λ sweep + DMRG params
    # -----------------------
    λs = collect(0.0:0.05:1.0)
    nλ = length(λs)

    nsweeps = 80
    maxdim  = 1

    # print essentials at λ=1.0 if present; else at max(λs)
    λ_target = 1.0
    iλ_print = findfirst(x -> isapprox(x, λ_target; atol=1e-12, rtol=0.0), λs)
    if iλ_print === nothing
        λ_print = maximum(λs)
        iλ_print = argmin(abs.(λs .- λ_print))
    else
        λ_print = λs[iλ_print]  # will be 1.0
    end
    @info "Per-seed essentials will be printed at" λ_print

    # -----------------------
    # Output folder (ALWAYS under project root)
    # -----------------------
    ROOT = normpath(joinpath(@__DIR__, ".."))     # project root: QiILS_ITensor/
    RESULTS_DIR = joinpath(ROOT, "results")       # QiILS_ITensor/results
    weight_tag = weighted ? "weighted" : "unweighted"

    save_dir = joinpath(
        RESULTS_DIR,
        "itensor_random_regular_N$(N)_k$(k)_graphs$(length(graph_seeds))_$(weight_tag)"
    )
    mkpath(save_dir)

    @info "Project ROOT" ROOT
    @info "Saving results under" save_dir

    # λ-grid tag + checkpoint path
    λmin = first(λs); λmax = last(λs); dλ = (length(λs) > 1 ? (λs[2] - λs[1]) : 0.0)
    tag = "lam$(replace(@sprintf("%.3f", λmin), "."=>"p"))_to_$(replace(@sprintf("%.3f", λmax), "."=>"p"))_d$(replace(@sprintf("%.3f", dλ), "."=>"p"))"
    ckpt_path = joinpath(save_dir, "checkpoint_energy2_cut_hat_ratio_$(tag)_maxdim$(maxdim)_nsw$(nsweeps).json")

    println("▶ Starting sweep: $(length(graph_seeds)) graphs × $(nλ) λ-values (total runs = $(length(graph_seeds)*nλ))")
    println("   DMRG: nsweeps=$nsweeps, maxdim=$maxdim")
    println("   Checkpoint: $ckpt_path")

    # -----------------------
    # Graph file helpers
    # -----------------------
    graphs_base = joinpath(ROOT, "graphs")
    @inline function graph_path_qiils(N::Int, k::Int, seed::Int, weighted::Bool)
        dir_path = joinpath(graphs_base, string(N), string(k))
        wtag = weighted ? "weighted" : "unweighted"
        filename = "graph_N$(N)_k$(k)_seed$(seed)_seedb$(seed)_$(wtag).txt"
        return joinpath(dir_path, filename)
    end

    # -----------------------
    # Accumulators (maybe loaded from checkpoint)
    # -----------------------
    sum_Eλ     = zeros(Float64, nλ); sum_Eλ2    = zeros(Float64, nλ)
    sum_E1     = zeros(Float64, nλ); sum_E12    = zeros(Float64, nλ)
    sum_cut    = zeros(Float64, nλ); sum_cut2   = zeros(Float64, nλ)
    sum_devZ   = zeros(Float64, nλ); sum_devZ2  = zeros(Float64, nλ)
    sum_devTh  = zeros(Float64, nλ); sum_devTh2 = zeros(Float64, nλ)
    sum_t      = zeros(Float64, nλ); sum_t2     = zeros(Float64, nλ)
    sum_ratio  = zeros(Float64, nλ); sum_ratio2 = zeros(Float64, nλ)
    n_ratio    = zeros(Int, nλ)

    n_graphs = 0
    start_index = 1

    # -----------------------
    # Resume from checkpoint if present (and compatible)
    # -----------------------
    if isfile(ckpt_path)
        ck = JSON.parsefile(ckpt_path)
        if ck["N"] == N && ck["k"] == k && ck["weighted_flag"] == weighted &&
           ck["dmrg_nsweeps"] == nsweeps && ck["dmrg_maxdim"] == maxdim &&
           length(ck["λs"]) == length(λs)

            sum_Eλ    .= Float64.(ck["sum_Eλ"]);    sum_Eλ2   .= Float64.(ck["sum_Eλ2"])
            sum_E1    .= Float64.(ck["sum_E1"]);    sum_E12   .= Float64.(ck["sum_E12"])
            sum_cut   .= Float64.(ck["sum_cut"]);   sum_cut2  .= Float64.(ck["sum_cut2"])
            sum_devZ  .= Float64.(ck["sum_devZ"]);  sum_devZ2 .= Float64.(ck["sum_devZ2"])
            sum_devTh .= Float64.(ck["sum_devTh"]); sum_devTh2 .= Float64.(ck["sum_devTh2"])
            sum_t     .= Float64.(ck["sum_t"]);     sum_t2    .= Float64.(ck["sum_t2"])
            sum_ratio .= Float64.(ck["sum_ratio"]); sum_ratio2 .= Float64.(ck["sum_ratio2"])
            n_ratio   .= Int.(ck["n_ratio"])

            n_graphs = Int(ck["n_graphs"])
            start_index = Int(ck["next_seed_index"])
            println("↻ Resuming from checkpoint: n_graphs=$n_graphs, next_seed_index=$start_index")
        else
            println("⚠ Found checkpoint but it does not match current run parameters; ignoring it.")
        end
    end

    # -----------------------
    # Outer loop over graphs (resume-safe)
    # -----------------------
    for idx in start_index:length(graph_seeds)
        gs = graph_seeds[idx]
        println("\n► graph seed = $gs  (graph $idx / $(length(graph_seeds)))")

        gpath = graph_path_qiils(N, k, gs, weighted)
        if !isfile(gpath)
            println("    … graph missing; creating: $gpath")
            create_and_save_graph_QiILS(N, k, gs; weighted=weighted, base_path=graphs_base)
            @assert isfile(gpath) "Graph creation failed; expected file not found: $gpath"
            println("    ✔ created graph file: $gpath")
        else
            println("    ✔ graph file exists: $gpath")
        end

        wg = load_graph(path=gpath, weighted=weighted)
        println("    nv(wg) = ", nv(wg), ", ne(wg) = ", ne(wg))

        solpath = QiILS_ITensor.solution_file_path(N, k, gs; weighted=weighted)
        optimal_cut = load_optimal_cut(solpath)
        if optimal_cut !== nothing
            println("    ✔ loaded optimal_cut = $optimal_cut")
        else
            println("    ⚠ no stored optimal_cut for this graph (ratio will be skipped for it)")
        end

        for (iλ, λ) in enumerate(λs)
            Random.seed!(gs * 10_000 + Int(round(λ * 1000)))

            t0 = time()
            Eλ, E1, cut_hat, devZ_abs, devTheta_abs = itensor_minimize_then_measure(
                wg, Float64(λ);
                nsweeps = nsweeps,
                maxdim  = maxdim,
                weighted = weighted,
                do_debug_lambda0 = (idx == 1),
            )
            dt = time() - t0

            sum_Eλ[iλ]  += Eλ;          sum_Eλ2[iλ]  += Eλ^2
            sum_E1[iλ]  += E1;          sum_E12[iλ]  += E1^2
            sum_cut[iλ] += cut_hat;     sum_cut2[iλ] += cut_hat^2
            sum_devZ[iλ]  += devZ_abs;     sum_devZ2[iλ]  += devZ_abs^2
            sum_devTh[iλ] += devTheta_abs; sum_devTh2[iλ] += devTheta_abs^2
            sum_t[iλ] += dt;           sum_t2[iλ]   += dt^2

            ratio_here = nothing
            if optimal_cut !== nothing
                ratio_here = cut_hat / optimal_cut
                n_ratio[iλ] += 1
                sum_ratio[iλ]  += ratio_here
                sum_ratio2[iλ] += ratio_here^2
            end

            if iλ == iλ_print
                rstr = ratio_here === nothing ? "none" : @sprintf("%.6f", ratio_here)
                @printf("    [seed=%d | λ=%.3f] E(λ)=%.6f  E(1)=%.6f  cut_hat=%.6f  ratio=%s  t=%.3fs\n",
                        gs, λ_print, Eλ, E1, cut_hat, rstr, dt)
            end
        end

        n_graphs += 1

        # -----------------------
        # CHECKPOINT SAVE (single rolling file)
        # -----------------------
        ckpt = Dict(
            "N" => N,
            "k" => k,
            "weighted_flag" => weighted,
            "graph_seeds" => graph_seeds,
            "dmrg_nsweeps" => nsweeps,
            "dmrg_maxdim" => maxdim,
            "λs" => λs,

            "n_graphs" => n_graphs,
            "next_seed_index" => idx + 1,

            "sum_Eλ" => sum_Eλ, "sum_Eλ2" => sum_Eλ2,
            "sum_E1" => sum_E1, "sum_E12" => sum_E12,
            "sum_cut" => sum_cut, "sum_cut2" => sum_cut2,
            "sum_devZ" => sum_devZ, "sum_devZ2" => sum_devZ2,
            "sum_devTh" => sum_devTh, "sum_devTh2" => sum_devTh2,
            "sum_t" => sum_t, "sum_t2" => sum_t2,
            "sum_ratio" => sum_ratio, "sum_ratio2" => sum_ratio2,
            "n_ratio" => n_ratio,

            "timestamp_utc" => Dates.format(Dates.now(Dates.UTC), Dates.ISODateTimeFormat),
        )
        atomic_json_write(ckpt_path, ckpt)
    end

    # -----------------------
    # Final aggregation + JSON
    # -----------------------
    results = Vector{Dict}(undef, nλ)

    for (iλ, λ) in enumerate(λs)
        meanEλ, errEλ   = mean_and_stderr(sum_Eλ[iλ],   sum_Eλ2[iλ],   n_graphs)
        meanE1, errE1   = mean_and_stderr(sum_E1[iλ],   sum_E12[iλ],   n_graphs)
        meanCut, errCut = mean_and_stderr(sum_cut[iλ],  sum_cut2[iλ],  n_graphs)
        meanZ, errZ     = mean_and_stderr(sum_devZ[iλ], sum_devZ2[iλ], n_graphs)
        meanTh, errTh   = mean_and_stderr(sum_devTh[iλ], sum_devTh2[iλ], n_graphs)
        meanT, errT     = mean_and_stderr(sum_t[iλ],    sum_t2[iλ],    n_graphs)

        ratio_mean = "none"
        ratio_stderr = "none"
        if n_ratio[iλ] > 0
            μr, er = mean_and_stderr(sum_ratio[iλ], sum_ratio2[iλ], n_ratio[iλ])
            ratio_mean = μr
            ratio_stderr = er
        end

        results[iλ] = Dict(
            "λ_sweep" => λ,
            "ngraphs" => n_graphs,

            "energy_lambda_mean" => meanEλ,
            "energy_lambda_stderr" => errEλ,

            "energy_cost_lambda1_mean" => meanE1,
            "energy_cost_lambda1_stderr" => errE1,

            "cut_hat_mean" => meanCut,
            "cut_hat_stderr" => errCut,

            "approx_ratio_mean" => ratio_mean,
            "approx_ratio_stderr" => ratio_stderr,
            "n_ratio" => n_ratio[iλ],

            "devZ_abs_mean" => meanZ,
            "devZ_abs_stderr" => errZ,

            "devTheta_abs_mean" => meanTh,
            "devTheta_abs_stderr" => errTh,

            "runtime_s_mean" => meanT,
            "runtime_s_stderr" => errT,
        )
    end

    save_data = Dict(
        "experiment_solver" => "itensor_minimize_then_measure",
        "init_state" => "minus_product_state",
        "graph_type" => "random_regular",
        "N" => N,
        "k" => k,
        "weighted_flag" => weighted,
        "graph_seeds" => graph_seeds,

        "dmrg_nsweeps" => nsweeps,
        "dmrg_maxdim" => maxdim,

        "λs" => λs,
        "results_per_lambda" => results,

        "cut_hat_notes" => Dict(
            "definition" => "cut_hat = (Wtot - E(1))/2, with Wtot = sum_{edges} w_ij",
            "ratio" => "approx_ratio = cut_hat / optimal_cut (from solutions/random_regular/N/k/akmaxdata_...json)",
        ),
        "dev_metric_notes" => Dict(
            "devZ_abs" => "mean_i |<Z_i>|",
            "devTheta_abs" => "mean_i | 0.5*acos(|<Z_i>|) - π/4 |",
        ),
        "timestamp_utc" => Dates.format(Dates.now(Dates.UTC), Dates.ISODateTimeFormat),
    )

    out_path = joinpath(
        save_dir,
        "qiils_itensor_lambda_sweep_energy2_cut_hat_ratio_$(tag)_ngraphs$(n_graphs)_maxdim$(maxdim)_nsw$(nsweeps).json"
    )

    open(out_path, "w") do io
        JSON.print(io, save_data)
    end

    println("\n✔ Aggregated results saved to: $out_path")
    println("✔ Checkpoint saved to: $ckpt_path")
    println("Done.")
end

main()