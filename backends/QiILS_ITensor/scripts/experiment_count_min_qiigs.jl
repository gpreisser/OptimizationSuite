# scripts/qiils_itensor_lambda_sweep_uniques_skeleton.jl
#
# Skeleton for:
#   - outer loop: graphs (seeds)
#   - inner loop: λ ∈ 0.0:0.05:1.0
#   - inner-inner: many random |↑/↓> initial product states per (graph, λ)
#
# For each (graph, λ):
#   - run solver from many initial states
#   - record:
#       * E(λ) (solver objective), E(1) (final cost quality), cut_hat
#       * grad_norm (||∇|| from your gradient-based solver)
#       * unique minima counts:
#           (A) by continuous signature (mz = ⟨Z⟩ vector, up to tol)
#           (B) by discrete cut bitstring (threshold sign(mz), canonicalize global flip)
#
# Output:
#   - rolling checkpoint after each graph seed (resume-safe)
#   - final JSON with per-(graph,λ) uniqueness summary + optional top-K solutions
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

# ============================================================
# Helpers: observables + energies
# ============================================================

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

# ============================================================
# Helpers: initial product states |↑/↓> (0 or π/2)
# ============================================================

"""
Build a random Z-basis product MPS.
Each site is either "Up" or "Dn" (with fallbacks).
"""
function random_updown_product_mps(sites::Vector{<:Index}; p_up::Float64=0.5, rng::AbstractRNG=Random.default_rng())
    N = length(sites)
    labels = Vector{String}(undef, N)
    for i in 1:N
        labels[i] = (rand(rng) < p_up) ? "Up" : "Dn"
    end

    # Try common label conventions
    for (up, dn) in (("Up","Dn"), ("Up","Down"), ("0","1"), ("↑","↓"))
        try
            labs = [lab == "Up" ? up : dn for lab in labels]
            return productMPS(sites, labs)
        catch
        end
    end
    error("Could not build random |Up/Dn> product MPS. Please confirm the state labels for siteinds(\"Qubit\", N).")
end

# ============================================================
# Uniqueness logic
# ============================================================

# ----- A) uniqueness by continuous signature (e.g. mz) -----

"""
Return true if `sig` is NEW (not within tol of any existing rep); if new, store it.
Uses infinity norm max_i |sig[i]-rep[i]|.
"""
function is_new_continuous_signature!(reps::Vector{Vector{Float64}}, sig::Vector{Float64}; tol::Float64=1e-3)
    for r in reps
        if maximum(abs.(sig .- r)) < tol
            return false
        end
    end
    push!(reps, copy(sig))
    return true
end

# ----- B) uniqueness by discrete cut bitstring -----

function spins_from_mz(mz::Vector{Float64})
    s = Vector{Int8}(undef, length(mz))
    @inbounds for i in eachindex(mz)
        s[i] = (mz[i] >= 0) ? Int8(1) : Int8(-1)
    end
    return s
end

function canonicalize_global_flip!(s::Vector{Int8})
    if s[1] == Int8(-1)
        @inbounds for i in eachindex(s)
            s[i] = -s[i]
        end
    end
    return s
end

function spin_key(s::Vector{Int8})
    io = IOBuffer()
    @inbounds for i in eachindex(s)
        write(io, s[i] == Int8(1) ? '+' : '-')
    end
    return String(take!(io))
end

# ============================================================
# Solver wrapper (SWAP THIS LATER)
# ============================================================

"""
Run the current solver for a single (wg, λ) from a given initial state ψ0.

For now: DMRG minimize H(λ) starting from ψ0.
Since DMRG is not gradient-based, we set grad_norm = "none".

When you swap to your gradient solver, return a Float64 grad_norm.
"""
function run_solver_minimize_then_measure(
    wg,
    λ::Float64,
    ψ0::MPS,
    sites::Vector{<:Index};
    nsweeps::Int = 80,
    maxdim::Int = 64,
    weighted::Bool = false,
)
    Hλ = QiILS_ITensor.build_H_mpo(wg, sites, Float64(λ); weighted=weighted)
    H1 = QiILS_ITensor.build_H_mpo(wg, sites, 1.0; weighted=weighted)

    Eλ, ψ = dmrg(Hλ, ψ0; nsweeps=nsweeps, maxdim=maxdim, outputlevel=0)

    E1 = energy_expectation(H1, ψ)
    mz = local_expect_Z(ψ, sites)

    Wtot = total_edge_weight(wg)
    cut_hat = (Wtot - E1) / 2

    grad_norm = "none"   # placeholder until you plug in the gradient solver

    return Eλ, E1, cut_hat, grad_norm, mz, ψ
end

# ============================================================
# JSON utilities
# ============================================================

function atomic_json_write(path::AbstractString, data)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        JSON.print(io, data)
    end
    mv(tmp, path; force=true)
    return nothing
end

# ============================================================
# Main
# ============================================================

function main()
    println("====================================================")
    println("  λ-sweep UNIQUE-SOLUTIONS skeleton (+grad_norm)     ")
    println("====================================================")

    # -----------------------
    # Graph distribution
    # -----------------------
    N = 50
    k = 4
    weighted = false
    graph_seeds = collect(1:100)

    # -----------------------
    # λ sweep
    # -----------------------
    λs = collect(0.0:0.05:1.0)
    nλ = length(λs)

    # -----------------------
    # Trials per (graph, λ)
    # -----------------------
    n_inits = 50
    p_up    = 0.5

    # -----------------------
    # Solver params (placeholder: DMRG)
    # -----------------------
    nsweeps = 80
    maxdim  = 8

    # -----------------------
    # Uniqueness tolerances
    # -----------------------
    tol_mz = 1e-3

    # keep only top-K by E1 for debugging/inspection
    topK = 5

    # -----------------------
    # Output folder under ROOT
    # -----------------------
    ROOT = normpath(joinpath(@__DIR__, ".."))
    RESULTS_DIR = joinpath(ROOT, "results")
    weight_tag = weighted ? "weighted" : "unweighted"

    save_dir = joinpath(
        RESULTS_DIR,
        "uniques_random_regular_N$(N)_k$(k)_graphs$(length(graph_seeds))_$(weight_tag)"
    )
    mkpath(save_dir)

    @info "Project ROOT" ROOT
    @info "Saving results under" save_dir

    λmin = first(λs); λmax = last(λs); dλ = (length(λs) > 1 ? (λs[2] - λs[1]) : 0.0)
    tag = "lam$(replace(@sprintf("%.3f", λmin), "."=>"p"))_to_$(replace(@sprintf("%.3f", λmax), "."=>"p"))_d$(replace(@sprintf("%.3f", dλ), "."=>"p"))"

    ckpt_path = joinpath(save_dir, "checkpoint_uniques_$(tag)_ninits$(n_inits)_maxdim$(maxdim)_nsw$(nsweeps).json")

    println("▶ Starting sweep: $(length(graph_seeds)) graphs × $(nλ) λ-values × $(n_inits) initials")
    println("   Solver params: nsweeps=$nsweeps, maxdim=$maxdim")
    println("   tol_mz=$tol_mz, topK=$topK")
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
    # Resume state
    # -----------------------
    per_graph_results = Dict{String,Any}()  # key = string(seed)
    start_index = 1

    if isfile(ckpt_path)
        ck = JSON.parsefile(ckpt_path)
        if ck["N"] == N && ck["k"] == k && ck["weighted_flag"] == weighted &&
           ck["nsweeps"] == nsweeps && ck["maxdim"] == maxdim &&
           ck["n_inits"] == n_inits && isapprox(ck["p_up"], p_up; atol=0.0, rtol=0.0) &&
           length(ck["λs"]) == length(λs)

            per_graph_results = Dict{String,Any}(ck["per_graph_results"])
            start_index = Int(ck["next_seed_index"])
            println("↻ Resuming from checkpoint: next_seed_index=$start_index (loaded $(length(per_graph_results)) graph entries)")
        else
            println("⚠ Found checkpoint but it does not match current run parameters; ignoring it.")
        end
    end

    # -----------------------
    # Outer loop over graphs
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

        # optional: load optimal cut
        solpath = QiILS_ITensor.solution_file_path(N, k, gs; weighted=weighted)
        optimal_cut = load_optimal_cut(solpath)

        sites = siteinds("Qubit", N)

        graph_entry = Dict(
            "graph_seed" => gs,
            "N" => N,
            "k" => k,
            "weighted_flag" => weighted,
            "λs" => λs,
            "n_inits" => n_inits,
            "p_up" => p_up,
            "tol_mz" => tol_mz,
            "solver_params" => Dict("nsweeps" => nsweeps, "maxdim" => maxdim),
            "optimal_cut" => (optimal_cut === nothing ? "none" : optimal_cut),
            "per_lambda" => Vector{Any}(undef, nλ),
        )

        for (iλ, λ) in enumerate(λs)
            unique_mz_reps = Vector{Vector{Float64}}()
            cut_counts = Dict{String,Int}()
            cut_bestE1 = Dict{String,Float64}()

            # gradient-norm summary per λ (per graph)
            grad_vals = Float64[]
            n_grad = 0

            top = Vector{Dict{String,Any}}()

            sumEλ = 0.0; sumE1 = 0.0; sumCut = 0.0
            t_sum = 0.0

            for trial in 1:n_inits
                seed_here = gs * 1_000_000 + Int(round(λ * 10_000)) * 10_000 + trial
                rng = MersenneTwister(seed_here)

                ψ0 = random_updown_product_mps(sites; p_up=p_up, rng=rng)

                t0 = time()
                Eλ, E1, cut_hat, grad_norm, mz, _ψ = run_solver_minimize_then_measure(
                    wg, Float64(λ), ψ0, sites;
                    nsweeps=nsweeps, maxdim=maxdim, weighted=weighted
                )
                dt = time() - t0
                t_sum += dt

                sumEλ += Eλ; sumE1 += E1; sumCut += cut_hat

                # grad_norm bookkeeping (skip if "none")
                if grad_norm isa Number
                    push!(grad_vals, Float64(grad_norm))
                    n_grad += 1
                end

                is_new_continuous_signature!(unique_mz_reps, mz; tol=tol_mz)

                s = spins_from_mz(mz)
                canonicalize_global_flip!(s)
                key = spin_key(s)

                cut_counts[key] = get(cut_counts, key, 0) + 1
                if !haskey(cut_bestE1, key) || E1 < cut_bestE1[key]
                    cut_bestE1[key] = E1
                end

                push!(top, Dict(
                    "trial" => trial,
                    "Eλ" => Eλ,
                    "E1" => E1,
                    "cut_hat" => cut_hat,
                    "grad_norm" => grad_norm,
                    "cut_key" => key,
                    "dt_s" => dt,
                ))
                sort!(top, by = x -> x["E1"])
                if length(top) > topK
                    resize!(top, topK)
                end
            end

            n_unique_mz  = length(unique_mz_reps)
            n_unique_cut = length(cut_counts)

            best_cut_key = "none"
            best_cut_E1  = Inf
            for (key, e1) in cut_bestE1
                if e1 < best_cut_E1
                    best_cut_E1 = e1
                    best_cut_key = key
                end
            end

            ratio_best = "none"
            if optimal_cut !== nothing && isfinite(best_cut_E1)
                Wtot = total_edge_weight(wg)
                cut_hat_best = (Wtot - best_cut_E1) / 2
                ratio_best = cut_hat_best / optimal_cut
            end

            grad_mean = "none"
            grad_min  = "none"
            if n_grad > 0
                grad_mean = mean(grad_vals)
                grad_min  = minimum(grad_vals)
            end

            graph_entry["per_lambda"][iλ] = Dict(
                "λ" => λ,
                "n_trials" => n_inits,

                "n_unique_mz" => n_unique_mz,
                "n_unique_cut" => n_unique_cut,

                "Eλ_mean" => sumEλ / n_inits,
                "E1_mean" => sumE1 / n_inits,
                "cut_hat_mean" => sumCut / n_inits,

                "grad_norm_mean" => grad_mean,
                "grad_norm_min" => grad_min,
                "n_grad_recorded" => n_grad,

                "best_cut_key_by_E1" => best_cut_key,
                "best_cut_E1" => (isfinite(best_cut_E1) ? best_cut_E1 : "none"),
                "best_cut_ratio" => ratio_best,

                "cut_counts" => cut_counts,
                "cut_bestE1" => cut_bestE1,

                "topK_by_E1" => top,

                "runtime_total_s" => t_sum,
            )

            @printf("    [λ=%.3f] unique_mz=%d unique_cut=%d  meanE1=%.6f  bestE1=%s  grad_mean=%s  t=%.2fs\n",
                    λ, n_unique_mz, n_unique_cut, sumE1/n_inits,
                    (isfinite(best_cut_E1) ? @sprintf("%.6f", best_cut_E1) : "none"),
                    (grad_mean == "none" ? "none" : @sprintf("%.3e", grad_mean)),
                    t_sum)
        end

        per_graph_results[string(gs)] = graph_entry

        ckpt = Dict(
            "N" => N,
            "k" => k,
            "weighted_flag" => weighted,
            "graph_seeds" => graph_seeds,
            "λs" => λs,
            "n_inits" => n_inits,
            "p_up" => p_up,
            "tol_mz" => tol_mz,
            "nsweeps" => nsweeps,
            "maxdim" => maxdim,

            "per_graph_results" => per_graph_results,

            "next_seed_index" => idx + 1,
            "timestamp_utc" => Dates.format(Dates.now(Dates.UTC), Dates.ISODateTimeFormat),
        )
        atomic_json_write(ckpt_path, ckpt)
        println("    ✔ checkpoint saved")
    end

    out_path = joinpath(save_dir, "uniques_results_$(tag)_ninits$(n_inits)_maxdim$(maxdim)_nsw$(nsweeps).json")

    save_data = Dict(
        "experiment" => "lambda_sweep_uniques_skeleton",
        "graph_type" => "random_regular",
        "N" => N,
        "k" => k,
        "weighted_flag" => weighted,
        "graph_seeds" => graph_seeds,
        "λs" => λs,
        "n_inits" => n_inits,
        "p_up" => p_up,
        "tol_mz" => tol_mz,
        "solver_placeholder" => "dmrg_minimize_Hlambda",
        "solver_params" => Dict("nsweeps" => nsweeps, "maxdim" => maxdim),
        "per_graph_results" => per_graph_results,
        "timestamp_utc" => Dates.format(Dates.now(Dates.UTC), Dates.ISODateTimeFormat),
    )

    open(out_path, "w") do io
        JSON.print(io, save_data)
    end

    println("\n✔ Final results saved to: $out_path")
    println("✔ Checkpoint saved to: $ckpt_path")
    println("Done.")
end

main()