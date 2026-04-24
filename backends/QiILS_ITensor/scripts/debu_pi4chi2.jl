# scripts/debug_chi2_lambda1_devtheta.jl
#
# Debug χ=2 at λ=1.0 : why devTheta_abs_mean can be < π/4 (often even lower than χ=1)
#
# Adds χ>1 diagnostics:
#   - entanglement proxy: bond entropies (if available) + linkdim summary
#   - edge correlators: ⟨Z_i Z_j⟩ on edges touching "bad" sites (|Z_i|<1)
#   - classical rounding: round from ⟨Z⟩ to a bitstring and compute cut_round, compare to cut_hat from E(1)
#
# IMPORTANT: saves under ROOT/results/debug_devtheta (not pwd-dependent)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JSON
using Printf
using Statistics
using Random
using Dates

using QiILS_ITensor
using Graphs
using ITensors
using ITensorMPS

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(ROOT, "results")
const GRAPHS_DIR  = joinpath(ROOT, "graphs")

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

function energy_expectation(H::MPO, ψ::MPS)
    return real(inner(ψ, Apply(H, ψ)))
end

function total_edge_weight(wg)
    W = 0.0
    for e in edges(wg)
        i, j = src(e), dst(e)
        W += wg.weights[i, j]
    end
    return W
end

@inline function graph_path_qiils(N::Int, k::Int, seed::Int, weighted::Bool)
    dir_path = joinpath(GRAPHS_DIR, string(N), string(k))
    wtag = weighted ? "weighted" : "unweighted"
    filename = "graph_N$(N)_k$(k)_seed$(seed)_seedb$(seed)_$(wtag).txt"
    return joinpath(dir_path, filename)
end

# atomic JSON write: write tmp then rename
function atomic_json_write(path::AbstractString, data)
    mkpath(dirname(path))
    tmp = path * ".tmp"
    open(tmp, "w") do io
        JSON.print(io, data)  # no NaN allowed; we store missing as null
    end
    mv(tmp, path; force=true)
    return nothing
end

# Try to compute bond entanglement entropies (API differs across ITensorMPS versions).
# Returns Vector{Union{Missing,Float64}} so JSON stays valid (missing -> null).
function bond_entropies(ψ::MPS)
    N = length(ψ)
    S = Vector{Union{Missing,Float64}}(undef, max(N - 1, 0))
    for b in 1:(N-1)
        val::Union{Missing,Float64} = missing
        # Try common function names:
        try
            val = Float64(ITensorMPS.entanglement_entropy(ψ, b))
        catch
            try
                val = Float64(ITensorMPS.entropy(ψ, b))
            catch
                val = missing
            end
        end
        S[b] = val
    end
    return S
end

# Two-point correlator ⟨Z_i Z_j⟩
function expect_ZZ(ψ::MPS, sites::Vector{<:Index}, i::Int, j::Int)
    Zi = op("Z", sites, i)
    Zj = op("Z", sites, j)
    ψ2 = apply(Zi, apply(Zj, ψ))
    return real(inner(ψ, ψ2))
end

# Round from ⟨Z⟩ to a classical ±1 spin assignment (tie-break random)
function round_spins_from_mz(mz::AbstractVector{<:Real}; tol::Real=1e-8, rng=Random.default_rng())
    N = length(mz)
    s = Vector{Int}(undef, N)
    for i in 1:N
        if mz[i] > tol
            s[i] = +1
        elseif mz[i] < -tol
            s[i] = -1
        else
            s[i] = rand(rng, Bool) ? +1 : -1
        end
    end
    return s
end

# Compute (weighted) cut value for ±1 spins: cut counts edges with s_i != s_j
function cut_value(wg, s::Vector{Int})
    cut = 0.0
    for e in edges(wg)
        i, j = src(e), dst(e)
        if s[i] != s[j]
            cut += wg.weights[i, j]
        end
    end
    return cut
end

# ---------------------------
# Diagnostic run: χ=2, λ=1.0
# ---------------------------
function main()
    println("========================================================")
    println("  DEBUG χ=2 at λ=1.0 : why devTheta_abs_mean can drop")
    println("  - runs 20 graph seeds, single λ")
    println("========================================================")

    # experiment choices
    N = 50
    k = 3
    weighted = false

    graph_seeds = collect(1:20)

    # DMRG params
    λ = 1.0
    nsweeps = 80
    maxdim  = 2  # χ=2

    # Explicit sweeps (no hidden defaults)
    sw = Sweeps(nsweeps)
    maxdim!(sw, maxdim)
    cutoff!(sw, 0.0)
    noise!(sw, 0.0)

    # save path
    save_dir = joinpath(RESULTS_DIR, "debug_devtheta")
    mkpath(save_dir)
    out_path = joinpath(save_dir,
        "debug_chi2_lambda1_N$(N)_k$(k)_graphs$(length(graph_seeds))_nsw$(nsweeps)_$(weighted ? "weighted" : "unweighted").json"
    )

    per_seed = Vector{Dict{String,Any}}()

    # thresholds for “is it basically ±1?”
    tolZ = 1e-6
    tolZ_loose = 1e-3

    # expected graph stats (unweighted regular)
    m_expected = N * k ÷ 2
    W_expected = weighted ? nothing : float(m_expected)  # 75.0

    for (idx, gs) in enumerate(graph_seeds)
        println("\n► graph seed = $gs  (", idx, " / ", length(graph_seeds), ")")

        gpath = graph_path_qiils(N, k, gs, weighted)
        if !isfile(gpath)
            println("    … graph missing; creating: $gpath")
            create_and_save_graph_QiILS(N, k, gs; weighted=weighted, base_path=GRAPHS_DIR)
            @assert isfile(gpath) "Graph creation failed: $gpath"
        end

        wg = load_graph(path=gpath, weighted=weighted)

        # ---- graph sanity ----
        deg = degree(wg)
        dmin, dmax = minimum(deg), maximum(deg)
        niso = count(==(0), deg)
        nwrong = count(!=(k), deg)
        println("    degree stats: min=$dmin max=$dmax  n_isolated=$niso  n_deg!=k=$nwrong")

        # edges sanity
        m_actual = ne(wg)
        @assert m_actual == m_expected "Edge count mismatch in seed=$gs: ne=$m_actual expected=$m_expected"

        # weight sanity (for unweighted)
        Wtot = total_edge_weight(wg)
        if !weighted
            println("    Wtot = $Wtot (expected $(W_expected))")
            @assert isapprox(Wtot, W_expected; atol=1e-12) "Total weight mismatch in seed=$gs: Wtot=$Wtot expected=$W_expected"
        end

        sites = siteinds("Qubit", nv(wg))
        Hλ = QiILS_ITensor.build_H_mpo(wg, sites, Float64(λ); weighted=weighted)
        H1 = QiILS_ITensor.build_H_mpo(wg, sites, 1.0; weighted=weighted)

        ψ0 = minus_product_mps(sites)

        # run DMRG
        Eλ, ψ = dmrg(Hλ, ψ0, sw; outputlevel=0)

        # Ensure χ≤2 held (could be <2 if it collapses)
        ld = linkdims(ψ)
        max_ld = isempty(ld) ? 1 : maximum(ld)
        @assert max_ld ≤ 2 "Expected χ≤2 but max linkdim = $max_ld"
        mean_ld = isempty(ld) ? 1.0 : mean(Float64.(ld))

        # energies
        E1 = energy_expectation(H1, ψ)
        ΔE = abs(Eλ - E1)

        # locals
        mz = local_expect_Z(ψ, sites)
        mx = local_expect_X(ψ, sites)

        abs_mz = abs.(mz)
        abs_mx = abs.(mx)

        mean_absZ = mean(abs_mz)
        min_absZ  = minimum(abs_mz)
        max_absZ  = maximum(abs_mz)

        mean_absX = mean(abs_mx)
        min_absX  = minimum(abs_mx)
        max_absX  = maximum(abs_mx)

        # angle metric
        θeff = 0.5 .* acos.(clamp.(abs_mz, 0.0, 1.0))
        devTheta_abs = mean(abs.(θeff .- (π/4)))

        # “classicality”
        n_strict = count(z -> (1.0 - z) ≤ tolZ, abs_mz)
        n_loose  = count(z -> (1.0 - z) ≤ tolZ_loose, abs_mz)
        frac_strict = n_strict / length(abs_mz)
        frac_loose  = n_loose  / length(abs_mz)

        target = π/4
        delta_from_target = devTheta_abs - target

        # entanglement (if available)
        S = bond_entropies(ψ)
        Sfinite = [Float64(x) for x in S if x !== missing]
        Smax  = isempty(Sfinite) ? missing : maximum(Sfinite)
        Smean = isempty(Sfinite) ? missing : mean(Sfinite)

        # cut mapping from energy
        cut_hat = (Wtot - E1)/2

        # classical rounding from mz
        rng = MersenneTwister(9999 + gs)
        s_round = round_spins_from_mz(mz; tol=1e-8, rng=rng)
        cut_round = cut_value(wg, s_round)

        println(@sprintf("    linkdims: max=%d mean=%.3f", max_ld, mean_ld))
        if Smax === missing
            println("    entanglement: (not available in this ITensorMPS version)")
        else
            println(@sprintf("    entanglement: Smax=%.6f  Smean=%.6f", Float64(Smax), Float64(Smean)))
        end
        println(@sprintf("    mean|Z|=%.6f  min|Z|=%.6f  max|Z|=%.6f", mean_absZ, min_absZ, max_absZ))
        println(@sprintf("    mean|X|=%.6f  min|X|=%.6f  max|X|=%.6f", mean_absX, min_absX, max_absX))
        println(@sprintf("    devTheta_abs=%.6f   target π/4=%.6f   (dev-target)=%.6f",
                         devTheta_abs, target, delta_from_target))
        println(@sprintf("    frac near-classical: strict=%.2f loose=%.2f", frac_strict, frac_loose))
        println(@sprintf("    E(λ)=%.6f  E(1)=%.6f  |Eλ-E1|=%.3e", Eλ, E1, ΔE))
        println(@sprintf("    cut_hat(from E1)=%.1f   cut_round(from mz)=%.1f", cut_hat, cut_round))

        # ---- forensic dump for bad sites + ZZ correlators on incident edges ----
        bad = findall(z -> z < 1.0 - 1e-12, abs_mz)
        bad_edge_ZZ = Vector{Dict{String,Any}}()
        if !isempty(bad)
            println("    nonclassical sites (|Z|<1): ", bad)
            for i in bad
                nbrs = neighbors(wg, i)
                println("      i=$i  Z=$(mz[i]) |Z|=$(abs_mz[i])   X=$(mx[i]) |X|=$(abs_mx[i])   deg=$(degree(wg, i))  nbrs=$(nbrs)")

                hi = 0.0
                println("        incident weights, neighbor Z, and <ZiZj>:")
                for j in nbrs
                    wij = wg.weights[i, j]
                    zz  = expect_ZZ(ψ, sites, i, j)
                    hi += wij * mz[j]
                    println("          j=$j  w_ij=$(wij)  Zj=$(mz[j])  <ZiZj>=$(zz)")
                    push!(bad_edge_ZZ, Dict(
                        "i" => i, "j" => j,
                        "w_ij" => wij,
                        "Zi" => mz[i], "Zj" => mz[j],
                        "ZiZj" => zz,
                    ))
                end
                println("        hi = Σ w_ij * Zj = $hi")
            end
        end

        push!(per_seed, Dict(
            "graph_seed" => gs,
            "lambda" => λ,
            "N" => N,
            "k" => k,
            "weighted_flag" => weighted,
            "dmrg_nsweeps" => nsweeps,
            "dmrg_maxdim_cap" => maxdim,

            "degree_min" => dmin,
            "degree_max" => dmax,
            "n_isolated" => niso,
            "n_deg_not_k" => nwrong,
            "ne" => m_actual,
            "ne_expected" => m_expected,
            "Wtot" => Wtot,
            "Wtot_expected_unweighted" => (weighted ? missing : W_expected),

            "E_lambda" => Eλ,
            "E1_cost" => E1,
            "abs_Elambda_minus_E1" => ΔE,
            "cut_hat" => cut_hat,
            "cut_round" => cut_round,

            "mean_absZ" => mean_absZ,
            "min_absZ" => min_absZ,
            "max_absZ" => max_absZ,

            "mean_absX" => mean_absX,
            "min_absX" => min_absX,
            "max_absX" => max_absX,

            "devTheta_abs" => devTheta_abs,
            "devTheta_target_pi_over_4" => target,
            "devTheta_minus_target" => delta_from_target,

            "n_sites_near_classical_strict" => n_strict,
            "n_sites_near_classical_loose" => n_loose,
            "frac_sites_near_classical_strict" => frac_strict,
            "frac_sites_near_classical_loose" => frac_loose,

            "n_sites" => length(abs_mz),
            "linkdims" => ld,
            "max_linkdim" => max_ld,
            "mean_linkdim" => mean_ld,

            "bond_entropies" => S,
            "Smax" => Smax,
            "Smean" => Smean,

            "bad_sites_absZ_lt_1" => bad,
            "bad_edges_ZZ" => bad_edge_ZZ,
        ))
    end

    # aggregate summary
    devs = [Float64(d["devTheta_abs"]) for d in per_seed]
    mean_dev = mean(devs)
    std_dev  = std(devs)

    mean_absZs = [Float64(d["mean_absZ"]) for d in per_seed]
    mean_absXs = [Float64(d["mean_absX"]) for d in per_seed]
    ΔEs = [Float64(d["abs_Elambda_minus_E1"]) for d in per_seed]
    cuts_hat = [Float64(d["cut_hat"]) for d in per_seed]
    cuts_round = [Float64(d["cut_round"]) for d in per_seed]

    # entanglement aggregates (skip missing)
    Smaxs = Float64[]
    Smeans = Float64[]
    for d in per_seed
        smax = d["Smax"]
        smean = d["Smean"]
        if smax !== missing
            push!(Smaxs, Float64(smax))
        end
        if smean !== missing
            push!(Smeans, Float64(smean))
        end
    end

    summary = Dict(
        "what" => "debug χ=2, λ=1.0 devTheta_abs vs π/4 (adds entanglement, ZZ correlators on bad edges, and rounding)",
        "timestamp_utc" => Dates.format(Dates.now(Dates.UTC), Dates.ISODateTimeFormat),
        "N" => N,
        "k" => k,
        "weighted_flag" => weighted,
        "graph_seeds" => graph_seeds,
        "dmrg_nsweeps" => nsweeps,
        "dmrg_maxdim_cap" => maxdim,
        "lambda" => λ,
        "target_pi_over_4" => π/4,

        "mean_devTheta_abs_over_seeds" => mean_dev,
        "std_devTheta_abs_over_seeds" => std_dev,
        "mean_meanAbsZ_over_seeds" => mean(mean_absZs),
        "mean_meanAbsX_over_seeds" => mean(mean_absXs),
        "mean_abs_Elambda_minus_E1_over_seeds" => mean(ΔEs),
        "max_abs_Elambda_minus_E1_over_seeds" => maximum(ΔEs),

        "mean_cut_hat_over_seeds" => mean(cuts_hat),
        "mean_cut_round_over_seeds" => mean(cuts_round),

        "mean_Smax_over_seeds" => (isempty(Smaxs) ? missing : mean(Smaxs)),
        "mean_Smean_over_seeds" => (isempty(Smeans) ? missing : mean(Smeans)),

        "per_seed" => per_seed,
    )

    atomic_json_write(out_path, summary)
    println("\n✔ Saved debug summary → ", out_path)
end

main()