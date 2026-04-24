# scripts/inspect_two_minima_case.jl
#
# Purpose:
#   Inspect the first two representative minima found for one graph and one λ,
#   before doing any NEB.
#
# What it does:
#   - loads the one-graph minima-scan JSON
#   - extracts the first two reduced minima at the chosen λ
#   - reloads the graph
#   - compares the two minima in angle space and spin space
#   - recomputes gradient/Hessian diagnostics
#   - prints a compact summary
#
# Notes:
#   - This script assumes the minima-scan JSON already contains
#     "results_per_lambda" with "reduced_minima"
#   - It uses the continuous angle vectors saved as representative_theta
#   - No plotting yet

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using QiIGS
using JSON
using Printf
using LinearAlgebra
using SparseArrays

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

function theta_distance_modpi(
    θ1::AbstractVector{<:AbstractFloat},
    θ2::AbstractVector{<:AbstractFloat},
)
    @assert length(θ1) == length(θ2)
    s = 0.0
    @inbounds for i in eachindex(θ1)
        d = abs(mod(Float64(θ1[i]) - Float64(θ2[i]), pi))
        d = min(d, pi - d)
        s += d^2
    end
    return sqrt(s)
end

function theta_distance_modpi_shifted(
    θ1::AbstractVector{<:AbstractFloat},
    θ2::AbstractVector{<:AbstractFloat},
    shift::Float64,
)
    @assert length(θ1) == length(θ2)
    s = 0.0
    @inbounds for i in eachindex(θ1)
        d = abs(mod(Float64(θ1[i]) - (Float64(θ2[i]) + shift), pi))
        d = min(d, pi - d)
        s += d^2
    end
    return sqrt(s)
end

function hamming_distance(s1::AbstractVector{<:Integer}, s2::AbstractVector{<:Integer})
    @assert length(s1) == length(s2)
    c = 0
    @inbounds for i in eachindex(s1)
        c += (s1[i] != s2[i])
    end
    return c
end

function spin_overlap(s1::AbstractVector{<:Integer}, s2::AbstractVector{<:Integer})
    @assert length(s1) == length(s2)
    acc = 0.0
    @inbounds for i in eachindex(s1)
        acc += Float64(s1[i]) * Float64(s2[i])
    end
    return acc / length(s1)
end

# Continuous objective consistent with your gradient:
#
#   grad_i = -2 λ a_i sin(2θ_i) - 2 (1-λ) cos(2θ_i),
#   a_i = Σ_j W_ij cos(2θ_j)
#
# One compatible energy is:
#
#   F(θ; λ) = λ Σ_{i<j} W_ij cos(2θ_i) cos(2θ_j) - (1-λ) Σ_i sin(2θ_i)
#
function continuous_energy(
    W::SparseMatrixCSC{T},
    θ::AbstractVector{T},
    λ::T,
) where {T<:AbstractFloat}
    N = length(θ)
    @assert size(W, 1) == N && size(W, 2) == N

    c2 = Vector{T}(undef, N)
    s2 = Vector{T}(undef, N)
    @inbounds for i in 1:N
        c2[i] = cos(2 * θ[i])
        s2[i] = sin(2 * θ[i])
    end

    Epair = zero(T)
    @inbounds for col in 1:N
        for idx in W.colptr[col]:(W.colptr[col + 1] - 1)
            row = W.rowval[idx]
            if row < col
                Epair += W.nzval[idx] * c2[row] * c2[col]
            end
        end
    end

    Efield = zero(T)
    @inbounds for i in 1:N
        Efield += s2[i]
    end

    return λ * Epair - (one(T) - λ) * Efield
end

function get_result_for_lambda(results_per_lambda, λtarget::Float64; atol::Float64=1e-12)
    for entry in results_per_lambda
        λ = Float64(entry["lambda"])
        if isapprox(λ, λtarget; atol=atol, rtol=0.0)
            return entry
        end
    end
    error("No entry found for λ = $λtarget")
end

function vector_float(x)
    return Float64.(x)
end

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

function main()
    println("==============================================================")
    println("  Inspect two minima for one graph and one λ")
    println("==============================================================")

    # -----------------------
    # User choices
    # -----------------------
    N = 50
    k = 3
    weighted = false
    graph_seed = 1
    λtarget = 0.275

    # Path to the minima-scan JSON you just created
    ROOT = normpath(joinpath(@__DIR__, ".."))
    MINSCAN_DIR = joinpath(ROOT, "results", "minima_scan_onegraph")

    mins_path = joinpath(
        MINSCAN_DIR,
        "onegraph_minima_scan_N50_k3_seed1_unweighted_lam0p000_to_1p000_d0p025_conv1e-08_abin0p01_afliptrue_hess_on_htol1e-08_gfiltertrue_gnmax1e-06_ninit2000_outer1_inner5000.json"
    )

    ROOT_QIILS = "/Users/guillermo.preisser/Projects/QiILS_ITensor"
    GRAPHS_ROOT = joinpath(ROOT_QIILS, "graphs")

    # -----------------------
    # Load minima scan
    # -----------------------
    @assert isfile(mins_path) "Minima-scan JSON not found: $mins_path"
    data = JSON.parsefile(mins_path)

    results = data["results_per_lambda"]
    entry = get_result_for_lambda(results, λtarget)

    red = entry["reduced_minima"]
    nred = length(red)

    println("Minima-scan path: $mins_path")
    @printf("Requested λ = %.4f\n", λtarget)
    @printf("Reduced minima found at this λ = %d\n", nred)

    nred >= 2 || error("Need at least two reduced minima at λ = $λtarget")

    m1 = red[1]
    m2 = red[2]

    θ1 = vector_float(m1["representative_theta"])
    θ2 = vector_float(m2["representative_theta"])

    # -----------------------
    # Load graph
    # -----------------------
    gpath = QiIGS.graph_path(N, k, graph_seed; weighted=weighted, graphs_root=GRAPHS_ROOT)
    @assert isfile(gpath) "Graph file not found: $gpath"
    W = QiIGS.load_weight_matrix(gpath)
    Wsp = (W isa SparseMatrixCSC) ? SparseMatrixCSC{Float64, Int}(W) : sparse(Float64.(W))

    # -----------------------
    # Recompute diagnostics
    # -----------------------
    gn1 = QiIGS.grad_norm(θ1, λtarget, Wsp)
    gn2 = QiIGS.grad_norm(θ2, λtarget, Wsp)

    hs1 = QiIGS.hessian_summary(Wsp, θ1, λtarget; tol=1e-8)
    hs2 = QiIGS.hessian_summary(Wsp, θ2, λtarget; tol=1e-8)

    s1 = QiIGS.round_configuration(θ1)
    s2 = QiIGS.round_configuration(θ2)

    Eround1 = QiIGS.energy_from_spins(Wsp, s1)
    Eround2 = QiIGS.energy_from_spins(Wsp, s2)

    Econt1 = continuous_energy(Wsp, θ1, λtarget)
    Econt2 = continuous_energy(Wsp, θ2, λtarget)

    dθ = theta_distance_modpi(θ1, θ2)
    dθ_shift = theta_distance_modpi_shifted(θ1, θ2, pi / 2)

    ham = hamming_distance(s1, s2)
    ov = spin_overlap(s1, s2)

    # -----------------------
    # Print summary
    # -----------------------
    println()
    println("Graph path:")
    println(gpath)

    println()
    println("Top two reduced minima from saved scan:")
    @printf("  min #1  count = %d  saved_gn = %.6e  saved_mineig = %.6e  saved_Eround = %.6f\n",
        Int(m1["count"]),
        Float64(m1["best_grad_norm"]),
        Float64(m1["best_hess_mineig"]),
        Float64(m1["rounded_energy"]),
    )
    @printf("  min #2  count = %d  saved_gn = %.6e  saved_mineig = %.6e  saved_Eround = %.6f\n",
        Int(m2["count"]),
        Float64(m2["best_grad_norm"]),
        Float64(m2["best_hess_mineig"]),
        Float64(m2["rounded_energy"]),
    )

    println()
    println("Recomputed diagnostics:")
    @printf("  continuous energy min #1 = %.12f\n", Econt1)
    @printf("  continuous energy min #2 = %.12f\n", Econt2)
    @printf("  ΔE_cont = %.12e\n", Econt2 - Econt1)

    @printf("  rounded energy min #1    = %.12f\n", Eround1)
    @printf("  rounded energy min #2    = %.12f\n", Eround2)
    @printf("  ΔE_round = %.12e\n", Eround2 - Eround1)

    println()
    @printf("  grad norm min #1         = %.12e\n", gn1)
    @printf("  grad norm min #2         = %.12e\n", gn2)

    println()
    @printf("  Hessian min #1: mineig = %.12e, maxeig = %.12e, cond = %.12e, is_minimum = %s\n",
        Float64(hs1[:hess_mineig]),
        Float64(hs1[:hess_maxeig]),
        Float64(hs1[:hess_cond]),
        string(hs1[:hess_is_minimum]),
    )
    @printf("  Hessian min #2: mineig = %.12e, maxeig = %.12e, cond = %.12e, is_minimum = %s\n",
        Float64(hs2[:hess_mineig]),
        Float64(hs2[:hess_maxeig]),
        Float64(hs2[:hess_cond]),
        string(hs2[:hess_is_minimum]),
    )

    println()
    println("Pairwise comparison:")
    @printf("  angle distance mod π           = %.12f\n", dθ)
    @printf("  angle distance after +π/2 shift= %.12f\n", dθ_shift)
    @printf("  Hamming distance of rounded spins = %d / %d\n", ham, length(s1))
    @printf("  spin overlap                      = %.12f\n", ov)

    println()
    println("First 12 rounded spins:")
    println("  s1 = ", collect(s1[1:min(end, 12)]))
    println("  s2 = ", collect(s2[1:min(end, 12)]))

    println()
    println("First 12 angles:")
    println("  θ1 = ", θ1[1:min(end, 12)])
    println("  θ2 = ", θ2[1:min(end, 12)])

    println()
    println("==============================================================")
    println("Interpretation guide:")
    println("  - If both Hessians are minima and grad norms are small,")
    println("    these are good endpoint candidates.")
    println("  - If ΔE_cont ≈ 0, the pair is nearly degenerate in the")
    println("    continuous landscape.")
    println("  - If Hamming distance > 0, they correspond to different")
    println("    rounded spin states.")
    println("==============================================================")
end

main()