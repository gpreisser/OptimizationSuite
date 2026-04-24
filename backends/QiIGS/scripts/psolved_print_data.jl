# scripts/print_panel_c_psolved.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using JSON
using Printf

# ------------------------------------------------------------
# CHANGE THIS PATH IF NEEDED
# ------------------------------------------------------------
const JSON_PATH = "/Users/guillermo.preisser/Projects/QiIGS/results/qiigs_unique_minima_N50_k3_graphs100_unweighted/qiigs_unique_ratio_meanbest_succ_grad_lam0p000_to_1p000_d0p025_conv1e-08_abin0p01_hess_on_htol1e-08_ngraphs100_ninit10000_outer1_inner5000_thr0p999.json"

# ------------------------------------------------------------
# helper
# ------------------------------------------------------------
function getnum(entry, key::AbstractString)
    haskey(entry, key) || return NaN
    v = entry[key]
    (v === nothing || v == "none") && return NaN
    return Float64(v)
end

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
function main()
    println("Using file:")
    println(JSON_PATH)
    println()

    data = JSON.parsefile(JSON_PATH)
    rpl = data["results_per_lambda"]

    println("============================================================")
    println("PANEL (c): P_solved (mean)")
    println("============================================================")
    println("Columns:")
    println("λ, success_rate_mean, success_rate_stderr")
    println("------------------------------------------------------------")

    for entry in rpl
        λ  = getnum(entry, "λ")
        ps = getnum(entry, "success_rate_mean")
        pe = getnum(entry, "success_rate_stderr")

        if isfinite(λ) && isfinite(ps)
            @printf("%.4f  %.8f  %.8f\n", λ, ps, pe)
        end
    end

    println("============================================================")

    # --------------------------------------------------------
    # also print focused window (same as your panel b focus)
    # --------------------------------------------------------
    println()
    println("============================================================")
    println("FOCUS WINDOW: 0.15 ≤ λ ≤ 0.40")
    println("============================================================")

    for entry in rpl
        λ  = getnum(entry, "λ")
        ps = getnum(entry, "success_rate_mean")
        pe = getnum(entry, "success_rate_stderr")

        if isfinite(λ) && 0.15 <= λ <= 0.40
            @printf("λ = %.4f | P_solved = %.8f ± %.8f\n", λ, ps, pe)
        end
    end

    println("============================================================")

    # --------------------------------------------------------
    # step-to-step changes (very useful)
    # --------------------------------------------------------
    println()
    println("============================================================")
    println("STEP-TO-STEP CHANGES in P_solved")
    println("============================================================")

    λs = Float64[]
    ps = Float64[]

    for entry in rpl
        λ  = getnum(entry, "λ")
        p  = getnum(entry, "success_rate_mean")
        if isfinite(λ) && isfinite(p)
            push!(λs, λ)
            push!(ps, p)
        end
    end

    for i in 2:length(λs)
        @printf("λ: %.4f -> %.4f | ΔP_solved = %+ .8f\n",
            λs[i-1], λs[i],
            ps[i] - ps[i-1])
    end

    println("============================================================")
end

main()