# scripts/debug_measure.jl
#
# julia --project=. scripts/debug_measure.jl
#
# Debug checks per λ:
#  - Δθ_max vs tolerance for first few sweeps
#  - dev_meanabs + detailed δ=θ-π/4 stats
#  - measurement stochasticity (measure twice from same θ)
#
# IMPORTANT: we call QiILS.sweep_pass! (qualified), not sweep_pass! in Main.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using QiILS
using Graphs
using Statistics
using Printf

const PI_OVER_4 = π/4

# --- adjust if your graph naming differs ---
@inline function graph_path_qiils(graphs_base::AbstractString, N::Int, k::Int, seed::Int)
    dir_path = joinpath(graphs_base, string(N), string(k))
    filename = "graph_N$(N)_k$(k)_seed$(seed)_seedb$(seed).txt"
    return joinpath(dir_path, filename)
end

function ensure_graph!(graphs_base::AbstractString, N::Int, k::Int, seed::Int; weighted::Bool=false)
    gpath = graph_path_qiils(graphs_base, N, k, seed)
    if !isfile(gpath)
        println("… graph missing; creating: $gpath")
        QiILS.create_and_save_graph_QiILS(N, k, seed; weighted=weighted, base_path=graphs_base)
        @assert isfile(gpath) "Graph creation failed; expected file not found: $gpath"
        println("✔ created graph file: $gpath")
    else
        println("✔ graph file exists: $gpath")
    end
    return gpath
end

function measure_cut_twice(wg, θ::Vector{Float64})
    θ_meas1 = QiILS.finaltheta(θ)
    spins1  = QiILS.angles_to_spins(θ_meas1)
    cut1    = QiILS.maxcut_value(wg, spins1)

    θ_meas2 = QiILS.finaltheta(θ)
    spins2  = QiILS.angles_to_spins(θ_meas2)
    cut2    = QiILS.maxcut_value(wg, spins2)

    return cut1, cut2, (spins1 == spins2)
end

function qiils_minimize_then_measure_debug(
    wg,
    λ::Float64,
    gvec::Vector{Float64};
    θ0::Union{Nothing,Vector{Float64}} = nothing,
    sweeps::Int = 80,
    angle_conv::Float64 = 1e-20,
    use_scaled_convergence::Bool = true,
    print_first_sweeps::Int = 5,
)
    N = nv(wg)

    θ = isnothing(θ0) ? fill(PI_OVER_4, N) : copy(θ0)

    θ_old = similar(θ)
    cos2θ = cos.(2 .* θ)
    sin2θ = sin.(2 .* θ)

    total_sweeps_done = 0
    last_Δθ_max = NaN
    last_scaled_tol = NaN

    for sweep in 1:sweeps
        θ_old .= θ

        # ✅ QUALIFIED CALL
        QiILS.sweep_pass!(N, wg, λ, θ, gvec, cos2θ, sin2θ)

        total_sweeps_done += 1

        Δθ_max = maximum(abs.(θ .- θ_old))
        last_Δθ_max = Δθ_max

        if use_scaled_convergence
            scaled_tol = max(angle_conv * mean(abs.(θ .- PI_OVER_4)), 1e-12)
            last_scaled_tol = scaled_tol
            if sweep ≤ print_first_sweeps
                @printf("    sweep %d: Δθ_max=%.3e  scaled_tol=%.3e\n", sweep, Δθ_max, scaled_tol)
            end
            Δθ_max < scaled_tol && break
        else
            if sweep ≤ print_first_sweeps
                @printf("    sweep %d: Δθ_max=%.3e  angle_conv=%.3e\n", sweep, Δθ_max, angle_conv)
            end
            Δθ_max < angle_conv && break
        end
    end

    δ = θ .- PI_OVER_4
    dev_meanabs = mean(abs.(δ))

    δ_min     = minimum(δ)
    δ_max     = maximum(δ)
    δ_maxabs  = maximum(abs.(δ))
    frac_pos  = mean(δ .> 0.0)
    min_margin = minimum(abs.(δ))

    cut1, cut2, same_spins = measure_cut_twice(wg, θ)

    return (; cut1, cut2, same_spins,
            dev_meanabs, δ_min, δ_max, δ_maxabs, frac_pos, min_margin,
            total_sweeps_done, last_Δθ_max, last_scaled_tol)
end

function main()
    println("====================================================")
    println("  Debug: dev vs cut changes (QiILS)")
    println("====================================================")

    N = 50
    k = 3
    weighted = false
    graph_seed = 1

    λs = collect(0.0:0.1:1.0)

    max_sweeps = 80
    angle_conv = 1e-20
    use_scaled_convergence = true

    graphs_base = joinpath(@__DIR__, "..", "graphs")

    println("► graph seed = $graph_seed")
    gpath = ensure_graph!(graphs_base, N, k, graph_seed; weighted=weighted)
    wg = QiILS.load_graph(path=gpath, weighted=weighted)
    println("    nv(wg) = ", nv(wg), ", ne(wg) = ", ne(wg))

    for λ in λs
        println("\nλ=$(round(λ, digits=3))")

        # fresh per λ so nothing carries over
        gvec = zeros(Float64, nv(wg))

        out = qiils_minimize_then_measure_debug(
            wg, λ, gvec;
            θ0 = nothing,
            sweeps = max_sweeps,
            angle_conv = angle_conv,
            use_scaled_convergence = use_scaled_convergence,
            print_first_sweeps = 5,
        )

        @printf("    dev_meanabs=%.3e | δ_min=%.3e δ_max=%.3e δ_maxabs=%.3e\n",
                out.dev_meanabs, out.δ_min, out.δ_max, out.δ_maxabs)
        @printf("    frac(θ>π/4)=%.3f | min_margin_to_boundary=%.3e\n",
                out.frac_pos, out.min_margin)

        println("    measure twice: cut1=$(out.cut1), cut2=$(out.cut2), spins_equal=$(out.same_spins)")
        @printf("    sweeps_done=%d | last Δθ_max=%.3e | last_scaled_tol=%.3e\n",
                out.total_sweeps_done, out.last_Δθ_max, out.last_scaled_tol)

        if (!out.same_spins) || (out.cut1 != out.cut2)
            println("    ⚠ Measurement appears STOCHASTIC (same θ -> different spins/cut).")
        end
        if out.min_margin < 1e-14
            println("    ⚠ Angles extremely close to π/4 boundary; tiny noise can flip spins.")
        end
    end

    println("\nDone.")
end

# ✅ Avoid Julia 1.12 world-age / Revise weirdness in scripts
Base.invokelatest(main)