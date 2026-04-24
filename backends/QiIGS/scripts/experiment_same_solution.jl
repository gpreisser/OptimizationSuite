# scripts/inspect_transition_angles_same_spin.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using QiIGS
using Printf
using Statistics

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

function angle_key(θ::AbstractVector{<:AbstractFloat}; δ::Float64=1e-3, flip_equiv::Bool=false)
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
    return min(q0, q1)
end

spin_key_from_config(config) = QiIGS.spin_config_key(config)

# distance on circle modulo pi
@inline function circdist_pi(a::Float64, b::Float64)
    d = mod(a - b, pi)
    return min(d, pi - d)
end

function mean_circdist_pi(θ1::AbstractVector, θ2::AbstractVector)
    @assert length(θ1) == length(θ2)
    return mean(circdist_pi(Float64(θ1[i]), Float64(θ2[i])) for i in eachindex(θ1))
end

function mean_circdist_pi_shift(θ1::AbstractVector, θ2::AbstractVector, shift::Float64)
    @assert length(θ1) == length(θ2)
    return mean(circdist_pi(Float64(θ1[i]), Float64(θ2[i]) + shift) for i in eachindex(θ1))
end

function summary_compare(θ1, θ2)
    d0   = mean_circdist_pi(θ1, θ2)
    d90  = mean_circdist_pi_shift(θ1, θ2, Float64(pi/2))
    d180 = mean_circdist_pi_shift(θ1, θ2, Float64(pi))
    return d0, d90, d180
end

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

function main()
    N = 50
    k = 3
    weighted = false

    graph_seed = 1
    λ = 0.275
    n_inits = 2000

    iterations = 1
    inner_iterations = 5000
    tao = 0.1
    angle_conv = 1e-8
    init_mode = :uniform
    save_params = true

    angle_bin = 1e-2

    compute_hessian = false
    hessian_tol = 1e-8

    ROOT_QIILS = "/Users/guillermo.preisser/Projects/QiILS_ITensor"
    GRAPHS_ROOT = joinpath(ROOT_QIILS, "graphs")
    SOLUTIONS_ROOT = joinpath(ROOT_QIILS, "solutions")

    gpath = QiIGS.graph_path(N, k, graph_seed; weighted=weighted, graphs_root=GRAPHS_ROOT)
    @assert isfile(gpath) "Graph file not found: $gpath"
    W = QiIGS.load_weight_matrix(gpath)

    spath = QiIGS.akmax_solution_path(N, k, graph_seed; weighted=weighted, solutions_root=SOLUTIONS_ROOT)
    opt = QiIGS.load_optimal_cut(spath)
    opt === nothing && error("No optimal cut found at $spath")

    # spin_key => angle_raw_key => representative theta
    grouped = Dict{UInt64, Dict{Any, Vector{Float64}}}()

    # also store counts
    counts = Dict{UInt64, Dict{Any, Int}}()

    for r in 1:n_inits
        seed = graph_seed * 1_000_000 + r * 10_000 + Int(round(λ * 1000))

        sol = QiIGS.solve!(
            W, N;
            solver = :grad,
            seed = seed,
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

        θc = get(sol.metadata, :theta_converged, Float64[])
        isempty(θc) && error("Missing :theta_converged in metadata")

        skey = spin_key_from_config(sol.configuration)
        akey = angle_key(θc; δ=angle_bin, flip_equiv=false)

        if !haskey(grouped, skey)
            grouped[skey] = Dict{Any, Vector{Float64}}()
            counts[skey] = Dict{Any, Int}()
        end

        if !haskey(grouped[skey], akey)
            grouped[skey][akey] = collect(Float64.(θc))
            counts[skey][akey] = 0
        end
        counts[skey][akey] += 1
    end

    # sort spin groups by total count
    spin_totals = [(skey, sum(values(counts[skey]))) for skey in keys(counts)]
    sort!(spin_totals, by = x -> -x[2])

    println("======================================================")
    @printf("Inspecting λ = %.4f, graph seed = %d\n", λ, graph_seed)
    println("======================================================")

    for (skey, total_runs) in spin_totals
        akeys = collect(keys(grouped[skey]))
        nang = length(akeys)

        @printf("\nspin_key = %s, total_runs = %d, n_distinct_angle_minima = %d\n",
            string(skey), total_runs, nang)

        for (j, ak) in enumerate(akeys)
            @printf("  angle minimum %d: count = %d\n", j, counts[skey][ak])
        end

        if nang >= 2
            println("  Pairwise comparisons:")
            for i in 1:nang-1
                for j in i+1:nang
                    θ1 = grouped[skey][akeys[i]]
                    θ2 = grouped[skey][akeys[j]]
                    d0, d90, d180 = summary_compare(θ1, θ2)

                    @printf("    (%d,%d): mean dist = %.6e,  shift π/2 = %.6e,  shift π = %.6e\n",
                        i, j, d0, d90, d180)

                    println("    first 10 angles of solution ", i, ":")
                    println("      ", join([@sprintf("%.6f", x) for x in θ1[1:10]], ", "))
                    println("    first 10 angles of solution ", j, ":")
                    println("      ", join([@sprintf("%.6f", x) for x in θ2[1:10]], ", "))
                end
            end
        end
    end
end

main()