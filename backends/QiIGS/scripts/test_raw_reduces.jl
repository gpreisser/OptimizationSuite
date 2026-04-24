# scripts/debug_angles_raw_vs_reduced_simple.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using QiIGS
using Printf

# ------------------------------------------------------------
# Angle-key hashing
# ------------------------------------------------------------

@inline wrap_pi(x::Float64) = mod(x, pi)

@inline function quantize_angle(x::Float64, δ::Float64)
    return Int32(floor(x / δ + 0.5))
end

@inline function fnv1a64_combine(h::UInt64, x::UInt64)
    h ⊻= x
    return h * 0x100000001b3
end

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

    q1 = qvec_for(pi/2)
    return min(q0, q1)   # lexicographic canonical representative
end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

function main()
    N = 50
    k = 3
    weighted = false

    graph_seeds = 1:100
    λs = 0.0:0.1:1.0

    n_inits = 10_000

    iterations = 1
    inner_iterations = 5000
    tao = 0.1
    angle_conv = 0.01
    init_mode = :uniform
    save_params = true

    angle_bin = 2e-2

    ROOT_QIILS = "/Users/guillermo.preisser/Projects/QiILS_ITensor"
    GRAPHS_ROOT = joinpath(ROOT_QIILS, "graphs")

    println("===================================================")
    println("DEBUG raw vs reduced angle counts")
    println("Stops at first impossible event: reduced > raw")
    println("===================================================")

    for gs in graph_seeds
        println("\n► graph seed = $gs")

        gpath = QiIGS.graph_path(N, k, gs; weighted=weighted, graphs_root=GRAPHS_ROOT)
        W = QiIGS.load_weight_matrix(gpath)

        for λ in λs
            println("   λ = $(round(λ, digits=4))")

            seen_angles_raw = Set{NTuple{50,Int32}}()
            seen_angles_reduced = Set{NTuple{50,Int32}}()

            for r in 1:n_inits
                run_seed = gs * 1_000_000 + r * 10_000 + Int(round(λ * 1000))

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

                θc = get(sol.metadata, :theta_converged, Float64[])
                isempty(θc) && error("Missing :theta_converged")

                kraw = angle_key(θc; δ=angle_bin, flip_equiv=false)
                kred = angle_key(θc; δ=angle_bin, flip_equiv=true)

                push!(seen_angles_raw, kraw)
                push!(seen_angles_reduced, kred)

                nraw = length(seen_angles_raw)
                nred = length(seen_angles_reduced)

                if r == 1 || r % 1000 == 0
                    @printf("      init=%d   raw=%d   reduced=%d\n", r, nraw, nred)
                end

                if nred > nraw
                    println("\n===================================")
                    println("IMPOSSIBLE EVENT FOUND")
                    println("===================================")
                    println("graph_seed = $gs")
                    println("λ          = $λ")
                    println("init       = $r")
                    println("run_seed   = $run_seed")
                    println("raw        = $nraw")
                    println("reduced    = $nred")
                    println("angle_bin  = $angle_bin")

                    println("\nCurrent θ first 12:")
                    println(round.(θc[1:min(12, end)], digits=6))

                    println("\nCurrent raw key:")
                    println(kraw)

                    println("\nCurrent reduced key:")
                    println(kred)

                    return
                end
            end

            @printf("   final at λ=%.4f   raw=%d   reduced=%d\n",
                    λ, length(seen_angles_raw), length(seen_angles_reduced))
        end
    end

    println("\nNo impossible event found.")
end

main()