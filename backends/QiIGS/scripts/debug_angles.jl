# scripts/debug_angle_raw_vs_reduced.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using QiIGS
using Printf

N = 50
k = 3
weighted = false
graph_seed = 4

λ = 0.8
n_inits = 10000

iterations = 1
inner_iterations = 5000
tao = 0.1
angle_conv = 0.01
init_mode = :uniform
save_params = true

angle_bin = 1e-2

ROOT_QIILS = "/Users/guillermo.preisser/Projects/QiILS_ITensor"
GRAPHS_ROOT = joinpath(ROOT_QIILS, "graphs")

@inline function canonical_angle(x::Float64)
    y = mod(x, pi)
    if abs(y) < 1e-12 || abs(y - pi) < 1e-12
        return 0.0
    end
    if abs(y - pi/2) < 1e-12
        return pi/2
    end
    return y
end

@inline function quantize_angle(x::Float64, δ::Float64)
    Int32(round(canonical_angle(x) / δ))
end

function angle_key(θ::AbstractVector{<:AbstractFloat}; δ::Float64=1e-2, flip_equiv::Bool=true)
    function qvec_for(shift::Float64)
        q = Vector{Int32}(undef, length(θ))
        @inbounds for i in eachindex(θ)
            q[i] = quantize_angle(Float64(θ[i]) + shift, δ)
        end
        return q
    end

    q0 = qvec_for(0.0)
    if !flip_equiv
        return Tuple(q0)
    end

    q1 = qvec_for(pi/2)
    return Tuple(q1 < q0 ? q1 : q0)   # lexicographic canonical representative
end

function main()
    gpath = QiIGS.graph_path(N, k, graph_seed; weighted=weighted, graphs_root=GRAPHS_ROOT)
    W = QiIGS.load_weight_matrix(gpath)

    seen_raw = Dict{Any,Tuple{Int,Int,Vector{Float64}}}()
    seen_red = Dict{Any,Tuple{Int,Int,Vector{Float64}}}()

    for r in 1:n_inits
        run_seed = graph_seed * 1_000_000 + r * 10_000 + Int(round(λ * 1000))

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

        θc = sol.metadata[:theta_converged]

        kraw = angle_key(θc; δ=angle_bin, flip_equiv=false)
        kred = angle_key(θc; δ=angle_bin, flip_equiv=true)

        if !haskey(seen_raw, kraw)
            seen_raw[kraw] = (r, run_seed, copy(θc))
        end
        if !haskey(seen_red, kred)
            seen_red[kred] = (r, run_seed, copy(θc))
        end

        nr = length(seen_raw)
        nd = length(seen_red)
        @printf("run %4d   raw = %4d   reduced = %4d\n", r, nr, nd)

        if nd > nr
            println("\nIMPOSSIBLE EVENT FOUND")
            println("run = $r, seed = $run_seed")
            println("raw = $nr, reduced = $nd")

            println("\nCurrent θ first 10:")
            @printf("[%s]\n", join([@sprintf("%.6f", x) for x in θc[1:min(10,end)]], ", "))

            println("\nCurrent raw key first 10:")
            println(collect(kraw)[1:min(10,end)])

            println("\nCurrent reduced key first 10:")
            println(collect(kred)[1:min(10,end)])

            return
        end
    end

    println("\nNo impossible event found.")
end

main()