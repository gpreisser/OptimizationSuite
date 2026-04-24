# scripts/debug_lambda1_reduced_vs_spin.jl
#
# Check at λ = 1.0:
#   - unique rounded spin configs
#   - unique raw angle configs
#   - unique reduced angle configs (θ ~ θ + π/2)
#
# Uses angle_conv = 1e-6.
# No files are saved.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "QiIGS"))

using QiIGS
using Printf

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

    q1 = qvec_for(pi / 2)
    return min(q0, q1)
end

function pm1_graph_path(graphs_root::AbstractString, N::Int, k::Int, seed::Int)
    return joinpath(
        graphs_root,
        string(N),
        string(k),
        "graph_N$(N)_k$(k)_seed$(seed)_seedb$(seed)_pm1weighted.txt",
    )
end

function main()
    N = 12
    k = 3
    graph_seed = 1
    λ = 1.0
    n_inits = 1000

    iterations = 1
    inner_iterations = 5000
    tao = 0.1
    angle_conv = 1e-6
    init_mode = :uniform
    save_params = true

    angle_bin = 2e-2
    seed_salt = 0

    ROOT_QIILS = "/Users/guillermo.preisser/Projects/QiILS_ITensor"
    GRAPHS_ROOT = joinpath(ROOT_QIILS, "graphs")

    gpath = pm1_graph_path(GRAPHS_ROOT, N, k, graph_seed)
    isfile(gpath) || error("Graph file not found: $gpath")
    W = QiIGS.load_weight_matrix(gpath)

    keyT = typeof(angle_key(zeros(Float64, N); δ=angle_bin, flip_equiv=false))

    seen_spins = Set{UInt64}()
    seen_angles_raw = Set{keyT}()
    seen_angles_reduced = Set{keyT}()

    println("===================================================")
    println(" λ = 1.0 reduced-angle vs spin count debug")
    println("===================================================")
    println("graph_seed   = ", graph_seed)
    println("n_inits      = ", n_inits)
    println("angle_bin    = ", angle_bin)
    println("angle_conv   = ", angle_conv)
    println()

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
        )

        push!(seen_spins, QiIGS.spin_config_key(sol.configuration))

        θc = get(sol.metadata, :theta_converged, Float64[])
        isempty(θc) && error("Missing :theta_converged in sol.metadata")

        push!(seen_angles_raw, angle_key(θc; δ=angle_bin, flip_equiv=false))
        push!(seen_angles_reduced, angle_key(θc; δ=angle_bin, flip_equiv=true))
    end

    nspin = length(seen_spins)
    nraw  = length(seen_angles_raw)
    nred  = length(seen_angles_reduced)

    println("N_unique_spin           = ", nspin)
    println("N_unique_angle_raw      = ", nraw)
    println("N_unique_angle_reduced  = ", nred)
    println()
    println("raw - spin      = ", nraw - nspin)
    println("reduced - spin  = ", nred - nspin)
end

main()