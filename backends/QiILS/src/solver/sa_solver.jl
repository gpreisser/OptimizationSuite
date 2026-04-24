#############################
# Simulated Annealing (SA)
# Baseline solver for MaxCut
#############################

using Random
using Graphs
using ProgressMeter
"""
    sa_maxcut(
        wg;
        β0=0.01,
        βf=5.0,
        sweeps=10_000,
        seed=1,
        init_spins=nothing,
        schedule=:linear,
    )

Simulated Annealing for MaxCut.

Conventions
-----------
- spins s ∈ {+1, -1}
- objective: maximize MaxCut
- 1 sweep = N attempted single-spin flips
- acceptance:
    Δcut ≥ 0        → accept
    Δcut < 0        → accept with prob exp(β * Δcut)

Returns
-------
(best_cut, best_spins, best_cut_history, sweeps_done)
"""
function sa_maxcut(
    wg;
    β0::Float64 = 0.01,
    βf::Float64 = 5.0,
    sweeps::Int = 10_000,
    seed::Int = 1,
    init_spins::Union{Nothing,Vector{Int8}} = nothing,
    schedule::Symbol = :linear,   # :linear or :geometric
)

    rng = MersenneTwister(seed)
    N = nv(wg)

    # ------------------
    # Initialization
    # ------------------
    s = isnothing(init_spins) ?
        Int8.(rand(rng, Bool, N) .* 2 .- 1) :
        copy(init_spins)

    cut = maxcut_value(wg, s)

    best_cut = cut
    best_spins = copy(s)

    best_cut_history = Vector{Float64}(undef, sweeps)

    # ------------------
    # β schedule
    # ------------------
    β_of(t) = schedule === :geometric ?
        β0 * (βf / β0)^((t - 1) / max(sweeps - 1, 1)) :
        β0 + (βf - β0) * ((t - 1) / max(sweeps - 1, 1))

    # ------------------
    # SA loop
    # ------------------
    prog = Progress(sweeps; desc="SA Sweeps")
    for sweep in 1:sweeps
        β = β_of(sweep)

        # one sweep = N attempted flips
        for i in randperm(rng, N)
            Δcut = delta_cut_flip(wg, s, i)

            if (Δcut ≥ 0.0) || (rand(rng) < exp(β * Δcut))
                s[i] = Int8(-s[i])
                cut += Δcut

                if cut > best_cut
                    best_cut = cut
                    best_spins .= s

                    #println("New BEST found at sweep $sweep: cut = $best_cut")
                    #println("Verifying cut from stored spins = ", maxcut_value(wg, best_spins))
                    #println("----------------------------------------------")
                end
            end
        end

        best_cut_history[sweep] = best_cut
        next!(prog)     # one tick per sweep
    end
    finish!(prog)
    return best_cut, best_spins, best_cut_history, sweeps
end
