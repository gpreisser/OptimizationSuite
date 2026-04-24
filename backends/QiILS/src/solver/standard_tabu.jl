#############################
# Tabu Search (TS) - full scan
# More standard baseline for MaxCut
#############################

using Random
using Graphs
using ProgressMeter
using Statistics

# ---------------------------------------------------------
# Tabu Search: full scan, 1 sweep = 1 move
# ---------------------------------------------------------

"""
    tabu_maxcut_fullscan(
        wg;
        sweeps=10_000,
        tenure=25,
        seed=1,
        init_spins=nothing,
        verbose=false,
    )

Tabu Search for MaxCut using a full neighborhood scan.

Conventions
-----------
- spins s ∈ {+1,-1}
- maximize MaxCut
- 1 sweep = 1 tabu move
- each sweep evaluates all N possible single-spin flips
- tabu: forbids re-flipping vertex i until `step >= tabu_until[i]`
- aspiration: allow tabu move if it yields a new global best

Parameters
----------
- sweeps: number of tabu moves
- tenure: tabu tenure in moves
- verbose: print whenever a new best is found

Returns
-------
(
    best_cut,
    best_spins,
    best_cut_history,
    sweeps_done,
    sweep_times_sec,
    avg_time_per_sweep_sec,
    median_time_per_sweep_sec,
    total_runtime_sec,
)
"""
function tabu_maxcut_fullscan(
    wg;
    sweeps::Int = 10_000,
    tenure::Int = 25,
    seed::Int = 1,
    init_spins::Union{Nothing,Vector{Int8}} = nothing,
    verbose::Bool = false,
)
    rng = MersenneTwister(seed)
    N = nv(wg)

    # ---- init spins ----
    s = isnothing(init_spins) ?
        Int8.(rand(rng, Bool, N) .* 2 .- 1) :
        copy(init_spins)

    cut = maxcut_value(wg, s)

    best_cut = cut
    best_spins = copy(s)

    # ---- tabu bookkeeping ----
    tabu_until = fill(0, N)

    # ---- gain array: gain[i] = Δcut if flip i ----
    gain = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        gain[i] = delta_cut_flip(wg, s, i)
    end

    best_cut_history = Vector{Float64}(undef, sweeps)
    sweep_times_sec  = Vector{Float64}(undef, sweeps)

    prog = Progress(sweeps; desc = "Tabu Full-Scan Sweeps")

    total_t0 = time_ns()
    step = 0

    for sweep in 1:sweeps
        t0 = time_ns()
        step += 1

        # --- full scan over all vertices ---
        best_i = 0
        best_Δ = -Inf

        @inbounds for i in 1:N
            Δ = gain[i]

            admissible = (step >= tabu_until[i]) || (cut + Δ > best_cut)

            if admissible && (Δ > best_Δ)
                best_Δ = Δ
                best_i = i
            end
        end

        # fallback
        if best_i == 0
            @inbounds for i in 1:N
                Δ = gain[i]
                if Δ > best_Δ
                    best_Δ = Δ
                    best_i = i
                end
            end
        end

        # --- apply move ---
        i = best_i
        si_old = s[i]

        cut += gain[i]
        s[i] = Int8(-si_old)

        tabu_until[i] = step + tenure

        gain[i] = -gain[i]

        @inbounds for j in neighbors(wg, i)
            w = wg.weights[i, j]
            gain[j] += -2.0 * w * (s[j] * si_old)
        end

        if cut > best_cut
            best_cut = cut
            best_spins .= s

            if verbose
                println("New BEST found at sweep $sweep (step $step): cut = $best_cut")
                println("Verifying cut from stored spins = ", maxcut_value(wg, best_spins))
                println("----------------------------------------------")
            end
        end

        best_cut_history[sweep] = best_cut
        sweep_times_sec[sweep]  = (time_ns() - t0) / 1e9

        next!(prog)
    end

    finish!(prog)

    total_runtime_sec = (time_ns() - total_t0) / 1e9
    avg_time_per_sweep_sec = sum(sweep_times_sec) / sweeps
    median_time_per_sweep_sec = median(sweep_times_sec)

    return (
        best_cut,
        best_spins,
        best_cut_history,
        sweeps,
        sweep_times_sec,
        avg_time_per_sweep_sec,
        median_time_per_sweep_sec,
        total_runtime_sec,
    )
end
