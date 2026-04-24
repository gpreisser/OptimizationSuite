#############################
# Tabu Search (TS)
# Baseline solver for MaxCut
#############################

using Random
using Graphs
using ProgressMeter

# ---------------------------------------------------------
# Tabu Search
# ---------------------------------------------------------

"""
    tabu_maxcut(
        wg;
        sweeps=10_000,
        tenure=25,
        candidate_size=0,
        seed=1,
        init_spins=nothing,
        verbose=false,
    )

Tabu Search for MaxCut.

Conventions
-----------
- spins s ∈ {+1,-1}
- maximize MaxCut
- 1 sweep = N tabu steps (N single-spin flips)
- tabu: forbids re-flipping vertex i until `step >= tabu_until[i]`
- aspiration: allow tabu move if it yields a new global best

Parameters
----------
- sweeps: number of sweeps (each sweep = N steps)
- tenure: tabu tenure in *steps*
- candidate_size:
    0  -> evaluate all vertices each step
    >0 -> evaluate a random subset of that size each step
- verbose: print whenever a new best is found

Returns
-------
(best_cut, best_spins, best_cut_history, sweeps_done,
 sweep_times_sec, avg_time_per_sweep_sec)
"""
function tabu_maxcut(
    wg;
    sweeps::Int = 10_000,
    tenure::Int = 25,
    candidate_size::Int = 0,
    seed::Int = 1,
    init_spins::Union{Nothing,Vector{Int8}} = nothing,
    verbose::Bool = false,
)
    rng = MersenneTwister(seed)
    N = nv(wg)
    steps_total = sweeps * N

    # ---- init spins ----
    s = isnothing(init_spins) ?
        Int8.(rand(rng, Bool, N) .* 2 .- 1) :
        copy(init_spins)

    cut = maxcut_value(wg, s)

    best_cut = cut
    best_spins = copy(s)

    # ---- tabu bookkeeping ----
    tabu_until = fill(0, N)  # tabu_until[i] = step index (inclusive threshold)

    # ---- gain array: gain[i] = Δcut if flip i ----
    gain = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        gain[i] = delta_cut_flip(wg, s, i)
    end

    best_cut_history = Vector{Float64}(undef, sweeps)
    sweep_times_sec = Vector{Float64}(undef, sweeps)

    prog = Progress(sweeps; desc="Tabu Sweeps")

    step = 0
    for sweep in 1:sweeps
        t0 = time_ns()

        for _ in 1:N
            step += 1

            # --- choose candidate set ---
            candidates = if candidate_size <= 0 || candidate_size >= N
                1:N
            else
                # allow duplicates to avoid O(N) randperm each step (fine for baseline)
                rand(rng, 1:N, candidate_size)
            end

            # --- pick best admissible move ---
            best_i = 0
            best_Δ = -Inf

            @inbounds for i in candidates
                Δ = gain[i]

                # admissible if not tabu, or aspiration (would beat global best)
                admissible = (step >= tabu_until[i]) || (cut + Δ > best_cut)

                if admissible && (Δ > best_Δ)
                    best_Δ = Δ
                    best_i = i
                end
            end

            # fallback: if all were tabu and no aspiration triggered, pick best in candidates anyway
            if best_i == 0
                @inbounds for i in candidates
                    Δ = gain[i]
                    if Δ > best_Δ
                        best_Δ = Δ
                        best_i = i
                    end
                end
            end

            # --- apply move (flip best_i) ---
            i = best_i
            si_old = s[i]

            # update objective
            cut += gain[i]

            # flip spin
            s[i] = Int8(-si_old)

            # mark tabu
            tabu_until[i] = step + tenure

            # update gains efficiently:
            # gain[i] flips sign
            gain[i] = -gain[i]

            @inbounds for j in neighbors(wg, i)
                w = wg.weights[i, j]
                gain[j] += -2.0 * w * (s[j] * si_old)
            end

            # update global best
            if cut > best_cut
                best_cut = cut
                best_spins .= s

                if verbose
                    println("New BEST found at sweep $sweep (step $step): cut = $best_cut")
                    println("Verifying cut from stored spins = ", maxcut_value(wg, best_spins))
                    println("----------------------------------------------")
                end
            end
        end

        best_cut_history[sweep] = best_cut
        sweep_times_sec[sweep] = (time_ns() - t0) / 1e9

        next!(prog)
    end

    finish!(prog)

    avg_time_per_sweep_sec = sum(sweep_times_sec) / sweeps

    return best_cut, best_spins, best_cut_history, sweeps,
           sweep_times_sec, avg_time_per_sweep_sec
end
