#############################
# QiILS Solver Core
# - mixing
# - angle projection
# - sweep_pass! (forward + backward)
# - MaxCut evaluation
# - high-level qiils_solve with convergence
#############################

"""
    mixing(N, θ, percentage, seed, loop)

Randomly flip a fraction of angles around π/4 to escape local minima.

Each flip maps θ[i] ↦ π/2 - θ[i], which effectively swaps the
"bit" encoded by that site.

The randomness is deterministic given (`seed`, `loop`).
"""
function mixing(N::Int,
                θ::Vector{Float64},
                percentage::Float64,
                seed::Int,
                loop::Int)

    # how many sites to flip
    sitesflip = max(Int(floor(N * percentage)), 1)

    # deterministic randomness per (seed, loop)
    Random.seed!(seed * loop)

    # choose distinct indices to flip
    idxs = randperm(N)[1:sitesflip]

    @inbounds for idx in idxs
        θ[idx] = π/2 - θ[idx]
    end

    return θ
end

########## Angle projection & spins ##########

"""
    finaltheta(θ)

Project continuous angles onto {0, π/2} sitewise:

- if θ[i] > π/4 → π/2
- else          → 0.0
"""
function finaltheta(vectheta::Vector{Float64})::Vector{Float64}
    finaltheta = Vector{Float64}(undef, length(vectheta))
    @inbounds for i in eachindex(vectheta)
        finaltheta[i] = vectheta[i] > π/4 ? π/2 : 0.0
    end
    return finaltheta
end


"""
    angles_to_spins(θ_meas)

Convert projected angles in {0, π/2} to Ising spins in {+1, -1}:

- θ = 0.0   → +1
- θ = π/2   → -1
"""
function angles_to_spins(θ_meas::Vector{Float64})
    spins = Vector{Float64}(undef, length(θ_meas))
    @inbounds for i in eachindex(θ_meas)
        spins[i] = (θ_meas[i] == 0.0) ? 1.0 : -1.0
    end
    return spins
end


########## MaxCut evaluation ##########

"""
    maxcut_value(wg, spins)

Compute the MaxCut value of a spin configuration s ∈ {+1, -1}:

    Cut(s) = Σ_{(i,j)∈E} w_ij * (1 - s_i s_j) / 2

This is the standard weighted MaxCut objective.
"""
function maxcut_value(wg, spins)
    cut = 0.0
    for e in edges(wg)
        i, j = src(e), dst(e)
        w = wg.weights[i, j]
        cut += w * (1 - spins[i] * spins[j]) / 2
    end
    return cut
end


########## (Optional) continuous energy ##########

"""
    compute_continuous_energy(wg, angles, λ, gvec)

Continuous energy used during the analog part of the algorithm:

E(θ; λ) = λ * Σ_edges w_ij cos(2θ_i)cos(2θ_j)
          - (1-λ) * Σ_i [ sin(2θ_i) + g_i cos(2θ_i) ]

This is NOT used by `qiils_solve` directly, but is useful for diagnostics
or plotting the analog energy landscape.
"""
function compute_continuous_energy(wg,
                                   angles::Vector{Float64},
                                   λ::Float64,
                                   gvec::Vector{Float64})
    N = length(angles)
    cos2θ = cos.(2 .* angles)
    sin2θ = sin.(2 .* angles)

    local_term = 0.0
    @inbounds for i in 1:N
        local_term += sin2θ[i] + gvec[i] * cos2θ[i]
    end

    edge_term = 0.0
    for e in edges(wg)
        i, j = src(e), dst(e)
        w = wg.weights[i, j]
        edge_term += w * cos2θ[i] * cos2θ[j]
    end

    return λ * edge_term - (1 - λ) * local_term
end


########## Single sweep (forward + backward) ##########

"""
    sweep_pass!(N, wg, λ, θ, gvec, cos2θ, sin2θ)

Perform one full QiILS sweep (forward + backward) in-place.

Arguments:
- N        :: Int
- wg       :: SimpleWeightedGraph
- λ        :: Float64     (annealing parameter in [0,1])
- θ        :: Vector{Float64}      current angles (updated in-place)
- gvec     :: Vector{Float64}      local field vector
- cos2θ    :: Vector{Float64}      cos(2θ) cache (updated in-place)
- sin2θ    :: Vector{Float64}      sin(2θ) cache (updated in-place)

Returns: nothing (θ, cos2θ, sin2θ mutated in-place).
"""
function sweep_pass!(N::Int,
                     wg,
                     λ::Float64,
                     θ::Vector{Float64},
                     gvec::Vector{Float64},
                     cos2θ::Vector{Float64},
                     sin2θ::Vector{Float64})

    B = 1 - λ

    ###########
    # Forward sweep
    ###########
    @inbounds for i in 1:N
        # a = Σ_j w_ij * cos(2θ_j)
        a = 0.0
        for j in neighbors(wg, i)
            a += wg.weights[i, j] * cos2θ[j]
        end

        A = λ * a - (1 - λ) * gvec[i]

        θ_i_new = π/4 + 0.5 * atan(A, B)

        θ[i] = θ_i_new
        cos2θ[i] = cos(2 * θ_i_new)
        sin2θ[i] = sin(2 * θ_i_new)
    end

    ###########
    # Backward sweep
    ###########
    @inbounds for i in N:-1:1
        a = 0.0
        for j in neighbors(wg, i)
            a += wg.weights[i, j] * cos2θ[j]
        end

        A = λ * a - (1 - λ) * gvec[i]

        θ_i_new = π/4 + 0.5 * atan(A, B)

        θ[i] = θ_i_new
        cos2θ[i] = cos(2 * θ_i_new)
        sin2θ[i] = sin(2 * θ_i_new)
    end

    return nothing
end


########## High-level solver ##########

"""
Returns:
- best_cut_history::Vector{Float64}
- best_angles::Vector{Float64}
- total_sweeps_done::Int
"""
function qiils_solve(
    wg,
    λ_sweep::Float64,
    gvec::Vector{Float64},
    attempts::Int = 20,
    sweeps_per_attempt::Int = 80,
    percentage::Float64 = 0.3,
    seed::Int = 1,
    θ0::Union{Nothing,Vector{Float64}} = nothing,
    angle_conv::Float64 = 1e-6,
    use_scaled_convergence::Bool = true,
)

    N = nv(wg)

    # --- initial angles ---
    θ = isnothing(θ0) ? [rand() * (π/2) for _ in 1:N] : copy(θ0)

    # preallocate old-theta buffer
    θ_old = similar(θ)

    # trig caches
    cos2θ = cos.(2 .* θ)
    sin2θ = sin.(2 .* θ)

    best_cut = -Inf
    best_angles = finaltheta(θ)

    best_cut_history = Vector{Float64}(undef, attempts)

    # NEW: cumulative sweeps up to end of each attempt
    sweeps_cumsum = Vector{Int}(undef, attempts)
    total_sweeps_done = 0

    # progress bar by attempts
    prog = Progress(attempts; desc="QiILS Attempts")

    for attempt in 1:attempts
        # ---- Sweeps with convergence test ----
        for sweep in 1:sweeps_per_attempt
            θ_old .= θ

            sweep_pass!(N, wg, λ_sweep, θ, gvec, cos2θ, sin2θ)
            total_sweeps_done += 1

            Δθ_max = maximum(abs.(θ .- θ_old))

            if use_scaled_convergence
                scaled_tol = max(angle_conv * mean(abs.(θ .- π/4)), 1e-12)
                if Δθ_max < scaled_tol
                    break
                end
            else
                if Δθ_max < angle_conv
                    break
                end
            end
        end

        # ---- Measurement ----
        θ_meas = finaltheta(θ)
        spins = angles_to_spins(θ_meas)

        # ---- MaxCut evaluation ----
        cut_val = maxcut_value(wg, spins)

        if cut_val > best_cut
            best_cut = cut_val
            best_angles = copy(θ_meas)
            # best_spins = copy(spins)  # keep if you want to store/verify later
        end

        best_cut_history[attempt] = best_cut
        sweeps_cumsum[attempt] = total_sweeps_done

        # ---- Mixing ----
        θ = mixing(N, θ, percentage, seed, attempt)
        cos2θ .= cos.(2 .* θ)
        sin2θ .= sin.(2 .* θ)

        next!(prog)
    end

    finish!(prog)

    # total sweeps is sweeps_cumsum[end]
    return best_cut_history, best_angles, sweeps_cumsum
end