using ITensors        # ITensor, Index, OpSum, MPO, op, apply, orthogonalize!
using ITensorMPS      # MPS, randomMPS, dmrg, siteinds
using Random          # Random.seed!, rand
using Graphs          # edges(), src(), dst()
using ProgressMeter
using Printf          # For @printf

"""
    mix_mps_state(state, hilbert, sitesflip, ll)

Apply random X flips to an MPS to introduce noise/mixing.

This version:
  • applies `sitesflip` random X gates
  • uses deterministic seeding via (ii, ll)
  • returns a canonicalized MPS (important for DMRG/TDVP/sampling)

Arguments:
  state      :: MPS
  hilbert    :: Vector{Index}
  sitesflip  :: Int
  ll         :: Int   (loop index)

Returns:
  Modified MPS (the same variable name in caller gets updated)
"""
function mix_mps_state(state::MPS,
                        hilbert::Vector{<:Index},
                        sitesflip::Int)

    N = length(state)

    for ii in 1:sitesflip
        # Use current random state (no re-seeding)
        tmpr = rand(1:N)
        Xop  = op("X", hilbert, tmpr)
        state = apply(Xop, state)
    end

    orthogonalize!(state, 1)
    return state
end


###############################################################
# Convert sampled MPS outputs {1,2} → spins {+1,-1} or bits {0,1}
###############################################################

"""
    sample_to_spins(sample)

Convert an MPS sample vector in {1,2} (ITensor convention)
to Ising spins in {+1, -1}.

Mapping:
    1 → +1
    2 → -1
"""
function sample_to_spins(sample::Vector{Int})
    N = length(sample)
    spins = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        spins[i] = (sample[i] == 1) ? +1 : -1
    end
    return spins
end


"""
    sample_to_bits(sample)

Convert an MPS sample vector in {1,2} to classical bits {0,1}.

Mapping:
    1 → 0
    2 → 1
"""
function sample_to_bits(sample::Vector{Int})
    N = length(sample)
    bits = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        bits[i] = (sample[i] == 1) ? 0 : 1
    end
    return bits
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



"""
    build_H_mpo(wg, hilbert, λ; weighted=false)

Construct the MPO for the adiabatic MaxCut Hamiltonian:

    H(λ) = (1-λ) * Σ_i X_i  +  λ * Σ_(i,j) w_ij Z_i Z_j
"""
function build_H_mpo(wg, hilbert, λ::Float64; weighted=false)
    os = OpSum()

    # Local transverse field term
    hloc = (1 - λ)
    for i in vertices(wg)
        os += hloc, "X", i
    end

    # ZZ interactions
    for e in edges(wg)
        i, j = src(e), dst(e)
        w = weighted ? wg.weights[i,j] : 1.0
        os += λ * w, "Z", i, "Z", j
    end

    return MPO(os, hilbert)
end
function qiils_itensor_solver(wg;
    lambda_sweep         = 0.5,
    attempts             = 20,
    sweeps_per_attempt   = 10,
    maxdim               = 20,
    percentage           = 0.2,
    sampler              = nothing,
    sample_mode          = :entangled,
    init_psi             = nothing,
    return_psi           = false,
    hilbert              = nothing,
    weighted             = true,
    verbose              = false,
    mix_strategy         = :best,  # ← NEW PARAMETER: :best, :current, or :random
)

    N = nv(wg)
    hilbert = hilbert === nothing ? siteinds("Qubit", N) : hilbert
    H = build_H_mpo(wg, hilbert, Float64(lambda_sweep); weighted=weighted)
    psi = init_psi === nothing ? randomMPS(hilbert; linkdims=maxdim) : init_psi

    if sample_mode == :local && sampler === nothing
        sampler = LocalSampler(hilbert)
    end

    best_cut     = -Inf
    best_spins   = nothing
    best_psi     = nothing
    best_history = Vector{Float64}(undef, attempts)
    cut_history  = Vector{Float64}(undef, attempts)
    energy_history = Vector{Float64}(undef, attempts)

    prog = Progress(attempts; desc="QiILS-ITensor", enabled=!verbose)

    for attempt in 1:attempts
        # DMRG optimization
        energy, psi = dmrg(H, psi;
            nsweeps=sweeps_per_attempt,
            maxdim=maxdim,
            outputlevel=0,
        )
        
        energy_history[attempt] = energy

        # Sample and evaluate
        sample = sample_mps(psi; mode=sample_mode, sampler=sampler)
        spins  = sample_to_spins(sample)
        cutval = maxcut_value(wg, spins)

        cut_history[attempt] = cutval

        # Update best if better
        if cutval > best_cut
            best_cut   = cutval
            best_spins = copy(spins)
            best_psi   = copy(psi)
        end
        best_history[attempt] = best_cut

        # Diagnostic output
        if verbose
            @printf("Attempt %2d: energy=%.6f  cut=%.2f  best=%.2f\n", 
                    attempt, energy, cutval, best_cut)
        end

        # ← CHOOSE MIXING STRATEGY
      
if mix_strategy == :best
    # Mix from best state found so far (exploitation)
    if best_psi !== nothing
        psi = mix_mps_state(best_psi, hilbert, floor(Int, percentage*N))  # ← Remove 'attempt'
    else
        psi = mix_mps_state(psi, hilbert, floor(Int, percentage*N))  # ← Remove 'attempt'
    end
    
elseif mix_strategy == :current
    # Mix from current state (exploration, can drift away from best)
    psi = mix_mps_state(psi, hilbert, floor(Int, percentage*N))  # ← Remove 'attempt'
            
        elseif mix_strategy == :random
            # Complete restart from random (maximum exploration)
            psi = randomMPS(hilbert; linkdims=maxdim)
            
        else
            error("Unknown mix_strategy: $mix_strategy. Use :best, :current, or :random")
        end

        !verbose && next!(prog)
    end

    !verbose && finish!(prog)

    if return_psi
        return best_history, cut_history, best_spins, best_psi, hilbert, energy_history
    else
        return best_history, cut_history, best_spins, energy_history
    end
end