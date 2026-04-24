###############################################################
# SAMPLING UTILITIES (QiILS_ITensor)
#
# Provides:
#   - LocalSampler      : precomputes ⟨n| projectors for each site
#   - sample_truefast   : fast local independent sampling (no entanglement)
#   - sample_exact_mps  : correct sequential conditional sampling (with entanglement)
#   - sample_mps        : unified interface (exact, local, or entangled)
#
# NOTE:
#   All of this code is evaluated inside the QiILS_ITensor module
#   because sampling.jl is included from QiILS_ITensor.jl.
###############################################################

using ITensors
using Random
using ITensorMPS   # <-- THIS brings in MPS, MPO, etc.

###############################################################
# LocalSampler
###############################################################

"""
    LocalSampler(sites)

Build a sampling helper that precomputes, for each site index,
the set of bra projectors ⟨n| required for fast local sampling.

Arguments:
  sites :: Vector{Index}   (usually siteinds("Qubit", N))

Fields:
  sites      :: Vector{Index}
  proj_dag   :: proj_dag[j][n] = ⟨n| for site j
  probs      :: reusable probability buffer (avoids allocations)
"""
struct LocalSampler
    sites::Vector{Index}
    proj_dag::Vector{Vector{ITensor}}
    probs::Vector{Float64}
end


function LocalSampler(sites::Vector{<:Index})
    N = length(sites)
    proj_dag = Vector{Vector{ITensor}}(undef, N)
    maxd = 0

    for j in 1:N
        s = sites[j]
        d = dim(s)
        proj_dag[j] = Vector{ITensor}(undef, d)
        maxd = max(maxd, d)

        # precompute bra projectors ⟨n|
        for n in 1:d
            ket = ITensor(s)
            ket[s => n] = 1.0
            proj_dag[j][n] = dag(ket)
        end
    end

    return LocalSampler(sites, proj_dag, zeros(Float64, maxd))
end



###############################################################
# sample_truefast (fast approximate sampling - NO ENTANGLEMENT)
###############################################################

"""
    sample_truefast(sampler::LocalSampler, mps::MPS)

Fast independent-site sampling of an MPS:

    P_j(n) = || ⟨n| mps[j] ||²

Properties:
  - Very fast
  - Ignores entanglement
  - Good for quick heuristics when entanglement is not critical
  - Returns Vector{Int} of 1/2 labels (ITensor convention)

WARNING: This is NOT mathematically exact sampling from the quantum state.
Use sample_exact_mps for correct sampling that respects entanglement.
"""
function sample_truefast(sampler::LocalSampler, mps::MPS)
    N = length(mps)
    result = Vector{Int}(undef, N)

    probs = sampler.probs
    proj_dag = sampler.proj_dag

    @inbounds for j in 1:N
        m_j = mps[j]
        d = length(proj_dag[j])
        total = 0.0

        # compute local probabilities P_j(n)
        for n in 1:d
            v = m_j * proj_dag[j][n]   # ⟨n|ψ_j⟩   local contraction
            pn = norm(v)^2            # probability
            probs[n] = pn
            total += pn
        end

        # normalize
        invT = 1 / total
        for n in 1:d
            probs[n] *= invT
        end

        # sample local distribution
        r = rand()
        cum = 0.0
        for n in 1:d
            cum += probs[n]
            if r <= cum
                result[j] = n
                break
            end
        end
    end

    return result
end


###############################################################
# sample_exact_mps (correct entanglement-respecting sampling)
###############################################################

"""
    sample_exact_mps(mps::MPS; rng=Random.GLOBAL_RNG)

Correct sequential conditional sampling from an MPS that respects entanglement.

This implements the standard quantum measurement process:
  - Sample site 1 from its marginal distribution
  - Collapse the wavefunction based on the measurement
  - Sample site 2 from its conditional distribution P(site2 | site1_result)
  - Continue sequentially through all sites

Properties:
  - Mathematically correct sampling from |ψ⟩
  - Respects all quantum correlations and entanglement
  - Returns Vector{Int} of 1/2 labels (ITensor convention)

Requirements:
  - MPS must be in left-canonical form with orthocenter at site 1
  - MPS must be normalized

Arguments:
  mps :: MPS
  rng :: AbstractRNG (optional, defaults to global RNG)

Returns:
  Vector{Int} - sampled configuration
"""
function sample_exact_mps(mps::MPS; rng::AbstractRNG=Random.GLOBAL_RNG)
    N = length(mps)
    
    # Verify MPS is properly prepared
    if orthocenter(mps) != 1
        error("sample_exact_mps: MPS must have orthocenter at site 1. Call orthogonalize!(mps, 1) first.")
    end
    if abs(1.0 - norm(mps[1])) > 1E-8
        error("sample_exact_mps: MPS is not normalized, norm=$(norm(mps[1]))")
    end
    
    result = zeros(Int, N)
    A = mps[1]
    
    for j in 1:N
        s = siteind(mps, j)
        d = dim(s)
        
        # Compute conditional probabilities for site j
        pdisc = 0.0
        r = rand(rng)
        
        # Will need n, An, and pn below
        n = 1
        An = ITensor()
        pn = 0.0
        
        while n <= d
            # Create projector |n⟩⟨n|
            projn = ITensor(s)
            projn[s => n] = 1.0
            
            # Apply measurement projector
            An = A * dag(projn)
            pn = real(scalar(dag(An) * An))
            
            pdisc += pn
            
            # Sample: stop when random number exceeds cumulative probability
            (r < pdisc) && break
            
            n += 1
        end
        
        result[j] = n
        
        # Collapse wavefunction and prepare for next site
        if j < N
            A = mps[j + 1] * An
            A *= (1.0 / sqrt(pn))  # renormalize
        end
    end
    
    return result
end


###############################################################
# Unified sampling API
###############################################################

"""
    sample_mps(mps; mode=:entangled, sampler=nothing, rng=Random.GLOBAL_RNG)

Unified interface for sampling an MPS.

Arguments:
  mps       :: MPS
  mode      :: Symbol - sampling method to use:
               :entangled (default) - Correct sequential conditional sampling (respects entanglement)
               :exact               - Uses ITensor's built-in sample() function
               :local               - Fast approximate sampling (ignores entanglement)
  sampler   :: LocalSampler  (required for :local mode)
  rng       :: AbstractRNG   (optional, for :entangled mode)

Returns:
  Vector{Int} of 1/2 labels.

Recommendations:
  - Use :entangled (default) for correct quantum sampling with entanglement
  - Use :exact if you want ITensor's built-in implementation
  - Use :local only if you need speed and don't care about entanglement correlations
"""
function sample_mps(mps::MPS; mode=:entangled, sampler=nothing, rng::AbstractRNG=Random.GLOBAL_RNG)
    if mode === :entangled
        return sample_exact_mps(mps; rng=rng)
        
    elseif mode === :exact
        return sample(mps)  # ITensor's built-in
        
    elseif mode === :local
        sampler === nothing &&
            error("sample_mps: must pass sampler=LocalSampler(...) for :local mode")
        return sample_truefast(sampler, mps)
        
    else
        error("sample_mps: unknown sampling mode $(mode). Use :entangled, :exact, or :local")
    end
end