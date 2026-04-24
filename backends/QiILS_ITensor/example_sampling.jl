#!/usr/bin/env julia

using ITensors
using Random
using BenchmarkTools
using ITensorMPS

# ==============================================================================
# 1. Original local sampler (slow, allocating)
# ==============================================================================
function sample_local_probs(m::MPS)
    N = length(m)
    result = Vector{Int}(undef, N)

    for j in 1:N
        s = siteind(m, j)
        d = dim(s)
        probs = zeros(Float64, d)

        for n in 1:d
            proj = ITensor(s)
            proj[s => n] = 1.0
            v = m[j] * dag(proj)
            probs[n] = real(scalar(dag(v) * v))
        end

        total = sum(probs)
        probs ./= (total == 0 ? 1 : total)

        r = rand()
        cumulative = 0.0
        for n in 1:d
            cumulative += probs[n]
            if r <= cumulative
                result[j] = n
                break
            end
        end
    end

    return result
end

# ==============================================================================
# 2. Fast sampler (your current optimization)
# ==============================================================================
function sample_local_probs_fast(m::MPS)
    N = length(m)
    sites = siteinds(m)
    maxd = maximum(dim.(sites))

    probs = Vector{Float64}(undef, maxd)
    proj_dag = Vector{Vector{ITensor}}(undef, N)

    # Preallocate projectors per site
    for j in 1:N
        s = sites[j]
        d = dim(s)
        proj_dag[j] = Vector{ITensor}(undef, d)
        for n in 1:d
            t = ITensor(s)
            t[s => n] = 1.0
            proj_dag[j][n] = dag(t)
        end
    end

    result = Vector{Int}(undef, N)

    for j in 1:N
        m_j = m[j]
        d = length(proj_dag[j])

        total = 0.0
        @inbounds for n in 1:d
            v = m_j * proj_dag[j][n]
            pn = norm(v)^2
            probs[n] = pn
            total += pn
        end

        if total == 0
            invd = 1 / d
            @inbounds for n in 1:d
                probs[n] = invd
            end
        else
            invT = 1 / total
            @inbounds for n in 1:d
                probs[n] *= invT
            end
        end

        r = rand()
        cum = 0.0
        @inbounds for n in 1:d
            cum += probs[n]
            if r <= cum
                result[j] = n
                break
            end
        end
    end

    return result
end

# ==============================================================================
# 3. TRUE FAST SAMPLER (zero-allocation version)
#    Precomputes projectors ONCE, reuses buffers
# ==============================================================================

struct LocalSampler
    sites::Vector{Index}
    proj_dag::Vector{Vector{ITensor}}
    probs::Vector{Float64}
end

"""
    LocalSampler(hilbert)

Precompute dag(|n⟩) for every site. Build once and reuse.
"""
function LocalSampler(hilbert::Vector{<:Index})
    N = length(hilbert)
    proj_dag = Vector{Vector{ITensor}}(undef, N)
    maxd = 0

    # Precompute projectors
    for j in 1:N
        s = hilbert[j]
        d = dim(s)
        maxd = max(maxd, d)

        proj_dag[j] = Vector{ITensor}(undef, d)
        for n in 1:d
            t = ITensor(s)
            t[s => n] = 1.0
            proj_dag[j][n] = dag(t)
        end
    end

    probs = Vector{Float64}(undef, maxd)
    return LocalSampler(hilbert, proj_dag, probs)
end

LocalSampler(m::MPS) = LocalSampler(siteinds(m))

"""
    sample_local_probs!(sampler, m, result)

Zero-allocation fast local sampling.
"""
function sample_local_probs!(sampler::LocalSampler, m::MPS, result::Vector{Int})
    N = length(m)
    proj_dag = sampler.proj_dag
    probs = sampler.probs

    for j in 1:N
        m_j = m[j]
        d = length(proj_dag[j])

        total = 0.0
        @inbounds for n in 1:d
            v = m_j * proj_dag[j][n]
            pn = norm(v)^2
            probs[n] = pn
            total += pn
        end

        if total == 0.0
            invd = 1.0 / d
            @inbounds for n in 1:d
                probs[n] = invd
            end
        else
            invT = 1.0 / total
            @inbounds for n in 1:d
                probs[n] *= invT
            end
        end

        r = rand()
        cum = 0.0
        @inbounds for n in 1:d
            cum += probs[n]
            if r <= cum
                result[j] = n
                break
            end
        end
    end
    return result
end

"""
Convenience wrapper that allocates the output vector.
"""
function sample_local_probs(sampler::LocalSampler, m::MPS)
    result = Vector{Int}(undef, length(m))
    return sample_local_probs!(sampler, m, result)
end

# ==============================================================================
# MAIN
# ==============================================================================
function main()
    Random.seed!(123)

    N = 50
    maxdim1 = 20

    hilbert = siteinds("Qubit", N)
    psi = randomMPS(hilbert; linkdims=maxdim1)

    # Build true fast sampler ONCE
    sampler = LocalSampler(hilbert)

    println("\nChecking sample outputs (one sample each):")
    println("original → ", sample_local_probs(psi))
    println("fast     → ", sample_local_probs_fast(psi))
    println("truefast → ", sample_local_probs(sampler, psi))

    println("\nBenchmarking...")

    @btime sample_local_probs($psi)
    @btime sample_local_probs_fast($psi)
    @btime sample_local_probs($sampler, $psi)
end

main()