using ITensors
using Random, Statistics

# --- Simplified versions of the three samplers -------------------------
function sample_original(m::MPS)
    N = length(m)
    result = Vector{Int}(undef, N)
    for j in 1:N
        s = siteind(m,j)
        d = dim(s)
        probs = zeros(Float64, d)
        for n in 1:d
            proj = ITensor(s)
            proj[s=>n] = 1.0
            v = m[j] * dag(proj)
            probs[n] = norm(v)^2
        end
        probs ./= sum(probs)
        r = rand()
        c = 0
        for n in 1:d
            c += probs[n]
            if r ≤ c
                result[j] = n
                break
            end
        end
    end
    return result
end

# Fast version (local buffers)
function sample_fast(m::MPS)
    N = length(m)
    sites = siteinds(m)
    maxd = maximum(dim.(sites))
    probs = zeros(Float64, maxd)
    proj_dag = [ [dag( (t = ITensor(s); t[s=>n]=1; t) ) for n in 1:dim(s)] 
                 for s in sites ]
    result = Vector{Int}(undef, N)
    for j in 1:N
        d = length(proj_dag[j])
        total = 0.0
        for n in 1:d
            v = m[j] * proj_dag[j][n]
            pn = norm(v)^2
            probs[n] = pn
            total += pn
        end
        probs[1:d] ./= sum(probs[1:d])
        r = rand()
        c = 0.0
        for n in 1:d
            c += probs[n]
            if r ≤ c
                result[j] = n
                break
            end
        end
    end
    return result
end

# TRUE FAST sampler
struct LocalSampler
    sites::Vector{Index}
    proj_dag::Vector{Vector{ITensor}}
    probs::Vector{Float64}
end

function LocalSampler(sites::Vector{<:Index})
    proj_dag = Vector{Vector{ITensor}}(undef, length(sites))
    maxd = 0
    for j in 1:length(sites)
        s = sites[j]
        d = dim(s)
        maxd = max(maxd, d)
        proj_dag[j] = [dag( (t=ITensor(s); t[s=>n]=1; t) ) for n in 1:d]
    end
    return LocalSampler(sites, proj_dag, zeros(Float64, maxd))
end

function sample_truefast(sampler::LocalSampler, m::MPS)
    N = length(m)
    result = Vector{Int}(undef, N)
    probs = sampler.probs
    for j in 1:N
        m_j = m[j]
        d = length(sampler.proj_dag[j])
        total = 0.0
        for n in 1:d
            v = m_j * sampler.proj_dag[j][n]
            pn = norm(v)^2
            probs[n] = pn
            total += pn
        end
        probs[1:d] ./= total
        r = rand()
        c = 0.0
        for n in 1:d
            c += probs[n]
            if r ≤ c
                result[j] = n
                break
            end
        end
    end
    return result
end

# -----------------------------------------------------------------------
# TESTING: many samples → histogram comparison
# -----------------------------------------------------------------------

function test_histograms(; N=3, maxdim=4, trials=10000)
    println("Building MPS…")
    sites = siteinds("Qubit", N)
    psi = randomMPS(sites; linkdims=maxdim)

    sampler = LocalSampler(sites)

    # histograms for site 1 only (you can generalize easily)
    hist_orig  = zeros(2)
    hist_fast  = zeros(2)
    hist_true  = zeros(2)

    println("Sampling $trials trials…")

    for _ in 1:trials
        s1 = sample_original(psi)[1]
        hist_orig[s1] += 1

        s2 = sample_fast(psi)[1]
        hist_fast[s2] += 1

        s3 = sample_truefast(sampler, psi)[1]
        hist_true[s3] += 1
    end

    # normalize
    hist_orig ./= trials
    hist_fast ./= trials
    hist_true ./= trials

    println("\n=== HISTOGRAMS (probabilities) for site 1 ===")
    println("original : ", hist_orig)
    println("fast     : ", hist_fast)
    println("truefast : ", hist_true)

    println("\nDifference fast - original : ", hist_fast - hist_orig)
    println("Difference true - original : ", hist_true - hist_orig)
end

# Run test
test_histograms()