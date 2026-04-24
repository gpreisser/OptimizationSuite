###############################################################
# Compare Exact ITensor Sampling vs Local (Fast) Sampling
# Includes:
#   - sample_original
#   - sample_fast
#   - sample_truefast (LocalSampler)
# Tests:
#   1. Product state → SAME
#   2. GHZ state     → DIFFERENT
# Also benchmarks each method (only once)
###############################################################

using ITensors
using Random
using Statistics
using BenchmarkTools
using Printf


###############################################################
# 0. TrueFast Sampler (LocalSampler)
###############################################################

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
        maxd = max(maxd, d)

        proj_dag[j] = Vector{ITensor}(undef, d)
        for n in 1:d
            t = ITensor(s)
            t[s => n] = 1.0
            proj_dag[j][n] = dag(t)
        end
    end

    return LocalSampler(sites, proj_dag, zeros(Float64, maxd))
end

function sample_truefast(sampler::LocalSampler, m::MPS)
    N = length(m)
    result = Vector{Int}(undef, N)

    probs = sampler.probs
    proj_dag = sampler.proj_dag

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

        probs[1:d] ./= total

        r = rand()
        cum = 0.0
        for n in 1:d
            cum += probs[n]
            if r ≤ cum
                result[j] = n
                break
            end
        end
    end

    return result
end


# Turn [1,2,1] into "010" (subtract 1 to map 1→0, 2→1)
bits_to_string(bs::Vector{Int}) = join(string.(bs .- 1))

###############################################################
# 1. Original sampler (slow, allocating)
###############################################################
function sample_original(m::MPS)
    N = length(m)
    result = Vector{Int}(undef, N)
    for j in 1:N
        s = siteind(m, j)
        d = dim(s)
        probs = zeros(Float64, d)

        for n in 1:d
            t = ITensor(s)
            t[s=>n] = 1.0
            v = m[j] * dag(t)
            probs[n] = norm(v)^2
        end
        probs ./= sum(probs)

        r = rand()
        cum = 0.0
        for n in 1:d
            cum += probs[n]
            if r ≤ cum
                result[j] = n
                break
            end
        end
    end

    return result
end

###############################################################
# 2. Fast (preallocates projectors each call)
###############################################################
function sample_fast(m::MPS)
    N = length(m)
    sites = siteinds(m)
    maxd = maximum(dim.(sites))
    probs = zeros(Float64, maxd)

    proj_dag = [ [dag((t=ITensor(s); t[s=>n]=1; t)) for n in 1:dim(s)] for s in sites ]

    result = Vector{Int}(undef, N)

    for j in 1:N
        m_j = m[j]
        d = length(proj_dag[j])

        total = 0.0
        for n in 1:d
            v = m_j * proj_dag[j][n]
            pn = norm(v)^2
            probs[n] = pn
            total += pn
        end
        probs[1:d] ./= total

        r = rand()
        cum = 0.0
        for n in 1:d
            cum += probs[n]
            if r ≤ cum
                result[j] = n
                break
            end
        end
    end

    return result
end

###############################################################
# ============= CASE 1: PRODUCT STATE =========================
###############################################################

function test_product_state()
    println("\n==============================")
    println("TEST 1: PRODUCT STATE (should match)")
    println("==============================")

    sites = siteinds("Qubit", 3)

    # |010⟩ product state
   function basis_ket(s::Index, n::Int)
    t = ITensor(s)
    t[s => n] = 1.0
    return t
end

psi = MPS([
    basis_ket(sites[1], 1),   # |0⟩
    basis_ket(sites[2], 2),   # |1⟩
    basis_ket(sites[3], 1)    # |0⟩
])
    orthogonalize!(psi, 1)

    sampler = LocalSampler(sites)

    println("Exact  sample: ", sample(psi))
    println("Local TrueFast: ", sample_truefast(sampler, psi))
end


###############################################################
# ============= CASE 2: GHZ STATE =============================
###############################################################

function test_ghz()
    println("\n==============================")
    println("TEST 2: GHZ ENTANGLED STATE (should DIFFER)")
    println("==============================")

    sites = siteinds("Qubit", 3)

    psi0 = productMPS(sites, "0")
    psi1 = productMPS(sites, "1")
    psi = psi0 + psi1
    normalize!(psi)
    orthogonalize!(psi, 1)

    sampler = LocalSampler(sites)

    println("Exact  sample: ", sample(psi))
    println("Local TrueFast: ", sample_truefast(sampler, psi))
end



function bitstring_histogram_ghz(; trials=20_000)
    println("\n==============================")
    println("Bitstring histogram (GHZ, exact vs local)")
    println("==============================")

    # 3-qubit GHZ state
    sites = siteinds("Qubit", 3)
    psi0 = productMPS(sites, "0")
    psi1 = productMPS(sites, "1")
    psi = psi0 + psi1
    normalize!(psi)
    orthogonalize!(psi, 1)

    sampler = LocalSampler(sites)

    # frequency maps: bitstring "000", "001", ..., "111"
    hist_exact = Dict{String, Int}()
    hist_local = Dict{String, Int}()

    # initialize all 8 bitstrings to 0 (optional but nice for display)
    for b1 in 0:1, b2 in 0:1, b3 in 0:1
        key = string(b1, b2, b3)
        hist_exact[key] = 0
        hist_local[key] = 0
    end

    for _ in 1:trials
        # exact sampling of the full MPS
        s_exact = sample(psi)
        key_exact = bits_to_string(s_exact)
        hist_exact[key_exact] += 1

        # local TrueFast sampling
        s_local = sample_truefast(sampler, psi)
        key_local = bits_to_string(s_local)
        hist_local[key_local] += 1
    end

    # normalize to probabilities
    for (k, v) in hist_exact
        hist_exact[k] = v
    end
    for (k, v) in hist_local
        hist_local[k] = v
    end

    println("\nBitstring  |  Exact (p)   |  Local TrueFast (p)")
    println("-----------+-------------+--------------------")
    for key in sort(collect(keys(hist_exact)))
        p_exact = hist_exact[key] / trials
        p_local = hist_local[key] / trials
        @printf("   %s     |  %7.4f    |    %7.4f\n", key, p_exact, p_local)
    end
end
###############################################################
# ============ HISTOGRAM COMPARISON ===========================
###############################################################

function histogram_test(; trials=20000)
    println("\n==============================")
    println("Histogram comparison (entangled GHZ)")
    println("==============================")

    sites = siteinds("Qubit", 3)

    psi = normalize!(productMPS(sites, "0") + productMPS(sites, "1"))
    orthogonalize!(psi, 1)

    sampler = LocalSampler(sites)

    hist_exact = zeros(2)
    hist_true  = zeros(2)

    for _ in 1:trials
        s1 = sample(psi)[1]               # exact
        hist_exact[s1] += 1

        s2 = sample_truefast(sampler, psi)[1]
        hist_true[s2] += 1
    end

    hist_exact ./= trials
    hist_true  ./= trials

    println("\nExact sample probabilities:    ", hist_exact)
    println("Local TrueFast probabilities: ", hist_true)
    println("Difference (local - exact):    ", hist_true - hist_exact)
end


###############################################################
# ============ BENCHMARK ======================================
###############################################################

function benchmark_once()
    println("\n==============================")
    println("Benchmarking (N=20, bond=10)")
    println("==============================")

    N = 20
    sites = siteinds("Qubit", N)
    psi = randomMPS(sites; linkdims=10)
    sampler = LocalSampler(sites)

    println("\nSample original:")
    @btime sample_original($psi)

    println("\nSample fast:")
    @btime sample_fast($psi)

    println("\nSample truefast:")
    @btime sample_truefast($sampler, $psi)
end

###############################################################
# MAIN
###############################################################
function main()
    test_product_state()
    test_ghz()
    histogram_test()
    bitstring_histogram_ghz()
    benchmark_once()
end

main()