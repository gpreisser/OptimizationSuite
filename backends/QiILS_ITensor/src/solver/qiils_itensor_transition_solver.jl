using ITensors
using ITensorMPS
using Statistics

# ------------------------------------------------------------
# Helper: build the |+>^{⊗N} product MPS (X-polarized state)
# ------------------------------------------------------------
function x_product_mps(sites::Vector{<:Index}, which::Symbol)
    N = length(sites)
    labels = which === :plus ? ("+", "Plus", "plus") : ("-", "Minus", "minus")
    for lab in labels
        try
            return productMPS(sites, fill(lab, N))
        catch
        end
    end
    error("Could not build X-$(which) product MPS. Check valid state labels for siteinds(\"Qubit\", N).")
end

# ------------------------------------------------------------
# Helper: compute <Z_i> for all sites, minimal + robust
# ------------------------------------------------------------
function local_expect_Z(ψ::MPS, sites::Vector{<:Index})
    # Try the fast built-in route first
    try
        return ITensorMPS.expect(ψ, "Z")
    catch
        # fall back below
    end

    N = length(sites)
    mz = Vector{Float64}(undef, N)

    # Fallback: <ψ|Z_i|ψ> by applying a 1-site operator and taking overlap
    # This is O(N^2)ish but fine for N=50.
    for i in 1:N
        Zi  = op("Z", sites, i)         # 1-site ITensor operator
        ψZi = apply(Zi, ψ)              # apply operator to MPS
        mz[i] = real(inner(ψ, ψZi))
    end
    return mz
end

# ------------------------------------------------------------
# ITensor analogue of qiils_minimize_then_measure
# One run: init in X product state, DMRG minimize, measure energy + dev
# ------------------------------------------------------------
"""
    qiils_itensor_minimize_then_measure(
        wg,
        λ::Float64;
        nsweeps::Int = 80,
        maxdim::Int = 64,
        hilbert = nothing,
        weighted::Bool = true,
        return_psi::Bool = false,
    )

Run a single DMRG minimization of H(λ) starting from the X product state |+>^{⊗N}.
Return:
  energy,
  devZ_abs      = mean_i |<Z_i>|,
  devTheta_abs  = mean_i | 0.5*acos(|<Z_i>|) - π/4 |,
  (optionally) best_psi, hilbert, mz
"""
function qiils_itensor_minimize_then_measure(
    wg,
    λ::Float64;
    nsweeps::Int = 80,
    maxdim::Int = 64,
    hilbert = nothing,
    weighted::Bool = true,
    return_psi::Bool = false,
)

    N = nv(wg)
    sites = hilbert === nothing ? siteinds("Qubit", N) : hilbert

    # Build H(λ) using your existing constructor
    H = build_H_mpo(wg, sites, Float64(λ); weighted=weighted)
    Hcost = build_H_mpo(wg, sites, 1.0; weighted=weighted)

    # X-polarized init: |+>^{⊗N}
    ψ0 = x_product_mps(sites, :minus)
    

    # Single DMRG minimization
    energy, ψ = dmrg(H, ψ0; nsweeps=nsweeps, maxdim=maxdim, outputlevel=0)
    energy_cost = real(inner(ψ, Apply(Hcost, ψ)))

    

    # Measure <Z_i>
    mz = local_expect_Z(ψ, sites)
    abs_mz = abs.(mz)

    # Metrics
    devZ_abs = mean(abs_mz)

    # angle-like mapping: θ_i = 0.5*acos(|<Z_i>|), dev = mean |θ_i - π/4|
    # (clamp for safety against tiny numerical overshoots outside [-1,1])
    θeff = 0.5 .* acos.(clamp.(abs_mz, 0.0, 1.0))
    devTheta_abs = mean(abs.(θeff .- (π/4)))

    if return_psi
        return energy, energy_cost, devZ_abs, devTheta_abs, ψ, sites, mz
    else
        return energy, energy_cost, devZ_abs, devTheta_abs
    end
end