using Random
using LinearAlgebra
using SparseArrays
using Statistics

# -----------------------------
# Init (angles)
# -----------------------------

function rand_updown_angles!(rng::AbstractRNG, θ::Vector{T}) where {T<:AbstractFloat}
    @inbounds for i in eachindex(θ)
        θ[i] = rand(rng, Bool) ? T(0) : T(pi/2)
    end
    return nothing
end

function rand_uniform_angles!(rng::AbstractRNG, θ::Vector{T}; θmax = pi/2) where {T<:AbstractFloat}
    rand!(rng, θ)
    θ .*= T(θmax)
    return nothing
end

# -----------------------------
# Rounding angles -> spins
# -----------------------------

function round_configuration(θ::AbstractVector{T}) where {T<:AbstractFloat}
    conf = Vector{Int8}(undef, length(θ))
    thr = T(pi/4)

    @inbounds for i in eachindex(θ)
        θi = mod(θ[i], T(pi))
        conf[i] = (θi < thr) ? Int8(1) : Int8(-1)
    end

    return conf
end

# -----------------------------
# Angle-space diagnostics
# -----------------------------

function angle_deviation_metrics(θ::AbstractVector{T}) where {T<:AbstractFloat}
    N = length(θ)

    sum_abs_z_like = zero(T)
    sum_dev_theta = zero(T)

    @inbounds for i in 1:N
        zi = abs(cos(2 * θ[i]))
        zi = clamp(zi, zero(T), one(T))
        θeff = acos(zi) / 2
        sum_abs_z_like += zi
        sum_dev_theta += abs(θeff - (T(pi) / 4))
    end

    devZ_abs = sum_abs_z_like / N
    devTheta_abs = sum_dev_theta / N
    return devZ_abs, devTheta_abs
end

# -----------------------------
# Energy from spins / angles
# -----------------------------

function energy_from_spins(W::SparseMatrixCSC{T}, s::AbstractVector{<:Integer}) where {T<:AbstractFloat}
    N = length(s)
    @assert size(W, 1) == N && size(W, 2) == N
    E = zero(T)
    @inbounds for col in 1:N
        for idx in W.colptr[col]:(W.colptr[col + 1] - 1)
            row = W.rowval[idx]
            if row < col
                E += W.nzval[idx] * T(s[row]) * T(s[col])
            end
        end
    end
    return E
end

function energy_from_angles(W::SparseMatrixCSC{T}, θ::AbstractVector{T}) where {T<:AbstractFloat}
    s = round_configuration(θ)
    return energy_from_spins(W, s)
end

# Pre-rounding continuous energy evaluated with the final Hamiltonian λ = 1
function energy_lambda1_preround(
    W::SparseMatrixCSC{T},
    θ::AbstractVector{T},
) where {T<:AbstractFloat}
    N = length(θ)
    @assert size(W, 1) == N && size(W, 2) == N

    c2 = Vector{T}(undef, N)
    @inbounds for i in 1:N
        c2[i] = cos(2 * θ[i])
    end

    E = zero(T)
    @inbounds for col in 1:N
        for idx in W.colptr[col]:(W.colptr[col + 1] - 1)
            row = W.rowval[idx]
            if row < col
                E += W.nzval[idx] * c2[row] * c2[col]
            end
        end
    end
    return E
end

# -----------------------------
# Sweep + gradient norm
# -----------------------------

function sweep!(
    θold::AbstractVector{T},
    λ::T,
    W::SparseMatrixCSC{T},
    tao::T,
    θnew::AbstractVector{T},
    c2::AbstractVector{T},
) where {T<:AbstractFloat}

    N = length(θold)
    s2 = zero(T)

    @inbounds for i in 1:N
        c2[i] = cos(2 * θold[i])
    end

    @inbounds for i in 1:N
        a = zero(T)

        for idx in W.colptr[i]:(W.colptr[i + 1] - 1)
            j = W.rowval[idx]
            w = W.nzval[idx]
            a += w * c2[j]
        end

        θ = θold[i]
        s = sin(2 * θ)
        c = c2[i]

        grad = -2 * λ * a * s - 2 * (1 - λ) * c

        s2 += grad * grad
        θnew[i] = θ - tao * grad
    end

    return θnew, sqrt(s2)
end

function grad_norm(θ::AbstractVector{T}, λ::T, W::SparseMatrixCSC{T}) where {T<:AbstractFloat}
    N = length(θ)
    s2 = zero(T)
    @inbounds for i in 1:N
        a = zero(T)
        for idx in W.colptr[i]:(W.colptr[i + 1] - 1)
            j = W.rowval[idx]
            w = W.nzval[idx]
            a += w * cos(2 * θ[j])
        end
        gi = -2 * λ * a * sin(2 * θ[i]) - 2 * (1 - λ) * cos(2 * θ[i])
        s2 += gi * gi
    end
    return sqrt(s2)
end

# -----------------------------
# Hessian helpers
# -----------------------------

function hessian_angles(
    W::SparseMatrixCSC{T},
    θ::AbstractVector{T},
    λ::T,
) where {T<:AbstractFloat}

    N = length(θ)
    @assert size(W, 1) == N && size(W, 2) == N

    s2θ = Vector{T}(undef, N)
    c2θ = Vector{T}(undef, N)
    a   = zeros(T, N)

    @inbounds for i in 1:N
        s2θ[i] = sin(2 * θ[i])
        c2θ[i] = cos(2 * θ[i])
    end

    @inbounds for i in 1:N
        ai = zero(T)
        for idx in W.colptr[i]:(W.colptr[i + 1] - 1)
            j = W.rowval[idx]
            ai += W.nzval[idx] * c2θ[j]
        end
        a[i] = ai
    end

    H = zeros(T, N, N)

    @inbounds for i in 1:N
        H[i, i] = -4 * λ * a[i] * c2θ[i] + 4 * (1 - λ) * s2θ[i]
    end

    @inbounds for col in 1:N
        for idx in W.colptr[col]:(W.colptr[col + 1] - 1)
            row = W.rowval[idx]
            if row < col
                hij = 4 * λ * W.nzval[idx] * s2θ[row] * s2θ[col]
                H[row, col] = hij
                H[col, row] = hij
            end
        end
    end

    return H
end

function hessian_summary(
    W::SparseMatrixCSC{T},
    θ::AbstractVector{T},
    λ::T;
    tol::T = T(1e-8),
) where {T<:AbstractFloat}

    H = hessian_angles(W, θ, λ)
    vals = eigvals(Hermitian(H))

    mineig = minimum(vals)
    maxeig = maximum(vals)

    posvals = vals[vals .> tol]
    minpos = isempty(posvals) ? T(NaN) : minimum(posvals)
    npos = length(posvals)
    nnearzero = count(v -> abs(v) <= tol, vals)
    nneg = count(v -> v < -tol, vals)

    is_minimum = mineig > tol
    is_maximum = maxeig < -tol
    is_saddle = (mineig < -tol) && (maxeig > tol)
    is_degenerate = !(is_minimum || is_maximum || is_saddle)

    cond = is_minimum ? (maxeig / mineig) : T(NaN)
    cond_pos = (!isempty(posvals) && maxeig > tol && minpos > tol) ? (maxeig / minpos) : T(NaN)

    return Dict{Symbol, Any}(
        :hess_mineig => mineig,
        :hess_maxeig => maxeig,
        :hess_minpos => minpos,
        :hess_cond => cond,
        :hess_cond_pos => cond_pos,
        :hess_npos => npos,
        :hess_nneg => nneg,
        :hess_nnearzero => nnearzero,
        :hess_is_minimum => is_minimum,
        :hess_is_maximum => is_maximum,
        :hess_is_saddle => is_saddle,
        :hess_is_degenerate => is_degenerate,
        :hess_tol => tol,
    )
end

function prefixed_hessian_summary(
    prefix::Symbol,
    W::SparseMatrixCSC{T},
    θ::AbstractVector{T},
    λ::T;
    tol::T = T(1e-8),
) where {T<:AbstractFloat}
    raw = hessian_summary(W, θ, λ; tol=tol)
    out = Dict{Symbol, Any}()
    for (k, v) in raw
        out[Symbol(prefix, "_", k)] = v
    end
    return out
end

function empty_prefixed_hessian_summary(prefix::Symbol, tol)
    return Dict{Symbol, Any}(
        Symbol(prefix, "_hess_mineig") => NaN,
        Symbol(prefix, "_hess_maxeig") => NaN,
        Symbol(prefix, "_hess_minpos") => NaN,
        Symbol(prefix, "_hess_cond") => NaN,
        Symbol(prefix, "_hess_cond_pos") => NaN,
        Symbol(prefix, "_hess_npos") => 0,
        Symbol(prefix, "_hess_nneg") => 0,
        Symbol(prefix, "_hess_nnearzero") => 0,
        Symbol(prefix, "_hess_is_minimum") => false,
        Symbol(prefix, "_hess_is_maximum") => false,
        Symbol(prefix, "_hess_is_saddle") => false,
        Symbol(prefix, "_hess_is_degenerate") => false,
        Symbol(prefix, "_hess_tol") => tol,
    )
end

# -----------------------------
# Solver
# -----------------------------

function solve_grad!(
    W,
    N;
    rng = Random.GLOBAL_RNG,
    seed::Integer,
    lambda,
    iterations::Int = 1000,
    inner_iterations::Int = 1,
    tao = 0.1,
    angle_conv = 0.1,
    init_mode::Symbol = :updown,
    theta0 = nothing,
    save_params::Bool = false,
    progressbar::Bool = false,

    # Hessian at init and final point when enabled
    compute_hessian::Bool = false,
    hessian_tol = 1e-8,

    # optional curvature test for optimum-reaching runs only
    compute_optimum_curvature::Bool = false,
    optimal_energy = nothing,
    optimal_energy_atol = 1e-9,
)
    T = Float64
    λ = T(lambda)
    τ = T(tao)
    conv = T(angle_conv)
    htol = T(hessian_tol)
    eopt_atol = T(optimal_energy_atol)

    if compute_optimum_curvature && optimal_energy === nothing
        error("compute_optimum_curvature=true requires optimal_energy to be provided.")
    end

    Eopt_known = optimal_energy === nothing ? nothing : T(optimal_energy)

    Wsp = (W isa SparseMatrixCSC) ? SparseMatrixCSC{T, Int}(W) : sparse(T.(W))
    @assert size(Wsp, 1) == N && size(Wsp, 2) == N

    local_rng = MersenneTwister(seed)

    θ = if theta0 === nothing
        θinit = Vector{T}(undef, N)
        if init_mode == :updown
            rand_updown_angles!(local_rng, θinit)
        elseif init_mode == :uniform
            rand_uniform_angles!(local_rng, θinit; θmax = pi/2)
        else
            error("Unknown init_mode=$(init_mode). Use :updown or :uniform.")
        end
        θinit
    else
        length(theta0) == N || error("theta0 must have length N=$N.")
        T.(copy(theta0))
    end

    θ_init = copy(θ)

    gn_init = grad_norm(θ_init, λ, Wsp)

    θ_best = copy(θ_init)
    E_best = energy_from_angles(Wsp, θ_best)

    θ_old = similar(θ)
    θ_new = similar(θ)
    c2    = similar(θ)

    θ_last_converged = copy(θ_best)

    sweep_counter = 0
    sweep_list = Int64[]

    best_gn_max_inner = zero(T)
    best_gn_sum_inner = zero(T)
    best_gn_cnt_inner = 0

    t0 = time()

    for it in 1:iterations
        θ_old .= θ_best

        cand_gn_max_inner = zero(T)
        cand_gn_sum_inner = zero(T)
        cand_gn_cnt_inner = 0

        for p in 1:inner_iterations
            sweep_counter += 1

            _, gn_sweep = sweep!(θ_old, λ, Wsp, τ, θ_new, c2)

            cand_gn_sum_inner += gn_sweep
            cand_gn_cnt_inner += 1
            cand_gn_max_inner = max(cand_gn_max_inner, gn_sweep)

            maxΔ = 0.0
            @inbounds for i in 1:N
                d = abs(θ_new[i] - θ_old[i])
                maxΔ = max(maxΔ, d)
            end

            scale = 0.0
            @inbounds for i in 1:N
                scale += abs(θ_new[i] - (pi / 4))
            end
            scale /= N
            scale = max(scale, 1e-6)

            θ_old, θ_new = θ_new, θ_old

            if maxΔ < conv * scale
                break
            end
        end

        push!(sweep_list, sweep_counter)
        θ_last_converged .= θ_old

        E_new = energy_from_angles(Wsp, θ_old)

        if (E_new < E_best) || (it == 1 && best_gn_cnt_inner == 0)
            E_best = E_new
            θ_best .= θ_old

            best_gn_max_inner = cand_gn_max_inner
            best_gn_sum_inner = cand_gn_sum_inner
            best_gn_cnt_inner = cand_gn_cnt_inner
        end
    end

    runtime = time() - t0

    gn_final = grad_norm(θ_best, λ, Wsp)
    gn_mean_inner = (best_gn_cnt_inner > 0) ? (best_gn_sum_inner / best_gn_cnt_inner) : T(NaN)

    devZ_abs, devTheta_abs = angle_deviation_metrics(θ_best)
    E_lambda1_preround = energy_lambda1_preround(Wsp, θ_best)

    conf = round_configuration(θ_best)
    θ_out = save_params ? copy(θ_best) : T[]

    optimum_reached = if Eopt_known === nothing
        false
    else
        isapprox(E_best, Eopt_known; atol=eopt_atol, rtol=0.0)
    end

    md = Dict{Symbol, Any}(
        :runtime => runtime,
        :sweep_counter => sweep_counter,
        :sweep_list => sweep_list,
        :iterations => iterations,
        :inner_iterations => inner_iterations,
        :tao => τ,
        :lambda => λ,
        :angle_conv => conv,
        :init_mode => init_mode,

        :gn_init => gn_init,
        :gn_final => gn_final,

        :gn_max_inner => best_gn_max_inner,
        :gn_mean_inner => gn_mean_inner,
        :gn_cnt_inner => best_gn_cnt_inner,
        :inner_sweeps_used => best_gn_cnt_inner,

        :devZ_abs => devZ_abs,
        :devTheta_abs => devTheta_abs,
        :energy_lambda1_preround => E_lambda1_preround,

        :theta_best => copy(θ_best),
        :theta_converged => copy(θ_best),
        :theta_last_converged => copy(θ_last_converged),

        :compute_hessian => compute_hessian,

        :compute_optimum_curvature => compute_optimum_curvature,
        :optimal_energy_target => (Eopt_known === nothing ? NaN : Eopt_known),
        :optimal_energy_atol => eopt_atol,
        :optimal_energy_reached => optimum_reached,
    )

    if save_params
        md[:theta_init] = copy(θ_init)
    end

    # Hessian at the initial random state
    if compute_hessian
        merge!(md, prefixed_hessian_summary(:init, Wsp, θ_init, λ; tol=htol))
    end

    # Hessian at the accepted final state (kept under the original keys)
    if compute_hessian
        merge!(md, hessian_summary(Wsp, θ_best, λ; tol=htol))
    end

    # Optional optimum-curvature test at final state
    if compute_optimum_curvature
        if optimum_reached
            merge!(md, prefixed_hessian_summary(:optcurv, Wsp, θ_best, λ; tol=htol))
        else
            merge!(md, empty_prefixed_hessian_summary(:optcurv, htol))
        end
    end

    return Solution(T(E_best), conf, T(gn_final), θ_out, md)
end
