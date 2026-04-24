# src/solver.jl

using ProgressMeter

include(joinpath("solvers", "grad.jl"))
include(joinpath("solvers", "lbfgs.jl"))

"""
    solve!(W, N; solver=:grad, kwargs...)

Dispatcher for different solvers.
"""
function solve!(W, N; solver::Symbol = :grad, kwargs...)
    if solver === :grad
        return solve_grad!(W, N; kwargs...)
    elseif solver === :lbfgs
        return solve_lbfgs!(W, N; kwargs...)
    else
        error("Unknown solver=$(solver). Available: :grad, :lbfgs")
    end
end

function _total_edge_weight(W)
    return total_edge_weight_upper(W)
end

function _random_theta(N::Int, rng::AbstractRNG, init_mode::Symbol)
    θ = Vector{Float64}(undef, N)
    if init_mode == :updown
        rand_updown_angles!(rng, θ)
    elseif init_mode == :uniform
        rand_uniform_angles!(rng, θ; θmax=pi / 2)
    else
        error("Unknown init_mode=$(init_mode). Use :updown or :uniform.")
    end
    return θ
end

function _mix_theta(theta::AbstractVector{<:AbstractFloat}, percentage::Real, rng::AbstractRNG)
    mixed = Float64.(copy(theta))
    N = length(mixed)
    nflip = max(1, floor(Int, percentage * N))
    idxs = randperm(rng, N)[1:nflip]
    @inbounds for idx in idxs
        mixed[idx] = (pi / 2) - mixed[idx]
    end
    return mixed
end

function qiigs_solve(
    W,
    N;
    solver::Symbol = :grad,
    attempts::Int = 20,
    percentage::Real = 0.2,
    lambda,
    iterations::Int = 1000,
    inner_iterations::Int = 1,
    tao = 0.1,
    angle_conv = 0.1,
    seed::Integer = 1,
    init_mode::Symbol = :updown,
    mix_strategy::Symbol = :best,
    save_params::Bool = false,
    kwargs...,
)
    total_weight = _total_edge_weight(W)
    base_rng = MersenneTwister(seed)
    t0_total = time()

    best_cut = -Inf
    best_configuration = Vector{Int8}()
    best_theta = Vector{Float64}()
    current_theta0 = nothing

    best_history = Vector{Float64}(undef, attempts)
    cut_history = Vector{Float64}(undef, attempts)
    energy_history = Vector{Float64}(undef, attempts)
    grad_norm_history = Vector{Float64}(undef, attempts)

    metadata = Dict{Symbol, Any}(
        :solver => solver,
        :attempts => attempts,
        :percentage => percentage,
        :mix_strategy => mix_strategy,
        :seed => seed,
        :lambda => lambda,
    )
    attempt_metadata = Vector{Dict{Symbol, Any}}(undef, attempts)
    prog = Progress(attempts; desc="QiIGS Attempts", enabled=true)

    for attempt in 1:attempts
        attempt_seed = seed + attempt - 1
        sol = solve!(
            W,
            N;
            solver=solver,
            seed=attempt_seed,
            lambda=lambda,
            iterations=iterations,
            inner_iterations=inner_iterations,
            tao=tao,
            angle_conv=angle_conv,
            init_mode=init_mode,
            theta0=current_theta0,
            save_params=save_params,
            kwargs...,
        )

        cut = (total_weight - sol.energy) / 2
        cut_history[attempt] = cut
        energy_history[attempt] = sol.energy
        grad_norm_history[attempt] = sol.grad_norm

        if cut > best_cut
            best_cut = cut
            best_configuration = copy(sol.configuration)
            best_theta = haskey(sol.metadata, :theta_best) ? Float64.(copy(sol.metadata[:theta_best])) : Float64[]
        end
        best_history[attempt] = best_cut
        attempt_metadata[attempt] = copy(sol.metadata)

        mix_rng = MersenneTwister(seed * 10_000 + attempt)
        if attempt < attempts
            if mix_strategy == :best
                source_theta = isempty(best_theta) ? get(sol.metadata, :theta_best, Float64[]) : best_theta
                current_theta0 = isempty(source_theta) ? _random_theta(N, base_rng, init_mode) :
                    _mix_theta(source_theta, percentage, mix_rng)
            elseif mix_strategy == :current
                source_theta = get(sol.metadata, :theta_best, Float64[])
                current_theta0 = isempty(source_theta) ? _random_theta(N, base_rng, init_mode) :
                    _mix_theta(source_theta, percentage, mix_rng)
            elseif mix_strategy == :random
                current_theta0 = _random_theta(N, mix_rng, init_mode)
            else
                error("Unknown mix_strategy=$(mix_strategy). Use :best, :current, or :random.")
            end
        end

        next!(prog)
    end

    finish!(prog)

    metadata[:runtime] = time() - t0_total
    metadata[:best_cut] = best_cut
    metadata[:total_weight] = total_weight
    metadata[:attempt_metadata] = attempt_metadata

    return best_history,
           cut_history,
           best_configuration,
           best_theta,
           energy_history,
           grad_norm_history,
           metadata
end
