using Optim

function continuous_energy(W::SparseMatrixCSC{T}, θ::AbstractVector{T}, λ::T) where {T<:AbstractFloat}
    N = length(θ)
    @assert size(W, 1) == N && size(W, 2) == N

    c2 = Vector{T}(undef, N)
    local_term = zero(T)
    @inbounds for i in 1:N
        c2[i] = cos(2 * θ[i])
        local_term += sin(2 * θ[i])
    end

    edge_term = zero(T)
    @inbounds for col in 1:N
        for idx in W.colptr[col]:(W.colptr[col + 1] - 1)
            row = W.rowval[idx]
            if row < col
                edge_term += W.nzval[idx] * c2[row] * c2[col]
            end
        end
    end

    return λ * edge_term + (one(T) - λ) * local_term
end

function continuous_gradient!(G::AbstractVector{T}, W::SparseMatrixCSC{T}, θ::AbstractVector{T}, λ::T) where {T<:AbstractFloat}
    N = length(θ)
    @assert length(G) == N

    c2 = Vector{T}(undef, N)
    @inbounds for i in 1:N
        c2[i] = cos(2 * θ[i])
    end

    @inbounds for i in 1:N
        a = zero(T)
        for idx in W.colptr[i]:(W.colptr[i + 1] - 1)
            j = W.rowval[idx]
            a += W.nzval[idx] * c2[j]
        end
        G[i] = -2 * λ * a * sin(2 * θ[i]) - 2 * (one(T) - λ) * cos(2 * θ[i])
    end

    return G
end

function solve_lbfgs!(
    W,
    N;
    seed::Integer,
    lambda,
    theta0 = nothing,
    init_mode::Symbol = :updown,
    save_params::Bool = false,
    maxiter::Int = 1000,
    g_tol = 1e-8,
    f_tol = 0.0,
    kwargs...,
)
    T = Float64
    λ = T(lambda)

    Wsp = (W isa SparseMatrixCSC) ? SparseMatrixCSC{T, Int}(W) : sparse(T.(W))
    @assert size(Wsp, 1) == N && size(Wsp, 2) == N

    local_rng = MersenneTwister(seed)

    θ0 = if theta0 === nothing
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

    θ_init = copy(θ0)

    f(θ) = continuous_energy(Wsp, θ, λ)

    function g!(G, θ)
        continuous_gradient!(G, Wsp, θ, λ)
        return nothing
    end

    options = Optim.Options(
        iterations = maxiter,
        g_tol = T(g_tol),
        f_abstol = T(f_tol),
        store_trace = false,
        show_trace = false,
    )

    t0 = time()
    result = Optim.optimize(f, g!, θ0, Optim.LBFGS(), options)
    runtime = time() - t0

    θ_best = Float64.(Optim.minimizer(result))
    Gfinal = similar(θ_best)
    continuous_gradient!(Gfinal, Wsp, θ_best, λ)
    gn_final = norm(Gfinal)
    conf = round_configuration(θ_best)
    E_best = energy_from_spins(Wsp, conf)
    θ_out = save_params ? copy(θ_best) : T[]

    md = Dict{Symbol, Any}(
        :solver => :lbfgs,
        :optim_converged => Optim.converged(result),
        :optim_iterations => Optim.iterations(result),
        :optim_minimum => Optim.minimum(result),
        :f_calls => try Optim.f_calls(result) catch; nothing end,
        :g_calls => try Optim.g_calls(result) catch; nothing end,
        :runtime => runtime,
        :continuous_grad_norm_final => gn_final,
        :discrete_energy => E_best,
        :theta_best => copy(θ_best),
        :theta_converged => copy(θ_best),
        :theta_last_converged => copy(θ_best),
        :init_mode => init_mode,
        :lambda => λ,
    )

    if save_params
        md[:theta_init] = copy(θ_init)
    end

    return Solution(T(E_best), conf, T(gn_final), θ_out, md)
end
