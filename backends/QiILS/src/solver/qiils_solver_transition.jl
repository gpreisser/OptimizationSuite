using Statistics

function qiils_minimize_then_measure(
    wg,
    λ_min::Float64,
    gvec::Vector{Float64};
    θ0::Union{Nothing,Vector{Float64}} = nothing,
    sweeps::Int = 200,
    angle_conv::Float64 = 1e-6,
    use_scaled_convergence::Bool = true,
)
    N = nv(wg)

    θ = isnothing(θ0) ? fill(π/4, N) : copy(θ0)

    θ_old = similar(θ)
    cos2θ = cos.(2 .* θ)
    sin2θ = sin.(2 .* θ)

    total_sweeps_done = 0

    for sweep in 1:sweeps
        θ_old .= θ
        sweep_pass!(N, wg, λ_min, θ, gvec, cos2θ, sin2θ)
        total_sweeps_done += 1

        Δθ_max = maximum(abs.(θ .- θ_old))
        if use_scaled_convergence
            scaled_tol = max(angle_conv * mean(abs.(θ .- π/4)), 1e-12)
            Δθ_max < scaled_tol && break
        else
            Δθ_max < angle_conv && break
        end
    end

    # deviation from π/4 AFTER minimization (continuous angles)
    dev_meanabs = mean(abs.(θ .- π/4))
    # dev_rms = sqrt(mean((θ .- π/4).^2))  # optional alternative

    θ_meas = finaltheta(θ)
    spins  = angles_to_spins(θ_meas)
    cut_val = maxcut_value(wg, spins)

    return cut_val, dev_meanabs, total_sweeps_done
end