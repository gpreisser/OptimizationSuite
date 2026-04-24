using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using OptimizationSuite
using Printf

instance_type = :gset
gset = 12
seed = 2
attempts = 1000
percentage = 0.2

function print_run_block(backend_name::AbstractString, params::Vector{String})
    println("====================================================")
    println("Running backend: $(backend_name)")
    println("====================================================")
    println("Instance: Gset G$(gset)")
    println("Parameters:")
    for param in params
        println("  - $(param)")
    end
    println("====================================================")
end

configs = [
    (
        name = "qiils",
        params = [
            "lambda_sweep=0.29",
            "attempts=$(attempts)",
            "sweeps_per_attempt=100",
            "percentage=$(percentage)",
            "seed=$(seed)",
            "angle_conv=0.1",
        ],
        kwargs = (
            backend = :qiils,
            instance_type = instance_type,
            gset = gset,
            seed = seed,
            attempts = attempts,
            percentage = percentage,
            lambda_sweep = 0.29,
            sweeps_per_attempt = 100,
            angle_conv = 0.1,
        ),
    ),
    (
        name = "qiigs",
        params = [
            "solver=:grad",
            "lambda=0.5",
            "attempts=$(attempts)",
            "iterations=1000",
            "inner_iterations=100",
            "percentage=$(percentage)",
            "seed=$(seed)",
            "tao=0.1",
            "angle_conv=0.1",
            "init_mode=:uniform",
            "mix_strategy=:current",
        ],
        kwargs = (
            backend = :qiigs,
            instance_type = instance_type,
            gset = gset,
            seed = seed,
            attempts = attempts,
            percentage = percentage,
            # Alternative exploratory settings:
            # init_mode = :updown can start closer to discrete spin configurations.
            # mix_strategy = :best can improve quality by perturbing the best solution found so far.
            solver = :grad,
            lambda = 0.5,
            iterations = 1000,
            inner_iterations = 100,
            tao = 0.1,
            angle_conv = 0.1,
            init_mode = :uniform,
            mix_strategy = :current,
            save_params = true,
        ),
    ),
    # QiILS-ITensor (MPS backend)
    # More expressive but significantly slower.
    # Intended for exploratory / research use (e.g. studying bond-dimension effects),
    # not for lightweight comparisons, so it is disabled here.
    # (
    #     name = "qiils_itensor",
    #     params = [
    #         "lambda_sweep=0.5",
    #         "attempts=5",
    #         "sweeps_per_attempt=80",
    #         "percentage=$(percentage)",
    #         "maxdim=2",
    #         "sample_mode=:local",
    #     ],
    #     kwargs = (
    #         backend = :qiils_itensor,
    #         instance_type = instance_type,
    #         gset = gset,
    #         attempts = 5,
    #         percentage = percentage,
    #         lambda_sweep = 0.5,
    #         sweeps_per_attempt = 80,
    #         maxdim = 2,
    #         sample_mode = :local,
    #     ),
    # ),
]

results = NamedTuple[]

for config in configs
    print_run_block(config.name, config.params)
    local result
    runtime = @elapsed begin
        if config.name == "qiils"
            result = solve_instance(; config.kwargs...)
        else
            result = redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                    solve_instance(; config.kwargs...)
                end
            end
        end
    end
    best_cut = result["best_cut"]
    approx_ratio = result["approximation_ratio"]
    println("Finished $(config.name): best_cut=$(best_cut), ratio=$(approx_ratio === nothing ? "none" : approx_ratio), runtime=$(@sprintf("%.6f", runtime))s")
    push!(results, (
        backend = config.name,
        best_cut = best_cut,
        approximation_ratio = approx_ratio,
        runtime = runtime,
    ))
end

println("====================================================")
println("Comparison Summary")
println("====================================================")
println("backend | best_cut | approx_ratio | runtime")
for row in results
    best_cut = row.best_cut === nothing ? "none" : string(row.best_cut)
    approx_ratio = row.approximation_ratio === nothing ? "none" : string(row.approximation_ratio)
    runtime = @sprintf("%.6f", row.runtime)
    println("$(row.backend) | $(best_cut) | $(approx_ratio) | $(runtime)")
end
