using Test

using OptimizationSuite

@testset "OptimizationSuite smoke tests" begin
    graph, meta = load_instance_graph(instance_type = :gset, gset = 12)
    @test graph !== nothing
    @test meta["gset"] == 12

    qiils_result = solve_instance(
        backend = :qiils,
        instance_type = :gset,
        gset = 12,
        lambda_sweep = 0.29,
        attempts = 1,
        sweeps_per_attempt = 5,
        percentage = 0.2,
        seed = 2,
        angle_conv = 0.1,
    )

    @test haskey(qiils_result, "best_cut")
    @test haskey(qiils_result, "approximation_ratio")

    qiigs_result = solve_instance(
        backend = :qiigs,
        instance_type = :gset,
        gset = 12,
        solver = :grad,
        lambda = 0.5,
        attempts = 1,
        iterations = 10,
        inner_iterations = 5,
        percentage = 0.2,
        seed = 2,
        tao = 0.1,
        angle_conv = 0.1,
        init_mode = :uniform,
        mix_strategy = :current,
        save_params = false,
    )

    @test haskey(qiigs_result, "best_cut")
    @test haskey(qiigs_result, "approximation_ratio")
end
