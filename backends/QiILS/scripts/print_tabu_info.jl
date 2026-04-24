# scripts/check_standard_tabu_files.jl

base_dir = "/Users/guillermo.preisser/Projects/QiILS/results/StandardTabu_Gset12_tenures_weighted_ntrials100_sweeps1000000"
tenures = [5, 10, 20, 40, 80]

for tenure in tenures
    final_path = joinpath(
        base_dir,
        "tenure$(tenure)",
        "standard_tabu_results_G12_tenure$(tenure)_sweeps1000000_ntrials100.json",
    )

    checkpoint_path = joinpath(
        base_dir,
        "tenure$(tenure)",
        "standard_tabu_checkpoint_G12_tenure$(tenure)_sweeps1000000_ntrials100.json",
    )

    println("--------------------------------------------------")
    println("tenure = ", tenure)

    println("final exists      = ", isfile(final_path))
    println("final size        = ", isfile(final_path) ? filesize(final_path) : -1)
    println("final path        = ", final_path)

    println("checkpoint exists = ", isfile(checkpoint_path))
    println("checkpoint size   = ", isfile(checkpoint_path) ? filesize(checkpoint_path) : -1)
    println("checkpoint path   = ", checkpoint_path)
end