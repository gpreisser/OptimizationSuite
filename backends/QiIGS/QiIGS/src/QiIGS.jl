module QiIGS

include("types.jl")
include("graphio.jl")
include("solutionio.jl")
include("solver.jl")
include("metrics.jl")

export Solution, solve!, load_weight_matrix, graph_path,
       akmax_solution_path, load_optimal_cut, load_optimal_ising_energy,
       total_edge_weight_upper, cut_hat_and_ratio
export qiigs_solve
export canonical_spin_config
export spin_config_key

end
