module QiILS_ITensor

using ITensors
using ITensorMPS
using Random
using Graphs
using Printf

# Graphs
include("graphs/load_graph.jl")
export load_graph, create_and_save_graph_QiILS

# Solutions
include("graphs/load_solution.jl")
export solution_file_path, load_optimal_cut, get_gset_optimal_cut

# Sampling utilities (NOT a module)
include("sampling/sampling.jl")
export LocalSampler, sample_mps

# Solver (existing, multi-attempt + mixing)
include("solver/qiils_itensor_solver.jl")
export qiils_itensor_solver

# Transition / λ-sweep helper (NEW, single-shot, no mixing, X-init)
include("solver/qiils_itensor_transition_solver.jl")
export qiils_itensor_minimize_then_measure

end # module