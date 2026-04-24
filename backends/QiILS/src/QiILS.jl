module QiILS

using Random
using LinearAlgebra
using Graphs
using SimpleWeightedGraphs
using JSON
using Statistics
using ProgressMeter

# ------------------
# Graph-related code
# ------------------
include("graphs/gset_solutions.jl")   # only here
include("graphs/load_graph.jl")
include("graphs/load_solution.jl")

# ------------------
# Solver
# ------------------
include("solver/qiils_solver.jl")
include("solver/maxcut_helpers.jl")
include("solver/sa_solver.jl")
include("solver/tabu_solver.jl")
include("solver/qiils_solver_transition.jl")   # <-- add this
include("solver/standard_tabu.jl")  

export load_graph
export load_optimal_cut
export qiils_solve
export get_optimal_cut
export solution_file_path
export create_and_save_graph_QiILS
export sa_maxcut
export tabu_maxcut
export qiils_minimize_then_measure
export tabu_maxcut_fullscan
end
