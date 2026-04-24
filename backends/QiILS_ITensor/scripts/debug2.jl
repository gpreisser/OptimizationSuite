
using Printf

# Change this line to point to the NEW file:
path = "/Users/guillermo.preisser/Projects/QiILS_ITensor/results/N30/k3/seed1/unweighted/runs/lambda1_sw80_pct0.05_maxd1_modelocal/run_s0003.json"

d = JSON.parsefile(path)

println("Loaded: ", path)
println("Keys: ", sort(collect(keys(d))))

println("\nSaved summary fields:")
println("best_cut      = ", get(d, "best_cut", missing))
println("best_ratio    = ", get(d, "best_ratio", missing))
println("one_minus_r   = ", get(d, "one_minus_r", missing))
println("optimal_cut   = ", get(d, "optimal_cut", missing))
println("attempts      = ", get(d, "attempts", missing))

opt = Float64(get(d, "optimal_cut", NaN))

# ----------------------------
# Load histories (if present)
# ----------------------------
best_hist = haskey(d, "best_history") ? Float64.(d["best_history"]) : Float64[]
cut_hist  = haskey(d, "cut_history")  ? Float64.(d["cut_history"])  : Float64[]

println("\nHistory presence:")
println("has best_history? ", haskey(d, "best_history"), " | len = ", length(best_hist))
println("has cut_history?  ", haskey(d, "cut_history"),  " | len = ", length(cut_hist))

# ----------------------------
# Quick summaries
# ----------------------------
if !isempty(best_hist)
    println("\nBest-history summary:")
    println("first 10 = ", best_hist[1:min(10,end)])
    println("last  10 = ", best_hist[max(1,end-9):end])
    println("max(best_hist) = ", maximum(best_hist))
    println("best_hist[end] = ", best_hist[end])
end

if !isempty(cut_hist)
    println("\nCut-history (current) summary:")
    println("first 10 = ", cut_hist[1:min(10,end)])
    println("last  10 = ", cut_hist[max(1,end-9):end])
    println("max(cut_hist) = ", maximum(cut_hist))
    println("cut_hist[end] = ", cut_hist[end])
end

# ----------------------------
# Side-by-side table
# ----------------------------
if !isempty(best_hist) && !isempty(cut_hist)
    length(best_hist) == length(cut_hist) || error("Length mismatch: best_history=$(length(best_hist)) cut_history=$(length(cut_hist))")

    T = length(best_hist)
    println("\nPer-attempt table:")
    println("attempt |   cut(current)   |   best_so_far    | r(current) | 1-r(cur) | r(best) | 1-r(best)")
    println("-----------------------------------------------------------------------------------------------")

    for t in 1:T
        c = cut_hist[t]
        b = best_hist[t]
        rc = c / opt
        rb = b / opt
        @printf("%7d | %14.6f | %14.6f | %9.6f | %8.3e | %7.6f | %8.3e\n",
                t, c, b, rc, 1-rc, rb, 1-rb)
    end
elseif !isempty(best_hist) && isempty(cut_hist)
    println("\nNOTE: This run file has no 'cut_history' key yet.")
    println("      You only saved best_history (best-so-far).")
    println("      Re-run once after updating the saver to include cut_history.")
end

# ----------------------------
# Consistency check
# ----------------------------
if !isempty(best_hist) && !isnan(opt)
    r_from_hist = best_hist[end] / opt
    println("\nConsistency check (from best_history[end]):")
    println("r_from_hist   = ", r_from_hist)
    println("1-r_from_hist = ", 1 - r_from_hist)
end