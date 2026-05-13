#!/usr/bin/env julia
# End-to-end Julia entrypoint: read Python's market data → run classical solvers
# → write comparison_summary_julia.csv. Mirror of src/runner.py (classical side only).

using Pkg
Pkg.activate(dirname(@__FILE__); io=devnull)

using Statistics
using LinearAlgebra
using Printf
using CSV
using DataFrames

include(joinpath(@__DIR__, "src", "PortfolioOptimizer.jl"))
using .PortfolioOptimizer.Metrics
using .PortfolioOptimizer.MarketData
using .PortfolioOptimizer.Optimizer
using .PortfolioOptimizer.Output

const RISK_DISCLAIMER = """
RISK NOTICE — please read carefully:
  1. Past returns do NOT predict future results.
  2. Crypto assets are extremely volatile and can lose value rapidly.
  3. Quantum algorithms here only solve the QUBO under the given mu/Sigma; they do not forecast prices.
  4. This tool is for research/education only and is NOT investment advice."""

const CLASSIC_OBJECTIVES = (
    "maximize_sharpe", "minimize_volatility", "maximize_return",
    "minimize_cvar", "constrained_sharpe",
)

struct Args
    outputs::String
    step::Float64
    objective::String
    risk_free_rate::Float64
    risk_aversion::Float64
    max_drawdown::Float64
    verbose::Bool
end

function parse_args(argv::Vector{String})
    outputs = "outputs"
    step = 0.01
    objective = "maximize_sharpe"
    risk_free_rate = 0.0
    risk_aversion = 1.0
    max_drawdown = -0.5
    verbose = false

    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--outputs"
            outputs = argv[i + 1]; i += 2
        elseif a == "--step"
            step = parse(Float64, argv[i + 1]); i += 2
        elseif a == "--objective"
            objective = argv[i + 1]; i += 2
        elseif a == "--risk-free-rate"
            risk_free_rate = parse(Float64, argv[i + 1]); i += 2
        elseif a == "--risk-aversion"
            risk_aversion = parse(Float64, argv[i + 1]); i += 2
        elseif a == "--max-drawdown"
            max_drawdown = parse(Float64, argv[i + 1]); i += 2
        elseif a == "--verbose"
            verbose = true; i += 1
        elseif a == "--help" || a == "-h"
            print_help(); exit(0)
        else
            error("unknown argument: $a (try --help)")
        end
    end

    objective in CLASSIC_OBJECTIVES ||
        error("--objective must be one of $(CLASSIC_OBJECTIVES), got $objective")
    0 < step <= 0.5 || error("--step must be in (0, 0.5], got $step")

    return Args(outputs, step, objective, risk_free_rate, risk_aversion, max_drawdown, verbose)
end

function print_help()
    println("""
    Usage: julia --project=julia julia/run.jl [options]

    Reads basic_stats.json + returns.csv from --outputs (written by the
    Python pipeline with --save-intermediate), runs classical grid search +
    two Mean-Variance solvers (Optim.jl, JuMP+Ipopt) + benchmarks, writes
    comparison_summary_julia.csv into the same directory.

    Options:
      --outputs DIR             Output directory (default: outputs)
      --step FLOAT              Grid step over BTC weight (default: 0.01)
      --objective NAME          One of: $(join(CLASSIC_OBJECTIVES, ", "))
      --risk-free-rate FLOAT    (default: 0.0)
      --risk-aversion FLOAT     (default: 1.0)
      --max-drawdown FLOAT      Threshold for constrained_sharpe (default: -0.5)
      --verbose                 More logging
      -h, --help                Show this help
    """)
end

# Convert a single grid-row DataFrameRow into our standard summary NamedTuple
# (used so grid + MV + benchmarks all flow through Output.build_row).
function _row_to_metrics(row)
    return (
        annual_return = row.annual_return,
        annual_volatility = row.annual_volatility,
        sharpe_ratio = row.sharpe_ratio,
        sortino_ratio = row.sortino_ratio,
        max_drawdown = row.max_drawdown,
        var_95 = row.var_95,
        cvar_95 = row.cvar_95,
        final_cumulative_return = row.final_cumulative_return,
        objective_mv = row.objective_mv,
    )
end

function run_pipeline(args::Args)
    println("=" ^ 60)
    println("[julia] outputs   = $(args.outputs)")
    println("[julia] step      = $(args.step)")
    println("[julia] objective = $(args.objective)")
    println("=" ^ 60)

    market = load_market(args.outputs)
    factor = market.annualization_factor
    println("[julia] tickers = $(market.tickers)")
    println("[julia] window  = $(market.window_start) → $(market.window_end)")
    println("[julia] rows    = $(nrow(market.returns))")
    println("[julia] factor  = $factor")

    rows = Dict{Symbol,Any}[]

    # ---- 1. Grid search -------------------------------------------------
    print("[julia] grid search...")
    t0 = time()
    try
        grid, best = grid_search(market.returns, market.μ, market.Σ;
                                 step=args.step, objective=args.objective,
                                 risk_free_rate=args.risk_free_rate,
                                 risk_aversion=args.risk_aversion,
                                 max_drawdown=args.max_drawdown, factor=factor)
        elapsed = time() - t0
        CSV.write(joinpath(args.outputs, "julia_grid_search_results.csv"), grid)
        row = build_row(
            method="Classic Julia Grid Search",
            solver=@sprintf("grid step=%g obj=%s", args.step, args.objective),
            w=[best.btc_weight, best.bnb_weight],
            metrics=_row_to_metrics(best),
            runtime_seconds=elapsed,
        )
        push!(rows, row)
        flush_summary(rows, args.outputs)
        @printf(" best BTC=%.4f, %.1fms\n", best.btc_weight, elapsed * 1000)
    catch e
        elapsed = time() - t0
        push!(rows, build_empty_row(
            method="Classic Julia Grid Search",
            solver=@sprintf("grid step=%g obj=%s", args.step, args.objective),
            error="$(typeof(e).name.name): $e",
            runtime_seconds=elapsed,
        ))
        flush_summary(rows, args.outputs)
        println(" FAILED: $e")
    end

    # ---- 2. Mean-Variance (Optim.jl) ----------------------------------
    print("[julia] mean-variance (Optim.jl)...")
    t0 = time()
    try
        res = mean_variance_optim(market.returns, market.μ, market.Σ;
                                  objective=args.objective,
                                  risk_free_rate=args.risk_free_rate,
                                  risk_aversion=args.risk_aversion, factor=factor)
        elapsed = time() - t0
        if get(res, :skipped, false)
            push!(rows, build_empty_row(
                method="Classic Julia Mean-Variance",
                solver="Optim.jl LBFGS+softmax",
                error=res.reason, runtime_seconds=elapsed,
            ))
            println(" skipped: $(res.reason)")
        else
            w = res.w
            m = summarize_weights(w, market.returns, market.μ, market.Σ;
                                  risk_free_rate=args.risk_free_rate,
                                  risk_aversion=args.risk_aversion, factor=factor)
            push!(rows, build_row(
                method="Classic Julia Mean-Variance",
                solver=@sprintf("Optim.jl LBFGS+softmax obj=%s", args.objective),
                w=w, metrics=m, runtime_seconds=elapsed,
            ))
            @printf(" w=[%.4f, %.4f], %.1fms\n", w[1], w[2], elapsed * 1000)
        end
        flush_summary(rows, args.outputs)
    catch e
        elapsed = time() - t0
        push!(rows, build_empty_row(
            method="Classic Julia Mean-Variance",
            solver="Optim.jl LBFGS+softmax",
            error="$(typeof(e).name.name): $e", runtime_seconds=elapsed,
        ))
        flush_summary(rows, args.outputs)
        println(" FAILED: $e")
    end

    # ---- 3. Mean-Variance (JuMP + Ipopt) -----------------------------
    print("[julia] mean-variance (JuMP+Ipopt)...")
    t0 = time()
    try
        res = mean_variance_jump(market.returns, market.μ, market.Σ;
                                 objective=args.objective,
                                 risk_free_rate=args.risk_free_rate,
                                 risk_aversion=args.risk_aversion, factor=factor)
        elapsed = time() - t0
        if get(res, :skipped, false)
            push!(rows, build_empty_row(
                method="Classic Julia Mean-Variance",
                solver="JuMP+Ipopt",
                error=res.reason, runtime_seconds=elapsed,
            ))
            println(" skipped: $(res.reason)")
        else
            w = res.w
            m = summarize_weights(w, market.returns, market.μ, market.Σ;
                                  risk_free_rate=args.risk_free_rate,
                                  risk_aversion=args.risk_aversion, factor=factor)
            push!(rows, build_row(
                method="Classic Julia Mean-Variance",
                solver=@sprintf("JuMP+Ipopt obj=%s", args.objective),
                w=w, metrics=m, runtime_seconds=elapsed,
                note="status=$(res.status)",
            ))
            @printf(" w=[%.4f, %.4f], %.1fms\n", w[1], w[2], elapsed * 1000)
        end
        flush_summary(rows, args.outputs)
    catch e
        elapsed = time() - t0
        push!(rows, build_empty_row(
            method="Classic Julia Mean-Variance",
            solver="JuMP+Ipopt",
            error="$(typeof(e).name.name): $e", runtime_seconds=elapsed,
        ))
        flush_summary(rows, args.outputs)
        println(" FAILED: $e")
    end

    # ---- 4. Benchmarks --------------------------------------------------
    println("[julia] benchmarks...")
    bench = evaluate_benchmarks(market.returns, market.μ, market.Σ;
                                tickers=market.tickers,
                                risk_free_rate=args.risk_free_rate,
                                risk_aversion=args.risk_aversion, factor=factor)
    for (label, w) in bench
        m = summarize_weights(w, market.returns, market.μ, market.Σ;
                              risk_free_rate=args.risk_free_rate,
                              risk_aversion=args.risk_aversion, factor=factor)
        push!(rows, build_row(
            method=replace(label, r"^Benchmark " => "Julia Benchmark "),
            solver="fixed", w=w, metrics=m, runtime_seconds=0.0,
        ))
    end
    flush_summary(rows, args.outputs)

    # ---- summary --------------------------------------------------------
    println("=" ^ 60)
    println("[julia] $(length(rows)) row(s) written to comparison_summary_julia.csv")
    println("=" ^ 60)
    println(RISK_DISCLAIMER)
    println("=" ^ 60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_args(ARGS)
    run_pipeline(args)
end
