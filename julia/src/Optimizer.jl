module Optimizer

using DataFrames
using LinearAlgebra
using Optim
using JuMP
using Ipopt
using Printf

using ..Metrics

export grid_search, mean_variance_optim, mean_variance_jump, evaluate_benchmarks

# ---- grid search --------------------------------------------------------

function _grid_rows(R::DataFrame, μ::AbstractVector, Σ::AbstractMatrix;
                    step::Float64, risk_free_rate::Real, risk_aversion::Real,
                    factor::Int)
    n_assets = ncol(R)
    n_assets == 2 ||
        error("grid_search only supports N=2; got N=$n_assets")
    n_steps = round(Int, 1.0 / step) + 1
    rows = NamedTuple[]
    for k in 0:(n_steps - 1)
        w_btc = round(min(1.0, k * step), digits=10)
        w = [w_btc, 1.0 - w_btc]
        m = summarize_weights(w, R, μ, Σ;
                              risk_free_rate=risk_free_rate,
                              risk_aversion=risk_aversion,
                              factor=factor)
        push!(rows, merge(m, (btc_weight=w_btc, bnb_weight=1.0 - w_btc)))
    end
    return DataFrame(rows)
end

function _select_from_grid(grid::DataFrame, objective::AbstractString, max_drawdown::Float64)
    if objective == "maximize_sharpe"
        return grid[argmax(grid.sharpe_ratio), :]
    elseif objective == "minimize_volatility"
        return grid[argmin(grid.annual_volatility), :]
    elseif objective == "maximize_return"
        return grid[argmax(grid.annual_return), :]
    elseif objective == "minimize_cvar"
        return grid[argmin(grid.cvar_95), :]
    elseif objective == "constrained_sharpe"
        # max_drawdown is a non-positive threshold; require row's MaxDD >= threshold
        elig = grid[grid.max_drawdown .>= max_drawdown, :]
        if nrow(elig) == 0
            @warn "no grid candidates satisfy max_drawdown >= $max_drawdown; using full grid"
            elig = grid
        end
        return elig[argmax(elig.sharpe_ratio), :]
    else
        throw(ArgumentError("unknown objective: $objective"))
    end
end

"""
    grid_search(R, μ, Σ; step, objective, risk_free_rate, risk_aversion, max_drawdown, factor)

N=2 brute force scan of `w_BTC ∈ [0, 1]` with the given step. Mirrors
[src/optimizer.py:44](src/optimizer.py:44). Returns (grid_df, best_row).
"""
function grid_search(R::DataFrame, μ::AbstractVector, Σ::AbstractMatrix;
                     step::Float64=0.01, objective::AbstractString="maximize_sharpe",
                     risk_free_rate::Real=0.0, risk_aversion::Real=1.0,
                     max_drawdown::Float64=-1.0, factor::Int=365)
    grid = _grid_rows(R, μ, Σ; step=step, risk_free_rate=risk_free_rate,
                      risk_aversion=risk_aversion, factor=factor)
    best = _select_from_grid(grid, objective, max_drawdown)
    return grid, best
end

# ---- Optim.jl mean-variance --------------------------------------------

# softmax(z) maps R^N → simplex; we optimize unconstrained over z and
# read back w = softmax(z). This avoids needing equality+inequality
# constraints in Optim.jl, which only handles boxes natively.
function _softmax(z::AbstractVector{<:Real})
    m = maximum(z)
    e = exp.(z .- m)
    return e ./ sum(e)
end

"""
    mean_variance_optim(R, μ, Σ; objective, risk_free_rate, risk_aversion, factor)

Optim.jl LBFGS over softmax-parameterized simplex. Closest spiritual
analogue to scipy SLSQP — gradient-based continuous optimizer.
Returns a NamedTuple matching the Python result-dict shape.
"""
function mean_variance_optim(R::DataFrame, μ::AbstractVector, Σ::AbstractMatrix;
                             objective::AbstractString="maximize_sharpe",
                             risk_free_rate::Real=0.0, risk_aversion::Real=1.0,
                             factor::Int=365)
    if objective in ("minimize_cvar", "constrained_sharpe")
        return (skipped=true,
                reason="mean_variance_optim does not implement $objective; see grid result")
    end

    n = length(μ)
    obj_fn(z) = portfolio_objective(_softmax(z), R, μ, Σ;
                                    mode=objective,
                                    risk_aversion=risk_aversion,
                                    risk_free_rate=risk_free_rate,
                                    factor=factor)
    z0 = zeros(n)
    res = optimize(obj_fn, z0, LBFGS(); autodiff=:finite)
    Optim.converged(res) ||
        return (skipped=true, reason="Optim did not converge: $(Optim.iterations(res)) iters")
    w = _softmax(Optim.minimizer(res))
    return (w=w, iterations=Optim.iterations(res))
end

# ---- JuMP + Ipopt mean-variance ----------------------------------------

"""
    mean_variance_jump(R, μ, Σ; objective, risk_free_rate, risk_aversion, factor)

JuMP + Ipopt with explicit `sum(w) == 1` and `w ∈ [0,1]^N` constraints —
the direct analogue of scipy SLSQP's constraint API.
"""
function mean_variance_jump(R::DataFrame, μ::AbstractVector, Σ::AbstractMatrix;
                            objective::AbstractString="maximize_sharpe",
                            risk_free_rate::Real=0.0, risk_aversion::Real=1.0,
                            factor::Int=365)
    if objective in ("minimize_cvar", "constrained_sharpe")
        return (skipped=true,
                reason="mean_variance_jump does not implement $objective; see grid result")
    end

    n = length(μ)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, 0.0 <= w[1:n] <= 1.0, start = 1.0 / n)
    @constraint(model, sum(w) == 1.0)

    if objective == "maximize_sharpe"
        # Maximize Sharpe = minimize -(w'μ − rf) / sqrt(w'Σw + ε).
        # Direct nonlinear formulation — the ε regularizes the gradient
        # near w=0 without distorting the optimum (ε << annualized var).
        @objective(model, Min,
                   -(sum(μ[i] * w[i] for i in 1:n) - risk_free_rate) /
                    sqrt(sum(w[i] * Σ[i, j] * w[j] for i in 1:n, j in 1:n) + 1e-12))
    elseif objective == "minimize_volatility"
        @objective(model, Min, sum(w[i] * Σ[i, j] * w[j] for i in 1:n, j in 1:n))
    elseif objective == "maximize_return"
        @objective(model, Min, -sum(μ[i] * w[i] for i in 1:n))
    elseif objective == "maximize_return_minus_risk"
        @objective(model, Min,
                   -(sum(μ[i] * w[i] for i in 1:n) -
                     risk_aversion * sum(w[i] * Σ[i, j] * w[j] for i in 1:n, j in 1:n)))
    else
        throw(ArgumentError("unknown objective: $objective"))
    end

    optimize!(model)
    status = termination_status(model)
    if status ∉ (MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
        return (skipped=true, reason="JuMP/Ipopt status: $status")
    end
    w_opt = clamp.(value.(w), 0.0, 1.0)
    w_opt = w_opt ./ sum(w_opt)  # renormalize after clamp
    return (w=w_opt, status=string(status))
end

# ---- benchmarks --------------------------------------------------------

"""
    evaluate_benchmarks(R, μ, Σ; risk_free_rate, risk_aversion, factor)

Returns one weight vector per benchmark. For N=2 mirrors
[src/optimizer.py:215](src/optimizer.py:215): 100% asset 1, 100% asset 2,
50/50. For N>2 generalizes to N corner weights + equal-weight.
"""
function evaluate_benchmarks(R::DataFrame, μ::AbstractVector, Σ::AbstractMatrix;
                             tickers::AbstractVector{<:AbstractString},
                             risk_free_rate::Real=0.0, risk_aversion::Real=1.0,
                             factor::Int=365)
    n = length(μ)
    out = Tuple{String,Vector{Float64}}[]
    for i in 1:n
        w = zeros(n)
        w[i] = 1.0
        push!(out, ("Benchmark 100% $(tickers[i])", w))
    end
    if n == 2
        push!(out, ("Benchmark 50/50", [0.5, 0.5]))
    else
        push!(out, ("Benchmark equal-weight", fill(1.0 / n, n)))
    end
    return out
end

end # module
