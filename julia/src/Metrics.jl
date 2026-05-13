module Metrics

using Statistics
using DataFrames

export cumulative_returns, portfolio_returns, annualized_return,
       annualized_volatility, sharpe_ratio, sortino_ratio,
       max_drawdown, value_at_risk, conditional_value_at_risk,
       portfolio_objective, summarize_weights

function cumulative_returns(r::AbstractVector{<:Real})
    isempty(r) && return Float64[]
    return cumprod(1.0 .+ r) .- 1.0
end

function portfolio_returns(w::AbstractVector{<:Real}, R::DataFrame)
    n_assets = ncol(R)
    length(w) == n_assets ||
        throw(ArgumentError("weights length $(length(w)) != number of assets $n_assets"))
    return Matrix(R) * collect(Float64.(w))
end

function annualized_return(r::AbstractVector{<:Real}; factor::Int=365)
    isempty(r) && return NaN
    return mean(r) * factor
end

function annualized_volatility(r::AbstractVector{<:Real}; factor::Int=365)
    length(r) < 2 && return NaN
    return std(r; corrected=true) * sqrt(factor)
end

# Julia's std on a constant series returns ~1e-18 (rounding), not exact 0
# like NumPy. Treat anything below 1e-12 as "effectively zero" so the
# zero-vol branch fires consistently.
const VOL_EPSILON = 1e-12

function sharpe_ratio(r::AbstractVector{<:Real}; risk_free_rate::Real=0.0, factor::Int=365)
    vol = annualized_volatility(r; factor=factor)
    (isnan(vol) || abs(vol) < VOL_EPSILON) && return NaN
    return (annualized_return(r; factor=factor) - risk_free_rate) / vol
end

function sortino_ratio(r::AbstractVector{<:Real}; risk_free_rate::Real=0.0, factor::Int=365)
    length(r) < 2 && return NaN
    downside = filter(<(0.0), r)
    isempty(downside) && return NaN
    dvol = std(downside; corrected=true) * sqrt(factor)
    (isnan(dvol) || abs(dvol) < VOL_EPSILON) && return NaN
    return (annualized_return(r; factor=factor) - risk_free_rate) / dvol
end

function max_drawdown(r::AbstractVector{<:Real})
    isempty(r) && return NaN
    wealth = cumprod(1.0 .+ r)
    peak = accumulate(max, wealth)
    return minimum(wealth ./ peak .- 1.0)
end

# VaR / CVaR are reported as non-negative loss magnitudes — see metrics.py:100-122
function value_at_risk(r::AbstractVector{<:Real}; level::Real=0.95)
    isempty(r) && return NaN
    q = quantile(collect(Float64, r), 1.0 - level)
    return q < 0 ? -q : 0.0
end

function conditional_value_at_risk(r::AbstractVector{<:Real}; level::Real=0.95)
    isempty(r) && return NaN
    cutoff = quantile(collect(Float64, r), 1.0 - level)
    tail = filter(<=(cutoff), r)
    isempty(tail) && return NaN
    m = mean(tail)
    return m < 0 ? -m : 0.0
end

function portfolio_objective(w::AbstractVector{<:Real}, R::DataFrame,
                             μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real};
                             mode::AbstractString, risk_aversion::Real=1.0,
                             risk_free_rate::Real=0.0, factor::Int=365)
    daily = portfolio_returns(w, R)
    ann_ret = dot(w, μ)
    ann_var = dot(w, Σ * w)
    ann_vol = sqrt(max(ann_var, 0.0))
    if mode == "maximize_sharpe" || mode == "constrained_sharpe"
        ann_vol == 0.0 && return Inf
        return -(ann_ret - risk_free_rate) / ann_vol
    elseif mode == "minimize_volatility"
        return ann_vol
    elseif mode == "maximize_return"
        return -ann_ret
    elseif mode == "maximize_return_minus_risk"
        return -(ann_ret - risk_aversion * ann_var)
    elseif mode == "minimize_cvar"
        return conditional_value_at_risk(daily)
    else
        throw(ArgumentError("unknown objective mode: $mode"))
    end
end

# helper since Julia stdlib's `dot` is in LinearAlgebra (imported here for self-containment)
using LinearAlgebra: dot

"""
    summarize_weights(w, R, μ, Σ; risk_free_rate, risk_aversion, factor)

All metrics for one weight vector. Field names match
[src/metrics.py:185-195](src/metrics.py:185) so downstream CSV columns line up.
"""
function summarize_weights(w::AbstractVector{<:Real}, R::DataFrame,
                           μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real};
                           risk_free_rate::Real=0.0, risk_aversion::Real=1.0,
                           factor::Int=365)
    daily = portfolio_returns(w, R)
    cum = cumulative_returns(daily)
    return (
        annual_return = annualized_return(daily; factor=factor),
        annual_volatility = annualized_volatility(daily; factor=factor),
        sharpe_ratio = sharpe_ratio(daily; risk_free_rate=risk_free_rate, factor=factor),
        sortino_ratio = sortino_ratio(daily; risk_free_rate=risk_free_rate, factor=factor),
        max_drawdown = max_drawdown(daily),
        var_95 = value_at_risk(daily),
        cvar_95 = conditional_value_at_risk(daily),
        final_cumulative_return = isempty(cum) ? NaN : last(cum),
        objective_mv = risk_aversion * dot(w, Σ * w) - dot(w, μ),
    )
end

end # module
