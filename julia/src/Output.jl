module Output

using CSV
using DataFrames

export COLUMNS, build_row, build_empty_row, flush_summary

# Mirror of src/comparison.py::COLUMNS — order matters for CSV round-trip.
const COLUMNS = (
    :method, :solver, :btc_weight, :bnb_weight,
    :annual_return, :annual_volatility, :sharpe_ratio, :sortino_ratio,
    :max_drawdown, :var_95, :cvar_95, :objective, :final_cumulative_return,
    :runtime_seconds, :success, :error_message, :note,
)

"""
    build_row(; method, solver, w, metrics, runtime_seconds, note="")

Pack a successful result into a row dict matching the Python schema.
`metrics` is a NamedTuple from Metrics.summarize_weights.
For N=2 we project w[1]→btc_weight, w[2]→bnb_weight (Invariant: ticker
order is `BTC-USD, BNB-USD` per src/config.py::TICKERS_DEFAULT).
"""
function build_row(; method::AbstractString, solver::AbstractString,
                   w::AbstractVector, metrics::NamedTuple,
                   runtime_seconds::Real, note::AbstractString="")
    length(w) == 2 ||
        error("Output.build_row currently expects N=2 (schema has btc_weight/bnb_weight)")
    return Dict{Symbol,Any}(
        :method => method,
        :solver => solver,
        :btc_weight => Float64(w[1]),
        :bnb_weight => Float64(w[2]),
        :annual_return => metrics.annual_return,
        :annual_volatility => metrics.annual_volatility,
        :sharpe_ratio => metrics.sharpe_ratio,
        :sortino_ratio => metrics.sortino_ratio,
        :max_drawdown => metrics.max_drawdown,
        :var_95 => metrics.var_95,
        :cvar_95 => metrics.cvar_95,
        :objective => metrics.objective_mv,
        :final_cumulative_return => metrics.final_cumulative_return,
        :runtime_seconds => Float64(runtime_seconds),
        :success => true,
        :error_message => "",
        :note => note,
    )
end

"""
    build_empty_row(; method, solver, error, runtime_seconds=0.0)

Failure row — same shape, NaN numerics, `success=false`. Invariant 3:
every requested solver gets a row.
"""
function build_empty_row(; method::AbstractString, solver::AbstractString,
                         error::AbstractString, runtime_seconds::Real=0.0)
    return Dict{Symbol,Any}(
        :method => method,
        :solver => solver,
        :btc_weight => NaN,
        :bnb_weight => NaN,
        :annual_return => NaN,
        :annual_volatility => NaN,
        :sharpe_ratio => NaN,
        :sortino_ratio => NaN,
        :max_drawdown => NaN,
        :var_95 => NaN,
        :cvar_95 => NaN,
        :objective => NaN,
        :final_cumulative_return => NaN,
        :runtime_seconds => Float64(runtime_seconds),
        :success => false,
        :error_message => error,
        :note => "",
    )
end

"""
    flush_summary(rows, outputs_dir)

Re-write comparison_summary_julia.csv from current rows. Called after each
new row — same incremental-flush invariant as src/runner.py::_flush_summary
so a Ctrl-C / SLURM kill never leaves an inconsistent file.
"""
function flush_summary(rows::AbstractVector{<:AbstractDict}, outputs_dir::AbstractString)
    isempty(rows) && return nothing
    df = DataFrame([Symbol(c) => [get(r, c, missing) for r in rows] for c in COLUMNS])
    path = joinpath(outputs_dir, "comparison_summary_julia.csv")
    CSV.write(path, df)
    return path
end

end # module
