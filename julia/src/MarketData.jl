module MarketData

using JSON3
using CSV
using DataFrames
using Dates

export Market, load_market

struct Market
    tickers::Vector{String}
    returns::DataFrame
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    annualization_factor::Int
    window_start::String
    window_end::String
end

"""
    load_market(outputs_dir)

Read μ/Σ from `basic_stats.json` and daily returns from `returns.csv`
written by the Python pipeline. The Python side is the source of truth
for both — Invariant 1 says classical/quantum/Julia share the same μ/Σ.
"""
function load_market(outputs_dir::AbstractString)
    stats_path = joinpath(outputs_dir, "basic_stats.json")
    returns_path = joinpath(outputs_dir, "returns.csv")

    isfile(stats_path) ||
        error("missing $stats_path; run Python pipeline first")
    isfile(returns_path) ||
        error("missing $returns_path; run Python with --save-intermediate or default-dump enabled")

    stats = JSON3.read(read(stats_path, String))
    tickers = String.(stats["tickers"])
    n = length(tickers)

    μ = Float64[Float64(stats["annualized_return"][Symbol(t)]) for t in tickers]
    Σ_rows = stats["covariance"]
    Σ = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        Σ[i, j] = Float64(Σ_rows[i][j])
    end

    raw = CSV.read(returns_path, DataFrame)
    # Python writes Date as the index — first column after CSV round-trip.
    # We only need the ticker columns for portfolio_returns.
    returns = select(raw, tickers)

    return Market(
        tickers,
        returns,
        μ,
        Σ,
        Int(stats["annualization_factor"]),
        String(stats["start"]),
        String(stats["end"]),
    )
end

end # module
