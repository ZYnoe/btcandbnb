using Test
using Statistics
using DataFrames
using .Main.PortfolioOptimizer.Metrics

@testset "Metrics" begin
    # Fixed synthetic series — expected values pre-computed via Python
    # (src/metrics.py with the same inputs) to make this a true parity test
    r1 = [0.01, -0.02, 0.03, 0.005, -0.01]
    r2 = [0.02, -0.01, 0.01, 0.015, 0.00]
    R = DataFrame(BTC=r1, BNB=r2)
    μ = [mean(r1) * 365, mean(r2) * 365]
    Σ = cov(Matrix(R)) * 365
    w = [0.5, 0.5]

    m = summarize_weights(w, R, μ, Σ)

    @test m.annual_return ≈ 1.8249999999999997 atol=1e-12
    @test m.annual_volatility ≈ 0.2785004488326724 atol=1e-12
    @test m.sharpe_ratio ≈ 6.5529517372393515 atol=1e-12
    @test m.sortino_ratio ≈ 13.509256086106296 atol=1e-12
    @test m.max_drawdown ≈ -0.015000000000000013 atol=1e-12
    @test m.var_95 ≈ 0.012999999999999998 atol=1e-12
    @test m.cvar_95 ≈ 0.015 atol=1e-12
    @test m.final_cumulative_return ≈ 0.024818363974999924 atol=1e-12
    @test m.objective_mv ≈ -1.7474375000000002 atol=1e-12

    # Edge cases
    @test isnan(annualized_return(Float64[]))
    @test isnan(annualized_volatility([1.0]))
    @test isnan(sharpe_ratio(fill(0.01, 10)))  # vol == 0
    @test isnan(sortino_ratio([0.01, 0.02, 0.03]))  # no downside

    # VaR / CVaR sign convention: returned as non-negative magnitudes
    losses = [-0.10, -0.05, -0.02, 0.0, 0.01, 0.02, 0.05]
    @test value_at_risk(losses; level=0.95) >= 0.0
    @test conditional_value_at_risk(losses; level=0.95) >= 0.0
end
