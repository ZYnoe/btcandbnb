using Test
using Statistics
using DataFrames
using .Main.PortfolioOptimizer.Metrics
using .Main.PortfolioOptimizer.Optimizer

@testset "Optimizer" begin
    # Synthetic 2-asset returns where BTC has high positive mean + low vol,
    # BNB has negative mean + high vol. Sharpe(BTC) ≫ Sharpe(BNB).
    # Optimum will be near (but not necessarily exactly at) the BTC corner —
    # tiny mixing benefit from imperfect correlation is allowed.
    rng_btc = [0.010, 0.012, 0.011, 0.013, 0.010, 0.012, 0.011, 0.013, 0.010, 0.012]
    rng_bnb = [-0.005, 0.020, -0.030, 0.015, -0.010, 0.008, -0.025, 0.030, -0.015, 0.005]
    R = DataFrame(BTC=rng_btc, BNB=rng_bnb)
    μ = [mean(rng_btc) * 365, mean(rng_bnb) * 365]
    Σ = cov(Matrix(R)) * 365

    @testset "grid_search step granularity" begin
        grid, _ = grid_search(R, μ, Σ; step=0.05)
        @test nrow(grid) == 21  # 0, 0.05, ..., 1.0
    end

    @testset "grid_search picks near-BTC corner when BTC dominates" begin
        _, best = grid_search(R, μ, Σ; step=0.05, objective="maximize_sharpe")
        @test best.btc_weight > 0.9
    end

    @testset "mean_variance_optim converges to near-BTC corner" begin
        res = mean_variance_optim(R, μ, Σ; objective="maximize_sharpe")
        @test !get(res, :skipped, false)
        @test res.w[1] > 0.9
        @test res.w[2] < 0.1
        @test sum(res.w) ≈ 1.0 atol=1e-6
    end

    @testset "mean_variance_jump converges to near-BTC corner" begin
        res = mean_variance_jump(R, μ, Σ; objective="maximize_sharpe")
        @test !get(res, :skipped, false)
        @test res.w[1] > 0.9
        @test res.w[2] < 0.1
        @test sum(res.w) ≈ 1.0 atol=1e-6
    end

    @testset "evaluate_benchmarks produces N + 1 rows for N=2" begin
        bench = evaluate_benchmarks(R, μ, Σ; tickers=["BTC", "BNB"])
        @test length(bench) == 3
        @test bench[1][2] == [1.0, 0.0]
        @test bench[2][2] == [0.0, 1.0]
        @test bench[3][2] == [0.5, 0.5]
    end

    @testset "MV solvers skip on minimize_cvar" begin
        @test get(mean_variance_optim(R, μ, Σ; objective="minimize_cvar"), :skipped, false)
        @test get(mean_variance_jump(R, μ, Σ; objective="minimize_cvar"), :skipped, false)
    end

    @testset "grid_search errors on N>2" begin
        R3 = DataFrame(A=rng_btc, B=rng_bnb, C=rng_btc)
        μ3 = vcat(μ, μ[1])
        Σ3 = cov(Matrix(R3)) * 365
        @test_throws ErrorException grid_search(R3, μ3, Σ3; step=0.05)
    end
end
