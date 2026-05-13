using Test
using .Main.PortfolioOptimizer.Output

# Hardcoded reference from src/comparison.py::COLUMNS — if this drifts,
# downstream analyze.py merging will silently produce columns of NaN.
const PYTHON_COLUMNS = (
    :method, :solver, :btc_weight, :bnb_weight,
    :annual_return, :annual_volatility, :sharpe_ratio, :sortino_ratio,
    :max_drawdown, :var_95, :cvar_95, :objective, :final_cumulative_return,
    :runtime_seconds, :success, :error_message, :note,
)

@testset "Schema parity with comparison.py::COLUMNS" begin
    @test Output.COLUMNS == PYTHON_COLUMNS

    # build_row produces every column expected by Python
    fake_metrics = (
        annual_return = 0.1, annual_volatility = 0.2, sharpe_ratio = 0.5,
        sortino_ratio = 0.7, max_drawdown = -0.3, var_95 = 0.02, cvar_95 = 0.025,
        final_cumulative_return = 0.15, objective_mv = -0.05,
    )
    row = build_row(method="X", solver="Y", w=[0.6, 0.4],
                    metrics=fake_metrics, runtime_seconds=1.5)
    for c in PYTHON_COLUMNS
        @test haskey(row, c)
    end
    @test row[:success] === true
    @test row[:btc_weight] ≈ 0.6
    @test row[:bnb_weight] ≈ 0.4

    empty = build_empty_row(method="X", solver="Y", error="boom")
    @test empty[:success] === false
    @test empty[:error_message] == "boom"
    @test isnan(empty[:btc_weight])
end
