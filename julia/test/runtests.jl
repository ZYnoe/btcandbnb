using Test

# Activate parent project so deps resolve correctly
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

include(joinpath(@__DIR__, "..", "src", "PortfolioOptimizer.jl"))

@testset "PortfolioOptimizer.jl" begin
    include("test_metrics.jl")
    include("test_optimizer.jl")
    include("test_schema.jl")
end
