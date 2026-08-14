using EarthquakeDiffinitive
using Test

@testset "EarthquakeDiffinitive.jl" begin
    include("pore_pressure_test.jl")
    include("elasticity_test.jl")
end
