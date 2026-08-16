using EarthquakeDiffinitive
using Test

@testset "EarthquakeDiffinitive.jl" begin
    include("pore_pressure_test.jl")
    include("elasticity_test.jl")
    include("elasticity_split_node_test.jl")
    include("rate_state_friction_test.jl")
end
