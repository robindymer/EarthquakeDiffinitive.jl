module EarthquakeDiffinitive

using Diffinitive

include("PorePressure.jl")
using .PorePressure

include("Elasticity.jl")
using .Elasticity

end
