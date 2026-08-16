module EarthquakeDiffinitive

using Diffinitive

include("PorePressure.jl")
using .PorePressure

include("Elasticity.jl")
using .Elasticity

include("ElasticitySplitNode.jl")
using .ElasticitySplitNode

include("RateStateFriction.jl")
using .RateStateFriction

end
