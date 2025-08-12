module TCMGreensFunctions

# using DimensionalData
# using DimensionalData: @dim 
# using Unitful
# using AlgebraicArrays
using LinearAlgebra
# using Downloads
# using MAT
using Interpolations
using QuadGK
using AccurateArithmetic
# using CSV
using ForwardDiff
using FastGaussQuadrature
using LinearAlgebra
using Optim
import Optim: minimizer
using Distributions  # InverseGaussian
using BlackBoxOptim
using LeastSquaresOptim
export BoundaryPropagator, boundary_propagator_timeseries

export simpsons_integral
export IG_BP_Inversion_NLLS
export IG_BP_Inversion
export make_integration_panels, make_integrator
export convolve_at
export MaxEntDist
include("BoundaryPropagator.jl")
include("tracer_inverse_gaussian.jl")
include("InvertBP.jl")
include("tracer_maximum_entropy_distribution.jl")
end
