module TCMGreensFunctions

using Distributions
using Distributions: @check_args
using Distributions: @distr_support
# using DimensionalData
# using DimensionalData: @dim 
# using Unitful
# using AlgebraicArrays
using LinearAlgebra
using Interpolations
using QuadGK
using Integrals
import Distributions: mean, median, quantile, std, var, cov, cor, shape, params, pdf, InverseGaussian

export TracerInverseGaussian
export width
export # re-export from Distributions
    mean, median, quantile, std, var, cov, cor, shape, params
export # re-export from Distributions
    InverseGaussian, pdf

export BoundaryPropagator, boundary_propagator_timeseries

export integrate

#order of includes matters here
include("Integrations.jl") 
include("BoundaryPropagator.jl")
include("Convolutions.jl")
include("tracer_inverse_gaussian.jl")

end
