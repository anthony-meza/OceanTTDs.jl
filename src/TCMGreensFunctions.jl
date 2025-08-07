module TCMGreensFunctions

using Distributions
using Distributions: @check_args
using Distributions: @distr_support
# using DimensionalData
# using DimensionalData: @dim 
# using Unitful
# using AlgebraicArrays
using LinearAlgebra
# using Downloads
# using MAT
using Interpolations
using QuadGK
# using CSV

import Distributions: mean, median, quantile, std, var, cov, cor, shape, params, pdf, InverseGaussian
# import Base: +, alignment, zeros
# import DimensionalData: dims
# import LinearAlgebra: eigen

export TracerInverseGaussian
export width
export # re-export from Distributions
    mean, median, quantile, std, var, cov, cor, shape, params
export # re-export from Distributions
    InverseGaussian, pdf

export BoundaryPropagator, boundary_propagator_timeseries

export simpsons_integral

include("tracer_inverse_gaussian.jl")
include("BoundaryPropagator.jl")

end
