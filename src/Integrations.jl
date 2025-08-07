using QuadGK
# using BenchmarkTools

include("ExponentialIntegrations.jl")
include("SimpsonsIntegrations.jl")

"""
    indefinite_integral(f, lb; method = :quadgk, x_nodes=nothing)
Calculate the indefinite integral of a univariate function `f` starting from `lb`.
Returns a function that computes the integral from `lb` to any upper limit `x`.
"""
function indefinite_integral(f::Function, lb; method = :quadgk, x_nodes::Union{AbstractVector{<:Real}, Nothing}=nothing)
    return x -> integrate(f, lb, x; method=method, x_nodes=x_nodes)
end

"""
    integrate(f, lb, ub; method = :quadgk, x_nodes=nothing) 
Calculate the definite integral of a univariate function `f` from `lb` to `ub`.
Supports various methods and allows for infinite limits.
"""

function integrate(f::Function, lb, ub; 
                    method = :quadgk, 
                    x_nodes::Union{AbstractVector{<:Real}, Nothing}=nothing)
                    
    # Check if f is a univariate function
    try
        f(lb + 1e-12)
    catch
        error("f must be a univariate function (accepts one argument)")
    end

    if method == :quadgk
        result, err = quadgk(f, lb, ub, rtol=1e-8, atol=1e-8)
    elseif method == :simpsons
        result = simpsons_integration(f, lb, ub; N=1_001)
    elseif method == :exponential
        result = exponential_integration(f, lb, ub)
    else
        error("Method $method is not implemented")
    end
    return result
end


# using Distributions 
# f(x) = pdf(Normal(0, 1e6), x)
# lb = -Inf; ub = Inf
# @btime integrate(f, lb, ub; method = :quadgk)
# @btime integrate(f, lb, ub; method = :simpsons)
# @btime integrate(f, lb, ub; method = :exponential)  

#
# using Distributions
# using Distributions: @check_args
# using Distributions: @distr_support
# import Distributions: mean, median, quantile, std, var, cov, cor, shape, params, pdf, InverseGaussian
# include("tracer_inverse_gaussian.jl")
# export TracerInverseGaussian
# export width
# export # re-export from Distributions
#     mean, median, quantile, std, var, cov, cor, shape, params
# export # re-export from Distributions
#     InverseGaussian, pdf

# Gp(x, Γ, Δ) = pdf(InverseGaussian(TracerInverseGaussian(Γ, Δ)), x)  # Inv. Gauss. Boundary propagator function
# g(x) = Gp(x, 200.0, 50.0)
# lb = 0; ub = Inf

# @btime integrate(g, lb, ub; method = :quadgk)
# @btime integrate(g, lb, ub; method = :simpsons)
# @btime integrate(g, lb, ub; method = :exponential)  

# @btime integrate(g, lb, ub; method = :quadgk)
# @btime integrate(g, lb, ub; method = :simpsons)
# @btime integrate(g, lb, ub; method = :exponential)  