using DoubleExponentialFormulas
using Distributions


"""
    exponential_integration(f, lb, ub; x_nodes=nothing)
Calculate the definite integral of a univariate function `f` from `lb` to `ub`.
Supports infinite limits and uses exponential transformations for numerical stability.
"""
exponential_integration(f::Function, lb, ub) = quadde(f, lb, ub)[1]