module Optimizers

    using Optim
    using ForwardDiff
    using NonlinearSolve
    using Distributions
    import Distributions: convolve   # avoid importing `support` to prevent name clashes
    # include("../TracerObservations.jl")
    using ..stat_utils

    using ..TracerObservations
    using ..TTDs

    include("InverseGaussian.jl")
    
end