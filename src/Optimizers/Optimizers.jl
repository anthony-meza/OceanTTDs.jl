module Optimizers

    using Optim
    using ForwardDiff
    using NonlinearSolve
    using Distributions
    import Distributions: convolve   # avoid importing `support` to prevent name clashes
    # include("../TracerObservations.jl")
    using JuMP, Ipopt
    using ..stat_utils
    using LinearAlgebra
    using ..TracerObservations
    using ..TTDs
    using ..InversionResults

    include("InverseGaussian.jl")
    include("MaximumEntropy.jl")
    include("TCM.jl")

    # Re-export functions from included files
    export invert_tcm

end