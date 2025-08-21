module TTDs

    using Distributions
    using Distributions: @check_args
    using Distributions: @distr_support

    import Distributions: mean, median, quantile, std, var, cov, cor, shape, params, pdf
    import Distributions: insupport, support

    using ..TracerObservations

    export _is_finite_real, logsumexp_with_logw

    include("utils.jl")

    #export statements are included in the file
    include("tracer_inverse_gaussian.jl")

    include("tracer_max_ent.jl")

    include("tracer_max_ent_err.jl")
end