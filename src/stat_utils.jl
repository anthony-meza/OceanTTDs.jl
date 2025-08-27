module stat_utils
    using Statistics, LinearAlgebra, Distributions
    using CovarianceEstimation, Random
    
    export BootstrapShrinkageCovariance, pmf_from

    function is_natural_number(n)
        return isa(n, Integer) && n >= 0
    end

    function pmf_from(dist::UnivariateDistribution, support::AbstractVector, weights = nothing)
        ℓ = logpdf.(dist, support)        # log-pdf values

        # Apply weights if provided
        if weights !== nothing
            ℓ .+= log.(weights)           # add log-weights for numerical stability
        end

        m = maximum(ℓ)                    # shift for stability
        ℓ .= exp.(ℓ .- m)                 # reuse ℓ vector, exponentiate in-place
        s = sum(ℓ)                        # compute sum
        ℓ ./= s                           # normalize in-place

        ℓ[ℓ .< eps(eltype(ℓ))*maximum(ℓ)] .= 0
        ℓ ./= sum(ℓ)

        return ℓ                          # return reused vector
    end

    function BootstrapShrinkageCovariance(sampling_function::Function, 
                                sampling_distribution;  
                            total_samples = nothing)

        if !is_natural_number(total_samples)
            throw(ArgumentError("total_samples must be a positive integer"))
        end

        if !isa(sampling_distribution, Distribution)
            throw(ArgumentError("sampling_distribution must be a Distribution"))
        end

        feature_length = length(sampling_function(rand(sampling_distribution)))
        ensemble = Matrix{Float64}(undef, total_samples, feature_length)

        random_samples = rand(sampling_distribution, total_samples)

        for i in eachindex(random_samples)
            @views ensemble[i, :] = sampling_function(random_samples[i])
        end

        LSE = LinearShrinkage
        method = LSE(target=DiagonalCommonVariance(), shrinkage=:rblw)
        Σ = cov(method, ensemble)
        return Σ, pinv(Σ)
    end

end # module stat_utils