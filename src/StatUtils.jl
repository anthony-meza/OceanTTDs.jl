module StatUtils
    using Statistics, LinearAlgebra, Distributions
    using CovarianceEstimation, Random
    
    export BootstrapShrinkageCovariance, discretize_pdf

    function is_natural_number(n)
        return isa(n, Integer) && n >= 0
    end

    """
        discretize_pdf(dist::UnivariateDistribution, support::AbstractVector, weights=nothing)

    Convert a continuous probability distribution to a discrete PMF evaluated at support points.

    This function evaluates the PDF of a continuous distribution at discrete support points,
    applies optional quadrature weights for numerical integration, and normalizes to create
    a proper probability mass function. Uses log-space operations for numerical stability.

    # Arguments
    - `dist::UnivariateDistribution`: Continuous distribution to discretize
    - `support::AbstractVector`: Points where to evaluate the discrete PMF
    - `weights=nothing`: Optional quadrature weights for numerical integration

    # Returns
    - Vector of probabilities summing to 1, representing the discrete PMF

    # Example
    ```julia
    using Distributions
    dist = Normal(0, 1)
    support = -3:0.1:3
    pmf = discretize_pdf(dist, support)
    ```
    """
    function discretize_pdf(dist::UnivariateDistribution, support::AbstractVector, weights = nothing)
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

end # module StatUtils