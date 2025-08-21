module stat_utils
    using Statistics, LinearAlgebra, Distributions
    export cov_from_cor, invcov_from_cor, pmf_from

    function pmf_from(dist::UnivariateDistribution, support)
        ℓ = logpdf.(dist, support)        # log-pdf values
        m = maximum(ℓ)                    # shift for stability
        w = exp.(ℓ .- m)                  # exponentiate
        return w ./ sum(w)                # normalize
    end

    function cov_from_cor(X::AbstractMatrix{T}; dims::Int=2, tol::Real=0.0) where {T<:Real}
        # 1) Std per variable and raw correlation
        σ = vec(std(X; dims=dims))
        R = cor(X; dims=dims)

        # 2) Handle zero/near-zero variance variables
        #    Set correlation to zero and variance to one for constant variables
        constvar = σ .<= tol
        if any(constvar)
            R[constvar, :] .= 0
            R[:, constvar] .= 0
            @inbounds for i in findall(constvar)
                R[i,i] = 1
            end
        end

        # 3) Rescale to covariance: Σ = Dσ * R * Dσ
        Dσ = Diagonal(σ)
        Σ  = Dσ * R * Dσ
        p = size(Σ,1)

        if isposdef(Σ)
            
            return Symmetric(Σ, :U)
        else
            warning_string1 = "Covariance matrix not positive-definite, using ridge regularization to ensure positive-definiteness."
            warning_string2 = "\nThis may affect the covariance matrix's properties, especially if κ_target is too high."
            @warn warning_string1
            @warn warning_string2
            max_Σ = maximum(Σ)
            #calculates a mean variance-based ridge parameter
            λridge = ridge_for_kappa(Σ; κ_target = 1e8, eps = 1e-12)
            @warn ("Maximum value in covariance matrix is = $max_Σ")
            @warn ("Using ridge regularization with λ = $λridge")

            return Symmetric(Σ + λridge*I, :U)
        end
        return 
    end

    function invcov_from_cor(X::AbstractMatrix{T}; dims::Int=2) where {T<:Real}
        # std per variable (vector) and correlation matrix

        Σ = cov_from_cor(X; dims=dims)

        # isposdef(Σ) || throw(ArgumentError("Covariance matrix must be positive-definite."))
        
        F = cholesky(Σ)
        invF = inv(F)
        return Symmetric(invF)
    end


end # module stat_utils