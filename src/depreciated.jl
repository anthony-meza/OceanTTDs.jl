
function WelfordCovariance(data::Matrix)
    estimator = WelfordEstimate()
    update_batch!(estimator, data); 
    return get_covariance(estimator)
end

function is_natural_number(n)
    return isa(n, Integer) && n >= 0
end

function WelfordCovariance(sampling_function::Function, 
                            sampling_distribution;  
                          batch_size = nothing, 
                          total_samples = nothing)
    if !is_natural_number(batch_size)
        throw(ArgumentError("batch_size must be a positive integer"))
    end
    if !is_natural_number(total_samples)
        throw(ArgumentError("total_samples must be a positive integer"))
    end
    if total_samples % batch_size != 0
        throw(ArgumentError("batch_size ($batch_size) must divide total_samples ($total_samples) evenly"))
    end

    if !isa(sampling_distribution, Distribution)
        throw(ArgumentError("sampling_distribution must be a Distribution"))
    end

    feature_length = length(sampling_function(rand(sampling_distribution)))
    ensemble = Matrix{Float64}(undef, batch_size, feature_length)
    nstrides = div(total_samples, batch_size)
    random_samples = Vector{Float64}(undef, batch_size)
    estimator = WelfordEstimate()

    for k in 1:nstrides
        rand!(sampling_distribution, random_samples)
        for i in eachindex(random_samples)
            @views ensemble[i, :] = sampling_function(random_samples[i])
        end
        update_batch!(estimator, ensemble)
    end
    Σ = get_covariance(estimator)
    Σ = Symmetric((Σ + Σ') ./ 2, :U)  # Enforce symmetry and use upper triangle
    return Σ
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

