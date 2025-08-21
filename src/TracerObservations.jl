
module TracerObservations
    export TracerObservation, TracerEstimate, update_estimate!
    #############################
    # Lightweight data structs
    #############################

    struct TracerObservation{T,F<:Function}
        t_obs :: Vector{T}
        y_obs :: Vector{T}
        σ_obs :: Union{Nothing,Vector{T}}
        f_src :: Union{Nothing,F}   # must be provided for inversion
        nobs  :: Int
    end
    function TracerObservation(t_obs::Vector{T}, y_obs::Vector{T};
                            σ_obs::Union{Nothing,Vector{T}}=nothing,
                            f_src::Union{Nothing,F}=nothing) where {T<:Real,F<:Function}
        length(t_obs) == length(y_obs) || throw(ArgumentError("y_obs must match t_obs"))
        (σ_obs === nothing || length(σ_obs)==length(t_obs)) ||
            throw(ArgumentError("σ_obs must match t_obs"))
        TracerObservation{T,F}(t_obs, y_obs, σ_obs, f_src, length(t_obs))
    end

    mutable struct TracerEstimate{T}
        t_obs :: Vector{T}
        y_obs :: Vector{T}
        σ_obs :: Union{Nothing,Vector{T}}
        yhat  :: Vector{T}
        σhat  :: Vector{T}
        nobs  :: Int
    end
    function TracerEstimate(obs::TracerObservation{T};
                            init_yhat::Symbol=:zeros,
                            init_σhat::Symbol=:zeros) where {T<:Real}
        n = obs.nobs
        yhat = init_yhat === :zeros ? zeros(T, n) :
            init_yhat === :undef ? Vector{T}(undef, n) :
            throw(ArgumentError("init_yhat must be :zeros or :undef"))
        σhat = init_σhat === :zeros ? zeros(T, n) :
            init_σhat === :undef ? Vector{T}(undef, n) :
            throw(ArgumentError("init_σhat must be :zeros or :undef"))
        TracerEstimate{T}(obs.t_obs, obs.y_obs, obs.σ_obs, yhat, σhat, n)
    end

    update_estimate!(e::TracerEstimate, yhat::AbstractVector) = (length(yhat)==e.nobs || throw(ArgumentError("len mismatch")); e.yhat .= yhat; e)
end