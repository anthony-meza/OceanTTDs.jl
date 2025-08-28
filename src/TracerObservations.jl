
module TracerObservations
    export TracerObservation, TracerEstimate, update_estimate!
    #############################
    # Lightweight data structs
    #############################

    """
        TracerObservation{T,F<:Function}

    Container for tracer observation data including measurement values, uncertainties, and source functions.

    # Fields
    - `t_obs::Vector{T}`: Time points of observations
    - `y_obs::Vector{T}`: Observed tracer concentrations
    - `σ_obs::Union{Nothing,Vector{T}}`: Measurement uncertainties (optional)
    - `f_src::Union{Nothing,F}`: Source function for boundary conditions (required for inversion)
    - `nobs::Int`: Number of observations

    # Constructor
    ```julia
    TracerObservation(t_obs, y_obs; σ_obs=nothing, f_src=nothing)
    ```
    """
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

    """
        TracerEstimate{T}

    Mutable container for tracer model estimates and predictions.

    # Fields
    - `t_obs::Vector{T}`: Time points (copied from observations)
    - `y_obs::Vector{T}`: Original observed values (copied from observations)
    - `σ_obs::Union{Nothing,Vector{T}}`: Observation uncertainties (copied from observations)
    - `yhat::Vector{T}`: Model predictions
    - `σhat::Vector{T}`: Model prediction uncertainties
    - `nobs::Int`: Number of observations

    # Constructor
    ```julia
    TracerEstimate(obs::TracerObservation; init_yhat=:zeros, init_σhat=:zeros)
    ```
    """
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

    """
        update_estimate!(estimate::TracerEstimate, yhat::AbstractVector)

    Update the model predictions in a TracerEstimate object.

    # Arguments
    - `estimate::TracerEstimate`: The estimate object to update
    - `yhat::AbstractVector`: New predicted values (must match observation length)

    # Returns
    - Modified `TracerEstimate` object
    """
    update_estimate!(e::TracerEstimate, yhat::AbstractVector) = (length(yhat)==e.nobs || throw(ArgumentError("len mismatch")); e.yhat .= yhat; e)
end