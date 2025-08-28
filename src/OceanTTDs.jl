module OceanTTDs
    using LinearAlgebra
    using Distributions
    using FastGaussQuadrature
    using Reexport
    import Distributions: convolve   # avoid importing `support` to prevent name clashes

    # Core data structures
    include("TracerObservations.jl")
    @reexport using .TracerObservations

    # Transit Time Distribution types
    include("TTDs/TTDs.jl")
    @reexport using .TTDs

    # Integration and convolution (simplified from BoundaryPropagators)
    include("Integration.jl")
    include("Convolution.jl") 

    # Statistical utilities
    include("StatUtils.jl")
    @reexport using .StatUtils

    # Result containers
    include("InversionResults.jl")
    @reexport using .InversionResults

    # Optimization methods
    include("Optimizers/Optimizers.jl")
    @reexport using .Optimizers

    # Convenience functions
    export tracer_observation, uniform_prior

    """
        tracer_observation(times, concentrations, source_function; uncertainties=nothing)

    Create a TracerObservation from measurement data.

    # Arguments
    - `times`: Observation time points
    - `concentrations`: Measured tracer concentrations  
    - `source_function`: Boundary condition function f(t) for this tracer
    - `uncertainties`: Measurement uncertainties (optional, defaults to 10% of concentrations)

    # Returns
    TracerObservation object ready for inversion

    # Example
    ```julia
    # CFC-12 with exponential atmospheric growth
    cfc12_source = t -> (t >= 1930) ? exp(0.08 * (t - 1930)) : 0.0
    obs = tracer_observation(times, cfc12_data, cfc12_source)
    
    # Multiple tracers for joint inversion
    obs1 = tracer_observation(times1, cfc11_data, cfc11_source)  
    obs2 = tracer_observation(times2, sf6_data, sf6_source)
    result = max_ent_inversion([obs1, obs2]; support=τ_support, prior_distribution=prior)
    ```
    """
    function tracer_observation(times::Vector, concentrations::Vector, source_function::Function;
                               uncertainties=nothing)
        if uncertainties === nothing
            uncertainties = 0.1 * abs.(concentrations)
        end
        return TracerObservation(times, concentrations; σ_obs=uncertainties, f_src=source_function)
    end

    """
        uniform_prior(support)

    Create a uniform prior distribution over the given support points.

    # Arguments  
    - `support`: Transit time support points

    # Returns
    Uniform probability vector (all values equal, sum to 1)

    # Example
    ```julia
    τ_support = collect(0:50:2000)
    prior = uniform_prior(τ_support)
    result = fit_ttd(obs, method=:max_entropy, support=τ_support, prior_distribution=prior)
    ```
    """
    uniform_prior(support) = ones(length(support)) ./ length(support)


end
