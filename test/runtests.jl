using TCMGreensFunctions
using Test

# Common test parameters
const Γ_true = 25.0   # Mean transit time
const Δ_true = 25.0   # Width parameter
const times = collect(1.0:5.0:250.0)
const τm = 250_000.0  # Maximum age to consider

@testset "TracerObservations" begin
    # ---------------------------------------------------------------
    # These tests validate the core data structures used for storing
    # tracer measurements and model estimates.
    # ---------------------------------------------------------------
    
    # Create a simple source function (step function at t=0)
    # This represents a boundary condition at the ocean surface
    unit_source = x -> (x >= 0) ? 1.0 : 0.0
    
    # Test TracerObservation constructor with standard parameters
    # Creates an observation object from synthetic time series data
    obs_values = rand(length(times))
    obs = TracerObservation(times, obs_values; 
                           σ_obs = 0.1 .+ zeros(length(times)),  # Constant uncertainty
                           f_src = unit_source)                  # Source function
    
    # Verify the observation object stores data correctly
    @test obs.t_obs == times                 # Time points should match
    @test obs.y_obs == obs_values            # Values should match
    @test obs.nobs == length(times)          # Length should be correct
    @test obs.f_src === unit_source          # Source function should be stored
    
    # Test proper error handling for mismatched arrays
    # Should reject time/value arrays of different lengths
    @test_throws ArgumentError TracerObservation(times, obs_values[1:end-1])
    
    # Should reject uncertainty arrays of wrong length
    @test_throws ArgumentError TracerObservation(times, obs_values, σ_obs=zeros(length(times)-1))
    
    # Test TracerEstimate construction from observation
    # This creates an object for storing model results
    estimate = TracerEstimate(obs)
    @test estimate.t_obs == obs.t_obs        # Should copy observation time points
    @test estimate.y_obs == obs.y_obs        # Should copy observation values
    @test estimate.nobs == obs.nobs          # Length should match
end

@testset "TTDs - InverseGaussian" begin
    # ---------------------------------------------------------------
    # These tests validate the Inverse Gaussian transit time 
    # distribution implementation, which is a core component for 
    # modeling water mass age distributions.
    # ---------------------------------------------------------------
    
    # Create a TracerInverseGaussian TTD with specified parameters
    # The parameters represent:
    # - Γ: mean transit time (years)
    # - Δ: width parameter (controls the shape of the distribution)
    tig = TracerInverseGaussian(Γ_true, Δ_true)
    @test tig.Γ == Γ_true                    # Parameters should be stored exactly
    @test tig.Δ == Δ_true
    
    # Test that statistical properties match theoretical values
    @test mean(tig) ≈ Γ_true                 # Mean should match Γ parameter
    @test width(tig) ≈ Δ_true                # Width should match Δ parameter
    
    # Test PDF evaluation at various ages
    # The PDF should be positive and finite everywhere for τ > 0
    τ_values = [10.0, 25.0, 50.0, 100.0]     # Test at various transit times
    for τ in τ_values
        pdf_value = pdf(InverseGaussian(tig), τ)
        @test pdf_value >= 0.0               # PDF should be non-negative
        @test isfinite(pdf_value)            # PDF should be finite
    end
    
    # Wrap in InverseGaussian for convolution operations
    # This adapts our TTD to work with standard Distributions.jl functions
    gaussian_ttd = InverseGaussian(tig)
    @test typeof(gaussian_ttd) <: InverseGaussian
end

@testset "Integrator" begin
    # ---------------------------------------------------------------
    # These tests validate the numerical integration methods used
    # for calculating convolutions with transit time distributions.
    # Accurate integration is crucial for proper tracer modeling.
    # ---------------------------------------------------------------
    
    # Set up adaptive break points for integration panels
    # This creates a non-uniform grid that's denser near t=0 
    # and coarser at large transit times for computational efficiency

    simple_gauss_π = make_integrator(:gausslegendre, 1000, 0.0, π/2) #from 0 to π/2 100 points
    @test sum(cos.(simple_gauss_π.nodes) .* simple_gauss_π.weights) ≈ 1.0

    break_points = [
        1.2 * (times[end] - times[1]),  # First break: just beyond data range
        25_000.0                        # Second break: transition to long times
    ]
    
    # Define node density in each panel (more nodes = higher accuracy)
    nodes_points = Int.(round.([
        10.0 * break_points[1],                 # Dense in first panel (recent times)
        0.03 * (break_points[2] - break_points[1]), # Moderate in middle panel
        0.01 * (τm - break_points[2])           # Sparse in last panel (ancient times)
    ]))
    
    # Create integration panels spanning from 0 to τm
    panels = make_integration_panels(0., τm, break_points, nodes_points)
    @test length(panels) == 3               # Should create exactly 3 panels
    @test panels[1][1] == 0.0               # First panel should start at t=0
    @test panels[end][2] ≈ τm               # Last panel should end at τm
    
    # Test creation of different integration methods
    # Gauss-Legendre quadrature (high accuracy)
    gauss_integrator = make_integrator(:gausslegendre, panels)
    @test !isnothing(gauss_integrator)
    
    # Uniform grid integration (simpler but requires more points)
    uniform_integrator = make_integrator(:uniform, panels)
    @test !isnothing(uniform_integrator)
    
    # Simple Gauss integrator with single panel and fixed number of points
    simple_gauss = make_integrator(:gausslegendre, 100, 0.0, τm) #from 0 to τm 100 points
    @test !isnothing(simple_gauss)
end

@testset "Convolution" begin
    # ---------------------------------------------------------------
    # These tests validate the convolution operations that combine
    # transit time distributions with surface boundary conditions
    # to predict interior tracer concentrations.
    # ---------------------------------------------------------------
    
    # Set up objects for convolution
    # Create TTD (water mass age distribution)
    gaussian_ttd = InverseGaussian(TracerInverseGaussian(Γ_true, Δ_true))
    
    # Define source function (surface boundary condition)
    # Here we use a step function: tracer is present after t=0
    unit_source(x) = (x >= 0) ? 1.0 : 0.0

    # Create integration framework with adaptive node spacing
    # This defines how the numerical convolution integral is computed
    break_points = [1.2 * (times[end] - times[1]), 25_000.0]
    nodes_points = Int.(round.([
        10.0 * break_points[1],                 # Dense in recent times
        0.03 * (break_points[2] - break_points[1]), # Moderate in middle range
        0.01 * (τm - break_points[2])           # Sparse in ancient times
    ]))
    panels = make_integration_panels(0., τm, break_points, nodes_points)
    gauss_integrator = make_integrator(:gausslegendre, panels)
    
    # Test convolution operation across the full time series
    # This computes: C(t) = ∫ TTD(τ) * Source(t-τ) dτ
    ttd_results = convolve(gaussian_ttd, unit_source, times; 
                          τ_max=τm,               # Maximum age to consider
                          integrator = gauss_integrator)
    
    @test length(ttd_results) == length(times)   # Should match input length
    @test all(ttd_results .>= 0.0)               # Results should be non-negative
    
    # For a step function source and normalized TTD, results should 
    # approach 1.0 at later times as more water parcels had contact
    # with the tracer at the surface
    @test ttd_results[end] > ttd_results[1]      # Later times should have higher values
    @test ttd_results[end] < 1.01                # Should not exceed source value
    
    # Create observation object from the synthetic results
    # This wraps the convolution results in the standard data structure
    ttd_observations = TracerObservation(times, ttd_results; 
                                       σ_obs = 0.1 .+ zero(ttd_results), 
                                       f_src = unit_source)
    @test typeof(ttd_observations) <: TracerObservation
end

@testset "Inverse Gaussian Optimization" begin
    # Set up known TTD for generating synthetic data
    # In real applications, these parameters would be unknown
    gaussian_ttd = InverseGaussian(TracerInverseGaussian(Γ_true, Δ_true))
    
    # Define unit step function as tracer boundary condition
    unit_source = x -> (x >= 0) ? 1.0 : 0.0
    
    # Create integrator for high-precision convolution
    break_points = [1.2 * (times[end] - times[1]), 25_000.0]
    nodes_points = Int.(round.([
        10.0 * break_points[1],                  # Dense in recent times
        0.03 * (break_points[2] - break_points[1]), # Moderate in middle range
        0.01 * (τm - break_points[2])            # Sparse in ancient times
    ]))
    panels = make_integration_panels(0., τm, break_points, nodes_points)
    gauss_integrator = make_integrator(:gausslegendre, panels)
    
    # Create synthetic observations with known parameters
    # Forward simulation: TTD + Source → Tracer values
    ttd_results = convolve(gaussian_ttd, unit_source, times; 
                          τ_max=τm, 
                          integrator = gauss_integrator)
    
    # Package results as observation with measurement uncertainty
    ttd_observations = TracerObservation(times, ttd_results; 
                                       σ_obs = 0.1 .+ zero(ttd_results), 
                                       f_src = unit_source)
    obs_vec = [ttd_observations]  # Multiple tracers could be added here
    
    # Test optimization with vector of observations
    # Inverse problem: Tracer values → TTD parameters
    result = invert_inverse_gaussian(
        obs_vec; 
        τ_max = τm, 
        integrator = gauss_integrator,
        u0 = [50.0, 50.0]  # Start from different values to test optimization
    )
    
    # Extract values from InversionResult
    Γ̂, Δ̂ = result.parameters
    estimates = result.obs_estimates
    optimizer_results = result.optimizer_output
    
    # Check parameter recovery accuracy
    @test !isnothing(optimizer_results)      # Optimizer should complete successfully
    @test 20 < Γ̂ < 30                        # Mean should be near true value (25)
    @test 20 < Δ̂ < 30                        # Width should be near true value (25)
    
    # Test optimization with single observation instead of vector
    # This tests the convenience method for single tracer inversion
    result2 = invert_inverse_gaussian(
        ttd_observations; 
        τ_max = τm, 
        integrator = gauss_integrator
    )
    Γ̂2, Δ̂2 = result2.parameters
    estimates2 = result2.obs_estimates
    optimizer_results2 = result2.optimizer_output
    
    # Verify both methods give consistent results
    @test isapprox(Γ̂, Γ_true, rtol=0.01)        # Parameters should match between methods
    @test isapprox(Δ̂, Δ_true, rtol=0.01)
end

@testset "Inverse Gaussian Equal Vars Optimization" begin
    # Set up known TTD for generating synthetic data
    # We use the same parameters as before for comparison
    gaussian_ttd = InverseGaussian(TracerInverseGaussian(Γ_true, Δ_true))
    unit_source = x -> (x >= 0) ? 1.0 : 0.0
    
    # Create integrator for convolution
    break_points = [1.2 * (times[end] - times[1]), 25_000.0]
    nodes_points = Int.(round.([
        10.0 * break_points[1],                  # Dense in recent times
        0.03 * (break_points[2] - break_points[1]), # Moderate in middle range
        0.01 * (τm - break_points[2])            # Sparse in ancient times
    ]))
    panels = make_integration_panels(0., τm, break_points, nodes_points)
    gauss_integrator = make_integrator(:gausslegendre, panels)
    
    # Create synthetic observations using the full model
    # Note: We generate data with a model where Γ=Δ but will
    # try to recover using the constrained equal-vars model
    ttd_results = convolve(gaussian_ttd, unit_source, times; 
                           τ_max=τm, 
                           integrator = gauss_integrator)
    ttd_observations = TracerObservation(times, ttd_results; 
                                       σ_obs = 0.1 .+ zero(ttd_results), 
                                       f_src = unit_source)
    obs_vec = [ttd_observations]
    
    # Test equal vars optimization with vector input
    # This uses the constraint that Γ = Δ (one parameter)
    result3 = invert_inverse_gaussian_equalvars(
        obs_vec; 
        τ_max = τm, 
        integrator = gauss_integrator,
        u0 = [50.0]  # Start from a different value to test optimization
    )
    Γ̂3 = result3.parameters[1]
    estimates3 = result3.obs_estimates
    optimizer_results3 = result3.optimizer_output
    
    # Test optimization with single observation instead of vector
    # This tests the convenience method for single tracer inversion
    result4 = invert_inverse_gaussian_equalvars(
        ttd_observations; 
        τ_max = τm, 
        integrator = gauss_integrator
    )
    Γ̂4 = result4.parameters[1]
    estimates4 = result4.obs_estimates
    optimizer_results4 = result4.optimizer_output
    
    @test isapprox(Γ̂4, Γ_true, rtol=0.01)        # Parameters should match between methods
    @test isapprox(Γ̂4, Δ_true, rtol=0.01)

    @test isapprox(Γ̂3, Γ_true, rtol=0.01)        # Parameters should match between methods
    @test isapprox(Γ̂3, Δ_true, rtol=0.01)
end
