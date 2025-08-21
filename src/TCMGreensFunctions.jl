module TCMGreensFunctions
    using LinearAlgebra
    using Interpolations
    using FastGaussQuadrature
    using Distributions  # InverseGaussian
    using BlackBoxOptim
    using Reexport
    # using Downloads
    # using MAT
    # using QuadGK
    # using AccurateArithmetic
    # using CSV
    # using LeastSquaresOptim
    # using Optim
    # import Optim: minimizer
    # using DimensionalData
    # using DimensionalData: @dim 
    # using Unitful
    # using AlgebraicArrays
    # export TracerObservation, TracerEstimate
    include("TracerObservations.jl")
    @reexport using .TracerObservations

    include("TTDs/TTDs.jl")
    @reexport using .TTDs

    include("BoundaryPropagators/BoundaryPropagators.jl")
    @reexport using .BoundaryPropagators

    include("stat_utils.jl")
    @reexport using .stat_utils

    include("Optimizers/Optimizers.jl")
    @reexport using .Optimizers

end

using .TCMGreensFunctions
using Statistics, Distributions
using LinearAlgebra
# Common test parameters
const Γ_true = 25.0   # Mean transit time
const times = collect(1.0:5.0:250.0)
const τm = 250_000.0  # Maximum age to consider

# Set up known TTD for generating synthetic data
# We use the same parameters as before for comparison
gaussian_ttd = InverseGaussian(TracerInverseGaussian(Γ_true, 2*Γ_true)) 


unit_source = x -> (x >= 0) ? x^2 / 1e2 : 0.0

# Create integrator for convolution
break_points = [1.2 * (times[end] - times[1]), 25_000.0]
nodes_points = Int.(round.([
    1.0 * break_points[1],                  # Dense in recent times
    0.1 * (break_points[2] - break_points[1]), # Moderate in middle range
    0.01 * (τm - break_points[2])            # Sparse in ancient times
]))

panels = make_integration_panels(0., τm, break_points, nodes_points)
gauss_integrator = make_integrator(:gausslegendre, panels)

rand_times = sort(unique(rand(times, 3)))
nt = length(rand_times)

ttd_results = convolve(gaussian_ttd, unit_source, rand_times; 
                        τ_max=τm, 
                        integrator = gauss_integrator)
contamination_level = 5
ttd_results .+= rand(Normal(0, contamination_level), 3)

# Add noise to the results
ttd_observations = TracerObservation(rand_times, ttd_results; 
                                    σ_obs = contamination_level .+ zero(ttd_results), 
                                    f_src = unit_source)

optimizer_results3, Γ̂3, estimates3, d_template = invert_inverse_gaussian_equalvars(
    ttd_observations; 
    τ_max = τm, 
    integrator = gauss_integrator,
)


using Plots
scatter(estimates3[1].t_obs, estimates3[1].y_obs, label="Observed")
scatter!(estimates3[1].t_obs, estimates3[1].yhat, label="Predicted")

τs =  gauss_integrator.nodes
plot(τs, pdf.(d_template(Γ̂3), τs), label = "Estimated TTD")
plot!(τs, pdf.(gaussian_ttd, τs), xlims = (0, 100), label = "True TTD")



discrete_prior_variation(Γ) = pmf_from(d_template(Γ), τs)
TruncatedNormal(μ, σ) = truncated(Normal(μ, σ); lower = 1e-12)
Γ̂P = Γ̂3
param_mean = Γ̂P
param_uncertainty =  Γ̂P * 0.303

Γ_dist = TruncatedNormal(param_mean, param_uncertainty)

nensemble = length(τs) * 4; 
prior_ensemble = zeros(nensemble, length(τs))
for i in 1:nensemble
    prior_ensemble[i, :] .= discrete_prior_variation(rand(Γ_dist))
end

using CovarianceEstimation


LSE = LinearShrinkage
# - Chen target + shrinkage (using the more verbose call)
method = LSE(target=DiagonalCommonVariance(), shrinkage=:lw)
# method = LSE(target=CommonCovariance(), shrinkage=:lw)
# method = LSE(target=DiagonalUnitVariance(), shrinkage=:lw)
# method = AnalyticalNonlinearShrinkage(; alpha=1.0)
Σ1 = cov(method, prior_ensemble)

inv_Σ1 = inv(Σ1)

idx = 500
τs[idx]
scatter(inv_Σ1[idx, :], xlims = (idx - 10, idx + 10))  

diag(inv_Σ1)