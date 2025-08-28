import Pkg
Pkg.activate("./examples")
Pkg.instantiate()
Pkg.resolve()

using OceanTTDs
using Statistics, Distributions, LinearAlgebra,
       Random, JuMP, Ipopt
using PythonPlot

# Load the TTD construction utilities
include("ConstructKnownTTD.jl")

TruncatedNormal(μ, σ) = truncated(Normal(μ, σ); lower = 1e-12)

# Parameters for TTD setup
Γ_true = 300.0   # Mean transit time
times = collect(1.0:5.0:250.0)
τm = 250_000.0  # Maximum lag to consider

# Define source function (sqrt growth)
sqrt_source = x -> (x >= 0) ? sqrt(x) : 0.0

# Set up TTD using the utility function
ttd_setup = setup_inverse_gaussian_TTD(Γ_true, 2 * Γ_true, sqrt_source; 
                                       times=times, τ_max=τm)

# Extract components for convenience
gaussian_ttd = ttd_setup.ttd
gauss_integrator = ttd_setup.integrator
τs = ttd_setup.τ_nodes
nτ = length(τs)

# Create synthetic observations
n_obs = 5
noise_level = 1.0
tracer_obs = create_synthetic_observations(ttd_setup; 
                                         n_obs=n_obs, 
                                         noise_level=noise_level, 
                                         seed=123)  # Set seed for reproducibility


######## BEGIN ESTIMATION COMPARISON ############
IG_opt_results = invert_inverse_gaussian_equalvars(tracer_obs; τ_max = τm, integrator = gauss_integrator)

Γ̂_IG = IG_opt_results.parameters[1]
IG_estimates = IG_opt_results.obs_estimates
IG_dist = IG_opt_results.distribution

discrete_prior_variation(Γ::Real) = discretize_pdf(IG_dist(Γ), τs, gauss_integrator.weights)

#~factor of 2 variations (~Γ/2 <-> ~2 * Γ)
Γ_dist = TruncatedNormal(Γ̂_IG, Γ̂_IG* 0.303) 
nensemble = nτ * 5
Σ, PΣ = BootstrapShrinkageCovariance(discrete_prior_variation, Γ_dist; total_samples = nensemble)
G0 = discrete_prior_variation(Γ̂_IG)

max_ent_results = max_ent_inversion(tracer_obs; C0 = nothing, prior_distribution = G0, support = τs)
max_ent_distribution = max_ent_results.distribution
max_ent_estimates = max_ent_results.obs_estimates

err_support = [-3*mean(tracer_obs.σ_obs), 0, 3*mean(tracer_obs.σ_obs)]
err_prior = ones(length(err_support)) .+ 1/length(err_support)

gen_max_ent_results = gen_max_ent_inversion(tracer_obs; C0 = nothing, prior_distribution = G0, support = τs, 
error_support = err_support, error_prior = err_prior)
gen_max_ent_distribution = gen_max_ent_results.distribution
gen_max_ent_estimates = gen_max_ent_results.obs_estimates

tcm_opt_results = invert_tcm(tracer_obs; C0 = nothing, prior_distribution = G0, PΣ = PΣ, support = τs)
tcm_distribution = tcm_opt_results.distribution
tcm_estimates = tcm_opt_results.obs_estimates

fig, ax = subplots(1, 2)

# TTD plot
ax[0].plot(τs, discretize_pdf(gaussian_ttd, τs, gauss_integrator.weights), 
           label="True TTD", color="black", linewidth=4)
ax[0].plot(τs, tcm_distribution, label="TCM estimate", linewidth=2.5, color="red")
# ax[0].plot(τs, max_ent_distribution, label="Max Ent estimate", linewidth=2.5, color="green")
ax[0].plot(τs, gen_max_ent_distribution, label="Gen Max Ent estimate", linewidth=2.5, color="green")
ax[0].plot(τs, G0, label="IG Prior", linewidth=2.5, color="blue")
ax[0].set_xlim(0, 250)
ax[0].set_title("TTDs")

# Observations plot - scatter only
fig
# Get true observations (without noise) for comparison
tobs = tracer_obs.t_obs
true_observations = convolve(gaussian_ttd, ttd_setup.source, tobs; τ_max=τm, integrator=gauss_integrator)

ax[1].scatter(tobs, true_observations, s=50, c="black", marker="o", label="True Observations")
ax[1].errorbar(tobs, tracer_obs.y_obs, yerr = 2 * tracer_obs.σ_obs, fmt="o",
               label="Contaminated Observations", markersize=6, capsize=3, color="red")
ax[1].scatter(tobs, IG_estimates[1].yhat, s=40, marker="s", label="IG Prediction", alpha=0.8)
ax[1].scatter(tobs, tcm_estimates[1].yhat, s=40, marker="^", label="TCM Prediction", alpha=0.9)
ax[1].scatter(tobs, max_ent_estimates[1].yhat, s=40, marker="^", label="MaxEnt Prediction", alpha=0.9)
ax[1].set_title("Observations vs Predictions")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Concentration")

[a.legend() for a in ax]
[a.grid(alpha = 0.2) for a in ax]

tight_layout()
fig