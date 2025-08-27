import Pkg
Pkg.activate("./examples")
Pkg.instantiate()
Pkg.resolve()

using TCMGreensFunctions
using Statistics, Distributions, LinearAlgebra,
       Random, JuMP, Ipopt
using PythonPlot

TruncatedNormal(μ, σ) = truncated(Normal(μ, σ); lower = 1e-12)

Γ_true = 300.0   # Mean transit time
times = collect(1.0:5.0:250.0)
τm = 250_000.0  # Maximum lag to consider

# Set up known TTD for generating synthetic data
# We use the same parameters as before for comparison
gaussian_ttd = InverseGaussian(TracerInverseGaussian(Γ_true, 2 * Γ_true)) 
unit_source = x -> (x >= 0) ? sqrt(x) : 0.0

# Create integrator for continuous convolution
break_points = [1.2 * (times[end] - times[1]), 25_000.0]
nodes_points = Int.(round.([
    5.0 * break_points[1],                  # Dense in recent times
    0.1 * (break_points[2] - break_points[1]), # Moderate in middle range
    0.01 * (τm - break_points[2])            # Sparse in ancient times
]))


panels = make_integration_panels(0., τm, break_points, nodes_points)
gauss_integrator = make_integrator(:gausslegendre, panels)
τs =  gauss_integrator.nodes
nτ = length(τs)

#setup psuedo samples
n_obs = 5
tobs = sort(sample(times, n_obs, replace=false))
true_observations = convolve(gaussian_ttd, unit_source, tobs; τ_max=τm, integrator = gauss_integrator)
σ_obs = repeat([1.], n_obs)
contaminated_observations = @. true_observations + rand(Normal(0, σ_obs))

tracer_obs = TracerObservation(tobs, contaminated_observations; 
                                    σ_obs = σ_obs, 
                                    f_src = unit_source)


######## BEGIN ESTIMATION COMPARISON ############
IG_opt_results = invert_inverse_gaussian_equalvars(tracer_obs; τ_max = τm, integrator = gauss_integrator)

Γ̂_IG = IG_opt_results.parameters[1]
IG_estimates = IG_opt_results.obs_estimates
IG_dist = IG_opt_results.distribution

discrete_prior_variation(Γ::Real) = pmf_from(IG_dist(Γ), τs, gauss_integrator.weights)

#~factor of 2 variations (~Γ/2 <-> ~2 * Γ)
Γ_dist = TruncatedNormal(Γ̂_IG, Γ̂_IG* 0.303) 
nensemble = nτ * 5
Σ, PΣ = BootstrapShrinkageCovariance(discrete_prior_variation, Γ_dist; total_samples = nensemble)
G0 = discrete_prior_variation(Γ̂_IG)

max_ent_results = max_ent_inversion(tracer_obs; C0 = nothing, prior_distribution = G0, support = τs)
max_ent_distribution = max_ent_results.distribution
max_ent_estimates = max_ent_results.obs_estimates

err_support = [-3*mean(σ_obs), 0, 3*mean(σ_obs)]
err_prior = ones(length(err_support)) .+ 1/length(err_support)

gen_max_ent_results = gen_max_ent_inversion(tracer_obs; C0 = nothing, prior_distribution = G0, support = τs, 
error_support = err_support, error_prior = err_prior)
gen_max_ent_distribution = gen_max_ent_results.distribution
gen_max_ent_estimates = gen_max_ent_results.obs_estimates
gen_max_ent_estimates[1].y_obs
gen_max_ent_estimates[1].yhat

gen_max_ent_estimates[1].y_obs
gen_max_ent_estimates[1].yhat
true_observations

tcm_opt_results = invert_tcm(tracer_obs; C0 = nothing, prior_distribution = G0, PΣ = PΣ, support = τs)
tcm_distribution = tcm_opt_results.distribution
tcm_estimates = tcm_opt_results.obs_estimates

fig, ax = subplots(1, 2)

# TTD plot
ax[0].plot(τs, pmf_from(gaussian_ttd, τs, gauss_integrator.weights), 
           label="True TTD", color="black", linewidth=4)
ax[0].plot(τs, tcm_distribution, label="TCM estimate", linewidth=2.5, color="red")
# ax[0].plot(τs, max_ent_distribution, label="Max Ent estimate", linewidth=2.5, color="green")
ax[0].plot(τs, gen_max_ent_distribution, label="Gen Max Ent estimate", linewidth=2.5, color="green")
ax[0].plot(τs, G0, label="IG Prior", linewidth=2.5, color="blue")
ax[0].set_xlim(0, 250)
ax[0].set_title("TTDs")

# Observations plot - scatter only
fig
ax[1].scatter(tobs, true_observations, s=50, c="black", marker="o", label="True Observations")
ax[1].errorbar(tobs, contaminated_observations, yerr = 2 * σ_obs, fmt="o",
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