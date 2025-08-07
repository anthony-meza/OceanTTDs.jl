using Turing, QuadGK, Distributions, Interpolations, Random, StatsPlots

# --- 1. Synthetic Atmospheric History (CFC-like rise) ---
years = 0:1:100  # years from 1925 to 2025
Ca_values = 400 .* (1 .- exp.(-0.04 .* years))  # Synthetic CFC-like growth
Ca_interp = LinearInterpolation(years, Ca_values, extrapolation_bc=Interpolations.Flat())

# --- 2. IG-TTD Definition ---
function IG_TTD(τ, Γ, Δ)
    if τ <= 0
        return 0.0
    end
    coeff = sqrt(Γ / (4 * π * Δ^2 * τ^3))
    exponent = -((τ - Γ)^2) / (4 * Δ^2 * τ)
    return coeff * exp(exponent)
end

# --- 3. Convolution Model ---
function model_concentration(Ca_interp, Γ, Δ, t₀)
    integrand(τ) = Ca_interp(t₀ - τ) * IG_TTD(τ, Γ, Δ)
    result, _ = quadgk(integrand, 0, t₀; rtol=1e-5)
    return result
end

# --- 4. Generate Synthetic Observations ---
Γ_true = 40.0   # Mean age in years
Δ_true = 15.0   # Age spread in years
σ = 10.0         # Measurement noise

# t_obs = [25.0, 40.0, 60.0, 80.0]  # Observation years = 2010–2025
# n_obs = 10
# t_obs = sort(rand(Uniform(years[1], years[end]), n_obs))
t_obs = collect(years[2:end-1])
C_clean = [model_concentration(Ca_interp, Γ_true, Δ_true, t) for t in t_obs]
C_obs = [c + rand(Normal(0, σ)) for c in C_clean]  # Add noise

scatter(t_obs, C_obs)
plot!(t_obs, C_clean)

function gamma_params_from_mean_and_variance(mean, variance)
    # Calculate shape and scale parameters
    shape = (mean^2) / variance  # α = μ² / σ²
    scale = variance / mean      # β = σ² / μ
    return shape, scale
end

function gamma_from_mean_and_variance(mean, variance)
    # Calculate shape and scale parameters
    shape, scale = gamma_params_from_mean_and_variance(mean, variance)
    return Gamma(shape, scale)
end

# --- 5. Bayesian Model ---
@model function IG_TTD_fit_time_series(C_obs, t_obs, Ca_interp, σ)
    # Γ ~ Uniform(10.0, 80.0)     # Prior over Γ
    # Δ ~ Uniform(1.0, 50.0)      # Prior over Δ
    Γ ~ Gamma(2.0, 10.0)         # Prior belief: Γ is a positive value, Gamma distribution with mean ~ 20 years
    Δ ~ Gamma(2.0, 10.0)         # Prior belief: Δ is a positive value, Gamma distribution with mean ~ 20 years
    # Γ ~ gamma_from_mean_and_variance(50, 10)         # Prior belief: Γ is a positive value, Gamma distribution with mean ~ 20 years
    # Δ ~ gamma_from_mean_and_variance(50, 10)         # Prior belief: Δ is a positive value, Gamma distribution with mean ~ 20 years

    for i in eachindex(t_obs)
        C_model = model_concentration(Ca_interp, Γ, Δ, t_obs[i])
        C_obs[i] ~ Normal(C_model, σ)
    end
end

# --- 6. Run MCMC Sampling ---
Random.seed!(123)
model = IG_TTD_fit_time_series(C_obs, t_obs, Ca_interp, σ)
chain = sample(model, NUTS(2000, 0.65), 10000)

# --- 7. Plot and Summarize Results ---
stats = describe(chain)

using DataFrames  # Required to access rows by names
# Convert to DataFrames
summary_df = DataFrame(stats[1])  # Summary stats
quantiles_df = DataFrame(stats[2])  # Quantiles

# Display DataFrames to inspect their structure
println("Summary statistics DataFrame:")
println(summary_df)

println("\nQuantiles DataFrame:")
println(quantiles_df)

# Extract the posterior mean for Γ and Δ
mean_Γ = summary_df[summary_df[!, :parameters] .== :Γ, :mean][1]
mean_Δ = summary_df[summary_df[!, :parameters] .== :Δ, :mean][1]

# Extract the 95% credible interval for Γ and Δ
ci_Γ = quantiles_df[quantiles_df[!, :parameters] .== :Γ, [:"2.5%", :"97.5%"]]
ci_Δ = quantiles_df[quantiles_df[!, :parameters] .== :Δ, [:"2.5%", :"97.5%"]]

# Print the results
println("\nPosterior mean of Γ: ", round(mean_Γ, digits=2))
println("95% CI for Γ: (", round(ci_Γ[1, 1], digits=2), ", ", round(ci_Γ[1, 2], digits=2), ")")

println("\nPosterior mean of Δ: ", round(mean_Δ, digits=2))
println("95% CI for Δ: (", round(ci_Δ[1, 1], digits=2), ", ", round(ci_Δ[1, 2], digits=2), ")")

plot(chain)
