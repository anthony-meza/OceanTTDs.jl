using TCMGreensFunctions
using FiniteDiff, LinearAlgebra, Random, ForwardDiff

# Simple Jacobian Test with 3 observations
Random.seed!(42)

# Time support (3 observations only)
t_obs = [0.5, 1.0, 1.5]
τ_support = [0.1, 0.3, 0.5, 0.7, 0.9]

# Simple exponential source function
f_src(t) = t > 0 ? exp(-t) : 0.0

# True distribution (exponential decay)
true_dist = exp.(-τ_support) 
true_dist = true_dist ./ sum(true_dist)

# Generate synthetic observation data
y_true = [sum(true_dist .* f_src.(t .- τ_support)) for t in t_obs]
y_obs = y_true .+ 0.01 * randn(length(y_true))

# Create tracer observation
tracer_obs = TracerObservation(
    t_obs,
    y_obs,
    f_src = f_src,
    σ_obs = fill(0.01, length(y_obs))
)

# Prior distribution (uniform)
prior_dist = ones(length(τ_support)) / length(τ_support)

println("Simple Jacobian Test:")
println("Observations: $(length(t_obs))")
println("Support size: $(length(τ_support))")
println("t_obs = $t_obs")
println("y_obs = $(round.(y_obs, digits=4))")
println()
observations_vec = [tracer_obs]

# Setup for manual vs autodiff comparison
generate_MaxEntTTD(λ::AbstractVector) = MaxEntTTD(λ, 
                                    observations_vec,
                                    prior_dist,
                                    τ_support;
                                    weights = nothing,
                                    cache_pdf = true,
                                    debug_checks = false)
λ2i_map = LambdaIndexMap(observations_vec)
i2λ_map = IndexLambdaMap(observations_vec)

nobs = total_observations([tracer_obs])
test_λ = zeros(nobs)  # Test at λ = 0

println("Testing at λ = $test_λ")

function r!(R, λ, p)
    d = generate_MaxEntTTD(λ)
    @inbounds for k in eachindex(observations_vec)
        obs = observations_vec[k];
        ŷ = convolve(d.probs, obs.f_src, obs.t_obs; τ_support = τ_support, C0=0.0)
        obs_ranges = λ2i_map.ranges[k]
        R[obs_ranges] .= (ŷ .- obs.y_obs)
    end
end

# Your original manual Jacobian
function manual_jac!(J, λ, p)
    m = size(J, 1)
    n = size(J, 2)
    
    d = generate_MaxEntTTD(λ)
    
    @inbounds for i in 1:m
        tri = i2λ_map.tracer_indices[i]
        obi = i2λ_map.obs_indices[i]
        
        ti = tracer_obs.t_obs[obi]
        fi = tracer_obs.f_src
        
        for k in 1:n
            trk = i2λ_map.tracer_indices[k]
            ork = i2λ_map.obs_indices[k]
            
            tk = tracer_obs.t_obs[ork]
            fk = tracer_obs.f_src
            
            lagged_f_ti = fi.(ti .- d.support)
            lagged_f_tk = fk.(tk .- d.support)
            
            Ĉ_k = dot(lagged_f_tk, d.probs)
            A_i = dot(lagged_f_ti, d.probs)
            B_ik = dot(lagged_f_ti .* lagged_f_tk, d.probs)
            J[i,k] = (Ĉ_k * A_i - B_ik)
        end
    end
end

# Compute Jacobians
manual_J = zeros(nobs, nobs)
manual_jac!(manual_J, test_λ, nothing)

# Autodiff Jacobian using ForwardDiff

# Create a function that returns the residual vector
function residual_vec(λ)
    T = eltype(λ)  # This will be dual numbers during ForwardDiff
    R = zeros(T, nobs)
    r!(R, λ, nothing)
    return R
end

# Compute autodiff Jacobian
autodiff_J = ForwardDiff.jacobian(residual_vec, test_λ)

autodiff_J
manual_J
# Compute errors

error_matrix = abs.(manual_J - autodiff_J)
max_error = maximum(error_matrix)

println("\nManual Jacobian ($(nobs)x$(nobs)):")
display(round.(manual_J, digits=6))

println("\nAutodiff Jacobian ($(nobs)x$(nobs)):")
display(round.(autodiff_J, digits=6))

println("\nError Matrix:")
display(round.(error_matrix, digits=8))

# Test result
if max_error < 1e-6
    println("\n✅ Jacobian test PASSED: Max error $(max_error) < 1e-6")
else
    println("\n❌ Jacobian test FAILED: Max error $(max_error) >= 1e-6")
end

println("\n" * "="^50)
println("You can now experiment by changing:")
println("- t_obs: observation times")
println("- τ_support: transit time distribution support") 
println("- f_src function: source function")
println("- true_dist: true distribution shape")
println("="^50)