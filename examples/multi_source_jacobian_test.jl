using TCMGreensFunctions
using FiniteDiff, LinearAlgebra, Random, ForwardDiff

# Multi-source Jacobian Test with different units
Random.seed!(42)

# Time support
τ_support = [0.1, 0.3, 0.5, 0.7, 0.9]

# Define multiple source functions with different units/characteristics
# Source 1: Temperature tracer (exponential decay, units: °C)
f_temp(t) = t > 0 ? 10.0 * exp(-t/2.0) : 0.0

# Source 2: Chemical concentration (step function + decay, units: mg/L)  
f_chem(t) = t > 0 ? 5.0 * (t < 1.0 ? 1.0 : exp(-(t-1.0))) : 0.0

# Source 3: Flow rate tracer (sine wave decay, units: L/s)
f_flow(t) = t > 0 ? 2.0 * sin(π*t) * exp(-t/3.0) : 0.0

# Observation times for each tracer (can be different)
t_obs_temp = [0.2, 0.8, 1.5, 2.2]
t_obs_chem = [0.5, 1.0, 1.8]  
t_obs_flow = [0.3, 1.2, 2.0, 2.8, 3.5]

# True distribution (bi-modal for complexity)
true_dist = 0.6 * exp.(-2.0 * τ_support) + 0.4 * exp.(-0.5 * (τ_support .- 0.6).^2 / 0.1^2)
true_dist = true_dist ./ sum(true_dist)

# Generate synthetic observation data for each tracer
y_temp_true = [sum(true_dist .* f_temp.(t .- τ_support)) for t in t_obs_temp]
y_temp_obs = y_temp_true .+ 0.05 * abs.(y_temp_true) .* randn(length(y_temp_true))

y_chem_true = [sum(true_dist .* f_chem.(t .- τ_support)) for t in t_obs_chem] 
y_chem_obs = y_chem_true .+ 0.02 * abs.(y_chem_true) .* randn(length(y_chem_true))

y_flow_true = [sum(true_dist .* f_flow.(t .- τ_support)) for t in t_obs_flow]
y_flow_obs = y_flow_true .+ 0.03 * abs.(y_flow_true) .* randn(length(y_flow_true))

# Create tracer observations with different noise levels
tracer_temp = TracerObservation(
    t_obs_temp,
    y_temp_obs,
    f_src = f_temp,
    σ_obs = 0.05 * abs.(y_temp_obs)
)

tracer_chem = TracerObservation(
    t_obs_chem,
    y_chem_obs,
    f_src = f_chem,
    σ_obs = 0.02 * abs.(y_chem_obs)
)

tracer_flow = TracerObservation(
    t_obs_flow,
    y_flow_obs,
    f_src = f_flow,
    σ_obs = 0.03 * abs.(y_flow_obs)
)

# Prior distribution (uniform)
prior_dist = ones(length(τ_support)) / length(τ_support)

println("Multi-source Jacobian Test:")
println("Temperature observations: $(length(t_obs_temp)) (units: °C)")
println("Chemical observations: $(length(t_obs_chem)) (units: mg/L)")
println("Flow observations: $(length(t_obs_flow)) (units: L/s)")
println("Total observations: $(length(t_obs_temp) + length(t_obs_chem) + length(t_obs_flow))")
println("Support size: $(length(τ_support))")
println()
println("t_obs_temp = $t_obs_temp")
println("y_obs_temp = $(round.(y_temp_obs, digits=4))")
println("t_obs_chem = $t_obs_chem") 
println("y_obs_chem = $(round.(y_chem_obs, digits=4))")
println("t_obs_flow = $t_obs_flow")
println("y_obs_flow = $(round.(y_flow_obs, digits=4))")
println()

observations_vec = [tracer_temp, tracer_chem, tracer_flow]

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

nobs = total_observations(observations_vec)
test_λ = zeros(nobs)  # Test at λ = 0

println("Testing at λ = zeros($(nobs))")
println("Lambda-to-index mapping ranges:")
for (i, obs) in enumerate(observations_vec)
    println("  Tracer $i: $(λ2i_map.ranges[i])")
end
println()

function r!(R, λ, p)
    d = generate_MaxEntTTD(λ)
    @inbounds for k in eachindex(observations_vec)
        obs = observations_vec[k];
        ŷ = convolve(d.probs, obs.f_src, obs.t_obs; τ_support = τ_support, C0=0.0)
        obs_ranges = λ2i_map.ranges[k]
        R[obs_ranges] .= (ŷ .- obs.y_obs)
    end
end

function jac_r!(J, λ, p)
    m = total_observations(observations_vec)   # number of residuals (observations)
    n = length(λ)       # number of parameters
    
    # Check Jacobian matrix dimensions
    size(J) == (m, n) || throw(DimensionMismatch("Jacobian matrix size $(size(J)) != ($m, $n)"))

    # Max-Ent distribution evaluated on quadrature nodes
    d = generate_MaxEntTTD(λ)

    # J[i, k] = ∑_j f(t_i - τ_j) * d(τ_j) * ( Ĉ_k - f(t_k - τ_j) )
    @inbounds for i in 1:m               # rows: observations
        tri = i2λ_map.tracer_indices[i]
        obi = i2λ_map.obs_indices[i]

        ti = observations_vec[tri].t_obs[obi]
        fi = observations_vec[tri].f_src

        for k in 1:n                     # cols: surface sources
            trk = i2λ_map.tracer_indices[k]
            ork = i2λ_map.obs_indices[k]

            tk =  observations_vec[trk].t_obs[ork]
            fk = observations_vec[trk].f_src

            lagged_f_ti = fi.(ti .- d.support)
            lagged_f_tk = fk.(tk .- d.support)

            Ĉ_k = dot(lagged_f_tk, d.probs) 
            
            A_i = dot(lagged_f_ti, d.probs)
            B_ik = dot(lagged_f_ti .* lagged_f_tk, d.probs)
            J[i,k] = Ĉ_k * A_i - B_ik

        end
    end

end

# Compute manual Jacobian
manual_J = zeros(nobs, nobs)
jac_r!(manual_J, test_λ, nothing)

# Autodiff Jacobian using ForwardDiff
function residual_vec(λ)
    T = eltype(λ)  # This will be dual numbers during ForwardDiff
    R = zeros(T, nobs)
    r!(R, λ, nothing)
    return R
end

# Compute autodiff Jacobian
autodiff_J = ForwardDiff.jacobian(residual_vec, test_λ)
manual_J
# Compute errors
error_matrix = abs.(manual_J - autodiff_J)
max_error = maximum(error_matrix)
mean_error = sum(error_matrix) / length(error_matrix)

println("Manual Jacobian ($(nobs)x$(nobs)):")
display(round.(manual_J, digits=6))

println("\nAutodiff Jacobian ($(nobs)x$(nobs)):")
display(round.(autodiff_J, digits=6))

println("\nError Matrix:")
display(round.(error_matrix, digits=8))

println("\nError Statistics:")
println("  Max error: $(max_error)")
println("  Mean error: $(mean_error)")
println("  Error standard deviation: $(sqrt(sum((error_matrix .- mean_error).^2) / length(error_matrix)))")

# Test result
if max_error < 1e-6
    println("\n✅ Multi-source Jacobian test PASSED: Max error $(max_error) < 1e-6")
else
    println("\n❌ Multi-source Jacobian test FAILED: Max error $(max_error) >= 1e-6")
end

# Additional diagnostics
println("\n" * "="^60)
println("MULTI-SOURCE DIAGNOSTICS")
println("="^60)
println("Cross-tracer coupling analysis:")

# Analyze blocks of the Jacobian corresponding to different tracer combinations
temp_range = λ2i_map.ranges[1]
chem_range = λ2i_map.ranges[2] 
flow_range = λ2i_map.ranges[3]

println("\nJacobian block structure:")
println("Temperature-Temperature: rows $(temp_range), cols $(temp_range)")
println("Temperature-Chemical:    rows $(temp_range), cols $(chem_range)")
println("Temperature-Flow:        rows $(temp_range), cols $(flow_range)")
println("Chemical-Temperature:    rows $(chem_range), cols $(temp_range)")
println("Chemical-Chemical:       rows $(chem_range), cols $(chem_range)")
println("Chemical-Flow:           rows $(chem_range), cols $(flow_range)")
println("Flow-Temperature:        rows $(flow_range), cols $(temp_range)")
println("Flow-Chemical:           rows $(flow_range), cols $(chem_range)")
println("Flow-Flow:               rows $(flow_range), cols $(flow_range)")

println("\nBlock-wise max errors:")
println("Temp-Temp: $(maximum(error_matrix[temp_range, temp_range]))")
println("Temp-Chem: $(maximum(error_matrix[temp_range, chem_range]))")
println("Temp-Flow: $(maximum(error_matrix[temp_range, flow_range]))")
println("Chem-Temp: $(maximum(error_matrix[chem_range, temp_range]))")
println("Chem-Chem: $(maximum(error_matrix[chem_range, chem_range]))")
println("Chem-Flow: $(maximum(error_matrix[chem_range, flow_range]))")
println("Flow-Temp: $(maximum(error_matrix[flow_range, temp_range]))")
println("Flow-Chem: $(maximum(error_matrix[flow_range, chem_range]))")
println("Flow-Flow: $(maximum(error_matrix[flow_range, flow_range]))")

println("\n" * "="^60)
println("You can experiment by changing:")
println("- Source functions (f_temp, f_chem, f_flow): Different physical units/processes")
println("- Observation times: Different sampling strategies")
println("- Noise levels: Different measurement uncertainties") 
println("- true_dist: Different transit time distributions")
println("- Add more tracers with different characteristics")
println("="^60)