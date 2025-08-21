using LinearAlgebra
using Statistics
using Random

"""
    WelfordCovariance{T}

Implements Welford's algorithm for incremental covariance matrix updates.
Allows updating statistics with new batches of data without storing all historical data.

# Fields
- `n::Int`: Total number of samples seen
- `mean::Vector{T}`: Running mean vector
- `M2::Matrix{T}`: Sum of squares of differences from mean (covariance accumulator)
- `n_features::Int`: Number of features

# References
- Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products. Technometrics, 4(3), 419-420.
- Knuth, D. E. (1998). The Art of Computer Programming, Volume 2: Seminumerical Algorithms (3rd ed.). Addison-Wesley.
"""
mutable struct WelfordCovariance{T<:AbstractFloat}
    n::Int
    mean::Vector{T}
    M2::Matrix{T}
    n_features::Int
    
    function WelfordCovariance{T}(n_features::Int=0) where T<:AbstractFloat
        new{T}(0, Vector{T}(), Matrix{T}(undef, 0, 0), n_features)
    end
end

# Convenience constructor
WelfordCovariance(n_features::Int=0) = WelfordCovariance{Float64}(n_features)

"""
    initialize!(wc::WelfordCovariance{T}, n_features::Int) where T

Initialize the WelfordCovariance struct with the given number of features.
"""
function initialize!(wc::WelfordCovariance{T}, n_features::Int) where T
    wc.n_features = n_features
    wc.mean = zeros(T, n_features)
    wc.M2 = zeros(T, n_features, n_features)
    return wc
end

"""
    update_batch!(wc::WelfordCovariance{T}, X::AbstractMatrix{T}) where T

Update statistics with a new batch of data using Welford's algorithm.

# Arguments
- `wc::WelfordCovariance{T}`: The WelfordCovariance instance to update
- `X::AbstractMatrix{T}`: New batch of data, size (n_samples, n_features)

# Returns
- `NamedTuple`: Current statistics (mean, covariance, count)
"""
function update_batch!(wc::WelfordCovariance{T}, X::AbstractMatrix{<:Real}) where T
    X_converted = convert(Matrix{T}, X)
    n_samples, n_features = size(X_converted)
    
    # Initialize on first batch
    if wc.n == 0
        initialize!(wc, n_features)
    elseif n_features != wc.n_features
        throw(DimensionMismatch("Expected $(wc.n_features) features, got $n_features"))
    end
    
    # Process each sample in the batch
    for i in 1:n_samples
        sample = view(X_converted, i, :)
        update_single!(wc, sample)
    end
    
    return get_statistics(wc)
end

"""
    update_batch!(wc::WelfordCovariance{T}, X::AbstractVector{T}) where T

Update statistics with a single sample (vector input).
"""
function update_batch!(wc::WelfordCovariance{T}, X::AbstractVector{<:Real}) where T
    X_converted = convert(Vector{T}, X)
    return update_batch!(wc, reshape(X_converted, 1, :))
end

"""
    update_single!(wc::WelfordCovariance{T}, x::AbstractVector) where T

Update statistics with a single data point using Welford's algorithm.

# Arguments
- `wc::WelfordCovariance{T}`: The WelfordCovariance instance to update
- `x::AbstractVector`: Single data point, length n_features
"""
function update_single!(wc::WelfordCovariance{T}, x::AbstractVector) where T
    wc.n += 1
    
    # Update mean: new_mean = old_mean + (x - old_mean) / n
    delta = x - wc.mean
    wc.mean .+= delta ./ wc.n
    
    # Update M2 (sum of outer products of deviations)
    delta2 = x - wc.mean
    wc.M2 .+= delta * delta2'
    
    return nothing
end

"""
    get_covariance(wc::WelfordCovariance{T}; corrected::Bool=true) where T

Get current covariance matrix.

# Arguments
- `wc::WelfordCovariance{T}`: The WelfordCovariance instance
- `corrected::Bool`: If true, use sample covariance (n-1), else population covariance (n)

# Returns
- `Matrix{T}`: Covariance matrix
"""
function get_covariance(wc::WelfordCovariance{T}; corrected::Bool=true) where T
    if wc.n == 0
        return Matrix{T}(undef, 0, 0)
    end
    
    ddof = corrected ? 1 : 0
    if wc.n <= ddof
        return fill(T(NaN), wc.n_features, wc.n_features)
    end
    
    return wc.M2 ./ (wc.n - ddof)
end

"""
    get_correlation(wc::WelfordCovariance{T}) where T

Get current correlation matrix.

# Returns
- `Matrix{T}`: Correlation matrix
"""
function get_correlation(wc::WelfordCovariance{T}) where T
    cov_matrix = get_covariance(wc)
    std_devs = sqrt.(diag(cov_matrix))
    
    # Avoid division by zero
    std_devs = replace(x -> x == 0 ? one(T) : x, std_devs)
    
    return cov_matrix ./ (std_devs * std_devs')
end

"""
    get_statistics(wc::WelfordCovariance{T}) where T

Get all current statistics.

# Returns
- `NamedTuple`: Named tuple containing count, mean, covariance, correlation, and variance
"""
function get_statistics(wc::WelfordCovariance{T}) where T
    if wc.n == 0
        return (count=0, mean=Vector{T}(), covariance=Matrix{T}(undef, 0, 0), 
                correlation=Matrix{T}(undef, 0, 0), variance=Vector{T}())
    end
    
    cov_matrix = get_covariance(wc)
    return (
        count=wc.n,
        mean=copy(wc.mean),
        covariance=cov_matrix,
        correlation=get_correlation(wc),
        variance=diag(cov_matrix)
    )
end

"""
    merge!(wc1::WelfordCovariance{T}, wc2::WelfordCovariance{T}) where T

Merge statistics from another WelfordCovariance instance into the first one.
Useful for parallel computation or combining statistics from different sources.

Uses Chan's parallel algorithm for combining online statistics.

# Arguments
- `wc1::WelfordCovariance{T}`: Target instance (will be modified)
- `wc2::WelfordCovariance{T}`: Source instance (will not be modified)

# References
- Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). Algorithms for computing the sample variance. The American Statistician, 37(3), 242-247.
"""
function merge!(wc1::WelfordCovariance{T}, wc2::WelfordCovariance{T}) where T
    if wc2.n == 0
        return wc1
    end
    
    if wc1.n == 0
        wc1.n = wc2.n
        wc1.mean = copy(wc2.mean)
        wc1.M2 = copy(wc2.M2)
        wc1.n_features = wc2.n_features
        return wc1
    end
    
    if wc1.n_features != wc2.n_features
        throw(DimensionMismatch("Cannot merge WelfordCovariance instances with different numbers of features"))
    end
    
    # Combined count
    combined_n = wc1.n + wc2.n
    
    # Combined mean using Chan's algorithm
    delta = wc2.mean - wc1.mean
    combined_mean = wc1.mean + delta * wc2.n / combined_n
    
    # Combined M2 using parallel algorithm
    combined_M2 = wc1.M2 + wc2.M2 + (delta * delta') * (wc1.n * wc2.n / combined_n)
    
    # Update first instance
    wc1.n = combined_n
    wc1.mean = combined_mean
    wc1.M2 = combined_M2
    
    return wc1
end

"""
    merge(wc1::WelfordCovariance{T}, wc2::WelfordCovariance{T}) where T

Create a new WelfordCovariance instance by merging two existing ones.
"""
function merge(wc1::WelfordCovariance{T}, wc2::WelfordCovariance{T}) where T
    result = WelfordCovariance{T}()
    if wc1.n > 0
        result.n = wc1.n
        result.mean = copy(wc1.mean)
        result.M2 = copy(wc1.M2)
        result.n_features = wc1.n_features
    end
    merge!(result, wc2)
    return result
end

# Demo functions

"""
    demo_welford_covariance()

Demonstration of how Welford's covariance estimates improve with more data.
Shows convergence to true parameters as sample size increases.
"""
function demo_welford_covariance()
    println("How Covariance Estimates Improve with More Data")
    println("=" ^ 55)
    println("This demo shows how each new batch of data points improves our")
    println("estimate of the true covariance matrix using Welford's algorithm.\n")
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # True population parameters we're trying to estimate
    true_mean = [1.0, 2.0, 3.0]
    true_cov = [2.0 0.5 -0.3;
                0.5 1.5 0.2;
                -0.3 0.2 1.0]
    
    println("🎯 TRUE POPULATION PARAMETERS:")
    println("   Mean: $true_mean")
    println("   Covariance matrix:")
    for i in 1:3
        println("   $(true_cov[i,:])")
    end
    println()
    
    # Create multivariate normal distribution
    L = cholesky(true_cov).L
    
    # We'll show estimates at these cumulative sample sizes
    checkpoints = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 30000]
    
    welford = WelfordCovariance{Float64}()
    current_samples = 0
    
    println("📊 CONVERGENCE PROGRESS:")
    println("Sample Size | Mean Error | Cov Error  | Status")
    println("-" ^ 45)
    
    for target_samples in checkpoints
        # Generate new data points to reach target sample size
        new_samples_needed = target_samples - current_samples
        
        if new_samples_needed > 0
            # Generate batch data (multivariate normal)
            Z = randn(new_samples_needed, 3)
            batch_data = Z * L' .+ true_mean'
            
            # Update Welford estimates
            stats = update_batch!(welford, batch_data)
            current_samples = target_samples
        else
            stats = get_statistics(welford)
        end
        
        # Calculate estimation errors
        mean_error = maximum(abs.(stats.mean - true_mean))
        cov_error = maximum(abs.(stats.covariance - true_cov))
        
        # Status indicators
        status = ""
        if cov_error > 0.5
            status = "📈 Getting started"
        elseif cov_error > 0.2
            status = "🎯 Converging"
        elseif cov_error > 0.1
            status = "✨ Good accuracy"
        else
            status = "🎉 Excellent!"
        end
        
        println("$(lpad(target_samples, 11)) | $(rpad(round(mean_error, digits=4), 10)) | $(rpad(round(cov_error, digits=4), 10)) | $status")
        
        # Show detailed results at key milestones
        if target_samples in [50, 500, 5000]
            println("\n📋 DETAILED ESTIMATES AT $target_samples SAMPLES:")
            println("   Estimated mean: $(round.(stats.mean, digits=3))")
            println("   Estimated covariance:")
            for i in 1:3
                println("   $(round.(stats.covariance[i,:], digits=3))")
            end
            println("   💡 As we see more data, estimates get closer to true values!\n")
        end
    end
    
    # Final detailed comparison
    final_stats = get_statistics(welford)
    
    println("\n🏆 FINAL RESULTS AFTER $(final_stats.count) SAMPLES:")
    println("True vs Estimated Comparison:")
    println()
    
    println("MEANS:")
    println("True:      $(true_mean)")
    println("Estimated: $(round.(final_stats.mean, digits=4))")
    println("Difference: $(round.(abs.(final_stats.mean - true_mean), digits=4))")
    println()
    
    println("COVARIANCE MATRICES:")
    println("True:")
    display(true_cov)
    println("\nEstimated:")
    display(round.(final_stats.covariance, digits=4))
    println("\nAbsolute differences:")
    display(round.(abs.(final_stats.covariance - true_cov), digits=4))
    
    println("\n🔍 INTERPRETATION:")
    println("• Sample size = total number of individual data points (observations)")
    println("• Each data point is a 3-dimensional vector [x₁, x₂, x₃]")
    println("• With more samples, our estimates converge to the true population values")
    println("• Welford's algorithm maintains numerical stability throughout")
    
    final_mean_error = maximum(abs.(final_stats.mean - true_mean))
    final_cov_error = maximum(abs.(final_stats.covariance - true_cov))
    
    println("\n📈 FINAL ACCURACY:")
    println("• Maximum mean error: $(round(final_mean_error, digits=6))")
    println("• Maximum covariance error: $(round(final_cov_error, digits=6))")
    println("• Both errors should decrease as O(1/√n) with more samples")
end

demo_welford_covariance()