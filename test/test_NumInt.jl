
# — Examples —

# Finite
@show sampled_integration(x->sin(x), 0, π)         # ≈ 2.0

# Semi‐infinite
@show sampled_integration(x->exp(-x), 0, Inf)     # ≈ 1.0

# Full infinite
@show sampled_integration(x->exp(-x^2), -Inf, Inf) # ≈ √π ≈ 1.77245…

@show sampled_integration(x->exp(-x), 0, Inf)     # ≈ 1.0

 sampled_integration(x->exp(-x), 0, Inf) ≈ 1.0

 sampled_integration(x -> pdf(Normal(666, 100), x), -Inf, Inf) ≈ 1.0
 
 sampled_integration(x -> pdf(Normal(666, 100), x), -Inf, Inf)