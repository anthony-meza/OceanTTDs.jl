using TCMGreensFunctions
using Test

@testset "Integration Tests" begin

    ub = π/2
    lb = 0.0
    quadgk_int = integrate(cos, lb, ub; 
                           method = :quadgk)  # Example usage

    trap_int = integrate(cos, lb, ub; 
                         method = :trapezoidal, 
                         x_nodes=range(lb, ub, length=1000000))  # Example usage
    simp_int = integrate(cos, lb, ub; 
                         method = :simpsons, 
                         x_nodes=range(lb, ub, length=100))  # Example usage

    @test quadgk_int ≈ 1.0
    @test trap_int ≈ 1.0
    @test simp_int ≈ 1.0

    # ∫ cos(x) dt from 0 to t is sin(t) - sin(0) = sin(t)
    my_sin(x) = indefinite_integral(cos, 0)(x)
    x = 0:0.1:(2 * pi)
    @test all(my_sin.(x) .≈ sin.(x))

end