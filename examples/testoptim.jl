	using Random, LinearAlgebra
	using NonlinearSolve
	
	# --- make synthetic data -------------------------------------------------------
	Random.seed!(42)
	a_true, b_true = 2.0, -1.5
	x = range(0, 1, length=2)
	σσ = 0.03
	y = a_true .* exp.(b_true .* x) .+ σσ .* randn(length(x))
	σw = fill(σσ, length(x))  # weights (std dev); can vary per point
	
	# --- residual: r(u) = (y_model(u) - y) / σw  ----------------------------------
	# u = [a, b]
	function r!(R, u, p)
	    a, b = u
	    @inbounds for i in eachindex(x)
	        R[i] = (a * exp(b * x[i]) - y[i]) / σw[i]
	    end
	    return nothing
	end
	
	# (optional) analytic Jacobian of residuals J[i,j] = ∂r_i/∂u_j
	function jac_r!(J, u, p)
	    a, b = u
	    @inbounds for i in eachindex(x)
	        e = exp(b * x[i])
	        J[i, 1] = e / σw[i]         # ∂r_i/∂a
	        J[i, 2] = (a * x[i] * e) / σw[i]  # ∂r_i/∂b
	    end
	    return nothing
	end
	
	# --- build and solve -----------------------------------------------------------
	u0 = [1.0, 0.0]  # initial guess
	
	# With analytic Jacobian (fast & precise):
	fn = NonlinearFunction(r!, jac=jac_r!)

	prob = NonlinearProblem(fn, u0, nothing)
	sol  = solve(prob, LevenbergMarquardt(); maxiters=200)
	
	# Or, drop `jac=` and let AD compute it:
	# prob = NonlinearProblem(r!, u0, nothing; residual = true)
	# sol  = solve(prob, LevenbergMarquardt(); autodiff = AutoForwardDiff())
	
	println("True params:     a=$(a_true), b=$(b_true)")
	println("Estimated params: a=$(sol.u[1]), b=$(sol.u[2])")
	# println("Residual norm:    ||r||₂ = ", norm(sol.residual))
	println("Converged:        ", sol.retcode)
