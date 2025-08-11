function integrate_maxent(
    Gp_in::F,
    τ_max::T,
    integr = nothing,
    C0::T = zero(T)
) where {F<:Function, T<:Real}

    # acc = zero(T)
    integrand = Gp_in.(integr.nodes)
    acc = dot(integrand, integr.weights)

    return acc + C0
end

# === Unified constructor (entry point) ===
"""
    MaxEntDist(λ, t_obs, f_atm, μ, integrator; implementation=:direct)

Builds a callable MaxEnt distribution object.

- `implementation = :direct` → exp-space (stores `Z`)
- `implementation = :stable` → log-space (stores `log(Z)`) for numerical stability
"""
function MaxEntDist(λ::Vector{G},
                    t_obs::Vector{T},
                    f_atm,
                    μ,
                    integrator;
                    implementation::Symbol = :stable) where {T,G}

    if implementation === :direct
        IndefMENum(τ) = MaxEntFuncNumerator(τ, λ, t_obs, f_atm, μ)
        Z = integrate_maxent(IndefMENum, first(integrator.nodes), integrator, zero(G))
        return MaxEntDistDirect{T,G,typeof(f_atm),typeof(μ)}(λ, t_obs, f_atm, μ, Z)

    elseif implementation === :stable
        LoggedIndefMENum(τ) = MaxEntFuncNumeratorStable(τ, λ, t_obs, f_atm, μ)
        #direct implementation of a stable sum, can be slow because creating many arrays
        a = LoggedIndefMENum.(integrator.nodes) .+ log.(integrator.weights)
        a_max = maximum(a)
        a_stable_sum = sum(exp.(a .- a_max)) #sum_oro doesn't work with NLSolve
        Zlog = a_max + log(a_stable_sum)

        return MaxEntDistStable{T,G,typeof(f_atm),typeof(μ)}(λ, t_obs, f_atm, μ, Zlog)
        #faster implementation of a stable sum, also includes Neumaier summation
        # does a trick where we keep track of the running max and scaled sum
        #and update as we go (NOT WORKING)
        # m = oftype(zero(T), -Inf)   # running max
        # s = zero(T)                 # running scaled sum
        # @inbounds for k in eachindex(integrator.nodes)
        #     x = LoggedIndefMENum(integrator.nodes[k])  + log(integrator.weights[k])
        #     if x <= m
        #         s += exp(x - m)
        #     else
        #         s = s * exp(m - x) + one(T)
        #         m = x
        #     end
        # end
        # Zlog = m + log(s)
        

    else
        throw(ArgumentError("implementation must be :direct or :stable"))
    end
end


# === Direct (exp-space) struct ===
struct MaxEntDistDirect{T,G,F1,F2}
    λ::Vector{G}
    t_obs::Vector{T}
    f_atm::F1
    μ::F2
    Z::G
end

function (d::MaxEntDistDirect)(τ::T) where {T}
    MaxEntFuncNumerator(τ, d.λ, d.t_obs, d.f_atm, d.μ) / d.Z
end


# === Stable (log-space) struct ===
struct MaxEntDistStable{T,G,F1,F2}
    λ::Vector{G}
    t_obs::Vector{T}
    f_atm::F1
    μ::F2
    Zlog::G
end

function (d::MaxEntDistStable)(τ::T) where {T}
    exp(MaxEntFuncNumeratorStable(τ, d.λ, d.t_obs, d.f_atm, d.μ) - d.Zlog)
end


# === Helper functions ===
function MaxEntFuncNumerator(τ::T,
                             λ::Vector{G},
                             t_obs::Vector{T},
                             f_atm,
                             μ) where {T,G}
    #Naive summation
    # s = zero(T)
    # @inbounds @simd for i in eachindex(t_obs)
    #     s = muladd(λ[i], f_atm(t_obs[i] - τ), s)
    # end

    # Neumaier summation (more accurate)
    s = zero(T)
    c = zero(T)   # compensation
    @inbounds for i in eachindex(t_obs, λ)
        term = muladd(λ[i], f_atm(t_obs[i] - τ), zero(T))
        t = s + term
        if abs(s) >= abs(term)
            c += (s - t) + term
        else
            c += (term - t) + s
        end
        s = t
    end

    μ(τ) * exp(-s)
end

function MaxEntFuncNumeratorStable(τ::T,
                                   λ::Vector{G},
                                   t_obs::Vector{T},
                                   f_atm,
                                   μ) where {T,G}
    #direct summation
    # s = zero(T)
    # @inbounds @simd for i in eachindex(t_obs)
    #     # s = muladd(λ[i], f_atm(t_obs[i] - τ), s)
    # end

    # Neumaier summation (more accurate)
    s = zero(T)
    c = zero(T)   # compensation
    @inbounds for i in eachindex(t_obs, λ)
        term = muladd(λ[i], f_atm(t_obs[i] - τ), zero(T))
        t = s + term
        if abs(s) >= abs(term)
            c += (s - t) + term
        else
            c += (term - t) + s
        end
        s = t
    end

    return log(μ(τ)) - s
end


# s = 0.
# z = [1e6, 1e-11, -1e6]
# for i in eachindex(z)
#     s += z[i]
# end
# s

# s = 0.
# z = [1e6, 1e-11, -1e6]
# for i in eachindex(z)
#     sa, se = two_sum(s, z[i])
#     s = (sa)
# end
# s
# sum_oro(zi for zi in z)
# # using BenchmarkTools
# # x = collect(1:1e4)
# # y = zeros(length(x))
# # @btime @. y = sin(x) + 3*x^2;
# # @btime y .= sin.(x) .+ 3 .* x.^2;



# function logsumexp_online(a)
#     m = -Inf; s = 0.0
#     @inbounds for x in a
#         if x <= m
#             s += exp(x - m)
#         else
#             s = s * exp(m - x) + 1
#             m = x
#         end
#     end
#     m + log(s)
# end

# a = randn(1000) .+ 20 .* rand()  # arbitrary
# m = maximum(a)
# two_pass = m + log(sum(exp.(a .- m)))
# isapprox(logsumexp_online(a), two_pass; rtol=1e-12)