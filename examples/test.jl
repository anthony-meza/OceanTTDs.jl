import Pkg
Pkg.activate(".")

using TCMGreensFunctions
using Plots
using Distributions
using QuadGK

μ = 1; λ = 3 #mean, width 


# f = InverseGaussian(TracerInverseGaussian(μ, λ))
f = InverseGaussian(μ, λ)
x_pdf = collect(0:0.05:3)
f_pdf = pdf.(f, x_pdf)

plot(x_pdf, f_pdf)

f_cdf_Distributions = cdf.(f, x_pdf)
quadgk_int_and_error = quadgk.(x -> pdf(f, x), 0, x_pdf)
f_cdf_quadgk = [quadgk_int_and_error[i][1] for i in 1:length(x_pdf)]
quadgk_error = [quadgk_int_and_error[i][2] for i in 1:length(x_pdf)]

plot(x_pdf, f_cdf_Distributions, lw = 3)
plot!(x_pdf, f_cdf_quadgk, linestyle = :dash, color = "green", lw = 2)
