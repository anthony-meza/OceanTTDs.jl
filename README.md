# OceanTTDs.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://anthony-meza.github.io/OceanTTDs.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://anthony-meza.github.io/OceanTTDs.jl/dev/)
[![Build Status](https://github.com/anthony-meza/OceanTTDs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/anthony-meza/OceanTTDs.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/anthony-meza/OceanTTDs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/anthony-meza/OceanTTDs.jl)

**A Julia package for Transit Time Distribution (TTD) modeling and tracer inversion in oceanography**

OceanTTDs.jl provides a comprehensive toolkit for analyzing ocean tracer distributions using Transit Time Distributions (TTDs). The package implements multiple optimization methods including Time-Corrected Method (TCM), Maximum Entropy (MaxEnt), and Inverse Gaussian fitting for estimating water mass age distributions from tracer observations.

## Features

- **Transit Time Distributions**: Inverse Gaussian TTD modeling with flexible parameterization
- **Multiple Inversion Methods**:
  - Inverse Gaussian parameter fitting
  - Time-Corrected Method (TCM) with covariance regularization
  - Maximum Entropy inversion
  - Generalized Maximum Entropy with error modeling
- **Numerical Integration**: Adaptive quadrature methods optimized for convolution operations
- **Boundary Propagators**: Tools for modeling tracer boundary conditions and convolutions
- **Statistical Utilities**: Covariance estimation and regularization methods

## Installation

Install OceanTTDs.jl from the Julia package registry:

```julia
using Pkg
Pkg.add("OceanTTDs")
```

Or install the development version:

```julia
using Pkg
Pkg.add(url="https://github.com/anthony-meza/OceanTTDs.jl")
```

## Quick Start

Here's a simple example of fitting an Inverse Gaussian TTD to synthetic tracer data:

```julia
using OceanTTDs

# Define time points and parameters
times = collect(1.0:5.0:250.0)
Γ_true = 25.0   # Mean transit time
Δ_true = 25.0   # Width parameter
τ_max = 250_000.0

# Create synthetic observations
unit_source = x -> (x >= 0) ? 1.0 : 0.0  # Step function source
true_ttd = InverseGaussian(TracerInverseGaussian(Γ_true, Δ_true))

# Set up integration
break_points = [1.2 * (times[end] - times[1]), 25_000.0]
nodes_points = Int.(round.([10.0 * break_points[1], 0.03 * (break_points[2] - break_points[1]), 0.01 * (τ_max - break_points[2])]))
panels = make_integration_panels(0.0, τ_max, break_points, nodes_points)
integrator = make_integrator(:gausslegendre, panels)

# Generate synthetic data
synthetic_data = convolve(true_ttd, unit_source, times; τ_max=τ_max, integrator=integrator)
obs = TracerObservation(times, synthetic_data; σ_obs=0.1 .+ zero(synthetic_data), f_src=unit_source)

# Fit TTD parameters
result = invert_inverse_gaussian(obs; τ_max=τ_max, integrator=integrator)
Γ_fit, Δ_fit = result.parameters

println("True parameters: Γ=$Γ_true, Δ=$Δ_true")
println("Fitted parameters: Γ=$Γ_fit, Δ=$Δ_fit")
```

## Main Components

### Transit Time Distributions (`TTDs`)
- `TracerInverseGaussian`: Parameterized inverse Gaussian TTD
- `InverseGaussian`: Wrapper for integration with Distributions.jl

### Optimization Methods (`Optimizers`)
- `invert_inverse_gaussian`: Least squares fitting of inverse Gaussian parameters
- `invert_tcm`: Time-Corrected Method with prior covariance
- `max_ent_inversion`: Maximum entropy TTD estimation
- `gen_max_ent_inversion`: Generalized maximum entropy with error modeling

### Data Structures
- `TracerObservation`: Container for tracer measurements with uncertainties
- `TracerEstimate`: Results container for model predictions
- `InversionResult`: Complete inversion results with parameters and diagnostics

### Numerical Tools
- `make_integrator`: Adaptive quadrature setup for convolution integrals
- `convolve`: TTD convolution with boundary conditions
- `BootstrapShrinkageCovariance`: Statistical covariance estimation

## Examples

See the `examples/` directory for detailed usage examples including:
- Comparison of different inversion methods
- Handling of measurement uncertainties
- Advanced integration configurations

## Scientific Background

This package implements methods for analyzing oceanic tracer distributions using the Transit Time Distribution framework. TTDs describe the probability distribution of water parcel ages, providing insights into ocean circulation timescales and mixing processes.

## Contributing

Contributions are welcome! Please see the documentation for details on development setup and contribution guidelines.

## Citation

If you use this package in your research, please cite:

```
@software{OceanTTDs,
  author = {A. Meza},
  title = {OceanTTDs.jl: Transit Time Distribution modeling for oceanography},
  url = {https://github.com/anthony-meza/OceanTTDs.jl},
  version = {1.0.0-DEV},
}
```
