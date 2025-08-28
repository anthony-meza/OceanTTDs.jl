```@meta
CurrentModule = OceanTTDs
```

# OceanTTDs

Documentation for [OceanTTDs](https://github.com/anthony-meza/OceanTTDs.jl).

**A Julia package for inferring transit-time distribution (TTD) from oceanographic tracer observations**

## Features

- **Multiple Inversion Methods**: Inverse Gaussian parameter fitting, Time-Corrected Method (TCM), Maximum Entropy inversion
- **Numerical Integration**: Adaptive quadrature methods optimized for convolution operations  
- **Statistical Utilities**: Covariance estimation and regularization methods

## Main Functions

### Data Structures

```@docs
TracerObservation
InversionResult
```

### Optimization Methods

```@docs  
invert_inverse_gaussian
max_ent_inversion
```

### Utility Functions

```@docs
convolve
tracer_observation
uniform_prior
```

```@index
```
