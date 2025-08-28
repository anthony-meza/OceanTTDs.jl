```@meta
CurrentModule = OceanTTDs
```

# OceanTTDs

Documentation for [OceanTTDs](https://github.com/anthony-meza/OceanTTDs.jl).

**A Julia package for Transit Time Distribution (TTD) modeling and tracer inversion in oceanography**

## Overview

OceanTTDs.jl provides a comprehensive toolkit for analyzing ocean tracer distributions using Transit Time Distributions (TTDs). The package implements multiple optimization methods including Time-Corrected Method (TCM), Maximum Entropy (MaxEnt), and Inverse Gaussian fitting for estimating water mass age distributions from tracer observations.

## Features

- **Transit Time Distributions**: Inverse Gaussian TTD modeling with flexible parameterization
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
