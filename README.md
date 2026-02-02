# PINN-SIREN for 2D Helmholtz Equation

A Physics-Informed Neural Network (PINN) using the SIREN (Sinusoidal Representation Networks) architecture to solve 2D electromagnetic wave propagation governed by the Helmholtz equation.

## Overview

This project simulates electromagnetic wave propagation from a point source in 2D, with support for dielectric materials. The simulation solves the frequency-domain Helmholtz equation:

```
(-∇² - ε·ω²) Ez = -i·ω·Jz
```

Where:
- **Ez**: Complex electric field amplitude (z-component)
- **ε**: Permittivity (1 in free space, configurable in dielectric regions)
- **ω**: Angular frequency
- **Jz**: Current source density (Gaussian point source)

## Features

- **SIREN Architecture**: Uses sinusoidal activation functions ideal for representing wave phenomena
- **Complex Field Representation**: Outputs both real and imaginary parts of the electric field
- **Dielectric Support**: Includes a configurable dielectric circle with smooth boundary
- **Open Boundaries**: Implements Sommerfeld radiation condition for non-reflecting boundaries
- **Automatic Differentiation**: Computes spatial derivatives using PyTorch autograd

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python pinn_helmholtz.py
```

This will train the PINN with default settings and generate visualization plots.

### Custom Configuration

```python
from pinn_helmholtz import HelmholtzPINN, plot_results, plot_wave_propagation

# Create PINN with custom settings
pinn = HelmholtzPINN(
    domain_size=(-6.0, 6.0),     # Simulation domain
    omega=0.8,                    # Angular frequency
    source_position=(-3.0, 0.0),  # Point source location
    source_width=0.15,            # Gaussian source width
    use_dielectric=True,          # Enable dielectric circle
    dielectric_center=(1.5, 0.0), # Dielectric center
    dielectric_radius=1.5,        # Dielectric radius
    dielectric_eps=2.0,           # Dielectric permittivity
)

# Train
history = pinn.train(
    n_iterations=15000,
    n_domain=5000,
    n_boundary=1000,
    lambda_bc=10.0,
)

# Visualize
plot_results(pinn, history)
plot_wave_propagation(pinn)
```

### Free Space Configuration (No Dielectric)

```python
pinn = HelmholtzPINN(
    omega=0.5,
    source_position=(0.0, 0.0),
    use_dielectric=False,
)
```

## Architecture Details

### SIREN Network

The network uses sinusoidal activations with the following architecture:

- **Input**: 2D coordinates (x, y)
- **Hidden Layers**: 5 layers with 256 neurons each
- **Activation**: `sin(ω₀ · Wx + b)` with ω₀ = 30
- **Output**: 2 values (Re(Ez), Im(Ez))

### Physics Loss

The physics loss enforces the Helmholtz equation:

```
L_physics = ||(-∇² - ε·ω²)Ez_real||² + ||(-∇² - ε·ω²)Ez_imag + ω·Jz||²
```

### Boundary Loss (Sommerfeld Radiation Condition)

For open boundaries, the Sommerfeld condition ensures outgoing waves:

```
∂Ez/∂r + i·k·Ez = 0
```

Where k = ω√ε is the wavenumber.

## Output Files

- `helmholtz_results.png`: Training history, field components, magnitude, phase, and permittivity
- `wave_propagation.png`: Time-domain wave animation showing propagation at different phases

## Physical Interpretation

The simulation shows:

1. **Cylindrical waves** emanating from the point source
2. **Refraction** when waves enter the dielectric region (shorter wavelength due to higher ε)
3. **Reflection** at the dielectric boundary
4. **Proper absorption** at the domain boundaries (no artificial reflections)

## Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `omega` | Angular frequency | 0.1 - 1.0 |
| `domain_size` | Simulation domain extent | Ensure several wavelengths fit |
| `source_width` | Gaussian source FWHM | ~0.1 - 0.3 |
| `dielectric_eps` | Dielectric permittivity | 2.0 - 4.0 |
| `n_iterations` | Training iterations | 10000 - 20000 |
| `n_domain` | Collocation points per iteration | 4000 - 8000 |

## References

1. Sitzmann, V., et al. "Implicit Neural Representations with Periodic Activation Functions." NeurIPS 2020.
2. Raissi, M., et al. "Physics-informed neural networks." Journal of Computational Physics, 2019.

## License

MIT License
