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
- **Dielectric Support**: Includes a configurable dielectric circle with smooth boundary (bonus feature)
- **Open Boundaries**: Implements Sommerfeld radiation condition for non-reflecting boundaries
- **Finite Difference Derivatives**: Fast Laplacian computation via finite differences

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Demo (CPU-friendly)

For a fast demonstration on CPU:

```bash
python demo_quick.py --iterations 2000 --omega 1.0
```

Options:
- `--iterations`: Training iterations (default: 2000)
- `--omega`: Wave frequency, higher = more ripples (default: 1.0)
- `--no-dielectric`: Disable dielectric circle
- `--output`: Output image path

### Full Training (GPU recommended)

For best results with clear wave ripples, use GPU:

```bash
python pinn_helmholtz.py
```

**Note**: Full training on CPU is slow (~1-2 seconds/iteration). For production results:
- Use a CUDA-enabled GPU (10-50x faster)
- Or reduce `n_domain` and `n_boundary` for faster CPU training

### Custom Configuration

```python
from pinn_helmholtz import HelmholtzPINN, plot_results, plot_wave_propagation

# Create PINN with custom settings
pinn = HelmholtzPINN(
    domain_size=(-5.0, 5.0),      # Simulation domain
    omega=1.5,                     # Angular frequency (higher = shorter wavelength)
    source_position=(-2.0, 1.0),   # Point source location
    source_width=0.1,              # Gaussian source width
    source_amplitude=10.0,         # Source strength
    use_dielectric=True,           # Enable dielectric circle
    dielectric_center=(0.5, -0.5), # Dielectric center
    dielectric_radius=1.2,         # Dielectric radius
    dielectric_eps=2.0,            # Dielectric permittivity
)

# Train
history = pinn.train(
    n_iterations=5000,
    n_domain=1500,
    n_boundary=500,
    lambda_bc=5.0,
)

# Visualize
plot_results(pinn, history)
plot_wave_propagation(pinn)
```

### Free Space Configuration (No Dielectric)

```python
pinn = HelmholtzPINN(
    omega=1.5,
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

## Torchdiffeq GPU Example

This repository also includes an example demonstrating how to use `torchdiffeq` with GPU acceleration for learning dynamical systems using Neural ODEs.

### Usage

```bash
# Learn a damped harmonic oscillator
python torchdiffeq_gpu_example.py --system oscillator --epochs 500

# Learn the Lorenz attractor (chaotic system)
python torchdiffeq_gpu_example.py --system lorenz --epochs 1000

# Force CPU execution
python torchdiffeq_gpu_example.py --system oscillator --cpu
```

### Features

- **GPU Acceleration**: Automatically uses CUDA if available for faster training
- **Adjoint Method**: Memory-efficient backpropagation through ODE solves
- **Multiple Systems**: Damped harmonic oscillator and Lorenz attractor examples
- **Configurable**: Supports different ODE solvers (`dopri5`, `rk4`, `euler`, etc.)

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--system` | Dynamical system (`oscillator`, `lorenz`) | `oscillator` |
| `--epochs` | Training epochs | 500 |
| `--lr` | Learning rate | 0.01 |
| `--hidden` | Hidden layer dimension | 64 |
| `--layers` | Number of hidden layers | 2 |
| `--solver` | ODE solver | `dopri5` |
| `--no-adjoint` | Disable adjoint method | False |
| `--cpu` | Force CPU execution | False |

### Output Files

- `torchdiffeq_oscillator_results.png`: Results for the damped oscillator
- `torchdiffeq_lorenz_results.png`: Results for the Lorenz system

### Neural ODE Overview

Neural ODEs learn continuous-time dynamics by parameterizing the derivative function with a neural network:

```
dx/dt = f_θ(t, x)
```

where `f_θ` is a neural network. Training uses the adjoint sensitivity method for memory-efficient gradient computation.

## References

1. Sitzmann, V., et al. "Implicit Neural Representations with Periodic Activation Functions." NeurIPS 2020.
2. Raissi, M., et al. "Physics-informed neural networks." Journal of Computational Physics, 2019.
3. Chen, R. T. Q., et al. "Neural Ordinary Differential Equations." NeurIPS 2018.

## License

MIT License
