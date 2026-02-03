"""
Torchdiffeq GPU Example: Neural ODEs for Learning Dynamical Systems

This example demonstrates how to use torchdiffeq with GPU acceleration to:
1. Learn a Neural ODE model from trajectory data
2. Solve ODEs efficiently on GPU
3. Use adjoint method for memory-efficient backpropagation

The example learns the dynamics of a damped harmonic oscillator and the
Lorenz attractor, showcasing both simple and chaotic systems.

Reference: Chen et al., "Neural Ordinary Differential Equations", NeurIPS 2018.

Usage:
    python torchdiffeq_gpu_example.py --system oscillator --epochs 500
    python torchdiffeq_gpu_example.py --system lorenz --epochs 1000
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from typing import Tuple, Optional

# Import torchdiffeq
try:
    from torchdiffeq import odeint, odeint_adjoint
except ImportError:
    raise ImportError(
        "torchdiffeq is required for this example. "
        "Install it with: pip install torchdiffeq"
    )


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ==============================================================================
# True Dynamical Systems (for generating training data)
# ==============================================================================

class DampedHarmonicOscillator(nn.Module):
    """
    Damped harmonic oscillator: 
        dx/dt = v
        dv/dt = -k*x - c*v
    
    State: [x, v] (position, velocity)
    Parameters: k (spring constant), c (damping coefficient)
    """
    def __init__(self, k: float = 2.0, c: float = 0.5):
        super().__init__()
        self.k = k
        self.c = c
    
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Compute derivatives.
        
        Args:
            t: Current time (scalar tensor)
            state: Shape (batch_size, 2) with [x, v]
        
        Returns:
            Derivatives shape (batch_size, 2)
        """
        x = state[..., 0]
        v = state[..., 1]
        
        dx_dt = v
        dv_dt = -self.k * x - self.c * v
        
        return torch.stack([dx_dt, dv_dt], dim=-1)


class LorenzSystem(nn.Module):
    """
    Lorenz attractor - a classic chaotic system:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z
    
    State: [x, y, z]
    Parameters: sigma=10, rho=28, beta=8/3 (classic chaotic regime)
    """
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0):
        super().__init__()
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Compute derivatives.
        
        Args:
            t: Current time (scalar tensor)
            state: Shape (batch_size, 3) with [x, y, z]
        
        Returns:
            Derivatives shape (batch_size, 3)
        """
        x = state[..., 0]
        y = state[..., 1]
        z = state[..., 2]
        
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z
        
        return torch.stack([dx_dt, dy_dt, dz_dt], dim=-1)


# ==============================================================================
# Neural ODE Model
# ==============================================================================

class ODEFunc(nn.Module):
    """
    Neural network that learns the dynamics f(t, x) in dx/dt = f(t, x).
    
    Uses a simple MLP architecture with tanh activations.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64, n_layers: int = 2):
        """
        Args:
            state_dim: Dimension of the state space
            hidden_dim: Number of neurons in hidden layers
            n_layers: Number of hidden layers
        """
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights for better training
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute dx/dt = f(t, x)
        
        Note: We don't use t explicitly here (autonomous system),
        but the signature is required by torchdiffeq.
        
        Args:
            t: Current time (not used, but required by torchdiffeq)
            state: Current state tensor
        
        Returns:
            Time derivative of state
        """
        return self.net(state)


class NeuralODE(nn.Module):
    """
    Neural ODE wrapper that integrates ODEFunc over time.
    """
    def __init__(
        self, 
        func: nn.Module, 
        solver: str = 'dopri5',
        use_adjoint: bool = True
    ):
        """
        Args:
            func: The ODE function (derivative network)
            solver: ODE solver ('dopri5', 'rk4', 'euler', etc.)
            use_adjoint: Use adjoint method for memory-efficient backprop
        """
        super().__init__()
        self.func = func
        self.solver = solver
        self.use_adjoint = use_adjoint
        
        # Choose integration method
        self.integrate = odeint_adjoint if use_adjoint else odeint
    
    def forward(
        self, 
        x0: torch.Tensor, 
        t: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-6
    ) -> torch.Tensor:
        """
        Integrate the ODE from initial condition x0 over time points t.
        
        Args:
            x0: Initial state, shape (batch_size, state_dim)
            t: Time points to evaluate, shape (n_times,)
            rtol: Relative tolerance for solver
            atol: Absolute tolerance for solver
        
        Returns:
            Trajectory shape (n_times, batch_size, state_dim)
        """
        return self.integrate(
            self.func, x0, t,
            method=self.solver,
            rtol=rtol,
            atol=atol
        )


# ==============================================================================
# Data Generation
# ==============================================================================

def generate_oscillator_data(
    device: torch.device,
    n_trajectories: int = 50,
    n_points: int = 100,
    t_max: float = 10.0,
    noise_std: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate training data from damped harmonic oscillator.
    
    Args:
        device: Torch device (cuda/cpu)
        n_trajectories: Number of trajectories to generate
        n_points: Number of time points per trajectory
        t_max: Maximum time
        noise_std: Standard deviation of observation noise
    
    Returns:
        t: Time points (n_points,)
        x0: Initial conditions (n_trajectories, 2)
        trajectories: Noisy trajectories (n_points, n_trajectories, 2)
    """
    true_system = DampedHarmonicOscillator(k=2.0, c=0.5).to(device)
    
    t = torch.linspace(0, t_max, n_points, device=device)
    
    # Random initial conditions
    x0 = torch.randn(n_trajectories, 2, device=device) * 2.0
    
    # Generate trajectories
    with torch.no_grad():
        trajectories = odeint(true_system, x0, t, method='dopri5')
        # Add noise
        trajectories = trajectories + noise_std * torch.randn_like(trajectories)
    
    return t, x0, trajectories


def generate_lorenz_data(
    device: torch.device,
    n_trajectories: int = 20,
    n_points: int = 200,
    t_max: float = 4.0,
    noise_std: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate training data from Lorenz system.
    
    Args:
        device: Torch device (cuda/cpu)
        n_trajectories: Number of trajectories to generate
        n_points: Number of time points per trajectory
        t_max: Maximum time (shorter for Lorenz due to chaos)
        noise_std: Standard deviation of observation noise
    
    Returns:
        t: Time points (n_points,)
        x0: Initial conditions (n_trajectories, 3)
        trajectories: Noisy trajectories (n_points, n_trajectories, 3)
    """
    true_system = LorenzSystem().to(device)
    
    t = torch.linspace(0, t_max, n_points, device=device)
    
    # Random initial conditions near the attractor
    x0 = torch.randn(n_trajectories, 3, device=device)
    x0[:, 0] = x0[:, 0] * 5.0 + 1.0   # x around 1
    x0[:, 1] = x0[:, 1] * 5.0 + 1.0   # y around 1  
    x0[:, 2] = x0[:, 2] * 5.0 + 25.0  # z around 25
    
    # Generate trajectories
    with torch.no_grad():
        trajectories = odeint(true_system, x0, t, method='dopri5', rtol=1e-6, atol=1e-8)
        # Add noise
        trajectories = trajectories + noise_std * torch.randn_like(trajectories)
    
    return t, x0, trajectories


# ==============================================================================
# Training
# ==============================================================================

def train_neural_ode(
    model: NeuralODE,
    t: torch.Tensor,
    x0: torch.Tensor,
    target: torch.Tensor,
    n_epochs: int = 500,
    lr: float = 1e-2,
    subsample_time: int = 1,
    log_every: int = 50
) -> dict:
    """
    Train the Neural ODE model.
    
    Args:
        model: NeuralODE model to train
        t: Time points (n_times,)
        x0: Initial conditions (n_trajectories, state_dim)
        target: Target trajectories (n_times, n_trajectories, state_dim)
        n_epochs: Number of training epochs
        lr: Learning rate
        subsample_time: Subsample time points for faster training
        log_every: Logging frequency
    
    Returns:
        Training history dictionary
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    # Subsample time points for training
    t_train = t[::subsample_time]
    target_train = target[::subsample_time]
    
    history = {'loss': [], 'epoch': []}
    
    pbar = tqdm(range(n_epochs), desc="Training Neural ODE")
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(x0, t_train)
        
        # MSE loss
        loss = torch.mean((pred - target_train) ** 2)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        history['loss'].append(loss.item())
        history['epoch'].append(epoch)
        
        if (epoch + 1) % log_every == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4e}'})
    
    return history


# ==============================================================================
# Visualization
# ==============================================================================

def plot_oscillator_results(
    model: NeuralODE,
    t: torch.Tensor,
    x0: torch.Tensor,
    target: torch.Tensor,
    history: dict,
    save_path: str = "torchdiffeq_oscillator_results.png"
):
    """
    Plot results for the damped harmonic oscillator.
    """
    device = x0.device
    
    # Generate predictions
    with torch.no_grad():
        pred = model(x0, t)
    
    # Convert to numpy
    t_np = t.cpu().numpy()
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Training loss
    ax = axes[0, 0]
    ax.semilogy(history['loss'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # Position vs time (sample trajectories)
    ax = axes[0, 1]
    n_show = min(5, target_np.shape[1])
    for i in range(n_show):
        ax.plot(t_np, target_np[:, i, 0], 'b-', alpha=0.5, label='Target' if i == 0 else '')
        ax.plot(t_np, pred_np[:, i, 0], 'r--', alpha=0.5, label='Predicted' if i == 0 else '')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position x')
    ax.set_title('Position vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Velocity vs time
    ax = axes[0, 2]
    for i in range(n_show):
        ax.plot(t_np, target_np[:, i, 1], 'b-', alpha=0.5)
        ax.plot(t_np, pred_np[:, i, 1], 'r--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Velocity v')
    ax.set_title('Velocity vs Time')
    ax.grid(True, alpha=0.3)
    
    # Phase space
    ax = axes[1, 0]
    for i in range(n_show):
        ax.plot(target_np[:, i, 0], target_np[:, i, 1], 'b-', alpha=0.5)
        ax.plot(pred_np[:, i, 0], pred_np[:, i, 1], 'r--', alpha=0.5)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Velocity v')
    ax.set_title('Phase Space')
    ax.grid(True, alpha=0.3)
    
    # Error over time
    ax = axes[1, 1]
    error = np.mean((pred_np - target_np) ** 2, axis=(1, 2))
    ax.plot(t_np, error, 'g-')
    ax.set_xlabel('Time')
    ax.set_ylabel('MSE')
    ax.set_title('Error vs Time')
    ax.grid(True, alpha=0.3)
    
    # Learned vector field
    ax = axes[1, 2]
    x_range = np.linspace(-4, 4, 15)
    v_range = np.linspace(-4, 4, 15)
    X, V = np.meshgrid(x_range, v_range)
    points = torch.tensor(np.stack([X.flatten(), V.flatten()], axis=1), 
                         dtype=torch.float32, device=device)
    with torch.no_grad():
        derivatives = model.func(torch.tensor(0.0), points).cpu().numpy()
    DX = derivatives[:, 0].reshape(X.shape)
    DV = derivatives[:, 1].reshape(V.shape)
    ax.quiver(X, V, DX, DV, alpha=0.7)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Velocity v')
    ax.set_title('Learned Vector Field')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_lorenz_results(
    model: NeuralODE,
    t: torch.Tensor,
    x0: torch.Tensor,
    target: torch.Tensor,
    history: dict,
    save_path: str = "torchdiffeq_lorenz_results.png"
):
    """
    Plot results for the Lorenz system.
    """
    # Generate predictions
    with torch.no_grad():
        pred = model(x0, t)
    
    # Convert to numpy
    t_np = t.cpu().numpy()
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    fig = plt.figure(figsize=(16, 12))
    
    # Training loss
    ax = fig.add_subplot(2, 3, 1)
    ax.semilogy(history['loss'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # x vs time
    ax = fig.add_subplot(2, 3, 2)
    n_show = min(3, target_np.shape[1])
    for i in range(n_show):
        ax.plot(t_np, target_np[:, i, 0], 'b-', alpha=0.6, label='Target' if i == 0 else '')
        ax.plot(t_np, pred_np[:, i, 0], 'r--', alpha=0.6, label='Predicted' if i == 0 else '')
    ax.set_xlabel('Time')
    ax.set_ylabel('x')
    ax.set_title('x vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # y vs time
    ax = fig.add_subplot(2, 3, 3)
    for i in range(n_show):
        ax.plot(t_np, target_np[:, i, 1], 'b-', alpha=0.6)
        ax.plot(t_np, pred_np[:, i, 1], 'r--', alpha=0.6)
    ax.set_xlabel('Time')
    ax.set_ylabel('y')
    ax.set_title('y vs Time')
    ax.grid(True, alpha=0.3)
    
    # z vs time
    ax = fig.add_subplot(2, 3, 4)
    for i in range(n_show):
        ax.plot(t_np, target_np[:, i, 2], 'b-', alpha=0.6)
        ax.plot(t_np, pred_np[:, i, 2], 'r--', alpha=0.6)
    ax.set_xlabel('Time')
    ax.set_ylabel('z')
    ax.set_title('z vs Time')
    ax.grid(True, alpha=0.3)
    
    # 3D phase space (target)
    ax = fig.add_subplot(2, 3, 5, projection='3d')
    for i in range(n_show):
        ax.plot(target_np[:, i, 0], target_np[:, i, 1], target_np[:, i, 2], 
               'b-', alpha=0.6, linewidth=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Target Trajectories (3D)')
    
    # 3D phase space (predicted)
    ax = fig.add_subplot(2, 3, 6, projection='3d')
    for i in range(n_show):
        ax.plot(pred_np[:, i, 0], pred_np[:, i, 1], pred_np[:, i, 2], 
               'r-', alpha=0.6, linewidth=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Predicted Trajectories (3D)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Torchdiffeq GPU Example')
    parser.add_argument('--system', type=str, default='oscillator',
                       choices=['oscillator', 'lorenz'],
                       help='Dynamical system to learn (default: oscillator)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-2,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--hidden', type=int, default=64,
                       help='Hidden layer dimension (default: 64)')
    parser.add_argument('--layers', type=int, default=2,
                       help='Number of hidden layers (default: 2)')
    parser.add_argument('--solver', type=str, default='dopri5',
                       help='ODE solver (default: dopri5)')
    parser.add_argument('--no-adjoint', action='store_true',
                       help='Disable adjoint method')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU even if GPU is available')
    args = parser.parse_args()
    
    # Device selection
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Torchdiffeq GPU Example: Neural ODEs")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"System: {args.system}")
    print(f"Solver: {args.solver}")
    print(f"Adjoint: {not args.no_adjoint}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)
    
    if args.system == 'oscillator':
        # ====================================================================
        # Damped Harmonic Oscillator
        # ====================================================================
        print("\nGenerating oscillator training data...")
        t, x0, target = generate_oscillator_data(
            device=device,
            n_trajectories=50,
            n_points=100,
            t_max=10.0,
            noise_std=0.01
        )
        state_dim = 2
        
        print(f"Data shapes: t={t.shape}, x0={x0.shape}, target={target.shape}")
        
        # Create model
        func = ODEFunc(state_dim=state_dim, hidden_dim=args.hidden, n_layers=args.layers).to(device)
        model = NeuralODE(func, solver=args.solver, use_adjoint=not args.no_adjoint)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        
        # Train
        print("\nTraining...")
        history = train_neural_ode(
            model=model,
            t=t,
            x0=x0,
            target=target,
            n_epochs=args.epochs,
            lr=args.lr,
            subsample_time=2
        )
        
        # Plot results
        print("\nGenerating plots...")
        plot_oscillator_results(model, t, x0, target, history)
        
    elif args.system == 'lorenz':
        # ====================================================================
        # Lorenz System
        # ====================================================================
        print("\nGenerating Lorenz training data...")
        t, x0, target = generate_lorenz_data(
            device=device,
            n_trajectories=20,
            n_points=200,
            t_max=4.0,
            noise_std=0.5
        )
        state_dim = 3
        
        print(f"Data shapes: t={t.shape}, x0={x0.shape}, target={target.shape}")
        
        # Create model (larger network for chaotic system)
        func = ODEFunc(state_dim=state_dim, hidden_dim=args.hidden * 2, n_layers=args.layers + 1).to(device)
        model = NeuralODE(func, solver=args.solver, use_adjoint=not args.no_adjoint)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        
        # Train
        print("\nTraining...")
        history = train_neural_ode(
            model=model,
            t=t,
            x0=x0,
            target=target,
            n_epochs=args.epochs,
            lr=args.lr * 0.5,  # Lower learning rate for chaotic system
            subsample_time=2
        )
        
        # Plot results
        print("\nGenerating plots...")
        plot_lorenz_results(model, t, x0, target, history)
    
    print(f"\nFinal loss: {history['loss'][-1]:.4e}")
    print("\nDone!")
    
    # GPU memory usage (if applicable)
    if device.type == 'cuda':
        print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
