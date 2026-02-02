"""
Physics-Informed Neural Network (PINN) with SIREN architecture
for solving the 2D Helmholtz equation in electromagnetics.

The Helmholtz equation in frequency domain:
    (-∇² - ε*ω²) Ez = -i*ω*Jz

Where:
    - Ez: complex electric field amplitude (z-component)
    - ε: permittivity (1 in free space, can be 2 in dielectric region)
    - ω: angular frequency
    - Jz: current source density
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, Optional


# ==============================================================================
# SIREN Architecture
# ==============================================================================

class SineLayer(nn.Module):
    """
    Sinusoidal activation layer used in SIREN architecture.
    
    SIREN (Sinusoidal Representation Networks) uses sin activation functions
    which are ideal for representing signals and their derivatives.
    
    Reference: Sitzmann et al., "Implicit Neural Representations with 
    Periodic Activation Functions", NeurIPS 2020.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        is_first: bool = False, 
        omega_0: float = 30.0
    ):
        """
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to use bias
            is_first: Whether this is the first layer (affects initialization)
            omega_0: Frequency scaling factor
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights according to SIREN paper."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/in_features, 1/in_features]
                self.linear.weight.uniform_(-1 / self.in_features, 
                                            1 / self.in_features)
            else:
                # Hidden layers: uniform in [-sqrt(6/in_features)/omega_0, sqrt(6/in_features)/omega_0]
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    SIREN network for representing the complex electric field.
    
    Takes 2D coordinates (x, y) as input and outputs the real and 
    imaginary parts of Ez.
    """
    
    def __init__(
        self, 
        in_features: int = 2, 
        out_features: int = 2,  # Real and imaginary parts
        hidden_features: int = 256, 
        hidden_layers: int = 5,
        omega_0: float = 30.0,
        omega_hidden: float = 30.0
    ):
        """
        Args:
            in_features: Input dimension (2 for x, y coordinates)
            out_features: Output dimension (2 for real and imaginary parts)
            hidden_features: Number of neurons in hidden layers
            hidden_layers: Number of hidden layers
            omega_0: Frequency for first layer
            omega_hidden: Frequency for hidden layers
        """
        super().__init__()
        
        self.net = []
        
        # First layer
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=omega_0))
        
        # Hidden layers
        for _ in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=omega_hidden))
        
        # Final layer (linear, no activation)
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_features) / omega_hidden
            final_linear.weight.uniform_(-bound, bound)
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            coords: Tensor of shape (N, 2) with (x, y) coordinates
            
        Returns:
            Tensor of shape (N, 2) with (Ez_real, Ez_imag)
        """
        return self.net(coords)


# ==============================================================================
# Physics-Informed Loss Functions
# ==============================================================================

class HelmholtzPINN:
    """
    Physics-Informed Neural Network for solving the 2D Helmholtz equation.
    
    The equation: (-∇² - ε*ω²) Ez = -i*ω*Jz
    
    For open boundaries, we use the Sommerfeld radiation condition:
        ∂Ez/∂r + i*k*Ez → 0 as r → ∞
    where k = ω*√ε is the wavenumber.
    """
    
    def __init__(
        self,
        domain_size: Tuple[float, float] = (-5.0, 5.0),
        omega: float = 1.5,
        source_position: Tuple[float, float] = (0.0, 0.0),
        source_width: float = 0.1,
        source_amplitude: float = 10.0,
        use_dielectric: bool = True,
        dielectric_center: Tuple[float, float] = (0.0, 0.0),
        dielectric_radius: float = 1.5,
        dielectric_eps: float = 2.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            domain_size: (min, max) for both x and y
            omega: Angular frequency (higher = shorter wavelength, more ripples)
            source_position: (x, y) position of point source
            source_width: Width of Gaussian source (smaller = more point-like)
            source_amplitude: Amplitude of source
            use_dielectric: Whether to include dielectric circle
            dielectric_center: Center of dielectric circle
            dielectric_radius: Radius of dielectric circle
            dielectric_eps: Permittivity of dielectric (1.0 = free space)
            device: Computation device
        """
        self.domain_size = domain_size
        self.omega = omega
        self.source_position = torch.tensor(source_position, device=device)
        self.source_width = source_width
        self.source_amplitude = source_amplitude
        self.use_dielectric = use_dielectric
        self.dielectric_center = torch.tensor(dielectric_center, device=device)
        self.dielectric_radius = dielectric_radius
        self.dielectric_eps = dielectric_eps
        self.device = device
        
        # Wavenumber for reference
        self.k0 = omega  # k = omega * sqrt(eps), in free space sqrt(eps)=1
        self.wavelength = 2 * np.pi / self.k0
        
        # Create the SIREN network with appropriate frequency for wave patterns
        siren_omega = max(30.0, 5 * omega)
        self.model = SIREN(
            in_features=2,
            out_features=2,
            hidden_features=128,  # Smaller for faster training
            hidden_layers=4,
            omega_0=siren_omega,
            omega_hidden=siren_omega
        ).to(device)
        
        # Optimizer with higher learning rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1500, gamma=0.5
        )
    
    def get_permittivity(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Get permittivity at given coordinates.
        
        Args:
            coords: Tensor of shape (N, 2)
            
        Returns:
            Tensor of shape (N,) with permittivity values
        """
        eps = torch.ones(coords.shape[0], device=self.device)
        
        if self.use_dielectric:
            # Distance from dielectric center
            dist = torch.sqrt(
                (coords[:, 0] - self.dielectric_center[0])**2 + 
                (coords[:, 1] - self.dielectric_center[1])**2
            )
            # Smooth transition using tanh
            transition_width = 0.1
            mask = 0.5 * (1 - torch.tanh((dist - self.dielectric_radius) / transition_width))
            eps = 1.0 + (self.dielectric_eps - 1.0) * mask
        
        return eps
    
    def get_source(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Get source term Jz at given coordinates (Gaussian point source).
        
        Args:
            coords: Tensor of shape (N, 2)
            
        Returns:
            Tensor of shape (N,) with source values
        """
        dist_sq = (
            (coords[:, 0] - self.source_position[0])**2 + 
            (coords[:, 1] - self.source_position[1])**2
        )
        # Gaussian source - tighter for more point-like behavior
        source = torch.exp(-dist_sq / (2 * self.source_width**2))
        # Apply amplitude
        source = self.source_amplitude * source / (2 * np.pi * self.source_width**2)
        return source
    
    def compute_derivatives(
        self, 
        coords: torch.Tensor,
        use_finite_diff: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Ez and its spatial derivatives.
        
        Args:
            coords: Tensor of shape (N, 2)
            use_finite_diff: If True, use finite differences for Laplacian (faster on CPU)
            
        Returns:
            Ez_real, Ez_imag, laplacian_real, laplacian_imag, grad_Ez
        """
        if use_finite_diff:
            return self._compute_derivatives_fd(coords)
        else:
            return self._compute_derivatives_autodiff(coords)
    
    def _compute_derivatives_fd(
        self,
        coords: torch.Tensor,
        h: float = 0.02
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute derivatives using finite differences (faster on CPU).
        Gradients flow through all network evaluations for training.
        """
        # Evaluate at center and offset points (keep gradients for backprop)
        Ez_center = self.model(coords)
        
        # Offset points for finite differences
        coords_px = coords.clone()
        coords_px[:, 0] = coords_px[:, 0] + h
        coords_mx = coords.clone()
        coords_mx[:, 0] = coords_mx[:, 0] - h
        coords_py = coords.clone()
        coords_py[:, 1] = coords_py[:, 1] + h
        coords_my = coords.clone()
        coords_my[:, 1] = coords_my[:, 1] - h
        
        Ez_px = self.model(coords_px)
        Ez_mx = self.model(coords_mx)
        Ez_py = self.model(coords_py)
        Ez_my = self.model(coords_my)
        
        Ez_real = Ez_center[:, 0]
        Ez_imag = Ez_center[:, 1]
        
        # First derivatives (central difference)
        dEz_real_dx = (Ez_px[:, 0] - Ez_mx[:, 0]) / (2 * h)
        dEz_real_dy = (Ez_py[:, 0] - Ez_my[:, 0]) / (2 * h)
        dEz_imag_dx = (Ez_px[:, 1] - Ez_mx[:, 1]) / (2 * h)
        dEz_imag_dy = (Ez_py[:, 1] - Ez_my[:, 1]) / (2 * h)
        
        # Second derivatives (central difference for Laplacian)
        d2Ez_real_dx2 = (Ez_px[:, 0] - 2*Ez_center[:, 0] + Ez_mx[:, 0]) / (h * h)
        d2Ez_real_dy2 = (Ez_py[:, 0] - 2*Ez_center[:, 0] + Ez_my[:, 0]) / (h * h)
        d2Ez_imag_dx2 = (Ez_px[:, 1] - 2*Ez_center[:, 1] + Ez_mx[:, 1]) / (h * h)
        d2Ez_imag_dy2 = (Ez_py[:, 1] - 2*Ez_center[:, 1] + Ez_my[:, 1]) / (h * h)
        
        laplacian_real = d2Ez_real_dx2 + d2Ez_real_dy2
        laplacian_imag = d2Ez_imag_dx2 + d2Ez_imag_dy2
        
        grad_real = torch.stack([dEz_real_dx, dEz_real_dy], dim=1)
        grad_imag = torch.stack([dEz_imag_dx, dEz_imag_dy], dim=1)
        
        return Ez_real, Ez_imag, laplacian_real, laplacian_imag, (grad_real, grad_imag)
    
    def _compute_derivatives_autodiff(
        self, 
        coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute derivatives using automatic differentiation (more accurate, slower on CPU).
        """
        coords.requires_grad_(True)
        
        # Forward pass
        output = self.model(coords)
        Ez_real = output[:, 0]
        Ez_imag = output[:, 1]
        
        # First derivatives
        grad_real = torch.autograd.grad(
            Ez_real, coords, 
            grad_outputs=torch.ones_like(Ez_real),
            create_graph=True
        )[0]
        
        grad_imag = torch.autograd.grad(
            Ez_imag, coords,
            grad_outputs=torch.ones_like(Ez_imag),
            create_graph=True
        )[0]
        
        dEz_real_dx = grad_real[:, 0]
        dEz_real_dy = grad_real[:, 1]
        dEz_imag_dx = grad_imag[:, 0]
        dEz_imag_dy = grad_imag[:, 1]
        
        # Second derivatives (Laplacian)
        d2Ez_real_dx2 = torch.autograd.grad(
            dEz_real_dx, coords,
            grad_outputs=torch.ones_like(dEz_real_dx),
            create_graph=True
        )[0][:, 0]
        
        d2Ez_real_dy2 = torch.autograd.grad(
            dEz_real_dy, coords,
            grad_outputs=torch.ones_like(dEz_real_dy),
            create_graph=True
        )[0][:, 1]
        
        d2Ez_imag_dx2 = torch.autograd.grad(
            dEz_imag_dx, coords,
            grad_outputs=torch.ones_like(dEz_imag_dx),
            create_graph=True
        )[0][:, 0]
        
        d2Ez_imag_dy2 = torch.autograd.grad(
            dEz_imag_dy, coords,
            grad_outputs=torch.ones_like(dEz_imag_dy),
            create_graph=True
        )[0][:, 1]
        
        laplacian_real = d2Ez_real_dx2 + d2Ez_real_dy2
        laplacian_imag = d2Ez_imag_dx2 + d2Ez_imag_dy2
        
        return Ez_real, Ez_imag, laplacian_real, laplacian_imag, (grad_real, grad_imag)
    
    def physics_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute the physics loss based on the Helmholtz equation.
        
        Helmholtz equation: (-∇² - ε*ω²) Ez = -i*ω*Jz
        
        Separating real and imaginary parts:
            Real: -∇²Ez_real - ε*ω²*Ez_real = ω*Jz  (source term is imaginary: -i*ω*Jz)
            Imag: -∇²Ez_imag - ε*ω²*Ez_imag = 0
            
        Wait, let's be more careful. The source -i*ω*Jz where Jz is real:
            Real part of RHS: 0
            Imag part of RHS: -ω*Jz
            
        So:
            Real: -∇²Ez_real - ε*ω²*Ez_real = 0
            Imag: -∇²Ez_imag - ε*ω²*Ez_imag = -ω*Jz
        """
        Ez_real, Ez_imag, lap_real, lap_imag, _ = self.compute_derivatives(coords)
        
        eps = self.get_permittivity(coords)
        source = self.get_source(coords)
        k_sq = eps * self.omega**2
        
        # Helmholtz equation residuals
        # (-∇² - k²) Ez = -i*ω*Jz
        # Real part: -∇²Ez_real - k²*Ez_real = 0
        # Imag part: -∇²Ez_imag - k²*Ez_imag = -ω*Jz
        
        residual_real = -lap_real - k_sq * Ez_real
        residual_imag = -lap_imag - k_sq * Ez_imag + self.omega * source
        
        loss = torch.mean(residual_real**2) + torch.mean(residual_imag**2)
        return loss
    
    def boundary_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute the boundary loss using Sommerfeld radiation condition.
        
        The Sommerfeld condition: ∂Ez/∂r + i*k*Ez → 0
        
        In Cartesian coordinates at the boundary:
            (x/r)*∂Ez/∂x + (y/r)*∂Ez/∂y + i*k*Ez = 0
            
        Where r = sqrt(x² + y²) and k = ω*√ε
        """
        Ez_real, Ez_imag, _, _, (grad_real, grad_imag) = self.compute_derivatives(coords)
        
        # Distance from origin (assuming source is near origin)
        r = torch.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
        r = torch.clamp(r, min=1e-6)  # Avoid division by zero
        
        # Unit radial vector
        r_hat_x = coords[:, 0] / r
        r_hat_y = coords[:, 1] / r
        
        # Radial derivative: ∂Ez/∂r = r_hat · ∇Ez
        dEz_real_dr = r_hat_x * grad_real[:, 0] + r_hat_y * grad_real[:, 1]
        dEz_imag_dr = r_hat_x * grad_imag[:, 0] + r_hat_y * grad_imag[:, 1]
        
        # Wavenumber (use free space value at boundary)
        k = self.omega  # sqrt(1) = 1 for free space
        
        # Sommerfeld condition: ∂Ez/∂r + i*k*Ez = 0
        # Real: ∂Ez_real/∂r - k*Ez_imag = 0
        # Imag: ∂Ez_imag/∂r + k*Ez_real = 0
        
        sommerfeld_real = dEz_real_dr - k * Ez_imag
        sommerfeld_imag = dEz_imag_dr + k * Ez_real
        
        loss = torch.mean(sommerfeld_real**2) + torch.mean(sommerfeld_imag**2)
        return loss
    
    def sample_domain(self, n_points: int) -> torch.Tensor:
        """Sample random points in the domain."""
        # Simple uniform sampling for speed
        coords = torch.rand(n_points, 2, device=self.device)
        coords = coords * (self.domain_size[1] - self.domain_size[0]) + self.domain_size[0]
        return coords
    
    def sample_boundary(self, n_points: int) -> torch.Tensor:
        """Sample points on the boundary."""
        n_per_side = n_points // 4
        
        boundary_points = []
        
        # Bottom boundary (y = min)
        x = torch.linspace(self.domain_size[0], self.domain_size[1], n_per_side, device=self.device)
        y = torch.full_like(x, self.domain_size[0])
        boundary_points.append(torch.stack([x, y], dim=1))
        
        # Top boundary (y = max)
        y = torch.full_like(x, self.domain_size[1])
        boundary_points.append(torch.stack([x, y], dim=1))
        
        # Left boundary (x = min)
        y = torch.linspace(self.domain_size[0], self.domain_size[1], n_per_side, device=self.device)
        x = torch.full_like(y, self.domain_size[0])
        boundary_points.append(torch.stack([x, y], dim=1))
        
        # Right boundary (x = max)
        x = torch.full_like(y, self.domain_size[1])
        boundary_points.append(torch.stack([x, y], dim=1))
        
        return torch.cat(boundary_points, dim=0)
    
    def train_step(
        self, 
        n_domain: int = 4000, 
        n_boundary: int = 1000,
        lambda_bc: float = 10.0
    ) -> Tuple[float, float, float]:
        """
        Perform one training step.
        
        Args:
            n_domain: Number of domain points
            n_boundary: Number of boundary points
            lambda_bc: Weight for boundary condition loss
            
        Returns:
            total_loss, physics_loss, boundary_loss
        """
        self.optimizer.zero_grad()
        
        # Sample points
        domain_coords = self.sample_domain(n_domain)
        boundary_coords = self.sample_boundary(n_boundary)
        
        # Compute losses
        loss_physics = self.physics_loss(domain_coords)
        loss_bc = self.boundary_loss(boundary_coords)
        
        # Total loss
        total_loss = loss_physics + lambda_bc * loss_bc
        
        # Backward and optimize
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return total_loss.item(), loss_physics.item(), loss_bc.item()
    
    def train(
        self, 
        n_iterations: int = 10000,
        n_domain: int = 4000,
        n_boundary: int = 1000,
        lambda_bc: float = 10.0,
        log_every: int = 100
    ):
        """
        Train the PINN.
        
        Args:
            n_iterations: Number of training iterations
            n_domain: Number of domain points per iteration
            n_boundary: Number of boundary points per iteration
            lambda_bc: Weight for boundary condition loss
            log_every: Logging frequency
        """
        print(f"Training PINN on {self.device}")
        print(f"Domain: [{self.domain_size[0]}, {self.domain_size[1]}]²")
        print(f"Omega: {self.omega}, k0: {self.k0:.2f}")
        print(f"Wavelength: {self.wavelength:.2f}")
        print(f"Source at: ({self.source_position[0].item():.1f}, {self.source_position[1].item():.1f})")
        print(f"Dielectric: {self.use_dielectric} (eps={self.dielectric_eps}, r={self.dielectric_radius})")
        print("-" * 60)
        
        history = {"total": [], "physics": [], "boundary": []}
        
        pbar = tqdm(range(n_iterations), desc="Training")
        for i in pbar:
            total_loss, physics_loss, bc_loss = self.train_step(
                n_domain, n_boundary, lambda_bc
            )
            
            history["total"].append(total_loss)
            history["physics"].append(physics_loss)
            history["boundary"].append(bc_loss)
            
            if (i + 1) % log_every == 0:
                pbar.set_postfix({
                    "total": f"{total_loss:.2e}",
                    "physics": f"{physics_loss:.2e}",
                    "bc": f"{bc_loss:.2e}"
                })
        
        return history
    
    @torch.no_grad()
    def predict(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict Ez at given coordinates.
        
        Args:
            coords: Tensor of shape (N, 2)
            
        Returns:
            Ez_real, Ez_imag tensors of shape (N,)
        """
        self.model.eval()
        output = self.model(coords)
        return output[:, 0], output[:, 1]
    
    @torch.no_grad()
    def predict_grid(
        self, 
        resolution: int = 200
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict Ez on a regular grid.
        
        Args:
            resolution: Number of grid points per dimension
            
        Returns:
            x, y, Ez_real, Ez_imag as numpy arrays
        """
        self.model.eval()
        
        x = np.linspace(self.domain_size[0], self.domain_size[1], resolution)
        y = np.linspace(self.domain_size[0], self.domain_size[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        coords = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=1),
            dtype=torch.float32,
            device=self.device
        )
        
        Ez_real, Ez_imag = self.predict(coords)
        
        Ez_real = Ez_real.cpu().numpy().reshape(resolution, resolution)
        Ez_imag = Ez_imag.cpu().numpy().reshape(resolution, resolution)
        
        return X, Y, Ez_real, Ez_imag


# ==============================================================================
# Visualization
# ==============================================================================

def plot_results(
    pinn: HelmholtzPINN,
    history: dict,
    resolution: int = 200,
    save_path: Optional[str] = None
):
    """
    Plot the training history and field distribution.
    
    Args:
        pinn: Trained PINN model
        history: Training history dictionary
        resolution: Grid resolution for visualization
        save_path: Optional path to save the figure
    """
    X, Y, Ez_real, Ez_imag = pinn.predict_grid(resolution)
    
    # Compute field magnitude and phase
    Ez_mag = np.sqrt(Ez_real**2 + Ez_imag**2)
    Ez_phase = np.arctan2(Ez_imag, Ez_real)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Training loss
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.semilogy(history["total"], label="Total", alpha=0.7)
    ax1.semilogy(history["physics"], label="Physics", alpha=0.7)
    ax1.semilogy(history["boundary"], label="Boundary", alpha=0.7)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training History")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Real part of Ez
    ax2 = fig.add_subplot(2, 3, 2)
    vmax = np.max(np.abs(Ez_real))
    im2 = ax2.pcolormesh(X, Y, Ez_real, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading='auto')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Re(Ez)")
    ax2.set_aspect("equal")
    plt.colorbar(im2, ax=ax2)
    
    # Add dielectric circle
    if pinn.use_dielectric:
        circle = plt.Circle(
            (pinn.dielectric_center[0].cpu(), pinn.dielectric_center[1].cpu()),
            pinn.dielectric_radius,
            fill=False, color="black", linestyle="--", linewidth=2
        )
        ax2.add_patch(circle)
    
    # Mark source position
    ax2.plot(pinn.source_position[0].cpu(), pinn.source_position[1].cpu(), 
             "k*", markersize=15, label="Source")
    
    # Imaginary part of Ez
    ax3 = fig.add_subplot(2, 3, 3)
    vmax = np.max(np.abs(Ez_imag))
    im3 = ax3.pcolormesh(X, Y, Ez_imag, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading='auto')
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_title("Im(Ez)")
    ax3.set_aspect("equal")
    plt.colorbar(im3, ax=ax3)
    
    if pinn.use_dielectric:
        circle = plt.Circle(
            (pinn.dielectric_center[0].cpu(), pinn.dielectric_center[1].cpu()),
            pinn.dielectric_radius,
            fill=False, color="black", linestyle="--", linewidth=2
        )
        ax3.add_patch(circle)
    
    ax3.plot(pinn.source_position[0].cpu(), pinn.source_position[1].cpu(), 
             "k*", markersize=15)
    
    # Field magnitude
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.pcolormesh(X, Y, Ez_mag, cmap="hot", shading='auto')
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_title("|Ez| (Magnitude)")
    ax4.set_aspect("equal")
    plt.colorbar(im4, ax=ax4)
    
    if pinn.use_dielectric:
        circle = plt.Circle(
            (pinn.dielectric_center[0].cpu(), pinn.dielectric_center[1].cpu()),
            pinn.dielectric_radius,
            fill=False, color="white", linestyle="--", linewidth=2
        )
        ax4.add_patch(circle)
    
    ax4.plot(pinn.source_position[0].cpu(), pinn.source_position[1].cpu(), 
             "w*", markersize=15)
    
    # Field phase
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.pcolormesh(X, Y, Ez_phase, cmap="hsv", vmin=-np.pi, vmax=np.pi, shading='auto')
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")
    ax5.set_title("Phase(Ez)")
    ax5.set_aspect("equal")
    plt.colorbar(im5, ax=ax5, label="radians")
    
    if pinn.use_dielectric:
        circle = plt.Circle(
            (pinn.dielectric_center[0].cpu(), pinn.dielectric_center[1].cpu()),
            pinn.dielectric_radius,
            fill=False, color="black", linestyle="--", linewidth=2
        )
        ax5.add_patch(circle)
    
    ax5.plot(pinn.source_position[0].cpu(), pinn.source_position[1].cpu(), 
             "k*", markersize=15)
    
    # Permittivity distribution
    ax6 = fig.add_subplot(2, 3, 6)
    coords = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1),
        dtype=torch.float32,
        device=pinn.device
    )
    eps = pinn.get_permittivity(coords).cpu().numpy().reshape(resolution, resolution)
    im6 = ax6.pcolormesh(X, Y, eps, cmap="viridis", shading='auto')
    ax6.set_xlabel("x")
    ax6.set_ylabel("y")
    ax6.set_title("Permittivity ε(x,y)")
    ax6.set_aspect("equal")
    plt.colorbar(im6, ax=ax6)
    
    ax6.plot(pinn.source_position[0].cpu(), pinn.source_position[1].cpu(), 
             "r*", markersize=15, label="Source")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_wave_propagation(
    pinn: HelmholtzPINN,
    n_frames: int = 8,
    resolution: int = 200,
    save_path: Optional[str] = None
):
    """
    Plot the wave at different time phases.
    
    The time-domain field is: Ez(x,y,t) = Re[Ez(x,y) * exp(-i*ω*t)]
    
    Args:
        pinn: Trained PINN model
        n_frames: Number of time frames
        resolution: Grid resolution
        save_path: Optional path to save the figure
    """
    X, Y, Ez_real, Ez_imag = pinn.predict_grid(resolution)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    phases = np.linspace(0, 2*np.pi, n_frames, endpoint=False)
    
    # Find global vmax for consistent colorbar
    Ez_complex = Ez_real + 1j * Ez_imag
    vmax = np.max(np.abs(Ez_complex))
    
    for i, (ax, phase) in enumerate(zip(axes, phases)):
        # Time-domain field: Re[Ez * exp(-i*ω*t)] = Ez_real*cos(ωt) + Ez_imag*sin(ωt)
        Ez_t = Ez_real * np.cos(phase) + Ez_imag * np.sin(phase)
        
        im = ax.pcolormesh(X, Y, Ez_t, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading='auto')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"ωt = {phase:.2f} rad")
        ax.set_aspect("equal")
        
        if pinn.use_dielectric:
            circle = plt.Circle(
                (pinn.dielectric_center[0].cpu(), pinn.dielectric_center[1].cpu()),
                pinn.dielectric_radius,
                fill=False, color="black", linestyle="--", linewidth=1.5
            )
            ax.add_patch(circle)
        
        ax.plot(pinn.source_position[0].cpu(), pinn.source_position[1].cpu(), 
                "k*", markersize=10)
    
    plt.tight_layout()
    
    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Ez(t)")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    plt.show()


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Main function to train and visualize the PINN solution."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration for clear wave ripples
    # Higher omega = shorter wavelength = more visible ripples
    # Domain should contain several wavelengths
    config = {
        "domain_size": (-5.0, 5.0),  # Domain: [-5, 5] x [-5, 5]
        "omega": 1.8,  # Frequency for visible ripples (wavelength ~3.5)
        "source_position": (-2.0, 1.5),  # Source position (off-center, upper left)
        "source_width": 0.1,  # Tight Gaussian for point-like source
        "source_amplitude": 10.0,  # Strong source
        "use_dielectric": True,  # Include dielectric circle
        "dielectric_center": (0.5, -0.5),  # Center of dielectric (offset)
        "dielectric_radius": 1.2,  # Radius of dielectric
        "dielectric_eps": 2.0,  # Permittivity of dielectric
    }
    
    print("=" * 60)
    print("PINN-SIREN for 2D Helmholtz Equation")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    wavelength = 2 * np.pi / config["omega"]
    domain_span = config["domain_size"][1] - config["domain_size"][0]
    n_wavelengths = domain_span / wavelength
    print(f"\n  Wavelength: {wavelength:.2f}")
    print(f"  Number of wavelengths in domain: {n_wavelengths:.1f}")
    print("=" * 60)
    
    # Create PINN
    pinn = HelmholtzPINN(**config)
    
    # Training parameters (optimized for CPU)
    train_config = {
        "n_iterations": 5000,
        "n_domain": 1500,
        "n_boundary": 500,
        "lambda_bc": 5.0,
        "log_every": 100
    }
    
    # Train
    history = pinn.train(**train_config)
    
    # Plot results
    print("\nGenerating visualizations...")
    plot_results(pinn, history, resolution=256, save_path="helmholtz_results.png")
    plot_wave_propagation(pinn, n_frames=8, resolution=256, save_path="wave_propagation.png")
    
    print("\nTraining complete!")
    print(f"Final loss: {history['total'][-1]:.2e}")


if __name__ == "__main__":
    main()
