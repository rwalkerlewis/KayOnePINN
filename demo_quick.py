"""
Quick demonstration of PINN-SIREN for Helmholtz equation.
This script uses a smaller network and fewer points for faster CPU execution.

For best results (clear wave ripples), run on GPU or use the full pinn_helmholtz.py with more iterations.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Simple SIREN Implementation (minimal for speed)
# ============================================================================

class SineLayer(torch.nn.Module):
    def __init__(self, in_f, out_f, is_first=False, omega=30.0):
        super().__init__()
        self.omega = omega
        self.linear = torch.nn.Linear(in_f, out_f)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1/in_f, 1/in_f)
            else:
                self.linear.weight.uniform_(-np.sqrt(6/in_f)/omega, np.sqrt(6/in_f)/omega)
    
    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class SIREN(torch.nn.Module):
    def __init__(self, hidden=64, layers=3, omega=30.0):
        super().__init__()
        self.net = torch.nn.Sequential(
            SineLayer(2, hidden, is_first=True, omega=omega),
            *[SineLayer(hidden, hidden, omega=omega) for _ in range(layers)],
            torch.nn.Linear(hidden, 2)
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================================
# Training Configuration
# ============================================================================

def run_demo(
    n_iterations: int = 2000,
    omega: float = 1.0,
    use_dielectric: bool = True,
    save_path: str = "demo_result.png"
):
    """
    Run quick PINN demo for Helmholtz equation.
    
    Args:
        n_iterations: Number of training iterations
        omega: Wave angular frequency (higher = shorter wavelength, more ripples)
        use_dielectric: Whether to include dielectric circle
        save_path: Path to save result image
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Physical parameters
    domain = (-5.0, 5.0)
    source_pos = torch.tensor([-2.5, 1.5], device=device)
    source_width = 0.12
    source_amp = 8.0
    eps_diel = 2.0 if use_dielectric else 1.0
    diel_center = torch.tensor([0.5, -0.5], device=device)
    diel_radius = 1.2
    
    wavelength = 2 * np.pi / omega
    print(f"Omega: {omega}, Wavelength: {wavelength:.2f}")
    print(f"Dielectric: {use_dielectric} (eps={eps_diel})")
    
    # Create model - smaller network for fast CPU training
    siren_omega = max(15.0, 5 * omega)
    model = SIREN(hidden=64, layers=3, omega=siren_omega).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def get_permittivity(coords):
        if not use_dielectric:
            return torch.ones(coords.shape[0], device=device)
        dist = torch.sqrt((coords[:,0] - diel_center[0])**2 + (coords[:,1] - diel_center[1])**2)
        mask = 0.5 * (1 - torch.tanh((dist - diel_radius) / 0.1))
        return 1.0 + (eps_diel - 1.0) * mask
    
    def get_source(coords):
        dist_sq = (coords[:,0] - source_pos[0])**2 + (coords[:,1] - source_pos[1])**2
        return source_amp * torch.exp(-dist_sq / (2 * source_width**2))
    
    # Training with finite differences (fast on CPU)
    h = 0.04  # Finite difference step size
    n_pts = 300  # Collocation points per iteration
    n_bc = 80   # Boundary points
    
    losses = []
    print(f"\nTraining for {n_iterations} iterations...")
    
    for i in tqdm(range(n_iterations)):
        optimizer.zero_grad()
        
        # Sample domain points
        coords = torch.rand(n_pts, 2, device=device) * (domain[1] - domain[0]) + domain[0]
        
        # Compute field at stencil points for finite differences
        c = coords
        px = c.clone(); px[:,0] = px[:,0] + h
        mx = c.clone(); mx[:,0] = mx[:,0] - h
        py = c.clone(); py[:,1] = py[:,1] + h
        my = c.clone(); my[:,1] = my[:,1] - h
        
        Ez_c = model(c)
        Ez_px = model(px)
        Ez_mx = model(mx)
        Ez_py = model(py)
        Ez_my = model(my)
        
        # Laplacian via finite differences
        lap_r = (Ez_px[:,0] - 2*Ez_c[:,0] + Ez_mx[:,0] + Ez_py[:,0] - 2*Ez_c[:,0] + Ez_my[:,0]) / (h*h)
        lap_i = (Ez_px[:,1] - 2*Ez_c[:,1] + Ez_mx[:,1] + Ez_py[:,1] - 2*Ez_c[:,1] + Ez_my[:,1]) / (h*h)
        
        eps = get_permittivity(c)
        k_sq = eps * omega**2
        source = get_source(c)
        
        # Helmholtz residuals: (-∇² - k²)Ez = -iωJz
        # Real part: -lap_r - k²*Ez_r = 0
        # Imag part: -lap_i - k²*Ez_i = -ω*Jz
        res_r = -lap_r - k_sq * Ez_c[:,0]
        res_i = -lap_i - k_sq * Ez_c[:,1] + omega * source
        
        loss_phys = torch.mean(res_r**2) + torch.mean(res_i**2)
        
        # Boundary points (Sommerfeld radiation condition)
        bc = torch.rand(n_bc, 2, device=device) * (domain[1] - domain[0]) + domain[0]
        side = torch.randint(0, 4, (n_bc,), device=device)
        bc[side==0, 0] = domain[0]  # left
        bc[side==1, 0] = domain[1]  # right
        bc[side==2, 1] = domain[0]  # bottom
        bc[side==3, 1] = domain[1]  # top
        
        r = torch.sqrt(bc[:,0]**2 + bc[:,1]**2).clamp(min=0.1)
        r_hat = bc / r.unsqueeze(1)
        
        Ez_bc = model(bc)
        bc_p = (bc + h * r_hat).clamp(domain[0], domain[1])
        bc_m = (bc - h * r_hat).clamp(domain[0], domain[1])
        Ez_bp = model(bc_p)
        Ez_bm = model(bc_m)
        
        dEz_dr_r = (Ez_bp[:,0] - Ez_bm[:,0]) / (2*h)
        dEz_dr_i = (Ez_bp[:,1] - Ez_bm[:,1]) / (2*h)
        
        # Sommerfeld: ∂Ez/∂r + ikEz = 0
        k = omega
        som_r = dEz_dr_r - k * Ez_bc[:,1]
        som_i = dEz_dr_i + k * Ez_bc[:,0]
        
        loss_bc = torch.mean(som_r**2) + torch.mean(som_i**2)
        
        # Total loss
        loss = loss_phys + 5.0 * loss_bc
        loss.backward()
        optimizer.step()
        
        if i % 500 == 0:
            losses.append(loss.item())
            tqdm.write(f"Iter {i}: loss={loss.item():.4e}, phys={loss_phys.item():.4e}, bc={loss_bc.item():.4e}")
    
    # ========================================================================
    # Visualization
    # ========================================================================
    print("\nGenerating visualization...")
    
    res = 256
    x = np.linspace(domain[0], domain[1], res)
    y = np.linspace(domain[0], domain[1], res)
    X, Y = np.meshgrid(x, y)
    coords_grid = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), 
                               dtype=torch.float32, device=device)
    
    with torch.no_grad():
        Ez = model(coords_grid).cpu().numpy()
        eps_grid = get_permittivity(coords_grid).cpu().numpy()
        source_grid = get_source(coords_grid).cpu().numpy()
    
    Ez_r = Ez[:,0].reshape(res, res)
    Ez_i = Ez[:,1].reshape(res, res)
    eps_grid = eps_grid.reshape(res, res)
    source_grid = source_grid.reshape(res, res)
    
    # Create figure matching the reference style
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Permittivity
    ax = axes[0]
    ax.imshow(eps_grid, extent=[domain[0], domain[1], domain[0], domain[1]], 
              origin='lower', cmap='gray')
    ax.set_title('Permittivity')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Source
    ax = axes[1]
    ax.imshow(source_grid, extent=[domain[0], domain[1], domain[0], domain[1]], 
              origin='lower', cmap='hot')
    ax.set_title(f'Source at [{source_pos[0].cpu().item():.0f} {source_pos[1].cpu().item():.0f}]')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Real Ez
    ax = axes[2]
    vmax = np.percentile(np.abs(Ez_r), 99)
    ax.imshow(Ez_r, extent=[domain[0], domain[1], domain[0], domain[1]], 
              origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title('Real(E_z)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Imag Ez
    ax = axes[3]
    vmax = np.percentile(np.abs(Ez_i), 99)
    ax.imshow(Ez_i, extent=[domain[0], domain[1], domain[0], domain[1]], 
              origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title('Imag(E_z)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")
    
    # Also save training history
    plt.figure(figsize=(8, 5))
    plt.semilogy(range(0, n_iterations, 500), losses, 'b-o')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_history.png', dpi=150)
    print("Saved training_history.png")
    
    return model, losses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick PINN-SIREN Helmholtz Demo')
    parser.add_argument('--iterations', type=int, default=2000, 
                        help='Number of training iterations (default: 2000)')
    parser.add_argument('--omega', type=float, default=1.0,
                        help='Wave frequency (default: 1.0)')
    parser.add_argument('--no-dielectric', action='store_true',
                        help='Disable dielectric circle')
    parser.add_argument('--output', type=str, default='demo_result.png',
                        help='Output image path')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PINN-SIREN Quick Demo for 2D Helmholtz Equation")
    print("=" * 60)
    
    model, losses = run_demo(
        n_iterations=args.iterations,
        omega=args.omega,
        use_dielectric=not args.no_dielectric,
        save_path=args.output
    )
    
    print("\nDemo complete!")
    print(f"Final loss: {losses[-1]:.4e}")
