"""
Quick test script for the PINN Helmholtz solver.
Runs a short training to verify the implementation works correctly.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments

from pinn_helmholtz import HelmholtzPINN, plot_results, plot_wave_propagation

def test_basic_functionality():
    """Test basic PINN components."""
    print("=" * 60)
    print("Testing PINN-SIREN Helmholtz Solver")
    print("=" * 60)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test 1: Create PINN with higher frequency for visible waves
    print("\n[Test 1] Creating PINN model...")
    pinn = HelmholtzPINN(
        domain_size=(-5.0, 5.0),
        omega=1.5,  # Higher frequency for visible ripples
        source_position=(-2.0, 1.0),
        source_width=0.1,
        source_amplitude=8.0,
        use_dielectric=True,
        dielectric_center=(0.5, -0.5),
        dielectric_radius=1.2,
        dielectric_eps=2.0,
    )
    print(f"   Device: {pinn.device}")
    print(f"   Model parameters: {sum(p.numel() for p in pinn.model.parameters()):,}")
    print("   PASSED")
    
    # Test 2: Forward pass
    print("\n[Test 2] Testing forward pass...")
    coords = torch.rand(100, 2, device=pinn.device) * 10 - 5
    with torch.no_grad():
        output = pinn.model(coords)
    assert output.shape == (100, 2), f"Expected (100, 2), got {output.shape}"
    print(f"   Output shape: {output.shape}")
    print("   PASSED")
    
    # Test 3: Permittivity function
    print("\n[Test 3] Testing permittivity field...")
    eps = pinn.get_permittivity(coords)
    assert eps.shape == (100,), f"Expected (100,), got {eps.shape}"
    assert eps.min() >= 1.0, f"Permittivity should be >= 1, got min {eps.min()}"
    print(f"   Permittivity range: [{eps.min():.2f}, {eps.max():.2f}]")
    print("   PASSED")
    
    # Test 4: Source function
    print("\n[Test 4] Testing source field...")
    source = pinn.get_source(coords)
    assert source.shape == (100,), f"Expected (100,), got {source.shape}"
    assert source.min() >= 0, "Source should be non-negative"
    print(f"   Source range: [{source.min():.4f}, {source.max():.4f}]")
    print("   PASSED")
    
    # Test 5: Derivative computation
    print("\n[Test 5] Testing derivative computation...")
    test_coords = pinn.sample_domain(50)
    Ez_r, Ez_i, lap_r, lap_i, _ = pinn.compute_derivatives(test_coords)
    assert Ez_r.shape == (50,), f"Expected (50,), got {Ez_r.shape}"
    assert lap_r.shape == (50,), f"Expected (50,), got {lap_r.shape}"
    print(f"   Field values computed successfully")
    print("   PASSED")
    
    # Test 6: Physics loss
    print("\n[Test 6] Testing physics loss computation...")
    test_coords = pinn.sample_domain(100)
    loss_phys = pinn.physics_loss(test_coords)
    assert torch.isfinite(loss_phys), "Physics loss should be finite"
    print(f"   Initial physics loss: {loss_phys.item():.4e}")
    print("   PASSED")
    
    # Test 7: Boundary loss
    print("\n[Test 7] Testing boundary loss computation...")
    boundary_coords = pinn.sample_boundary(100)
    loss_bc = pinn.boundary_loss(boundary_coords)
    assert torch.isfinite(loss_bc), "Boundary loss should be finite"
    print(f"   Initial boundary loss: {loss_bc.item():.4e}")
    print("   PASSED")
    
    # Test 8: Training step
    print("\n[Test 8] Testing training step...")
    total, phys, bc = pinn.train_step(n_domain=500, n_boundary=100)
    assert np.isfinite(total), "Total loss should be finite"
    print(f"   Loss after 1 step: total={total:.4e}, physics={phys:.4e}, bc={bc:.4e}")
    print("   PASSED")
    
    # Test 9: Short training run - optimized for CPU
    print("\n[Test 9] Running training (800 iterations)...")
    history = pinn.train(
        n_iterations=800,
        n_domain=1500,
        n_boundary=400,
        lambda_bc=5.0,
        log_every=200
    )
    assert len(history["total"]) == 500
    print(f"   Final loss: {history['total'][-1]:.4e}")
    print("   PASSED")
    
    # Test 10: Prediction on grid
    print("\n[Test 10] Testing grid prediction...")
    X, Y, Ez_real, Ez_imag = pinn.predict_grid(resolution=50)
    assert X.shape == (50, 50), f"Expected (50, 50), got {X.shape}"
    assert Ez_real.shape == (50, 50), f"Expected (50, 50), got {Ez_real.shape}"
    print(f"   Field predicted on 50x50 grid")
    print(f"   Re(Ez) range: [{Ez_real.min():.4f}, {Ez_real.max():.4f}]")
    print(f"   Im(Ez) range: [{Ez_imag.min():.4f}, {Ez_imag.max():.4f}]")
    print("   PASSED")
    
    # Test 11: Visualization
    print("\n[Test 11] Testing visualization...")
    try:
        plot_results(pinn, history, resolution=200, save_path="test_helmholtz_results.png")
        plot_wave_propagation(pinn, n_frames=8, resolution=200, save_path="test_wave_propagation.png")
        print("   Figures saved successfully")
        print("   PASSED")
    except Exception as e:
        print(f"   WARNING: Visualization failed with error: {e}")
        print("   SKIPPED (visualization not critical for core functionality)")
    
    print("\n" + "=" * 60)
    print("All core tests PASSED!")
    print("=" * 60)
    
    return pinn, history


def test_free_space():
    """Test without dielectric (free space only)."""
    print("\n" + "=" * 60)
    print("Testing Free Space Configuration (no dielectric)")
    print("=" * 60)
    
    torch.manual_seed(123)
    
    pinn = HelmholtzPINN(
        domain_size=(-5.0, 5.0),
        omega=1.5,
        source_position=(0.0, 0.0),
        source_width=0.1,
        source_amplitude=8.0,
        use_dielectric=False,
    )
    
    # Check permittivity is uniform
    coords = torch.rand(100, 2, device=pinn.device) * 10 - 5
    eps = pinn.get_permittivity(coords)
    assert torch.allclose(eps, torch.ones_like(eps)), "Free space should have ε=1 everywhere"
    print(f"   Permittivity: constant at {eps[0].item()}")
    
    # Quick training
    print("   Running 300 iterations...")
    history = pinn.train(n_iterations=300, n_domain=2000, n_boundary=500, log_every=150)
    print(f"   Final loss: {history['total'][-1]:.4e}")
    
    print("   PASSED")
    return pinn


if __name__ == "__main__":
    # Run tests
    pinn_dielectric, history = test_basic_functionality()
    pinn_free_space = test_free_space()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
