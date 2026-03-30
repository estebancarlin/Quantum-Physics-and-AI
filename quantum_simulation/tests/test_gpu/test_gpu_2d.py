"""Tests évolution 2D GPU."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from quantum_simulation.utils.gpu_manager import GPU_AVAILABLE, cp
from quantum_simulation.core.state import WaveFunctionState2D
from quantum_simulation.dynamics.evolution import TimeEvolution
from quantum_simulation.core.operators import Hamiltonian


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU requis")
def test_split_operator_2d_cpu_gpu_equivalence():
    """Test équivalence CPU/GPU Split-Operator 2D."""
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    
    nx, ny = 256, 256
    x = np.linspace(-5e-9, 5e-9, nx)
    y = np.linspace(-5e-9, 5e-9, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # FIX: Gaussienne avec impulsion (pour tester courant)
    sigma = 1e-9
    kx0, ky0 = 5e9, 3e9  # Impulsion initiale
    psi0 = np.exp(-(X**2 + Y**2) / (2*sigma**2)) * np.exp(1j * (kx0 * X + ky0 * Y))
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * (x[1]-x[0]) * (y[1]-y[0]))
    
    state0 = WaveFunctionState2D(x, y, psi0)
    
    H = Hamiltonian(mass, hbar)
    H.dimension = 2
    H.potential = lambda x, y: 0.0  # Particule libre
    
    evolver = TimeEvolution(H, hbar)
    
    times = np.linspace(0, 1e-15, 10)
    
    # CPU
    print("\n  Évolution 2D CPU...")
    states_cpu = evolver.evolve_wavefunction_2d(
        state0, times, H, method='split_operator', use_gpu=False
    )
    
    # GPU
    print("  Évolution 2D GPU...")
    states_gpu = evolver.evolve_wavefunction_2d(
        state0, times, H, method='split_operator', use_gpu=True
    )
    
    # Comparaison état final
    error = np.max(np.abs(states_cpu[-1].wavefunction - states_gpu[-1].wavefunction))
    relative_error = error / np.max(np.abs(states_cpu[-1].wavefunction))
    
    print(f"\n  Split-Operator 2D ({nx}×{ny}):")
    print(f"    Erreur max : {error:.2e}")
    print(f"    Erreur relative : {relative_error:.2e}")
    print(f"    Norme CPU : {states_cpu[-1].norm():.10f}")
    print(f"    Norme GPU : {states_gpu[-1].norm():.10f}")
    
    assert relative_error < 1e-5, \
        f"Erreur CPU/GPU 2D trop grande : {relative_error:.2e}"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU requis")
def test_split_operator_2d_norm_conservation():
    """Test conservation norme Split-Operator 2D GPU."""
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    
    nx, ny = 256, 256
    x = np.linspace(-5e-9, 5e-9, nx)
    y = np.linspace(-5e-9, 5e-9, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # FIX: Gaussienne avec impulsion
    kx0, ky0 = 5e9, 3e9
    psi0 = np.exp(-(X**2 + Y**2) / (2*(1e-9)**2)) * np.exp(1j * (kx0 * X + ky0 * Y))
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * (x[1]-x[0]) * (y[1]-y[0]))
    
    state0 = WaveFunctionState2D(x, y, psi0)
    
    H = Hamiltonian(mass, hbar)
    H.dimension = 2
    H.potential = lambda x, y: 0.0
    
    evolver = TimeEvolution(H, hbar)
    
    times = np.linspace(0, 5e-15, 50)
    
    # Évolution GPU
    print("\n  Évolution 2D GPU (50 pas)...")
    states = evolver.evolve_wavefunction_2d(
        state0, times, H, method='split_operator', use_gpu=True
    )
    
    # Validation conservation
    norms = [state.norm() for state in states]
    max_deviation = max(abs(n - 1.0) for n in norms)
    
    print(f"\n  Conservation norme 2D GPU:")
    print(f"    Max déviation : {max_deviation:.2e}")
    print(f"    Norme initiale : {norms[0]:.10f}")
    print(f"    Norme finale : {norms[-1]:.10f}")
    
    assert max_deviation < 1e-6, \
        f"Norme 2D GPU mal conservée : {max_deviation:.2e}"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU requis")
@pytest.mark.slow
def test_split_operator_2d_speedup():
    """Test speedup GPU Split-Operator 2D."""
    import time
    
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    
    # Grille 512×512 (sweet spot GPU)
    nx, ny = 512, 512
    x = np.linspace(-5e-9, 5e-9, nx)
    y = np.linspace(-5e-9, 5e-9, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # FIX: Gaussienne avec impulsion
    kx0, ky0 = 5e9, 3e9
    psi0 = np.exp(-(X**2 + Y**2) / (2*(1e-9)**2)) * np.exp(1j * (kx0 * X + ky0 * Y))
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * (x[1]-x[0]) * (y[1]-y[0]))
    
    state0 = WaveFunctionState2D(x, y, psi0)
    
    H = Hamiltonian(mass, hbar)
    H.dimension = 2
    H.potential = lambda x, y: 0.0
    
    evolver = TimeEvolution(H, hbar)
    
    times = np.linspace(0, 5e-15, 50)  # 50 pas
    
    # CPU
    print(f"\n  Évolution 2D CPU ({nx}×{ny}, {len(times)} pas)...")
    t0 = time.time()
    states_cpu = evolver.evolve_wavefunction_2d(
        state0, times, H, method='split_operator', use_gpu=False
    )
    t_cpu = time.time() - t0
    
    # GPU (warm-up)
    print("  Warm-up GPU...")
    _ = evolver.evolve_wavefunction_2d(
        state0, times[:5], H, method='split_operator', use_gpu=True
    )
    
    # GPU
    print("  Évolution 2D GPU...")
    t0 = time.time()
    states_gpu = evolver.evolve_wavefunction_2d(
        state0, times, H, method='split_operator', use_gpu=True
    )
    cp.cuda.Stream.null.synchronize()
    t_gpu = time.time() - t0
    
    speedup = t_cpu / t_gpu
    
    print(f"\n  Speedup GPU Split-Operator 2D:")
    print(f"    CPU : {t_cpu:.2f}s")
    print(f"    GPU : {t_gpu:.2f}s")
    print(f"    Speedup : {speedup:.1f}×")
    
    # Validation
    error = np.max(np.abs(states_cpu[-1].wavefunction - states_gpu[-1].wavefunction))
    relative_error = error / np.max(np.abs(states_cpu[-1].wavefunction))
    print(f"    Erreur relative : {relative_error:.2e}")
    
    # Attendu : 8-15× pour 512×512
    assert speedup > 5.0, \
        f"Speedup 2D trop faible : {speedup:.1f}× (attendu > 5×)"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU requis")
def test_dashboard_2d_gpu():
    """Test dashboard 2D avec GPU."""
    from quantum_simulation.visualization.dashboard_2d import QuantumDashboard2D
    
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    
    nx, ny = 128, 128
    x = np.linspace(-5e-9, 5e-9, nx)
    y = np.linspace(-5e-9, 5e-9, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # FIX: État initial avec impulsion (courant non nul)
    kx0, ky0 = 5e9, 3e9
    psi0 = np.exp(-(X**2 + Y**2) / (2*(1e-9)**2)) * np.exp(1j * (kx0 * X + ky0 * Y))
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * (x[1]-x[0]) * (y[1]-y[0]))
    
    state0 = WaveFunctionState2D(x, y, psi0)
    
    H = Hamiltonian(mass, hbar)
    H.dimension = 2
    H.potential = lambda x, y: 0.0
    
    evolver = TimeEvolution(H, hbar)
    
    times = np.linspace(0, 1e-15, 10)
    
    # Évolution GPU
    print("\n  Évolution 2D pour dashboard GPU...")
    states = evolver.evolve_wavefunction_2d(
        state0, times, H, method='split_operator', use_gpu=True
    )
    
    # Dashboard GPU
    print("  Création dashboard GPU...")
    dashboard = QuantumDashboard2D(
        output_dir='quantum_simulation/results/test_gpu/',
        use_gpu=True
    )
    
    output_path = dashboard.create_evolution_dashboard(
        states, times, hbar, mass,
        output_name='test_dashboard_gpu.gif',
        fps=5
    )
    
    assert Path(output_path).exists(), "Dashboard non créé"
    print(f"  ✓ Dashboard GPU créé : {output_path}")


# ==================== Main ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])