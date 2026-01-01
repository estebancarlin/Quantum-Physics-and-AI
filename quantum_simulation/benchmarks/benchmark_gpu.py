"""
Benchmark complet CPU vs GPU.

Tests:
    1. Gradient/Laplacien 1D (différences finies)
    2. FFT 1D/2D
    3. Crank-Nicolson évolution
    4. Dashboard 2D
"""

import sys
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum_simulation.utils.gpu_manager import GPU_AVAILABLE, cp
from quantum_simulation.utils.numerical import (
    gradient_1d, laplacian_1d, fft_gradient, laplacian_2d_fft
)
from quantum_simulation.dynamics.evolution import TimeEvolution
from quantum_simulation.systems.free_particle import FreeParticle
from quantum_simulation.core.operators import Hamiltonian


def benchmark_gradient_1d():
    """Test 1 : Gradient 1D."""
    print("\n" + "="*70)
    print(" Test 1 : Gradient 1D (différences finies ordre 2)")
    print("="*70)
    
    sizes = [512, 1024, 2048, 4096, 8192, 16384]
    results = {'nx': [], 'cpu_ms': [], 'gpu_ms': [], 'speedup': []}
    
    for nx in sizes:
        x = np.linspace(-5e-9, 5e-9, nx)
        f = np.exp(-x**2 / (2*(1e-9)**2)) + 0j
        dx = x[1] - x[0]
        
        # CPU
        t0 = time.time()
        grad_cpu = gradient_1d(f, dx, use_gpu=False)
        t_cpu = (time.time() - t0) * 1000
        
        # GPU
        if GPU_AVAILABLE:
            # Warm-up
            _ = gradient_1d(f, dx, use_gpu=True)
            
            t0 = time.time()
            grad_gpu = gradient_1d(f, dx, use_gpu=True)
            t_gpu = (time.time() - t0) * 1000
            
            speedup = t_cpu / t_gpu
            error = np.max(np.abs(grad_cpu - grad_gpu))
            
            results['nx'].append(nx)
            results['cpu_ms'].append(t_cpu)
            results['gpu_ms'].append(t_gpu)
            results['speedup'].append(speedup)
            
            print(f"  nx={nx:5d} : CPU {t_cpu:7.2f}ms | GPU {t_gpu:7.2f}ms | "
                  f"Speedup {speedup:4.1f}× | Erreur {error:.2e}")
        else:
            print(f"  nx={nx:5d} : CPU {t_cpu:7.2f}ms (GPU non disponible)")
    
    return results


def benchmark_fft_2d():
    """Test 2 : FFT Laplacien 2D."""
    print("\n" + "="*70)
    print(" Test 2 : Laplacien 2D (FFT)")
    print("="*70)
    
    sizes = [128, 256, 512, 1024, 2048]
    results = {'size': [], 'cpu_ms': [], 'gpu_ms': [], 'speedup': []}
    
    for nx in sizes:
        psi = np.random.randn(nx, nx) + 1j * np.random.randn(nx, nx)
        dx = dy = 1e-9
        
        # CPU
        t0 = time.time()
        lap_cpu = laplacian_2d_fft(psi, dx, dy, use_gpu=False)
        t_cpu = (time.time() - t0) * 1000
        
        # GPU
        if GPU_AVAILABLE:
            # Warm-up
            _ = laplacian_2d_fft(psi, dx, dy, use_gpu=True)
            
            t0 = time.time()
            lap_gpu = laplacian_2d_fft(psi, dx, dy, use_gpu=True)
            t_gpu = (time.time() - t0) * 1000
            
            speedup = t_cpu / t_gpu
            error = np.max(np.abs(lap_cpu - lap_gpu))
            
            results['size'].append(f"{nx}×{nx}")
            results['cpu_ms'].append(t_cpu)
            results['gpu_ms'].append(t_gpu)
            results['speedup'].append(speedup)
            
            print(f"  {nx:4d}×{nx:4d} : CPU {t_cpu:8.1f}ms | GPU {t_gpu:8.1f}ms | "
                  f"Speedup {speedup:5.1f}× | Erreur {error:.2e}")
        else:
            print(f"  {nx:4d}×{nx:4d} : CPU {t_cpu:8.1f}ms (GPU non disponible)")
    
    return results


def benchmark_crank_nicolson():
    """Test 3 : Évolution Crank-Nicolson."""
    print("\n" + "="*70)
    print(" Test 3 : Évolution Crank-Nicolson (nx=4096, nt=100)")
    print("="*70)
    
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    
    nx = 4096
    x = np.linspace(-5e-9, 5e-9, nx)
    dx = x[1] - x[0]
    
    # État initial gaussien
    sigma_x = 2e-9
    k0 = 5e9
    psi0 = np.exp(-x**2 / (2*sigma_x**2)) * np.exp(1j * k0 * x)
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx)
    
    from quantum_simulation.core.state import WaveFunctionState
    state0 = WaveFunctionState(x, psi0)
    
    # Hamiltonien particule libre
    H = Hamiltonian(mass, hbar)
    H.dimension = 1
    H.potential = lambda x: 0.0  # Particule libre
    
    evolver = TimeEvolution(H, hbar)
    
    # Paramètres évolution
    t0 = 0.0
    t_final = 5e-15
    dt = 5e-17
    
    # CPU
    print("  CPU...")
    t0_cpu = time.time()
    state_cpu = evolver.evolve_wavefunction(state0, t0, t_final, dt, use_gpu=False)
    t_cpu = time.time() - t0_cpu
    
    # GPU
    if GPU_AVAILABLE:
        print("  GPU...")
        t0_gpu = time.time()
        state_gpu = evolver.evolve_wavefunction(state0, t0, t_final, dt, use_gpu=True)
        t_gpu = time.time() - t0_gpu
        
        speedup = t_cpu / t_gpu
        error = np.max(np.abs(state_cpu.wavefunction - state_gpu.wavefunction))
        
        print(f"\n  Résultats:")
        print(f"    CPU : {t_cpu:.2f}s")
        print(f"    GPU : {t_gpu:.2f}s")
        print(f"    Speedup : {speedup:.2f}×")
        print(f"    Erreur max : {error:.2e}")
        print(f"    Norme CPU : {state_cpu.norm():.10f}")
        print(f"    Norme GPU : {state_gpu.norm():.10f}")
        
        return {'cpu_s': t_cpu, 'gpu_s': t_gpu, 'speedup': speedup}
    else:
        print(f"  CPU : {t_cpu:.2f}s (GPU non disponible)")
        return None


def plot_results(results_grad, results_fft, results_cn):
    """Génère graphiques comparatifs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Graphique 1 : Gradient 1D
    if results_grad['speedup']:
        ax = axes[0]
        ax.plot(results_grad['nx'], results_grad['speedup'], 'o-', linewidth=2, markersize=8)
        ax.axhline(1.0, color='r', linestyle='--', label='Pas de gain')
        ax.set_xlabel('Taille grille nx')
        ax.set_ylabel('Speedup GPU vs CPU')
        ax.set_title('Gradient 1D (différences finies ordre 2)')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Graphique 2 : FFT 2D
    if results_fft['speedup']:
        ax = axes[1]
        sizes_numeric = [int(s.split('×')[0]) for s in results_fft['size']]
        ax.plot(sizes_numeric, results_fft['speedup'], 's-', linewidth=2, markersize=8, color='green')
        ax.axhline(1.0, color='r', linestyle='--', label='Pas de gain')
        ax.set_xlabel('Taille grille (nx=ny)')
        ax.set_ylabel('Speedup GPU vs CPU')
        ax.set_title('Laplacien 2D (FFT)')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Graphique 3 : Crank-Nicolson
    if results_cn:
        ax = axes[2]
        categories = ['Gradient\n1D\n(avg)', 'FFT 2D\n(avg)', 'Crank-\nNicolson\n(4096pts)']
        speedups = [
            np.mean(results_grad['speedup']),
            np.mean(results_fft['speedup']),
            results_cn['speedup']
        ]
        colors = ['blue', 'green', 'orange']
        bars = ax.bar(categories, speedups, color=colors, alpha=0.7)
        ax.axhline(1.0, color='r', linestyle='--', linewidth=2, label='Pas de gain')
        ax.set_ylabel('Speedup moyen GPU vs CPU')
        ax.set_title('Résumé Performances GPU')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Annotations valeurs
        for bar, val in zip(bars, speedups):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}×',
                   ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Sauvegarde
    output_dir = Path("quantum_simulation/results/benchmarks/")
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "benchmark_gpu_summary.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\n✓ Graphiques sauvegardés : {filepath}")
    
    plt.show()


def main():
    print("="*70)
    print(" BENCHMARK GPU - Quantum Simulation Framework")
    print("="*70)
    
    if not GPU_AVAILABLE:
        print("\n⚠️  GPU non disponible. Exécution benchmarks CPU uniquement.")
        print("   Installer CuPy: pip install cupy-cuda12x\n")
    
    # Tests
    results_grad = benchmark_gradient_1d()
    results_fft = benchmark_fft_2d()
    results_cn = benchmark_crank_nicolson()
    
    # Résumé
    print("\n" + "="*70)
    print(" RÉSUMÉ")
    print("="*70)
    
    if GPU_AVAILABLE:
        print(f"\n  Gradient 1D (moyen) : {np.mean(results_grad['speedup']):.1f}×")
        print(f"  FFT 2D (moyen)      : {np.mean(results_fft['speedup']):.1f}×")
        if results_cn:
            print(f"  Crank-Nicolson      : {results_cn['speedup']:.1f}×")
        
        # Génération graphiques
        plot_results(results_grad, results_fft, results_cn)
    else:
        print("\n  Benchmarks CPU uniquement complétés.")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()