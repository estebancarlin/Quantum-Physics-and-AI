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
    """Test 1 : Gradient 1D (multiples itérations)."""
    print("\n" + "="*70)
    print(" Test 1 : Gradient 1D (différences finies ordre 2)")
    print("="*70)
    
    sizes = [512, 1024, 2048, 4096, 8192, 16384]
    results = {'nx': [], 'cpu_ms': [], 'gpu_ms': [], 'speedup': []}
    
    for nx in sizes:
        x = np.linspace(-5e-9, 5e-9, nx)
        f = np.exp(-x**2 / (2*(1e-9)**2)) + 0j
        dx = x[1] - x[0]
        
        # FIX: Nombre itérations adapté à la taille
        n_iter = max(10, int(1000 / (nx / 1024)))  # Plus petite grille → plus d'itérations
        
        # CPU (accumuler temps sur n_iter)
        t0 = time.time()
        for _ in range(n_iter):
            grad_cpu = gradient_1d(f, dx, use_gpu=False)
        t_cpu_total = time.time() - t0
        t_cpu = (t_cpu_total / n_iter) * 1000  # Temps moyen en ms
        
        # GPU
        if GPU_AVAILABLE:
            # Warm-up
            _ = gradient_1d(f, dx, use_gpu=True)
            cp.cuda.Stream.null.synchronize()
            
            # Mesure
            t0 = time.time()
            for _ in range(n_iter):
                grad_gpu = gradient_1d(f, dx, use_gpu=True)
            cp.cuda.Stream.null.synchronize()
            t_gpu_total = time.time() - t0
            t_gpu = (t_gpu_total / n_iter) * 1000  # Temps moyen en ms
            
            speedup = t_cpu / t_gpu if t_gpu > 0.01 else float('nan')
            
            if not np.isnan(speedup):
                error = np.max(np.abs(grad_cpu - grad_gpu))
                
                results['nx'].append(nx)
                results['cpu_ms'].append(t_cpu)
                results['gpu_ms'].append(t_gpu)
                results['speedup'].append(speedup)
                
                print(f"  nx={nx:5d} : CPU {t_cpu:7.2f}ms | GPU {t_gpu:7.2f}ms | "
                      f"Speedup {speedup:4.2f}× | Erreur {error:.2e} ({n_iter} iter)")
            else:
                print(f"  nx={nx:5d} : Temps trop court (skip)")
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
        
        # CPU (moyenne 3 runs)
        times_cpu = []
        for _ in range(3):
            t0 = time.time()
            lap_cpu = laplacian_2d_fft(psi, dx, dy, use_gpu=False)
            times_cpu.append(time.time() - t0)
        t_cpu = np.median(times_cpu) * 1000
        
        # GPU
        if GPU_AVAILABLE:
            # Warm-up
            _ = laplacian_2d_fft(psi, dx, dy, use_gpu=True)
            cp.cuda.Stream.null.synchronize()
            
            times_gpu = []
            for _ in range(3):
                t0 = time.time()
                lap_gpu = laplacian_2d_fft(psi, dx, dy, use_gpu=True)
                cp.cuda.Stream.null.synchronize()
                times_gpu.append(time.time() - t0)
            t_gpu = np.median(times_gpu) * 1000
            
            if t_gpu < 0.01:
                speedup = float('nan')
                print(f"  {nx:4d}×{nx:4d} : CPU {t_cpu:8.1f}ms | GPU {t_gpu:8.1f}ms | Speedup N/A")
            else:
                speedup = t_cpu / t_gpu
                error = np.max(np.abs(lap_cpu - lap_gpu))
                
                results['size'].append(f"{nx}×{nx}")
                results['cpu_ms'].append(t_cpu)
                results['gpu_ms'].append(t_gpu)
                results['speedup'].append(speedup)
                
                print(f"  {nx:4d}×{nx:4d} : CPU {t_cpu:8.1f}ms | GPU {t_gpu:8.1f}ms | "
                      f"Speedup {speedup:5.2f}× | Erreur {error:.2e}")
        else:
            print(f"  {nx:4d}×{nx:4d} : CPU {t_cpu:8.1f}ms (GPU non disponible)")
    
    return results


def benchmark_crank_nicolson():
    """Test 3 : Évolution Crank-Nicolson."""
    print("\n" + "="*70)
    print(" Test 3 : Évolution Crank-Nicolson (nx=8192, nt=50)")
    print("="*70)
    
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    
    nx = 8192
    x = np.linspace(-5e-9, 5e-9, nx)
    dx = x[1] - x[0]
    
    # État initial gaussien
    sigma_x = 2e-9
    k0 = 5e9
    psi0 = np.exp(-x**2 / (2*sigma_x**2)) * np.exp(1j * k0 * x)
    
    # FIX: Triple normalisation pour garantir précision
    norm1 = np.sqrt(np.sum(np.abs(psi0)**2) * dx)
    psi0 /= norm1
    
    norm2 = np.sqrt(np.sum(np.abs(psi0)**2) * dx)
    if abs(norm2 - 1.0) > 1e-10:
        psi0 /= norm2
    
    # Vérification finale
    norm_final = np.sqrt(np.sum(np.abs(psi0)**2) * dx)
    print(f"  Norme état initial : {norm_final:.15f}")
    
    if abs(norm_final - 1.0) > 1e-9:
        print(f"  ⚠️  Ajustement final : {norm_final:.15f} → 1.0")
        psi0 /= norm_final
        norm_final = np.sqrt(np.sum(np.abs(psi0)**2) * dx)
        print(f"  Norme après ajustement : {norm_final:.15f}")
    
    from quantum_simulation.core.state import WaveFunctionState
    state0 = WaveFunctionState(x, psi0)
    
    # Validation
    assert state0.is_normalized(tolerance=1e-6), \
        f"État non normalisé : {state0.norm():.15f}"
    
    # Hamiltonien particule libre
    H = Hamiltonian(mass, hbar)
    H.dimension = 1
    H.potential = lambda x: 0.0
    
    evolver = TimeEvolution(H)
    
    # Paramètres évolution
    t0 = 0.0
    t_final = 2.5e-15
    dt = 5e-17
    
    n_steps = int((t_final - t0) / dt)
    
    # CPU
    print(f"  CPU ({nx} pts, {n_steps} pas)...")
    t0_cpu = time.time()
    state_cpu = evolver.evolve_wavefunction(state0, t0, t_final, dt, use_gpu=False)
    t_cpu = time.time() - t0_cpu
    
    # GPU
    if GPU_AVAILABLE:
        print("  GPU...")
        t0_gpu = time.time()
        state_gpu = evolver.evolve_wavefunction(state0, t0, t_final, dt, use_gpu=True)
        cp.cuda.Stream.null.synchronize()
        t_gpu = time.time() - t0_gpu
        
        if t_gpu < 0.01:
            print(f"\n  ⚠️  Temps GPU trop court pour mesure fiable ({t_gpu:.3f}s)")
            return None
        
        speedup = t_cpu / t_gpu
        error = np.max(np.abs(state_cpu.wavefunction - state_gpu.wavefunction))
        
        print(f"\n  Résultats:")
        print(f"    CPU : {t_cpu:.2f}s")
        print(f"    GPU : {t_gpu:.2f}s")
        print(f"    Speedup : {speedup:.2f}×")
        print(f"    Erreur max : {error:.2e}")
        print(f"    Norme CPU : {state_cpu.norm():.10f}")
        print(f"    Norme GPU : {state_gpu.norm():.10f}")
        
        # Analyse performance
        if speedup < 0.8:
            print(f"\n  ⚠️  GPU plus lent que CPU!")
            print(f"      Raison : CN 1D sparse peu parallélisable")
            print(f"      Overhead transferts > gain calcul")
            print(f"      Normal pour cette configuration")
        
        return {'cpu_s': t_cpu, 'gpu_s': t_gpu, 'speedup': speedup}
    else:
        print(f"  CPU : {t_cpu:.2f}s (GPU non disponible)")
        return None


def plot_results(results_grad, results_fft, results_cn):
    """Génère graphiques comparatifs."""
    # FIX: Filtrer NaN
    if not results_grad['speedup'] or all(np.isnan(s) for s in results_grad['speedup']):
        print("\n⚠️  Pas assez de données gradient pour graphique")
        return
    
    if not results_fft['speedup'] or all(np.isnan(s) for s in results_fft['speedup']):
        print("\n⚠️  Pas assez de données FFT pour graphique")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Graphique 1 : Gradient 1D
    ax = axes[0]
    valid_grad = [(nx, sp) for nx, sp in zip(results_grad['nx'], results_grad['speedup']) if not np.isnan(sp)]
    if valid_grad:
        nx_vals, sp_vals = zip(*valid_grad)
        ax.plot(nx_vals, sp_vals, 'o-', linewidth=2, markersize=8)
        ax.axhline(1.0, color='r', linestyle='--', label='Pas de gain')
        ax.set_xlabel('Taille grille nx')
        ax.set_ylabel('Speedup GPU vs CPU')
        ax.set_title('Gradient 1D (différences finies ordre 2)')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Graphique 2 : FFT 2D
    ax = axes[1]
    valid_fft = [(s, sp) for s, sp in zip(results_fft['size'], results_fft['speedup']) if not np.isnan(sp)]
    if valid_fft:
        sizes, sp_vals = zip(*valid_fft)
        sizes_numeric = [int(s.split('×')[0]) for s in sizes]
        ax.plot(sizes_numeric, sp_vals, 's-', linewidth=2, markersize=8, color='green')
        ax.axhline(1.0, color='r', linestyle='--', label='Pas de gain')
        ax.set_xlabel('Taille grille (nx=ny)')
        ax.set_ylabel('Speedup GPU vs CPU')
        ax.set_title('Laplacien 2D (FFT)')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Graphique 3 : Résumé
    ax = axes[2]
    categories = []
    speedups = []
    colors = []
    
    if valid_grad:
        categories.append('Gradient\n1D\n(avg)')
        speedups.append(np.mean([sp for _, sp in valid_grad]))
        colors.append('blue')
    
    if valid_fft:
        categories.append('FFT 2D\n(avg)')
        speedups.append(np.mean([sp for _, sp in valid_fft]))
        colors.append('green')
    
    if results_cn and not np.isnan(results_cn.get('speedup', float('nan'))):
        categories.append('Crank-\nNicolson')
        speedups.append(results_cn['speedup'])
        colors.append('orange')
    
    if categories:
        bars = ax.bar(categories, speedups, color=colors, alpha=0.7)
        ax.axhline(1.0, color='r', linestyle='--', linewidth=2, label='Pas de gain')
        ax.set_ylabel('Speedup moyen GPU vs CPU')
        ax.set_title('Résumé Performances GPU')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        for bar, val in zip(bars, speedups):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}×',
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