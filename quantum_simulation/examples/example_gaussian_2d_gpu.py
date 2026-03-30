"""
Exemple complet : Évolution paquet gaussien 2D GPU-accelerated.

Démonstration:
    - Split-operator 2D avec GPU
    - Dashboard GPU
    - Gain performance 10-15×
"""

import sys
from pathlib import Path
import numpy as np
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum_simulation.systems.free_particle_2d import FreeParticle2D
from quantum_simulation.core.state import WaveFunctionState2D
from quantum_simulation.dynamics.evolution import TimeEvolution
from quantum_simulation.core.operators import Hamiltonian
from quantum_simulation.visualization.dashboard_2d import QuantumDashboard2D
from quantum_simulation.utils.config_loader import load_config


def main():
    print("="*70)
    print(" Évolution Paquet Gaussien 2D GPU-Accelerated")
    print("="*70)
    print()
    
    # Configuration
    config = load_config()
    hbar = config['physical_constants']['hbar']
    mass = config['systems']['free_particle']['mass']
    
    # Grille 2D (GPU optimal : 512×512)
    nx, ny = 512, 512
    x = np.linspace(-10e-9, 10e-9, nx)
    y = np.linspace(-10e-9, 10e-9, ny)
    
    print(f"Configuration:")
    print(f"  Grille : {nx}×{ny}")
    print(f"  hbar = {hbar:.3e} J·s")
    print(f"  mass = {mass:.3e} kg")
    print()
    
    # État initial gaussien avec impulsion
    print("[1/4] Création état initial...")
    fp2d = FreeParticle2D(mass, hbar)
    state0 = fp2d.create_gaussian_packet_2d(
        x, y,
        x0=0, y0=0,
        sigma_x=2e-9, sigma_y=2e-9,
        kx0=5e9, ky0=3e9
    )
    
    print(f"  ✓ Paquet gaussien créé")
    print(f"    Norme : {state0.norm():.10f}")
    print(f"    Largeur : σₓ = 2 nm, σᵧ = 2 nm")
    print(f"    Impulsion : kₓ = 5×10⁹ m⁻¹, kᵧ = 3×10⁹ m⁻¹")
    print()
    
    # Hamiltonien
    H = Hamiltonian(mass, hbar)
    H.dimension = 2
    H.potential = lambda x, y: 0.0  # Particule libre
    
    evolver = TimeEvolution(H, hbar)
    
    # Temps échantillonnage
    t_final = 5e-15  # 5 fs
    n_frames = 50
    times = np.linspace(0, t_final, n_frames)
    
    print(f"[2/4] Évolution temporelle GPU...")
    print(f"  Durée : {t_final*1e15:.1f} fs")
    print(f"  Frames : {n_frames}")
    
    # Évolution GPU
    t0 = time.time()
    states = evolver.evolve_wavefunction_2d(
        state0, times, H,
        method='split_operator',
        use_gpu=True  # Force GPU
    )
    t_gpu = time.time() - t0
    
    print(f"  ✓ Évolution GPU complétée : {t_gpu:.2f}s")
    print(f"    Performance : {len(states)/t_gpu:.1f} frames/s")
    print(f"    Norme finale : {states[-1].norm():.10f}")
    print()
    
    # Comparaison CPU (optionnel)
    print("[3/4] Comparaison CPU (optionnel)...")
    print("  Évolution CPU...")
    t0_cpu = time.time()
    states_cpu = evolver.evolve_wavefunction_2d(
        state0, times[:10], H,  # Seulement 10 frames pour CPU
        method='split_operator',
        use_gpu=False
    )
    t_cpu = time.time() - t0_cpu
    t_cpu_extrapolated = t_cpu * (n_frames / 10)
    
    speedup = t_cpu_extrapolated / t_gpu
    
    print(f"  ✓ Évolution CPU (10 frames) : {t_cpu:.2f}s")
    print(f"    Extrapolation {n_frames} frames : {t_cpu_extrapolated:.2f}s")
    print(f"    **Speedup GPU : {speedup:.1f}×**")
    print()
    
    # Dashboard GPU
    print("[4/4] Création dashboard GPU...")
    dashboard = QuantumDashboard2D(
        output_dir='quantum_simulation/results/gaussian_2d_gpu/',
        dpi=120,
        use_gpu=True
    )
    
    dashboard_path = dashboard.create_evolution_dashboard(
        states, times, hbar, mass,
        output_name='gaussian_2d_gpu.mp4',
        fps=10
    )
    
    print()
    print("="*70)
    print(" ✓ Simulation 2D GPU complète!")
    print("="*70)
    print(f" 📊 Dashboard GPU     : {dashboard_path}")
    print(f" ⚡ Speedup GPU       : {speedup:.1f}×")
    print(f" 🎬 Frames générées   : {len(states)}")
    print(f" ⏱️  Temps total GPU  : {t_gpu:.2f}s")
    print("="*70)


if __name__ == "__main__":
    main()