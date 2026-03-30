# quantum_simulation/examples/example_gaussian_2d_evolution.py
"""
Exemple complet : Évolution paquet gaussien 2D libre.

Démonstration:
    - Création paquet gaussien 2D avec impulsion
    - Évolution temporelle (ADI ou split-operator)
    - Visualisations densités + animations
    - Validation conservation norme

Usage:
    python examples/example_gaussian_2d_evolution.py
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum_simulation.systems.free_particle_2d import FreeParticle2D
from quantum_simulation.core.state import WaveFunctionState2D
from quantum_simulation.dynamics.evolution import TimeEvolution
from quantum_simulation.visualization.viz_2d import QuantumVisualizer2D


def main():
    print("="*70)
    print(" Simulation Quantique 2D : Évolution Paquet Gaussien Libre")
    print("="*70)
    print()
    
    # Constantes physiques
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    
    # Grilles spatiales
    print("[1/6] Construction grilles...")
    x = np.linspace(-5e-8, 5e-8, 256)
    y = np.linspace(-5e-8, 5e-8, 256)
    print(f"  Grille: ({len(x)}, {len(y)}) points")
    print()
    
    # État initial
    print("[2/6] Préparation état initial...")
    fp2d = FreeParticle2D(mass, hbar)
    
    initial_state = fp2d.create_gaussian_packet_2d(
        x, y,
        x0=0.0, y0=0.0,
        sigma_x=4e-9, sigma_y=4e-9,
        kx0=5e9, ky0=3e9
    )
    
    print(f"  Norme initiale: {initial_state.norm():.10f}")
    print(f"  Impulsion: kₓ={5e9:.2e} rad/m, kᵧ={3e9:.2e} rad/m")
    print()
    
    # Hamiltonien
    print("[3/6] Définition hamiltonien...")
    hamiltonian = fp2d.create_hamiltonian_2d(x, y)
    
    E_expected = fp2d.energy_eigenvalue_2d(5e9, 3e9)
    print(f"  Énergie attendue: {E_expected:.6e} J")
    print()
    
    # Évolution temporelle
    print("[4/6] Évolution temporelle...")
    times = np.linspace(0, 5e-14, 500)
    
    evolver = TimeEvolution(hamiltonian)
    method = 'split_operator'
    print(f"  Méthode: {method}")
    print(f"  Temps: {times[0]:.2e} → {times[-1]:.2e} s ({len(times)} pas)")
    
    states = evolver.evolve_wavefunction_2d(
        initial_state,
        times,
        hamiltonian,
        method=method
    )
    
    print(f"  ✓ {len(states)} états calculés")
    
    # Vérification norme
    norms = [state.norm() for state in states]
    max_deviation = max(abs(n - 1.0) for n in norms)
    print(f"  Conservation norme: max|norm-1| = {max_deviation:.2e}")
    print()
    
    # ✅ Dashboard évolution (UNIQUE)
    print("[5/6] Génération dashboard interactif...")
    from quantum_simulation.visualization.dashboard_2d import QuantumDashboard2D
    
    dashboard = QuantumDashboard2D(output_dir='quantum_simulation/results/gaussian_2d/')
    
    dashboard_path = dashboard.create_evolution_dashboard(
        states=states,
        times=times,
        hbar=hbar,
        mass=mass,
        output_name='evolution_dashboard.gif',
        fps=10
    )
    print(f"  ✓ Dashboard: {dashboard_path}")
    print()
    
    # ✅ Visualisations statiques complémentaires
    print("[6/6] Visualisations complémentaires...")
    viz = QuantumVisualizer2D(output_dir='quantum_simulation/results/gaussian_2d/')
    
    # État initial
    print("  - Densité initiale...")
    viz.plot_density_2d(
        states[0],
        title="Densité t=0",
        save_name='density_t0'
    )
    
    # État final
    print("  - Densité finale...")
    viz.plot_density_2d(
        states[-1],
        title=f"Densité t={times[-1]*1e15:.2f} fs",
        save_name='density_final'
    )
    
    # Fonction d'onde 3D
    print("  - Fonction d'onde 3D...")
    viz.plot_wavefunction_3d(
        states[0],
        component='abs',
        title="Module |ψ(x,y,0)|",
        save_name='wavefunction_3d_t0'
    )
    
    # Marginales
    print("  - Distributions marginales...")
    viz.plot_marginal_distributions(
        states[-1],
        title="Marginales état final",
        save_name='marginals_final'
    )
    
    # Courant probabilité
    print("  - Courant probabilité...")
    viz.plot_probability_current_2d(
        states[-1],
        hbar,
        mass,
        title="Courant probabilité J(x,y)",
        save_name='current_final'
    )
    
    # ✅ Animation simple (optionnel, si ffmpeg disponible)
    print("  - Animation simple densité...")
    try:
        viz.create_animation_2d(
            states,
            times,
            output_name='evolution_simple.mp4',
            interval=100,
            fps=10
        )
    except Exception as e:
        print(f"    ⚠️ Animation simple skippée: {e}")
    
    print()
    print("="*70)
    print(" ✓ Simulation 2D complète!")
    print("="*70)
    print(f" 📊 Dashboard animé    : {dashboard_path}")
    print(f" 📁 Figures statiques  : quantum_simulation/results/gaussian_2d/")
    print("="*70)


if __name__ == "__main__":
    main()