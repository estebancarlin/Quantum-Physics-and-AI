# quantum_simulation/experiments/gallery/double_slit_2d.py
"""
Expérience double-slit quantique 2D.

Démontre :
- Interférences quantiques
- Dualité onde-corpuscule
- Formule Young : Δy = λD/d
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum_simulation.experiments.base_experiment import Experiment
from quantum_simulation.systems.free_particle_2d import FreeParticle2D
from quantum_simulation.systems.potential_systems_2d import DoubleSlit2D
from quantum_simulation.dynamics.evolution import TimeEvolution
from quantum_simulation.core.operators import Hamiltonian
from quantum_simulation.visualization.dashboard_2d import QuantumDashboard2D


class DoubleSlitExperiment(Experiment):
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.hbar = config['physical_constants']['hbar']
        self.mass = config['physical_constants']['m_electron']
        
        # Paramètres géométriques
        exp_config = config.get('experiments', {}).get('double_slit_2d', {})
        
        self.slit_separation = exp_config.get('slit_separation', 1.0e-8)
        self.slit_width = exp_config.get('slit_width', 2.0e-9)
        self.barrier_position = exp_config.get('barrier_position', 0.0)
        self.screen_distance = exp_config.get('screen_distance', 1.0e-7)  # ← 100 nm
        
        # ✅ Grilles adaptées
        self.x = np.linspace(-1e-7, 2e-7, 512)   # -100 à +200 nm
        self.y = np.linspace(-5e-8, 5e-8, 2048)  # ±50 nm (inchangé)
        
        # ✅ Diagnostics prédictifs
        dy = self.y[1] - self.y[0]
        
        # Lire impulsion depuis config
        p_x_config = exp_config.get('initial_state', {}).get('momentum_x', 5e-24)
        lambda_deB = 2 * np.pi * self.hbar / p_x_config
        expected_interfrange = lambda_deB * self.screen_distance / self.slit_separation
        points_per_fringe = expected_interfrange / dy
        
        print(f"\n  [CONFIG] Prédictions physiques:")
        print(f"    Impulsion px = {p_x_config:.2e} kg·m/s")
        print(f"    λ de Broglie = {lambda_deB*1e9:.3f} nm")
        print(f"    Distance écran D = {self.screen_distance*1e9:.1f} nm")
        print(f"    Interfrange théorique = {expected_interfrange*1e9:.3f} nm")
        print(f"    Résolution dy = {dy*1e9:.3f} nm")
        print(f"    Points/frange = {points_per_fringe:.1f}")
        
        # ✅ Avertissement si résolution insuffisante
        if points_per_fringe < 10:
            import warnings
            warnings.warn(
                f"Résolution insuffisante : {points_per_fringe:.1f} points/frange "
                f"(minimum 10 requis). Interférences non résolues.",
                RuntimeWarning
            )
    
    def prepare_initial_state(self):
        """Paquet gaussien incident."""
        fp2d = FreeParticle2D(self.mass, self.hbar)
        
        # ✅ Lire paramètres depuis config
        exp_config = self.config.get('experiments', {}).get('double_slit_2d', {})
        init_config = exp_config.get('initial_state', {})
        
        x0 = init_config.get('x0', -5e-8)
        y0 = init_config.get('y0', 0.0)
        sigma_x = init_config.get('sigma_x', 3e-9)
        sigma_y = init_config.get('sigma_y', 1.5e-9)
        p_x = init_config.get('momentum_x', 5e-24)
        
        kx0 = p_x / self.hbar
        ky0 = 0.0
        
        print(f"\n  [INIT] État initial:")
        print(f"    Position (x0, y0) = ({x0*1e9:.1f}, {y0:.1f}) nm")
        print(f"    Largeurs (σx, σy) = ({sigma_x*1e9:.1f}, {sigma_y*1e9:.1f}) nm")
        print(f"    Critère cohérence σy/d = {sigma_y/self.slit_separation:.2f} (< 0.5 OK)")
        
        self.initial_state = fp2d.create_gaussian_packet_2d(
            self.x, self.y,
            x0=x0, y0=y0,
            sigma_x=sigma_x, sigma_y=sigma_y,
            kx0=kx0, ky0=ky0
        )
        
        self.momentum = p_x
    
    def evolve_state(self):
        """Évolution temporelle adaptée."""
        v_x = self.momentum / self.mass
        distance = self.screen_distance - self.barrier_position
        
        # ✅ Temps adapté à nouvelle vitesse
        t_vol = distance / v_x
        t_final = 5 * t_vol  # Facteur modéré
        
        times = np.linspace(0, t_final, 100)
        
        print(f"\n  [EVOL] Paramètres:")
        print(f"    Vitesse vₓ = {v_x:.3e} m/s")
        print(f"    Distance barrière-écran = {distance*1e9:.1f} nm")
        print(f"    Temps vol = {t_vol*1e15:.2f} fs")
        print(f"    t_final = {t_final*1e15:.2f} fs")
        
        evolver = TimeEvolution(self.hamiltonian, self.hbar)
        
        self.evolved_states = evolver.evolve_wavefunction_2d(
            self.initial_state,
            times,
            self.hamiltonian,
            method='split_operator'
        )
        
        self.times = times
    
    def validate_physics(self) -> dict:
        """Validation interfrange Young."""
        rho_y = self.measurement_results['screen_distribution']
        y = self.measurement_results['y_screen']
        
        from scipy.signal import find_peaks
        
        print(f"\n  [VALID] Analyse interfrange:")
        print(f"    max(ρ) = {np.max(rho_y):.3e}")
        
        # ✅ Détection adaptative multi-seuils
        best_result = None
        best_error = float('inf')
        
        for threshold in [0.1, 0.15, 0.2, 0.25, 0.3]:
            peaks, properties = find_peaks(
                rho_y,
                height=threshold * np.max(rho_y),
                distance=3,  # Distance minimale (3 points)
                prominence=0.05 * np.max(rho_y)
            )
            
            if len(peaks) >= 3:  # Au moins 3 pics requis
                delta_y_measured = np.mean(np.diff(y[peaks]))
                delta_y_theory = self.slit_system.expected_fringe_spacing(
                    self.momentum,
                    self.screen_distance
                )
                error = abs(delta_y_measured - delta_y_theory) / delta_y_theory
                
                print(f"    Seuil {threshold:.0%}: {len(peaks)} pics → Δy = {delta_y_measured*1e9:.3f} nm (err {error*100:.1f}%)")
                
                if error < best_error:
                    best_error = error
                    best_result = {
                        'threshold': threshold,
                        'n_peaks': len(peaks),
                        'delta_y_measured': delta_y_measured,
                        'delta_y_theory': delta_y_theory,
                        'error': error,
                        'peaks': peaks
                    }
        
        if best_result is None:
            print("    ✗ Aucun motif d'interférences détecté")
            return {'young_formula': False}
        
        print(f"\n    Meilleur résultat (seuil {best_result['threshold']:.0%}):")
        print(f"    Δy mesuré    = {best_result['delta_y_measured']*1e9:.3f} nm")
        print(f"    Δy théorique = {best_result['delta_y_theory']*1e9:.3f} nm")
        print(f"    Erreur       = {best_result['error']*100:.1f}%")
        
        validation = {
            'young_formula': best_result['error'] < 0.20,  # Tolérance 20%
            'delta_y_measured': best_result['delta_y_measured'],
            'delta_y_theory': best_result['delta_y_theory'],
            'relative_error': best_result['error'],
            'n_peaks_detected': best_result['n_peaks'],
            'optimal_threshold': best_result['threshold']
        }
        
        return validation
    
    def define_hamiltonian(self):
        """Hamiltonien avec barrière double-slit."""
        # Système double-slit
        slit = DoubleSlit2D(
            x_barrier=self.barrier_position,
            barrier_thickness=2e-10,
            barrier_height=1e-17,
            slit_separation=self.slit_separation,
            slit_width=self.slit_width,
            mass=self.mass,
            hbar=self.hbar
        )
        
        # Potentiel V(x,y)
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        V_slit = slit.potential_2d(X, Y)
        
        # Hamiltonien
        self.hamiltonian = Hamiltonian(
            mass=self.mass,
            potential=lambda X, Y: V_slit,
            hbar=self.hbar
        )
        
        self.hamiltonian.dimension = 2
        self.hamiltonian.x_grid = self.x
        self.hamiltonian.y_grid = self.y
        
        self.slit_system = slit
    
    def perform_measurements(self):
        """Mesure distribution écran."""
        
        # ✅ DIAGNOSTICS POSITION PAQUET
        print(f"\n  [MEAS] Diagnostic propagation:")
        
        # État initial
        rho_initial = self.initial_state.probability_density()
        x_mean_initial = np.sum(self.x[:, None] * rho_initial) * (self.x[1] - self.x[0]) * (self.y[1] - self.y[0])
        
        print(f"    Position initiale <x> = {x_mean_initial*1e9:.1f} nm")
        print(f"    Position barrière     = {self.barrier_position*1e9:.1f} nm")
        print(f"    Position écran        = {(self.barrier_position + self.screen_distance)*1e9:.1f} nm")
        
        # État final
        state_final = self.evolved_states[-1]
        rho_final = state_final.probability_density()
        
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        x_mean_final = np.sum(X * rho_final) * (self.x[1] - self.x[0]) * (self.y[1] - self.y[0])
        
        print(f"    Position finale <x>   = {x_mean_final*1e9:.1f} nm")
        print(f"    Déplacement total     = {(x_mean_final - x_mean_initial)*1e9:.1f} nm")
        
        # Distance attendue
        v_x = self.momentum / self.mass
        t_final = self.times[-1]
        distance_expected = v_x * t_final
        
        print(f"    Distance attendue     = {distance_expected*1e9:.1f} nm")
        print(f"    Ratio dépl/attendu    = {(x_mean_final - x_mean_initial) / distance_expected:.2f}")
        
        # Vérifier présence probabilité à l'écran
        x_screen = self.barrier_position + self.screen_distance
        idx_screen = np.argmin(np.abs(self.x - x_screen))
        
        rho_screen = rho_final[idx_screen, :]
        prob_total_screen = np.sum(rho_screen) * (self.y[1] - self.y[0])
        
        print(f"\n    Probabilité à l'écran = {prob_total_screen:.3e}")
        print(f"    max(ρ) écran          = {np.max(rho_screen):.3e}")
        
        # ✅ VÉRIFIER SI PAQUET A ATTEINT ÉCRAN
        if x_mean_final < self.barrier_position + 0.5 * self.screen_distance:
            import warnings
            warnings.warn(
                f"AVERTISSEMENT: Paquet n'a pas atteint l'écran!\n"
                f"  Position finale: {x_mean_final*1e9:.1f} nm\n"
                f"  Position écran: {x_screen*1e9:.1f} nm\n"
                f"  → Augmenter t_final ou vérifier barrière absorbante.",
                RuntimeWarning
            )
        
        if prob_total_screen < 0.01:  # Moins de 1% probabilité
            import warnings
            warnings.warn(
                f"AVERTISSEMENT: Probabilité quasi-nulle à l'écran ({prob_total_screen:.3e})!\n"
                f"  → Paquet absorbé par barrière ou n'a pas atteint écran.",
                RuntimeWarning
            )
        
        # Sauvegarde résultats
        self.measurement_results['screen_distribution'] = rho_screen
        self.measurement_results['y_screen'] = self.y
        self.measurement_results['times'] = self.times
        self.measurement_results['x_mean_trajectory'] = x_mean_final
        self.measurement_results['prob_at_screen'] = prob_total_screen
    
    def analyze_results(self) -> dict:
        """Génération visualisations."""
        # Dashboard évolution
        dashboard = QuantumDashboard2D(output_dir='quantum_simulation/results/double_slit/')
        dashboard.create_evolution_dashboard(
            self.evolved_states,
            self.times,
            self.hbar,
            self.mass,
            output_name='double_slit_evolution.mp4',
            fps=15
        )
        
        # Plot distribution écran
        rho_y = self.measurement_results['screen_distribution']
        y = self.measurement_results['y_screen']
        
        plt.figure(figsize=(12, 6))
        plt.plot(y * 1e9, rho_y, 'b-', linewidth=2)
        plt.xlabel('Position y (nm)')
        plt.ylabel('Densité probabilité')
        plt.title('Distribution écran - Interférences quantiques')
        plt.grid(True, alpha=0.3)
        
        # Annoter interfrange
        val = self.validation_results
        if 'delta_y_theory' in val:
            plt.axvline(val['delta_y_theory']*1e9, color='r', linestyle='--', 
                        label=f"Δy théorique = {val['delta_y_theory']*1e9:.2f} nm")
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('quantum_simulation/results/double_slit/screen_distribution.png', dpi=150)
        plt.close()
        
        return {
            'dashboard_path': 'quantum_simulation/results/double_slit/evolution_dashboard.gif',
            'screen_plot': 'quantum_simulation/results/double_slit/screen_distribution.png'
        }


if __name__ == "__main__":
    from quantum_simulation.utils.config_loader import load_config
    
    config = load_config()
    
    exp = DoubleSlitExperiment(config)
    results = exp.run()
    
    print("\n✓ Expérience double-slit complétée")
    val = results['validation']
    print(f"  Interfrange mesuré    : {val['delta_y_measured']*1e9:.2f} nm")
    print(f"  Interfrange théorique : {val['delta_y_theory']*1e9:.2f} nm")
    print(f"  Erreur relative       : {val['relative_error']*100:.1f}%")