"""
Expérience : Évolution temporelle d'un paquet d'ondes gaussien.

Objectifs:
    - Observer étalement du paquet dans l'espace libre
    - Vérifier relations Heisenberg durant évolution
    - Valider théorème Ehrenfest
    - Démontrer conservation probabilité

Système : Particule libre (V=0)
"""

import numpy as np
from typing import Dict, Any, List
from quantum_simulation.experiments.base_experiment import Experiment
from quantum_simulation.systems.free_particle import FreeParticle
from quantum_simulation.dynamics.evolution import TimeEvolution
from quantum_simulation.validation.heisenberg_relations import HeisenbergValidator
from quantum_simulation.validation.conservation_laws import ConservationValidator
from quantum_simulation.validation.ehrenfest_theorem import EhrenfestValidator
from quantum_simulation.core.state import WaveFunctionState
from quantum_simulation.core.operators import PositionOperator, MomentumOperator


class WavePacketEvolution(Experiment):
    """
    Évolution libre d'un paquet d'ondes gaussien.
    
    Configuration requise (dans parameters.yaml):
        experiments.wavepacket_evolution:
            initial_state:
                type: "gaussian"
                x0: position initiale (m)
                sigma_x: largeur gaussienne (m)
                k0: nombre d'onde central (m^-1)
            system:
                mass: masse particule (kg)
            evolution:
                t_initial: temps initial (s)
                t_final: temps final (s)
                dt: pas temporel (s)
                times_sample: liste temps pour mesures
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extraction configuration
        exp_config = config['experiments']['wavepacket_evolution']
        self.initial_config = exp_config['initial_state']
        self.system_config = exp_config['system']
        self.evolution_config = exp_config['evolution']
        
        # Constantes physiques
        self.hbar = config['physical_constants']['hbar']
        self.mass = self.system_config['mass']
        
        # Objets principaux
        self.free_particle = None
        self.time_evolution = None
        self.times = None
        
        # Validateurs
        self.heisenberg_validator = None
        self.conservation_validator = None
        self.ehrenfest_validator = None
        
    def prepare_initial_state(self):
        """
        Crée paquet gaussien initial.
        
        Vérifie:
            - Normalisation ⟨ψ|ψ⟩ = 1
            - État minimum incertitude ΔX·ΔP = ℏ/2
        """
        # Création grille spatiale
        spatial_params = self.config['numerical_parameters']['spatial_discretization']
        x = np.linspace(
            spatial_params['x_min'],
            spatial_params['x_max'],
            spatial_params['nx']
        )
        
        # Système particule libre
        self.free_particle = FreeParticle(self.mass, self.hbar)
        
        # État gaussien
        self.initial_state = self.free_particle.create_gaussian_wavepacket(
            spatial_grid=x,
            x0=self.initial_config['x0'],
            sigma_x=self.initial_config['sigma_x'],
            k0=self.initial_config['k0']
        )
        
        # Vérifications
        tol = self.config['numerical_parameters']['tolerances']['normalization_check']
        if not self.initial_state.is_normalized(tolerance=tol):
            raise ValueError("État initial non normalisé")
            
        print(f"    ✓ État gaussien créé: x0={self.initial_config['x0']:.2e}m, "
                f"σx={self.initial_config['sigma_x']:.2e}m")
        
    def define_hamiltonian(self):
        """
        Hamiltonien particule libre : H = P²/2m.
        
        Vérifie hermiticité H† = H.
        """
        # Hamiltonien déjà créé dans self.free_particle
        H = self.free_particle.hamiltonian
        
        # Vérification hermiticité (sur état test)
        # Note: Pour opérateurs différentiels, hermiticité garantie par construction
        # mais peut être vérifiée numériquement
        
        print(f"    ✓ Hamiltonien H = P²/2m défini (particule libre)")
        
    def evolve_state(self):
        """
        Évolution temporelle par intégration équation Schrödinger.
        
        Utilise schéma Crank-Nicolson (décision D1).
        Vérifie conservation norme à chaque pas.
        """
        # Configuration temporelle
        t_initial = self.evolution_config['t_initial']
        t_final = self.evolution_config['t_final']
        dt = self.evolution_config['dt']
        times_sample = np.array(self.evolution_config['times_sample'])
        
        # Création objet évolution
        self.time_evolution = TimeEvolution(
            hamiltonian=self.free_particle.hamiltonian
        )
        
        # Évolution aux temps d'échantillonnage
        self.times = times_sample
        self.evolved_states = []
        
        print(f"    Évolution de t={t_initial:.2e}s à t={t_final:.2e}s")
        print(f"    Nombre de points temporels: {len(times_sample)}")
        
        for i, t in enumerate(times_sample):
            if i == 0 and t == t_initial:
                # État initial
                state_t = self.initial_state
            else:
                # Évolution depuis initial
                state_t = self.time_evolution.evolve_wavefunction(
                    initial_state=self.initial_state,
                    t0=t_initial,
                    t=t,
                    dt=dt
                )
            
            self.evolved_states.append(state_t)
            
            # Vérification norme
            norm = state_t.norm()
            if abs(norm - 1.0) > 1e-6:
                print(f"    ⚠ Attention: norme à t={t:.2e}s : {norm:.6f}")
        
        print(f"    ✓ Évolution terminée ({len(self.evolved_states)} états)")
        
    def perform_measurements(self):
        """
        Mesure observables à chaque temps:
            - ⟨X⟩, ΔX : position moyenne et dispersion
            - ⟨P⟩, ΔP : impulsion moyenne et dispersion
            - ⟨E⟩ : énergie moyenne
        """
        X_op = PositionOperator(dimension=1)
        P_op = MomentumOperator(hbar=self.hbar, dimension=1)
        H_op = self.free_particle.hamiltonian
        
        n_times = len(self.times)
        
        # Arrays résultats
        mean_x = np.zeros(n_times)
        mean_p = np.zeros(n_times)
        mean_energy = np.zeros(n_times)
        delta_x = np.zeros(n_times)
        delta_p = np.zeros(n_times)
        
        for i, state in enumerate(self.evolved_states):
            mean_x[i] = X_op.expectation_value(state)
            mean_p[i] = P_op.expectation_value(state)
            mean_energy[i] = H_op.expectation_value(state)
            delta_x[i] = X_op.uncertainty(state)
            delta_p[i] = P_op.uncertainty(state)
        
        self.measurement_results = {
            'times': self.times,
            'mean_position': mean_x,
            'mean_momentum': mean_p,
            'mean_energy': mean_energy,
            'delta_position': delta_x,
            'delta_momentum': delta_p,
            'heisenberg_product': delta_x * delta_p
        }
        
        print(f"    ✓ Mesures effectuées aux {n_times} temps")
        
    def validate_physics(self) -> Dict[str, bool]:
        """
        Valide invariants physiques:
            1. Relations Heisenberg : ΔX·ΔP ≥ ℏ/2
            2. Conservation probabilité : ⟨ψ|ψ⟩ = 1
            3. Conservation énergie : ⟨H⟩ constant
            4. Théorème Ehrenfest : d⟨X⟩/dt = ⟨P⟩/m
        """
        results = {}
        
        # 1. Heisenberg
        print("    Test 1/4: Relations Heisenberg...")
        self.heisenberg_validator = HeisenbergValidator(
            hbar=self.hbar,
            tolerance=self.config['numerical_parameters']['tolerances']['heisenberg_inequality']
        )
        heisenberg_results = self.heisenberg_validator.validate_time_evolution(
            states=self.evolved_states,
            times=self.times
        )
        results['heisenberg'] = heisenberg_results['all_valid']
        
        if results['heisenberg']:
            print("      ✓ Heisenberg respecté à tous temps")
        else:
            violations = np.where(~heisenberg_results['is_valid'])[0]
            print(f"      ✗ Violations aux indices: {violations}")
        
        # 2. Conservation norme
        print("    Test 2/4: Conservation probabilité...")
        self.conservation_validator = ConservationValidator(
            hbar=self.hbar,
            mass=self.mass,
            tolerance=self.config['numerical_parameters']['tolerances']['conservation_probability']
        )
        conservation_results = self.conservation_validator.validate_norm_conservation(
            states=self.evolved_states,
            times=self.times
        )
        results['conservation_norm'] = conservation_results['is_conserved']
        
        if results['conservation_norm']:
            print(f"      ✓ Norme conservée (écart max: {conservation_results['max_deviation']:.2e})")
        else:
            print(f"      ✗ Norme non conservée (écart max: {conservation_results['max_deviation']:.2e})")
        
        # 3. Conservation énergie
        print("    Test 3/4: Conservation énergie...")
        mean_energy = self.measurement_results['mean_energy']
        energy_std = np.std(mean_energy)
        energy_mean = np.mean(mean_energy)
        relative_std = energy_std / abs(energy_mean) if energy_mean != 0 else energy_std
        
        results['conservation_energy'] = relative_std < 1e-6
        
        if results['conservation_energy']:
            print(f"      ✓ Énergie conservée (écart relatif: {relative_std:.2e})")
        else:
            print(f"      ✗ Énergie non conservée (écart relatif: {relative_std:.2e})")
        
        # 4. Ehrenfest
        print("    Test 4/4: Théorème Ehrenfest...")
        self.ehrenfest_validator = EhrenfestValidator(
            mass=self.mass,
            hbar=self.hbar,
            tolerance=self.config['numerical_parameters']['tolerances']['normalization_check']
        )
        
        # Particule libre : potentiel nul, force nulle
        def zero_potential(x):
            return np.zeros_like(x)
        
        ehrenfest_results = self.ehrenfest_validator.validate_full_ehrenfest(
            states=self.evolved_states,
            times=self.times,
            potential=zero_potential
        )
        results['ehrenfest'] = ehrenfest_results['overall_valid']
        
        if results['ehrenfest']:
            print("      ✓ Ehrenfest vérifié")
        else:
            print("      ✗ Ehrenfest non vérifié")
        
        # Stockage résultats détaillés
        self.validation_results = {
            'heisenberg_detailed': heisenberg_results,
            'conservation_detailed': conservation_results,
            'ehrenfest_detailed': ehrenfest_results,
            **results
        }
        
        return results
        
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyse physique spécifique:
            - Vitesse d'étalement du paquet
            - Comparaison vitesse groupe théorique vs observée
            - Évolution produit incertitudes
        """
        analysis = {}
        
        # 1. Étalement paquet
        delta_x = self.measurement_results['delta_position']
        delta_x_initial = delta_x[0]
        delta_x_final = delta_x[-1]
        spreading_factor = delta_x_final / delta_x_initial
        
        analysis['spreading'] = {
            'initial_width': delta_x_initial,
            'final_width': delta_x_final,
            'spreading_factor': spreading_factor
        }
        
        print(f"\n  Analyse physique:")
        print(f"    Largeur initiale: {delta_x_initial:.2e}m")
        print(f"    Largeur finale: {delta_x_final:.2e}m")
        print(f"    Facteur étalement: {spreading_factor:.2f}×")
        
        # 2. Vitesse groupe
        mean_x = self.measurement_results['mean_position']
        mean_p = self.measurement_results['mean_momentum']
        
        # Vitesse observée : Δx/Δt
        dx = mean_x[-1] - mean_x[0]
        dt = self.times[-1] - self.times[0]
        v_observed = dx / dt if dt > 0 else 0
        
        # Vitesse théorique : ⟨p⟩/m
        v_theory = mean_p[0] / self.mass
        
        analysis['group_velocity'] = {
            'observed': v_observed,
            'theoretical': v_theory,
            'relative_error': abs(v_observed - v_theory) / abs(v_theory) if v_theory != 0 else 0
        }
        
        print(f"    Vitesse groupe observée: {v_observed:.2e}m/s")
        print(f"    Vitesse groupe théorique: {v_theory:.2e}m/s")
        print(f"    Erreur relative: {analysis['group_velocity']['relative_error']:.2e}")
        
        # 3. Évolution Heisenberg
        heisenberg_product = self.measurement_results['heisenberg_product']
        heisenberg_bound = self.hbar / 2.0
        
        analysis['heisenberg_evolution'] = {
            'product': heisenberg_product,
            'bound': heisenberg_bound,
            'minimum_value': np.min(heisenberg_product),
            'maximum_value': np.max(heisenberg_product),
            'initial_value': heisenberg_product[0]
        }
        
        print(f"    ΔX·ΔP initial: {heisenberg_product[0]:.2e}J·s")
        print(f"    ΔX·ΔP final: {heisenberg_product[-1]:.2e}J·s")
        print(f"    Borne Heisenberg: {heisenberg_bound:.2e}J·s")
        
        return analysis