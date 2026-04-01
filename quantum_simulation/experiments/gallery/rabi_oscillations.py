"""
Expérience : Oscillations de Rabi dans un système à deux niveaux.

Système : atome à deux niveaux |1⟩ (fondamental) et |2⟩ (excité)
en interaction avec un champ oscillant (approximation tournante RWA).

Démontre :
    - Population P₂(t) = (Ω_R/Ω)² sin²(Ω t/2)
    - Inversion complète à résonance (impulsion π)
    - Effet du désaccord δ = ω_drive − ω₀
    - Trajectoire sur la sphère de Bloch

Règle R11.3 — Source : [Tome 2, Chap. XIII, § D]
"""

import numpy as np
from quantum_simulation.experiments.base_experiment import Experiment
from quantum_simulation.dynamics.time_perturbation import RabiOscillations
from quantum_simulation.validation.tome2_invariants import TimeDependentValidator


class RabiOscillationsExperiment(Experiment):
    """
    Oscillations de Rabi pour plusieurs valeurs de désaccord.

    Pour chaque désaccord δ, calcule :
        - P₂(t) analytique via RabiOscillations.population_excited()
        - Trajectoire de Bloch via intégration des équations de Bloch

    Règle R11.3 — Source : [Tome 2, Chap. XIII, § D]
    """

    def __init__(self, config: dict):
        super().__init__(config)
        rabi_cfg = config.get('experiments', {}).get('rabi_oscillations', {})
        phys = config.get('physical_constants', {})

        self.omega_0 = rabi_cfg.get('transition_frequency', 1.0e10)   # rad/s
        self.omega_rabi = rabi_cfg.get('rabi_frequency', 1.0e9)        # rad/s
        self.detunings = rabi_cfg.get('detunings', [0.0, 0.5e9, 1.0e9])
        self.hbar = phys.get('hbar', 1.054571817e-34)

        # Durée : t_final_factor × T_π
        T_pi = np.pi / self.omega_rabi if self.omega_rabi > 1e-20 else 1e-9
        t_factor = rabi_cfg.get('t_final_factor', 10)
        self.t_final = t_factor * T_pi
        self.n_points = rabi_cfg.get('n_points', 500)
        self.t_values = np.linspace(0, self.t_final, self.n_points)

        # Stockage résultats par détuning
        self.rabi_solvers = {}
        self.populations = {}
        self.bloch_trajectories = {}

    # ------------------------------------------------------------------ #
    # Cycle de vie Experiment                                             #
    # ------------------------------------------------------------------ #

    def prepare_initial_state(self):
        """État initial : |1⟩ (fondamental), vecteur de Bloch (0, 0, -1)."""
        self.initial_bloch = np.array([0.0, 0.0, -1.0])
        self.initial_state = {
            'state': 'ground',
            'bloch_vector': self.initial_bloch.tolist(),
            'P2_initial': 0.0,
        }
        print(f"    ω₀ = {self.omega_0:.3e} rad/s")
        print(f"    Ω_R = {self.omega_rabi:.3e} rad/s")
        print(f"    T_π = {np.pi/self.omega_rabi:.3e} s")
        print(f"    Désaccords : {self.detunings}")

    def define_hamiltonian(self):
        """Instancie RabiOscillations pour chaque valeur de désaccord."""
        for delta in self.detunings:
            self.rabi_solvers[delta] = RabiOscillations(
                omega_0=self.omega_0,
                omega_rabi=self.omega_rabi,
                hbar=self.hbar,
                detuning=delta,
            )

    def evolve_state(self):
        """Résout les équations de Bloch pour chaque désaccord."""
        for delta, solver in self.rabi_solvers.items():
            # Population analytique
            P2 = solver.population_excited(self.t_values)
            self.populations[delta] = P2

            # Trajectoire de Bloch (intégration numérique)
            bloch_traj = solver.bloch_vector_evolution(self.initial_bloch, self.t_values)
            self.bloch_trajectories[delta] = bloch_traj

        self.evolved_states = [
            {
                'detuning': delta,
                'P2_max': float(np.max(self.populations[delta])),
                'Omega_R': float(self.rabi_solvers[delta].rabi_frequency_generalized()),
            }
            for delta in self.detunings
        ]

    def perform_measurements(self):
        """Mesure l'amplitude max, la période, et le temps d'impulsion π."""
        results = {}
        for delta, solver in self.rabi_solvers.items():
            Omega_R = solver.rabi_frequency_generalized()
            expected_max = (self.omega_rabi / Omega_R) ** 2 if Omega_R > 1e-20 else 0.0
            actual_max = float(np.max(self.populations[delta]))

            T_pi = solver.pi_pulse_time()
            if np.isfinite(T_pi) and T_pi <= self.t_final:
                idx_pi = np.argmin(np.abs(self.t_values - T_pi))
                P2_at_T_pi = float(self.populations[delta][idx_pi])
            else:
                P2_at_T_pi = None

            results[delta] = {
                'P2_max_analytical': expected_max,
                'P2_max_computed': actual_max,
                'pi_pulse_time': T_pi,
                'P2_at_T_pi': P2_at_T_pi,
                'generalized_rabi_freq': Omega_R,
            }

        self.measurement_results = results
        # Résumé
        delta0 = self.detunings[0]  # Désaccord = 0 si présent
        if delta0 in results:
            print(f"    À résonance (δ=0) : P₂_max = {results[delta0]['P2_max_computed']:.4f}")
            if results[delta0]['P2_at_T_pi'] is not None:
                print(f"    P₂(T_π) = {results[delta0]['P2_at_T_pi']:.4f}")

    def validate_physics(self) -> dict:
        """Valide les invariants R11.3."""
        validator = TimeDependentValidator()
        validations = {}

        for delta in self.detunings:
            P2 = self.populations[delta]
            solver = self.rabi_solvers[delta]
            Omega_R = solver.rabi_frequency_generalized()

            # P₂(t) ∈ [0, 1]
            prob_ok = validator.probability_bounds(P2)

            # Amplitude de Rabi
            amp_ok = validator.rabi_oscillation_amplitude(P2, self.omega_rabi, Omega_R)

            key = f'delta={delta:.2e}'
            validations[key + '_probability_bounds'] = prob_ok
            validations[key + '_amplitude_correct'] = amp_ok

        # Vérification inversion complète à résonance
        if 0.0 in self.detunings:
            P2_res = self.populations[0.0]
            T_pi = np.pi / self.omega_rabi
            if T_pi <= self.t_final:
                idx = np.argmin(np.abs(self.t_values - T_pi))
                P2_T_pi = float(P2_res[idx])
                validations['resonance_full_inversion'] = abs(P2_T_pi - 1.0) < 0.01
            else:
                validations['resonance_full_inversion'] = True

        return validations

    def analyze_results(self) -> dict:
        """Post-traitement : spectre de résonance, comparaison désaccords."""
        # Fréquence de Rabi généralisée vs désaccord
        omega_rabi_vs_delta = {
            delta: float(self.rabi_solvers[delta].rabi_frequency_generalized())
            for delta in self.detunings
        }

        # P_max vs désaccord (spectre Lorentzien)
        P_max_vs_delta = {
            delta: float(np.max(self.populations[delta]))
            for delta in self.detunings
        }

        # Trajectoires de Bloch au dict
        bloch_summary = {
            delta: {
                'initial': self.bloch_trajectories[delta][0].tolist(),
                'final': self.bloch_trajectories[delta][-1].tolist(),
                'min_w': float(np.min(self.bloch_trajectories[delta][:, 2])),
                'max_w': float(np.max(self.bloch_trajectories[delta][:, 2])),
            }
            for delta in self.detunings
        }

        return {
            'detunings': list(self.detunings),
            'omega_rabi': self.omega_rabi,
            'omega_0': self.omega_0,
            'T_pi': float(np.pi / self.omega_rabi),
            'generalized_rabi_freq': omega_rabi_vs_delta,
            'P_max_vs_delta': P_max_vs_delta,
            'bloch_summary': bloch_summary,
            'populations': {delta: P.tolist() for delta, P in self.populations.items()},
            't_values': self.t_values.tolist(),
        }
