"""
Expérience : Diffusion élastique sur un potentiel de Yukawa.

Potentiel : V(r) = −V₀ exp(−r/a) / r

Compare :
    - Méthode des ondes partielles (exacte)
    - Approximation de Born (1er ordre)

Vérifications physiques :
    - Théorème optique (exact pour ondes partielles)
    - 0 ≤ sin²(δₗ) ≤ 1
    - σ(θ) ≥ 0
"""

import numpy as np
from quantum_simulation.experiments.base_experiment import Experiment
from quantum_simulation.dynamics.scattering import PhaseShiftSolver, BornApproximation, CrossSection
from quantum_simulation.validation.tome2_invariants import ScatteringValidator


class ScatteringYukawa(Experiment):
    """
    Diffusion élastique sur le potentiel de Yukawa.

    V(r) = −V₀ exp(−r/a) / r

    Ce potentiel est le limite de champ lointain du potentiel de Coulomb écranté.
    Il converge analytiquement vers l'approximation de Born pour V₀ → 0.

    Règles R6.1–R6.4 — Source : [Tome 2, Chap. VIII, § C-2]
    """

    def __init__(self, config: dict):
        super().__init__(config)
        scatt_cfg = config.get('experiments', {}).get('scattering_yukawa', {})
        phys = config.get('physical_constants', {})

        self.V0 = scatt_cfg.get('V0', 1.6e-18)          # ~10 eV
        self.a = scatt_cfg.get('range_a', 1.0e-10)       # ~ a₀
        self.energy = scatt_cfg.get('energy', 1.6e-19)   # ~1 eV
        self.l_max = scatt_cfg.get('l_max', 5)
        self.n_theta = scatt_cfg.get('n_theta', 300)

        self.mass = phys.get('m_electron', 9.1093837015e-31)
        self.hbar = phys.get('hbar', 1.054571817e-34)

        # Grille radiale : r_min = 1% de a (évite la singularité 1/r de Yukawa),
        # r_max = 30*a (bien au-delà de la portée du potentiel)
        r_min = self.a * 1e-2   # 1% de a — IC ~ r^(l+1) encore valides (r << turning point)
        r_max = 30 * self.a
        self.r_grid = np.array([r_min, r_max])
        self.theta_grid = np.linspace(1e-3, np.pi - 1e-3, self.n_theta)

        # Objets solveurs (initialisés dans define_hamiltonian)
        self.phase_solver = None
        self.born_solver = None
        self.cross_section_obj = None
        self.phase_shifts = None

    # ------------------------------------------------------------------ #
    # Potentiel                                                           #
    # ------------------------------------------------------------------ #

    def _V_yukawa_scalar(self, r: float) -> float:
        """V(r) = −V₀ exp(−r/a) / r (scalaire)"""
        if r < 1e-20:
            return 0.0
        return -self.V0 * np.exp(-r / self.a) / r

    def _V_yukawa_array(self, r: np.ndarray) -> np.ndarray:
        """V(r) vectorisé"""
        r = np.asarray(r)
        safe_r = np.where(r < 1e-20, 1e-20, r)
        return -self.V0 * np.exp(-safe_r / self.a) / safe_r

    # ------------------------------------------------------------------ #
    # Cycle de vie Experiment                                             #
    # ------------------------------------------------------------------ #

    def prepare_initial_state(self):
        """Définit le potentiel et le vecteur d'onde incident."""
        self.k = np.sqrt(2 * self.mass * self.energy) / self.hbar
        print(f"    k = {self.k:.3e} m⁻¹, λ_dB = {2*np.pi/self.k:.3e} m")
        print(f"    V₀ = {self.V0:.3e} J, a = {self.a:.3e} m")
        self.initial_state = {
            'k': self.k,
            'energy_J': self.energy,
            'energy_eV': self.energy / 1.602e-19,
        }

    def define_hamiltonian(self):
        """Instancie PhaseShiftSolver et BornApproximation."""
        self.phase_solver = PhaseShiftSolver(
            mass=self.mass, hbar=self.hbar, energy=self.energy,
            potential=self._V_yukawa_scalar, r_grid=self.r_grid
        )
        self.born_solver = BornApproximation(
            mass=self.mass, hbar=self.hbar, energy=self.energy,
            potential_radial=self._V_yukawa_array
        )

    def evolve_state(self):
        """Calcule les déphasages δₗ pour l = 0..l_max."""
        print(f"    Calcul déphasages l = 0..{self.l_max}...")
        self.phase_shifts = self.phase_solver.compute_all_phase_shifts(
            self.l_max, convergence_threshold=1e-6
        )
        self.cross_section_obj = CrossSection(
            k=self.k,
            phase_shifts=self.phase_shifts,
            hbar=self.hbar,
        )
        self.evolved_states = [{'phase_shifts': self.phase_shifts}]

    def perform_measurements(self):
        """Calcule sections efficaces partielles et Born."""
        # Ondes partielles
        sigma_l = self.cross_section_obj.partial_wave_cross_sections()
        sigma_tot_pw = self.cross_section_obj.total_cross_section()
        sigma_diff_pw = self.cross_section_obj.differential_cross_section(self.theta_grid)
        f_pw = self.cross_section_obj.scattering_amplitude(self.theta_grid)

        # Approximation de Born (rapide avec faible r_max ~ 10a)
        r_max_born = min(10 * self.a, self.r_grid[-1])
        f_born = self.born_solver.scattering_amplitude(self.theta_grid, r_max=r_max_born)
        sigma_diff_born = np.abs(f_born) ** 2
        sigma_tot_born = self.born_solver.total_cross_section(r_max=r_max_born)

        self.measurement_results = {
            'phase_shifts': self.phase_shifts,
            'sigma_partial_wave': sigma_l,
            'sigma_tot_pw': sigma_tot_pw,
            'sigma_diff_pw': sigma_diff_pw,
            'sigma_diff_born': sigma_diff_born,
            'sigma_tot_born': sigma_tot_born,
            'f_pw': f_pw,
            'f_born': f_born,
            'theta_grid': self.theta_grid,
        }

        print(f"    σ_tot (ondes partielles) = {sigma_tot_pw:.3e} m²")
        print(f"    σ_tot (Born)             = {sigma_tot_born:.3e} m²")
        print(f"    Déphasages : " + ", ".join(f"δ_{l}={d:.3f}" for l, d in enumerate(self.phase_shifts)))

    def validate_physics(self) -> dict:
        """Valide les invariants physiques R6.1–R6.4."""
        validator = ScatteringValidator(tolerance=1e-4)

        # Théorème optique (ondes partielles : exact)
        f0_pw = self.cross_section_obj.scattering_amplitude(np.array([0.0]))[0]
        optical_pw = validator.optical_theorem(self.cross_section_obj, f0_pw, self.k)

        # Borne d'unitarité
        unit_ok = validator.unitarity_bound(self.phase_shifts)

        # σ(θ) ≥ 0
        sigma_pos = validator.differential_cs_nonnegative(
            self.measurement_results['sigma_diff_pw']
        )

        # Régime Born
        born_regime = validator.born_approximation_regime(
            self.V0, self.energy, self.a, self.mass, self.hbar
        )

        return {
            'optical_theorem_pw': optical_pw['is_valid'],
            'optical_theorem_relative_error': optical_pw['relative_error'],
            'unitarity_bound': unit_ok,
            'differential_cs_nonnegative': sigma_pos,
            'born_regime_valid': born_regime['is_valid'],
            'born_expansion_parameter': born_regime['expansion_parameter'],
        }

    def analyze_results(self) -> dict:
        """Post-traitement : convergence, comparaison Born/partiel, pic avant."""
        meas = self.measurement_results
        theta = meas['theta_grid']
        sigma_pw = meas['sigma_diff_pw']
        sigma_born = meas['sigma_diff_born']

        # Ratio σ_pw / σ_Born à θ = π/2
        pi_half_idx = np.argmin(np.abs(theta - np.pi / 2))

        ratio_pi_half = (float(sigma_pw[pi_half_idx]) /
                         max(float(sigma_born[pi_half_idx]), 1e-60))

        # Indice de convergence : σ_l / σ_tot
        sigma_l = meas['sigma_partial_wave']
        sigma_tot = meas['sigma_tot_pw']
        convergence = {l: float(sigma_l[l] / max(sigma_tot, 1e-60))
                       for l in range(len(sigma_l))}

        return {
            'sigma_tot_pw': meas['sigma_tot_pw'],
            'sigma_tot_born': meas['sigma_tot_born'],
            'ratio_sigma_pw_born': float(meas['sigma_tot_pw'] / max(meas['sigma_tot_born'], 1e-60)),
            'ratio_at_pi_half': ratio_pi_half,
            'partial_wave_convergence': convergence,
            'forward_peak_ratio': float(sigma_pw[0] / max(sigma_pw[pi_half_idx], 1e-60)),
        }
