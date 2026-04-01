"""
Expérience : Structure fine du niveau n=2 de l'hydrogène + effet Zeeman.

Démontre :
    - Levée de dégénérescence par la structure fine (W_mv + W_SO + W_D)
    - j est le bon nombre quantique (pas l seul)
    - Dégénérescence résiduelle 2s_{1/2} / 2p_{1/2} (Lamb shift ignoré)
    - Diagramme Zeeman en champ faible : sous-niveaux M_J

Règles R10.1, R10.3 — Source : [Tome 2, Chap. XII]
"""

import numpy as np
from quantum_simulation.experiments.base_experiment import Experiment
from quantum_simulation.systems.hydrogen_structure import HydrogenFineStructure
from quantum_simulation.systems.zeeman_stark import ZeemanEffect
from quantum_simulation.core.angular_momentum import AngularMomentumCoupling
from quantum_simulation.validation.tome2_invariants import PerturbationValidator, ClebschGordanValidator


class HydrogenFineStructureExperiment(Experiment):
    """
    Structure fine n=2 de l'hydrogène avec balayage du champ Zeeman.

    Espace n=2 : 8 états dégénérés dans l'approximation non relativiste.
        2s_{1/2} : l=0, j=1/2, M_J = ±1/2    (2 états)
        2p_{1/2} : l=1, j=1/2, M_J = ±1/2    (2 états)
        2p_{3/2} : l=1, j=3/2, M_J = ±3/2, ±1/2  (4 états)

    Règles R10.1, R10.3
    """

    def __init__(self, config: dict):
        super().__init__(config)
        hfs_cfg = config.get('experiments', {}).get('hydrogen_fine_structure', {})
        phys = config.get('physical_constants', {})

        self.n = hfs_cfg.get('n', 2)
        self.B_max = hfs_cfg.get('B_max', 1.0)  # Tesla
        self.n_B = hfs_cfg.get('n_B_points', 200)

        self.mass = phys.get('m_electron', 9.1093837015e-31)
        self.hbar = phys.get('hbar', 1.054571817e-34)
        self.c = phys.get('c', 2.99792458e8)
        self.e_charge = phys.get('e_charge', 1.602176634e-19)
        self.epsilon_0 = phys.get('epsilon_0', 8.854187817e-12)

        self.hfs = HydrogenFineStructure(
            mass_electron=self.mass, hbar=self.hbar, c=self.c,
            e_charge=self.e_charge, epsilon_0=self.epsilon_0
        )
        self.zeeman = ZeemanEffect(hbar=self.hbar)
        self.B_field_range = np.linspace(0, self.B_max, self.n_B)

        # Définition des 8 sous-niveaux n=2
        # Format : (label, l, j, M_J)
        self.n2_levels_def = [
            ('2s_1/2', 0, 0.5, +0.5),
            ('2s_1/2', 0, 0.5, -0.5),
            ('2p_1/2', 1, 0.5, +0.5),
            ('2p_1/2', 1, 0.5, -0.5),
            ('2p_3/2', 1, 1.5, +1.5),
            ('2p_3/2', 1, 1.5, +0.5),
            ('2p_3/2', 1, 1.5, -0.5),
            ('2p_3/2', 1, 1.5, -1.5),
        ]

        self.fine_structure_energies = {}
        self.zeeman_diagram = {}

    # ------------------------------------------------------------------ #
    # Cycle de vie Experiment                                             #
    # ------------------------------------------------------------------ #

    def prepare_initial_state(self):
        """Définit les 8 états du niveau n=2."""
        E0 = self.hfs.unperturbed_energy(self.n)
        print(f"    E(n={self.n})^0 = {E0 / self.e_charge:.6f} eV")
        print(f"    α² ≈ {HydrogenFineStructure.ALPHA**2:.2e} (régime perturbatif valide)")
        self.initial_state = {
            'n': self.n,
            'E0_eV': E0 / self.e_charge,
            'n_sublevels': 8,
        }

    def define_hamiltonian(self):
        """Calcule la matrice H_SF pour le niveau n=2."""
        # Les éléments sont diagonaux dans la base |n, l, j, M_J⟩
        # (la structure fine ne couple pas les différents (l, j))
        energies_SF = []
        for label, l, j, M_J in self.n2_levels_def:
            E = self.hfs.fine_structure_energy(self.n, l, j)
            energies_SF.append(E)
            self.fine_structure_energies[(label, l, j, M_J)] = E

        self.energies_SF = np.array(energies_SF)
        print(f"    Structure fine calculée pour {len(self.n2_levels_def)} sous-niveaux")

    def evolve_state(self):
        """
        Diagonalise H_SF à B=0 puis balaye B pour le diagramme Zeeman.
        (Pas d'évolution temporelle — problème stationnaire.)
        """
        # Niveaux à B=0 : dégénérescence 2s_{1/2}/2p_{1/2}
        unique_levels = {}
        for (label, l, j, M_J), E in self.fine_structure_energies.items():
            key = (l, j)
            if key not in unique_levels:
                unique_levels[key] = E

        # Diagramme Zeeman : E(B) pour chaque sous-niveau
        for (label, l, j, M_J), E_SF in self.fine_structure_energies.items():
            g_J = self.zeeman.lande_g_factor(l, 0.5, j)
            mu_B = self.zeeman.mu_B
            E_vs_B = E_SF + g_J * mu_B * M_J * self.B_field_range
            key = f"{label}_MJ={M_J:+.1f}"
            self.zeeman_diagram[key] = {
                'l': l, 'j': j, 'M_J': M_J, 'label': label,
                'g_J': g_J,
                'E_vs_B': E_vs_B,
                'E_SF': E_SF,
            }

        self.evolved_states = [
            {'label': f"{l}_{j}_MJ={M_J:+.1f}",
             'E_SF_eV': E / self.e_charge}
            for (label, l, j, M_J), E in self.fine_structure_energies.items()
        ]

    def perform_measurements(self):
        """Calcule les grandeurs physiques clés."""
        eV = self.e_charge
        spectrum = self.hfs.level_n2_spectrum()

        # Écart de structure fine
        E_2p_3_2 = spectrum['2p_3/2']
        E_2p_1_2 = spectrum['2p_1/2']
        E_2s_1_2 = spectrum['2s_1/2']
        delta_FS = E_2p_3_2 - E_2p_1_2

        # Facteurs de Landé
        g_J_half = self.zeeman.lande_g_factor(1, 0.5, 0.5)   # l=1, j=1/2
        g_J_3half = self.zeeman.lande_g_factor(1, 0.5, 1.5)  # l=1, j=3/2

        # Valeur de référence Dirac
        E_dirac_j_half = self.hfs.fine_structure_energy_dirac(self.n, 0.5)
        E_dirac_j_3half = self.hfs.fine_structure_energy_dirac(self.n, 1.5)

        self.measurement_results = {
            'spectrum_eV': {k: v for k, v in spectrum.items() if k.endswith('_eV')},
            'E_2s_1/2_eV': E_2s_1_2 / eV,
            'E_2p_1/2_eV': E_2p_1_2 / eV,
            'E_2p_3/2_eV': E_2p_3_2 / eV,
            'delta_FS_eV': delta_FS / eV,
            'delta_FS_expected_eV': 4.53e-5,  # valeur de référence
            'g_J_2p_1/2': g_J_half,
            'g_J_2p_3/2': g_J_3half,
            'E_dirac_j_half_eV': E_dirac_j_half / eV,
            'E_dirac_j_3half_eV': E_dirac_j_3half / eV,
            '2s_2p_degenerate': abs(E_2s_1_2 - E_2p_1_2) / eV,
        }

        print(f"    ΔE_FS(n=2) = {delta_FS/eV:.4e} eV  (référence : 4.53e-5 eV)")
        print(f"    g_J(2p_{1/2}) = {g_J_half:.4f}, g_J(2p_{3/2}) = {g_J_3half:.4f}")
        print(f"    2s₁/₂ - 2p₁/₂ dégénérescence : {abs(E_2s_1_2-E_2p_1_2)/eV:.2e} eV")

    def validate_physics(self) -> dict:
        """Valide les invariants R10.1, R10.3."""
        pert_validator = PerturbationValidator()
        cg_validator = ClebschGordanValidator()
        eV = self.e_charge

        validations = {}

        # 1. Régime perturbatif : α² ≈ 5e-5 ≪ 1
        E0 = self.hfs.unperturbed_energy(self.n)
        E_2p_3_2 = self.hfs.fine_structure_energy(self.n, 1, 1.5)
        first_order_correction = E_2p_3_2 - E0
        pert_check = pert_validator.perturbative_regime(
            first_order_correction, E0, threshold=0.01
        )
        validations['perturbative_regime'] = pert_check['is_valid']
        validations['perturbative_ratio'] = pert_check['ratio']

        # 2. j est le bon nombre quantique :
        # 2s_{1/2} et 2p_{1/2} ont le même j=1/2 → même énergie corrigée (Dirac)
        E_2s_1_2 = self.hfs.fine_structure_energy(self.n, 0, 0.5)
        E_2p_1_2 = self.hfs.fine_structure_energy(self.n, 1, 0.5)
        # Dans l'approximation Dirac, ils sont dégénérés
        E_dirac_j_half = self.hfs.fine_structure_energy_dirac(self.n, 0.5)
        validations['j_good_quantum_number_2s_2p'] = (
            abs(E_2s_1_2 - E_2p_1_2) / abs(E0) < 1e-6
        )

        # 3. Accord avec la formule de Dirac
        delta_direct = abs(E_2p_3_2 - E_2p_1_2) / eV
        delta_expected = 4.53e-5  # eV
        validations['fine_structure_splitting_correct'] = (
            abs(delta_direct - delta_expected) / delta_expected < 0.05
        )

        # 4. Facteur g de Landé via Clebsch-Gordan (unitarité)
        cg_check = cg_validator.unitarity(0.5, 0.5, tolerance=1e-6)
        validations['clebsch_gordan_unitary'] = cg_check['is_unitary']

        # 5. Facteur g_J(j=3/2, l=1, s=1/2) = 4/3
        g_J_3half = self.zeeman.lande_g_factor(1, 0.5, 1.5)
        validations['lande_g_factor_2p_3/2'] = abs(g_J_3half - 4.0 / 3.0) < 1e-6

        # 6. Facteur g_J(j=1/2, l=1, s=1/2) = 2/3
        g_J_half = self.zeeman.lande_g_factor(1, 0.5, 0.5)
        validations['lande_g_factor_2p_1/2'] = abs(g_J_half - 2.0 / 3.0) < 1e-6

        return validations

    def analyze_results(self) -> dict:
        """Post-traitement : diagramme de niveaux et diagramme Zeeman."""
        eV = self.e_charge
        E0 = self.hfs.unperturbed_energy(self.n)

        # Résumé du spectre
        level_diagram = []
        seen = set()
        for (label, l, j, M_J), E in self.fine_structure_energies.items():
            key = (l, j)
            if key not in seen:
                seen.add(key)
                level_diagram.append({
                    'level': f"{label}",
                    'l': l, 'j': j,
                    'E_eV': E / eV,
                    'correction_eV': (E - E0) / eV,
                })

        # Diagramme Zeeman : quelques sous-niveaux représentatifs
        zeeman_summary = {}
        for key, data in self.zeeman_diagram.items():
            zeeman_summary[key] = {
                'label': data['label'],
                'j': data['j'],
                'M_J': data['M_J'],
                'g_J': data['g_J'],
                'E_B0_eV': data['E_SF'] / eV,
                'E_Bmax_eV': float(data['E_vs_B'][-1]) / eV,
                'slope_eV_per_T': float((data['E_vs_B'][-1] - data['E_vs_B'][0])
                                        / (self.B_max * eV)) if self.B_max > 0 else 0.0,
            }

        return {
            'n': self.n,
            'level_diagram': level_diagram,
            'measurements': self.measurement_results,
            'zeeman_diagram_summary': zeeman_summary,
            'B_field_range': self.B_field_range.tolist(),
            'alpha_squared': HydrogenFineStructure.ALPHA ** 2,
        }
