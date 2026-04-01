"""
Effets Zeeman (champ magnétique) et Stark (champ électrique) sur l'hydrogène.

Règles implémentées :
    R10.3 - Effet Zeeman : couplage moment magnétique/champ B
    R10.4 - Effet Stark du niveau n=2 (champ électrique)

Sources : Cohen-Tannoudji, Tome 2, Chapitre XII
"""

import numpy as np
from quantum_simulation.core.angular_momentum import AngularMomentumCoupling


class ZeemanEffect:
    """
    Effet Zeeman sur les niveaux de l'hydrogène.

    Perturbation : W_Z = −μ·B = (μ_B/ℏ)(L + g_S S)·B

    Champ faible (B → 0) : j bon nombre quantique, états |n,l,j,M_J⟩
        ΔE = g_J μ_B M_J B

    Champ fort (Paschen-Back) : l, s bons nombres quantiques
        ΔE = μ_B B (M_L + g_S M_S)

    Règle R10.3 — Source : [Tome 2, Chap. XII, § E]
    """

    def __init__(self, hbar: float = 1.054571817e-34,
                 mu_B: float = 9.2740100783e-24,
                 g_L: float = 1.0,
                 g_S: float = 2.0023193043):
        """
        Args:
            hbar  : ℏ (J·s)
            mu_B  : magnéton de Bohr (J/T)
            g_L   : facteur g orbital (= 1)
            g_S   : facteur g de spin de l'électron (≈ 2.0023)
        """
        self.hbar = hbar
        self.mu_B = mu_B
        self.g_L = g_L
        self.g_S = g_S

    def lande_g_factor(self, L: float, S: float, J: float) -> float:
        """
        Facteur de Landé g_J.

        g_J = 1 + [J(J+1) + S(S+1) − L(L+1)] / [2J(J+1)]

        Source : [Tome 2, Chap. XII, § E-2]
        """
        coupling = AngularMomentumCoupling(L, S)
        return coupling.lande_g_factor(L, S, J)

    def perturbation_matrix_element(self, J: float, M_J: float,
                                     L: float, S: float, B_field: float) -> float:
        """
        ⟨J,M_J|W_Z|J,M_J⟩ = g_J μ_B M_J B  (champ faible)

        Règle R10.3

        Returns:
            Élément de matrice diagonal (J)
        """
        g_J = self.lande_g_factor(L, S, J)
        return float(g_J * self.mu_B * M_J * B_field)

    def weak_field_energies(self, n: int, L: float, J: float,
                             B_field: float,
                             E0: float = 0.0) -> dict:
        """
        Énergies des sous-niveaux en champ faible :
            E(n,L,J,M_J) = E₀ + g_J μ_B M_J B

        Args:
            n       : nombre quantique principal (pour contexte)
            L       : nombre quantique orbital
            J       : nombre quantique total j
            B_field : champ magnétique (T)
            E0      : énergie non perturbée (J)

        Returns:
            {M_J: énergie (J)} pour M_J = −J, −J+1, ..., J
        """
        g_J = self.lande_g_factor(L, 0.5, J)
        J_2 = int(round(2 * J))
        return {
            M_J_2 / 2: E0 + g_J * self.mu_B * (M_J_2 / 2) * B_field
            for M_J_2 in range(-J_2, J_2 + 2, 2)
        }

    def strong_field_energies(self, n: int, L: float, S: float = 0.5,
                               B_field: float = 1.0,
                               E0: float = 0.0) -> dict:
        """
        Énergies en régime Paschen-Back :
            E(M_L, M_S) = E₀ + μ_B B (M_L + g_S M_S)

        Args:
            n       : nombre quantique principal
            L       : nombre quantique orbital
            S       : spin (0.5 pour électron)
            B_field : champ magnétique (T)
            E0      : énergie non perturbée (J)

        Returns:
            {(M_L, M_S): énergie (J)}
        """
        L_2 = int(round(2 * L))
        S_2 = int(round(2 * S))
        result = {}
        for M_L_2 in range(-L_2, L_2 + 2, 2):
            for M_S_2 in range(-S_2, S_2 + 2, 2):
                M_L = M_L_2 / 2
                M_S = M_S_2 / 2
                E = E0 + self.mu_B * B_field * (M_L + self.g_S * M_S)
                result[(M_L, M_S)] = E
        return result

    def zeeman_diagram(self, L: float, J_values: list,
                        B_field_range: np.ndarray,
                        E0_dict: dict = None) -> dict:
        """
        Diagramme Zeeman : énergie des sous-niveaux (J, M_J) en fonction de B.

        Args:
            L            : nombre quantique orbital
            J_values     : liste des valeurs de J (ex. [0.5, 1.5])
            B_field_range: valeurs de champ (T)
            E0_dict      : {J: énergie non perturbée} (None → E0=0 pour tous)

        Returns:
            {(J, M_J): array d'énergies sur B_field_range}
        """
        if E0_dict is None:
            E0_dict = {J: 0.0 for J in J_values}

        result = {}
        for J in J_values:
            g_J = self.lande_g_factor(L, 0.5, J)
            E0 = E0_dict.get(J, 0.0)
            J_2 = int(round(2 * J))
            for M_J_2 in range(-J_2, J_2 + 2, 2):
                M_J = M_J_2 / 2
                energies = E0 + g_J * self.mu_B * M_J * B_field_range
                result[(J, M_J)] = energies
        return result

    def intermediate_field_diagonalization(self, L: float, S: float = 0.5,
                                            B_field: float = 1.0,
                                            fine_structure_energies: dict = None) -> np.ndarray:
        """
        Diagonalisation exacte de H_SF + H_Z dans le sous-espace n.

        Construit la matrice H dans la base |L, M_L; S=1/2, M_S⟩ puis diagonalise.

        Args:
            L                       : nombre quantique orbital
            S                       : spin (0.5)
            B_field                 : champ (T)
            fine_structure_energies : {(L, J, M_J): énergie SF} (None → H_SF = 0)

        Returns:
            Valeurs propres triées (J)
        """
        L_2 = int(round(2 * L))
        S_2 = int(round(2 * S))
        dim = (L_2 + 1) * (S_2 + 1)

        # Base : (M_L, M_S) ordonnée
        basis = []
        for M_L_2 in range(-L_2, L_2 + 2, 2):
            for M_S_2 in range(-S_2, S_2 + 2, 2):
                basis.append((M_L_2 / 2, M_S_2 / 2))

        H = np.zeros((dim, dim), dtype=complex)
        for i, (M_L, M_S) in enumerate(basis):
            # Terme Zeeman diagonal
            H[i, i] += self.mu_B * B_field * (M_L + self.g_S * M_S)

        return np.sort(np.linalg.eigvalsh(H))


class StarkEffect:
    """
    Effet Stark sur le niveau n=2 de l'hydrogène.

    Perturbation : W_Stark = e ε Z  (champ électrique ε selon z)

    Pour n=2, seul l'élément ⟨2s|eεz|2p, m=0⟩ = −3eεa₀ est non nul
    (règles de sélection : Δl = ±1, Δm = 0).

    Règle R10.4 — Source : [Tome 2, Chap. XII, § F]
    """

    def __init__(self, hbar: float = 1.054571817e-34,
                 e_charge: float = 1.602176634e-19,
                 a0: float = 5.29177210903e-11):
        """
        Args:
            hbar     : ℏ (J·s)
            e_charge : charge élémentaire (C)
            a0       : rayon de Bohr (m)
        """
        self.hbar = hbar
        self.e = e_charge
        self.a0 = a0

    def perturbation_matrix_n2(self, electric_field: float) -> np.ndarray:
        """
        Matrice de la perturbation W_Stark dans le sous-espace n=2.

        Base ordonnée : |2s⟩, |2p, m=0⟩, |2p, m=+1⟩, |2p, m=−1⟩

        Seul élément non nul : ⟨2s|eεz|2p,m=0⟩ = −3 e ε a₀

        Règle R10.4 — Source : [Tome 2, Chap. XII, § F-1]

        Returns:
            Matrice 4×4 réelle (J)
        """
        W12 = -3 * self.e * electric_field * self.a0
        W = np.zeros((4, 4), dtype=float)
        W[0, 1] = W12
        W[1, 0] = W12  # Hermitique
        return W

    def stark_energies_n2(self, electric_field: float,
                           E_n2: float = 0.0) -> np.ndarray:
        """
        Diagonalise H₀ + W_Stark dans l'espace n=2.

        Démontre l'effet Stark linéaire : dégénérescence 2s-2p levée au 1er ordre.

        Args:
            electric_field : champ électrique (V/m)
            E_n2           : énergie non perturbée E_{n=2} (J)

        Returns:
            4 valeurs propres triées (J)
        """
        W = self.perturbation_matrix_n2(electric_field)
        H = E_n2 * np.eye(4) + W
        return np.sort(np.linalg.eigvalsh(H))

    def polarizability_1s(self, epsilon_0: float = 8.854187817e-12) -> float:
        """
        Polarisabilité statique de l'état 1s :

        α_pol = (9/2) a₀³ (4πε₀)

        Énergie de Stark au 2ème ordre :
            ΔE = −(1/2) α_pol ε²

        Source : [Tome 2, Chap. XII, § F-2]

        Returns:
            α_pol en C²·s²/kg (= F·m²)
        """
        return float(4.5 * self.a0 ** 3 * 4 * np.pi * epsilon_0)

    def stark_energy_1s_second_order(self, electric_field: float,
                                      epsilon_0: float = 8.854187817e-12) -> float:
        """
        ΔE_Stark(1s) = −(1/2) α_pol ε²  (2ème ordre en ε).

        Returns:
            ΔE (J), négatif
        """
        alpha = self.polarizability_1s(epsilon_0)
        return float(-0.5 * alpha * electric_field ** 2)
