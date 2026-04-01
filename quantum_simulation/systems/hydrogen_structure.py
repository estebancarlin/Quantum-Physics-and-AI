"""
Structure fine et hyperfine de l'atome d'hydrogène.

Règles implémentées :
    R10.1 - Structure fine : correction masse-vitesse, couplage spin-orbite, terme de Darwin
    R10.2 - Structure hyperfine : couplage noyau-électron, transition à 21 cm

Sources : Cohen-Tannoudji, Tome 2, Chapitre XII
"""

import numpy as np


class HydrogenFineStructure:
    """
    Corrections de structure fine à l'atome d'hydrogène.

    H = H₀ + W_mv + W_SO + W_D

    W_mv : terme masse-vitesse (relativiste)
    W_SO : couplage spin-orbite L·S
    W_D  : terme de Darwin (contact, l=0 seulement)

    L'énergie corrigée dépend de n et j (pas séparément de l et m_j).

    Règle R10.1 — Source : [Tome 2, Chap. XII]
    """

    # Constante de structure fine (sans unité)
    ALPHA = 7.2973525693e-3

    def __init__(self, mass_electron: float = 9.1093837015e-31,
                 hbar: float = 1.054571817e-34,
                 c: float = 2.99792458e8,
                 e_charge: float = 1.602176634e-19,
                 epsilon_0: float = 8.854187817e-12):
        """
        Args:
            mass_electron : masse de l'électron (kg)
            hbar          : ℏ (J·s)
            c             : vitesse de la lumière (m/s)
            e_charge      : charge élémentaire (C)
            epsilon_0     : permittivité du vide (F/m)
        """
        self.m = mass_electron
        self.hbar = hbar
        self.c = c
        self.e = e_charge
        self.eps0 = epsilon_0

        # Énergie de Rydberg (J) : E_I = m_e e⁴ / (2 ℏ² (4πε₀)²)
        self.a0 = (4 * np.pi * epsilon_0 * hbar ** 2) / (mass_electron * e_charge ** 2)
        self.E_I = hbar ** 2 / (2 * mass_electron * self.a0 ** 2)  # = 13.6 eV en Joules
        self.E_Rydberg = self.E_I  # alias

    def unperturbed_energy(self, n: int) -> float:
        """
        E_n^0 = −E_I / n²  (en Joules)

        Source : [Tome 2, Chap. XII, § A-1]
        """
        return -self.E_I / n ** 2

    def mass_velocity_correction(self, n: int, l: int) -> float:
        """
        ⟨W_mv⟩_{n,l} = −(m_e c² α⁴ / 2) × [1/(n³(l+1/2)) − 3/(4n⁴)]

        Règle R10.1 — Source : [Tome 2, Chap. XII, § B-1]

        Returns:
            Correction (J), négative
        """
        term1 = 1.0 / (n ** 3 * (l + 0.5))
        term2 = 3.0 / (4 * n ** 4)
        return -0.5 * self.m * self.c ** 2 * self.ALPHA ** 4 * (term1 - term2)

    def spin_orbit_correction(self, n: int, l: int, j: float) -> float:
        """
        ⟨W_SO⟩_{n,l,j} = E_I α² / n³ × [j(j+1)−l(l+1)−3/4] / (l(l+1/2)(l+1))

        Retourne 0 pour l=0 (pas de couplage spin-orbite → terme de Darwin).

        Règle R10.1 — Source : [Tome 2, Chap. XII, § B-2]
        """
        if l == 0:
            return 0.0
        numerator = j * (j + 1) - l * (l + 1) - 0.75  # 3/4 = s(s+1) avec s=1/2
        denominator = l * (l + 0.5) * (l + 1)
        return self.E_I * self.ALPHA ** 2 / (2 * n ** 3) * numerator / denominator

    def darwin_correction(self, n: int, l: int) -> float:
        """
        ⟨W_D⟩ = E_I α² / (2n³)  pour l=0, sinon 0.

        Terme de contact : proportionnel à |ψ(0)|² = 1/(πn³ a₀³).

        Règle R10.1 — Source : [Tome 2, Chap. XII, § B-3]
        """
        if l != 0:
            return 0.0
        return self.E_I * self.ALPHA ** 2 / n ** 3

    def fine_structure_energy(self, n: int, l: int, j: float) -> float:
        """
        Énergie totale corrigée :
            E(n,l,j) = E_n^0 + W_mv + W_SO + W_D

        Équivalent à la formule de Dirac au premier ordre en α² :
            E_SF(n,j) = E_n^0 [1 + α²/n² (n/(j+1/2) − 3/4)]

        Règle R10.1 — Source : [Tome 2, Chap. XII, § C]

        Args:
            n : nombre quantique principal
            l : nombre quantique orbital
            j : nombre quantique total j = l ± 1/2

        Returns:
            Énergie (J)
        """
        E0 = self.unperturbed_energy(n)
        Wmv = self.mass_velocity_correction(n, l)
        Wso = self.spin_orbit_correction(n, l, j)
        Wd = self.darwin_correction(n, l)
        return E0 + Wmv + Wso + Wd

    def fine_structure_energy_dirac(self, n: int, j: float) -> float:
        """
        Formule de Dirac (exacte au 1er ordre en α²) :

        E_SF(n,j) = E_n^0 [1 + (α²/n²)(n/(j+1/2) − 3/4)]

        Source : [Tome 2, Chap. XII, § C-2]
        """
        E0 = self.unperturbed_energy(n)
        return E0 * (1.0 + (self.ALPHA ** 2 / n ** 2) * (n / (j + 0.5) - 0.75))

    def level_n2_spectrum(self) -> dict:
        """
        Structure fine du niveau n=2 de l'hydrogène.

        Niveaux :
            2s_{1/2} : l=0, j=1/2
            2p_{1/2} : l=1, j=1/2
            2p_{3/2} : l=1, j=3/2

        Note : 2s_{1/2} et 2p_{1/2} sont dégénérés dans cette approximation
               (le déplacement de Lamb est ignoré).

        Returns:
            dict {label: énergie_J}
        """
        eV = 1.602176634e-19
        return {
            '2s_1/2': self.fine_structure_energy(2, 0, 0.5),
            '2p_1/2': self.fine_structure_energy(2, 1, 0.5),
            '2p_3/2': self.fine_structure_energy(2, 1, 1.5),
            '2s_1/2_eV': self.fine_structure_energy(2, 0, 0.5) / eV,
            '2p_1/2_eV': self.fine_structure_energy(2, 1, 0.5) / eV,
            '2p_3/2_eV': self.fine_structure_energy(2, 1, 1.5) / eV,
            'splitting_eV': (self.fine_structure_energy(2, 1, 1.5) -
                             self.fine_structure_energy(2, 1, 0.5)) / eV,
        }


# --------------------------------------------------------------------------- #
# Structure hyperfine                                                           #
# --------------------------------------------------------------------------- #


class HydrogenHyperfine:
    """
    Structure hyperfine de l'hydrogène : couplage électron-proton.

    H_hf = A I·S  où A est la constante hyperfine.

    Pour l'état 1s : F = 0 (singulet) ou F = 1 (triplet).
    La transition F=1→F=0 donne la raie à 21 cm.

    Règle R10.2 — Source : [Tome 2, Chap. XII, § D]
    """

    def __init__(self, mass_electron: float = 9.1093837015e-31,
                 mass_proton: float = 1.67262192369e-27,
                 hbar: float = 1.054571817e-34,
                 g_proton: float = 5.5857,
                 e_charge: float = 1.602176634e-19,
                 epsilon_0: float = 8.854187817e-12):
        """
        Args:
            mass_electron : m_e (kg)
            mass_proton   : m_p (kg)
            hbar          : ℏ (J·s)
            g_proton      : facteur g du proton (5.5857)
            e_charge      : charge élémentaire (C)
            epsilon_0     : permittivité du vide (F/m)
        """
        self.m_e = mass_electron
        self.m_p = mass_proton
        self.hbar = hbar
        self.g_p = g_proton

        # Constantes dérivées
        ALPHA = 7.2973525693e-3
        self.a0 = (4 * np.pi * epsilon_0 * hbar ** 2) / (mass_electron * e_charge ** 2)
        self.E_I = hbar ** 2 / (2 * mass_electron * self.a0 ** 2)
        self.ALPHA = ALPHA

    def hyperfine_coupling_1s(self) -> float:
        """
        Constante hyperfine pour l'état 1s :

        A = (16/3) g_p (m_e/m_p) α² E_I

        Règle R10.2 — Source : [Tome 2, Chap. XII, § D-2]

        Returns:
            A en Joules
        """
        return (8.0 / 3.0) * self.g_p * (self.m_e / self.m_p) * self.ALPHA ** 2 * self.E_I

    def hyperfine_energy(self, F: int, I: float = 0.5, S: float = 0.5) -> float:
        """
        Énergie hyperfine :
            E_hf(F) = A/2 × [F(F+1) − I(I+1) − S(S+1)]

        Pour hydrogène 1s : I = S = 1/2, F ∈ {0, 1}.

        Règle R10.2

        Args:
            F : nombre quantique total (I + S)
            I : spin nucléaire (1/2 pour proton)
            S : spin électronique (1/2)

        Returns:
            Énergie (J)
        """
        A = self.hyperfine_coupling_1s()
        return A / 2 * (F * (F + 1) - I * (I + 1) - S * (S + 1))

    def hyperfine_splitting(self) -> float:
        """
        Écart hyperfine : ΔE = E_hf(F=1) − E_hf(F=0)

        Returns:
            ΔE (J)
        """
        return self.hyperfine_energy(1) - self.hyperfine_energy(0)

    def transition_frequency_21cm(self) -> float:
        """
        Fréquence de la raie à 21 cm :
            ν = ΔE / h

        Valeur de référence : ν ≈ 1420.405751 MHz.

        Règle R10.2

        Returns:
            ν (Hz)
        """
        h = 2 * np.pi * self.hbar
        delta_E = self.hyperfine_splitting()
        return float(delta_E / h)

    def transition_wavelength_21cm(self) -> float:
        """
        Longueur d'onde : λ = c / ν

        Returns:
            λ (m)
        """
        c = 2.99792458e8
        return float(c / self.transition_frequency_21cm())
