"""
Couplage de moments angulaires : coefficients de Clebsch-Gordan et base couplée.

Règles implémentées :
    R8.1 - Addition de moments angulaires J = J₁ + J₂
    R8.2 - Coefficients de Clebsch-Gordan ⟨j₁,m₁;j₂,m₂|J,M⟩
    R8.3 - Facteur de Landé g_J et élément de matrice L·S

Sources : Cohen-Tannoudji, Tome 2, Chapitre X
"""

import numpy as np
from typing import Optional


# --------------------------------------------------------------------------- #
# Calcul des coefficients de Clebsch-Gordan                                   #
# --------------------------------------------------------------------------- #


class ClebschGordan:
    """
    Coefficients de Clebsch-Gordan ⟨j₁,m₁;j₂,m₂|J,M⟩.

    Convention de Condon-Shortley (coefficients réels).

    Règle R8.2 — Source : [Tome 2, Chap. X, § B-2]
    """

    @staticmethod
    def coefficient(j1: float, m1: float, j2: float, m2: float,
                    J: float, M: float) -> float:
        """
        Calcule ⟨j₁,m₁;j₂,m₂|J,M⟩.

        Retourne 0 si M ≠ m₁+m₂ ou si la règle du triangle est violée.

        Méthode : formule récursive de Racah si sympy absent, sinon sympy.
        """
        # Règle de sélection M = m1 + m2
        if abs(M - (m1 + m2)) > 1e-10:
            return 0.0
        # Règle du triangle
        if not ClebschGordan._triangle_rule(j1, j2, J):
            return 0.0
        # Bornes m
        if abs(m1) > j1 + 1e-10 or abs(m2) > j2 + 1e-10 or abs(M) > J + 1e-10:
            return 0.0

        try:
            from sympy.physics.quantum.cg import CG
            from sympy import S, Rational, sqrt as sym_sqrt
            val = CG(
                Rational(int(2 * j1), 2), Rational(int(2 * m1), 2),
                Rational(int(2 * j2), 2), Rational(int(2 * m2), 2),
                Rational(int(2 * J), 2), Rational(int(2 * M), 2)
            ).doit()
            return float(val)
        except ImportError:
            return ClebschGordan._recursive(j1, m1, j2, m2, J, M)

    @staticmethod
    def _triangle_rule(j1: float, j2: float, J: float) -> bool:
        """|j₁ − j₂| ≤ J ≤ j₁ + j₂ et J entier ou demi-entier cohérent."""
        if J < abs(j1 - j2) - 1e-10 or J > j1 + j2 + 1e-10:
            return False
        # J doit être entier ou demi-entier selon j1+j2
        expected_int = int(round(2 * (j1 + j2))) % 2
        actual_int = int(round(2 * J)) % 2
        return expected_int == actual_int

    @staticmethod
    def _recursive(j1: float, m1: float, j2: float, m2: float,
                   J: float, M: float) -> float:
        """
        Algorithme récursif via opérateurs échelle (fallback sans sympy).

        Construit la table complète pour (j1, j2) et retourne la valeur demandée.
        """
        table = ClebschGordan._build_table(j1, j2)
        key = (round(2 * m1), round(2 * m2), round(2 * J), round(2 * M))
        return table.get(key, 0.0)

    @staticmethod
    def _build_table(j1: float, j2: float) -> dict:
        """
        Construit toute la table CG pour (j1, j2) par récurrence sur J et M.

        Utilise les relations de récurrence :
            √[(J−M)(J+M+1)] ⟨m1,m2|J,M+1⟩ =
                √[(j1−m1)(j1+m1+1)] ⟨m1+1,m2|J,M⟩ +
                √[(j2−m2)(j2+m2+1)] ⟨m1,m2+1|J,M⟩
        """
        table = {}  # clé : (2m1, 2m2, 2J, 2M) → float

        def _cg(m1_2, m2_2, J_2, M_2):
            return table.get((m1_2, m2_2, J_2, M_2), 0.0)

        def _set(m1_2, m2_2, J_2, M_2, val):
            table[(m1_2, m2_2, J_2, M_2)] = val

        j1_2 = int(round(2 * j1))
        j2_2 = int(round(2 * j2))

        # Itération sur chaque valeur totale J
        J_min_2 = abs(j1_2 - j2_2)
        J_max_2 = j1_2 + j2_2

        for J_2 in range(J_min_2, J_max_2 + 2, 2):
            J_val = J_2 / 2
            # État maximal : M = J, normalisé
            # |J,J⟩ = α |j1,j1;j2,J-j1⟩ + ... (convention Condon-Shortley : coeff>0 pour m1 max)
            # Cas simple : utiliser descente à partir de l'état maximal
            # État |J,J⟩ : seul terme m1+m2=J contribue
            # Initialisation : |J_max, J_max⟩ = |j1,j1;j2,j2⟩ pour J_max = j1+j2
            if J_2 == J_max_2:
                # Un seul terme possible : m1=j1, m2=j2
                _set(j1_2, j2_2, J_2, J_2, 1.0)
            else:
                # Construction de |J,J⟩ par orthogonalité avec |J+2,J⟩ (J+1 en unités entières)
                # Utilisation de la relation d'orthogonalité
                M_2 = J_2
                norm_sq = 0.0
                coeffs = {}
                for m1_2 in range(-j1_2, j1_2 + 2, 2):
                    m2_2 = M_2 - m1_2
                    if abs(m2_2) > j2_2:
                        continue
                    # Coefficient par descente depuis J_2+2
                    # Utiliser les coefficients déjà calculés pour J_2+2
                    # CG pour |J+2, J⟩ → rotation par opérateur J−
                    # Relation : ⟨m1,m2|J,M-1⟩ = ... (Racah)
                    # Fallback: construction directe par récurrence sur M ↓
                    coeffs[(m1_2, m2_2)] = _get_initial_coeff(j1_2, j2_2, J_2, M_2,
                                                               m1_2, m2_2, table)
                # Normalisation
                norm_sq = sum(v ** 2 for v in coeffs.values())
                if norm_sq < 1e-30:
                    continue
                norm = np.sqrt(norm_sq)
                for (m1_2, m2_2), v in coeffs.items():
                    _set(m1_2, m2_2, J_2, J_2, v / norm)

            # Descente par opérateur J− : M de J à -J
            for M_2 in range(J_2 - 2, -J_2 - 2, -2):
                # J−|J,M+1⟩ = √[(J+M+1)(J-M)] |J,M⟩
                Jp1_2 = M_2 + 2
                prefactor_J = np.sqrt(((J_2 + Jp1_2) / 2) * ((J_2 - M_2) / 2))
                if prefactor_J < 1e-15:
                    continue
                for m1_2 in range(-j1_2, j1_2 + 2, 2):
                    m2_2 = M_2 - m1_2
                    m2p_2 = Jp1_2 - m1_2 - 2 + 2  # m2 correspondant à M+1 après descente m1
                    # Deux termes : J− = J1− ⊗ 1 + 1 ⊗ J2−
                    val = 0.0
                    # Terme 1 : J1− agit sur m1+1
                    m1a_2 = m1_2 + 2
                    m2a_2 = Jp1_2 - m1a_2
                    if abs(m1a_2) <= j1_2 and abs(m2a_2) <= j2_2:
                        f1 = np.sqrt(((j1_2 + m1a_2) / 2) * ((j1_2 - m1a_2) / 2 + 1))
                        val += f1 * _cg(m1a_2, m2a_2, J_2, Jp1_2)
                    # Terme 2 : J2− agit sur m2+1
                    m2b_2 = m2_2 + 2
                    m1b_2 = Jp1_2 - m2b_2
                    if abs(m1b_2) <= j1_2 and abs(m2b_2) <= j2_2:
                        f2 = np.sqrt(((j2_2 + m2b_2) / 2) * ((j2_2 - m2b_2) / 2 + 1))
                        val += f2 * _cg(m1b_2, m2b_2, J_2, Jp1_2)
                    _set(m1_2, m2_2, J_2, M_2, val / prefactor_J)

        return table

    @staticmethod
    def table(j1: float, j2: float) -> np.ndarray:
        """
        Matrice de changement de base uncoupled → coupled.

        Retourne array de forme ((2j1+1)*(2j2+1), (2j1+1)*(2j2+1)).
        Lignes indexées par (J,M) dans l'ordre croissant de J puis M.
        Colonnes indexées par (m1,m2) dans l'ordre lexicographique.
        """
        j1_2 = int(round(2 * j1))
        j2_2 = int(round(2 * j2))
        dim = (j1_2 + 1) * (j2_2 + 1)

        # Index colonnes : (m1, m2) ordre croissant
        uncoupled = []
        for m1_2 in range(-j1_2, j1_2 + 2, 2):
            for m2_2 in range(-j2_2, j2_2 + 2, 2):
                uncoupled.append((m1_2, m2_2))

        # Index lignes : (J, M) ordre croissant J, puis M
        coupled = []
        J_min_2 = abs(j1_2 - j2_2)
        J_max_2 = j1_2 + j2_2
        for J_2 in range(J_min_2, J_max_2 + 2, 2):
            for M_2 in range(-J_2, J_2 + 2, 2):
                coupled.append((J_2, M_2))

        mat = np.zeros((dim, dim))
        for row, (J_2, M_2) in enumerate(coupled):
            for col, (m1_2, m2_2) in enumerate(uncoupled):
                mat[row, col] = ClebschGordan.coefficient(
                    j1_2 / 2, m1_2 / 2, j2_2 / 2, m2_2 / 2,
                    J_2 / 2, M_2 / 2
                )
        return mat

    @staticmethod
    def validate_unitarity(j1: float, j2: float, tolerance: float = 1e-10) -> bool:
        """
        Vérifie que la table CG est orthogonale réelle (U^T U = I).

        Invariant R8.2 — Source : [Tome 2, Chap. X, § B-3]
        """
        U = ClebschGordan.table(j1, j2)
        product = U.T @ U
        I = np.eye(product.shape[0])
        return bool(np.max(np.abs(product - I)) < tolerance)

    @staticmethod
    def two_spins_half_table() -> np.ndarray:
        """
        Table CG analytique pour j₁ = j₂ = 1/2 (triplet + singulet).

        Valeurs de référence pour validation.
        """
        # Ordre lignes (J,M) : (1,1), (1,0), (1,-1), (0,0)
        # Ordre colonnes (m1,m2) : (+1/2,+1/2), (+1/2,-1/2), (-1/2,+1/2), (-1/2,-1/2)
        s2 = 1.0 / np.sqrt(2)
        return np.array([
            [1,   0,   0,   0],   # |1,1⟩ = |+,+⟩
            [0,   s2,  s2,  0],   # |1,0⟩ = (|+,-⟩ + |-,+⟩)/√2
            [0,   0,   0,   1],   # |1,-1⟩ = |-,-⟩
            [0,   s2, -s2,  0],   # |0,0⟩ = (|+,-⟩ - |-,+⟩)/√2
        ], dtype=float)

    @staticmethod
    def coupled_basis(j1: float, j2: float) -> dict:
        """
        Retourne {(J, M): {(m1, m2): coeff}} pour tous J, M valides.

        Règle R8.2
        """
        j1_2 = int(round(2 * j1))
        j2_2 = int(round(2 * j2))
        result = {}
        J_min_2 = abs(j1_2 - j2_2)
        J_max_2 = j1_2 + j2_2
        for J_2 in range(J_min_2, J_max_2 + 2, 2):
            for M_2 in range(-J_2, J_2 + 2, 2):
                J_val, M_val = J_2 / 2, M_2 / 2
                decomp = {}
                for m1_2 in range(-j1_2, j1_2 + 2, 2):
                    m2_2 = M_2 - m1_2
                    if abs(m2_2) > j2_2:
                        continue
                    c = ClebschGordan.coefficient(j1, m1_2 / 2, j2, m2_2 / 2, J_val, M_val)
                    if abs(c) > 1e-15:
                        decomp[(m1_2 / 2, m2_2 / 2)] = c
                result[(J_val, M_val)] = decomp
        return result


def _get_initial_coeff(j1_2, j2_2, J_2, M_2, m1_2, m2_2, table):
    """Helper pour construction récursive — retourne coefficient depuis table ou 0."""
    return table.get((m1_2, m2_2, J_2, M_2), 0.0)


# --------------------------------------------------------------------------- #
# Couplage de moments angulaires                                               #
# --------------------------------------------------------------------------- #


class AngularMomentumCoupling:
    """
    Couplage J = J₁ + J₂ : transformations entre bases couplée et découplée.

    Règle R8.1 — Source : [Tome 2, Chap. X, § A]

    Attributs :
        j1 : moment angulaire 1
        j2 : moment angulaire 2
    """

    def __init__(self, j1: float, j2: float):
        self.j1 = j1
        self.j2 = j2
        self._cg_cache: Optional[dict] = None

    def _get_cg_table(self) -> dict:
        if self._cg_cache is None:
            self._cg_cache = ClebschGordan.coupled_basis(self.j1, self.j2)
        return self._cg_cache

    def uncoupled_to_coupled(self, m1: float, m2: float) -> dict:
        """
        Exprime |j₁,m₁;j₂,m₂⟩ dans la base couplée {|J,M⟩}.

        Returns:
            {(J, M): coefficient}
        """
        result = {}
        j1_2 = int(round(2 * self.j1))
        j2_2 = int(round(2 * self.j2))
        M = m1 + m2
        J_min_2 = abs(j1_2 - j2_2)
        J_max_2 = j1_2 + j2_2
        for J_2 in range(J_min_2, J_max_2 + 2, 2):
            J_val = J_2 / 2
            if abs(M) > J_val + 1e-10:
                continue
            c = ClebschGordan.coefficient(self.j1, m1, self.j2, m2, J_val, M)
            if abs(c) > 1e-15:
                result[(J_val, M)] = c
        return result

    def coupled_to_uncoupled(self, J: float, M: float) -> dict:
        """
        Exprime |j₁,j₂;J,M⟩ dans la base découplée {|m₁,m₂⟩}.

        Returns:
            {(m1, m2): coefficient}
        """
        table = self._get_cg_table()
        return dict(table.get((J, M), {}))

    def total_angular_momentum_matrix(self, J: float) -> tuple:
        """
        Matrices J² et J_z dans le sous-espace couplé (J, M).

        Returns:
            (J_squared: ndarray(2J+1,2J+1), J_z: ndarray(2J+1,2J+1))
        """
        dim = int(round(2 * J)) + 1
        J_squared = J * (J + 1) * np.eye(dim)
        M_vals = np.linspace(-J, J, dim)
        J_z = np.diag(M_vals)
        return J_squared, J_z

    def lande_g_factor(self, L: float, S: float, J: float) -> float:
        """
        Facteur de Landé g_J.

        g_J = 1 + [J(J+1) + S(S+1) − L(L+1)] / [2J(J+1)]

        Règle R8.3 — Source : [Tome 2, Chap. X, § C-3]

        Args:
            L : nombre quantique orbital
            S : nombre quantique de spin (1/2 pour électron)
            J : nombre quantique total J = L + S

        Returns:
            g_J (sans unité)
        """
        if J < 1e-10:
            return 0.0  # État J=0 : pas de moment magnétique orbital
        return 1.0 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))

    def spin_orbit_matrix_element(self, n: int, l: float, j: float,
                                  hbar: float = 1.054571817e-34) -> float:
        """
        Élément de matrice diagonal de L·S dans la base |n,l,j⟩.

        ⟨n,l,j|L·S|n,l,j⟩ = (ℏ²/2)[j(j+1) − l(l+1) − s(s+1)]
        avec s = 1/2.

        Règle R8.1 — Source : [Tome 2, Chap. X, § B-1-b]

        Returns:
            Valeur de ⟨L·S⟩ en J·s² (= ℏ²)
        """
        s = 0.5  # spin électron
        return (hbar ** 2 / 2) * (j * (j + 1) - l * (l + 1) - s * (s + 1))
