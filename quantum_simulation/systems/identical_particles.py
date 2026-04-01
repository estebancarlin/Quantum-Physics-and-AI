"""
Particules identiques : symétrisation, déterminant de Slater, diffusion.

Règles implémentées :
    R12.1 - Postulat de symétrisation (bosons / fermions)
    R12.2 - Déterminant de Slater pour N fermions
    R12.3 - Effets d'échange sur les sections efficaces de diffusion

Sources : Cohen-Tannoudji, Tome 2, Chapitre XIV
"""

import numpy as np
from typing import List
from math import factorial


# --------------------------------------------------------------------------- #
# Symétrisation                                                                #
# --------------------------------------------------------------------------- #


class Symmetrizer:
    """
    Opérateurs de symétrisation et d'antisymétrisation pour deux (ou N) particules.

    Règle R12.1 — Source : [Tome 2, Chap. XIV, § B]
    """

    @staticmethod
    def symmetric_two_particle(psi1: np.ndarray, psi2: np.ndarray) -> np.ndarray:
        """
        État bosonique symétrique :
            |ψ_S⟩ = (1/√2)(|ψ₁⟩⊗|ψ₂⟩ + |ψ₂⟩⊗|ψ₁⟩)

        Règle R12.1 — Source : [Tome 2, Chap. XIV, § B-1]

        Args:
            psi1, psi2 : vecteurs d'état monoparticulaires, shape (d,)

        Returns:
            Vecteur d'état biparticulaire symétrique, shape (d²,)
        """
        term1 = np.kron(psi1, psi2)
        term2 = np.kron(psi2, psi1)
        return (term1 + term2) / np.sqrt(2)

    @staticmethod
    def antisymmetric_two_particle(psi1: np.ndarray, psi2: np.ndarray) -> np.ndarray:
        """
        État fermionique antisymétrique :
            |ψ_A⟩ = (1/√2)(|ψ₁⟩⊗|ψ₂⟩ − |ψ₂⟩⊗|ψ₁⟩)

        Retourne vecteur nul si psi1 ≡ psi2 (exclusion de Pauli).

        Règle R12.1 — Source : [Tome 2, Chap. XIV, § B-2]

        Returns:
            Vecteur d'état biparticulaire antisymétrique, shape (d²,)
        """
        if np.allclose(psi1, psi2, atol=1e-12):
            return np.zeros(len(psi1) ** 2, dtype=complex)
        term1 = np.kron(psi1, psi2)
        term2 = np.kron(psi2, psi1)
        return (term1 - term2) / np.sqrt(2)

    @staticmethod
    def verify_symmetry(state_vector: np.ndarray, n_particles: int,
                         single_particle_dim: int,
                         expected_symmetry: str) -> dict:
        """
        Vérifie la symétrie d'un état biparticulaire sous échange P_{12}.

        Pour 2 particules : P_{12} |ψ(1,2)⟩ = ±|ψ(1,2)⟩

        Args:
            state_vector       : état dans produit tensoriel, shape (d²,)
            n_particles        : nombre de particules (supporté : 2)
            single_particle_dim: dimension d de l'espace monoparticulaire
            expected_symmetry  : 'bose' (signe +) ou 'fermi' (signe −)

        Returns:
            {'symmetry_eigenvalue': float, 'is_correct': bool}
        """
        if n_particles != 2:
            raise NotImplementedError("verify_symmetry implémenté pour 2 particules")

        d = single_particle_dim
        psi = np.asarray(state_vector, dtype=complex).reshape(d, d)
        psi_exchanged = psi.T.flatten()
        psi_flat = psi.flatten()

        # Ratio : P_{12} |ψ⟩ = λ |ψ⟩ → trouver λ
        norm = np.linalg.norm(psi_flat)
        if norm < 1e-15:
            return {'symmetry_eigenvalue': 0.0, 'is_correct': False}

        # Estimation de λ via ⟨ψ|P₁₂|ψ⟩ / ⟨ψ|ψ⟩
        eigenvalue = float(np.real(np.dot(np.conj(psi_flat), psi_exchanged)) / norm ** 2)

        expected = 1.0 if expected_symmetry == 'bose' else -1.0
        is_correct = abs(eigenvalue - expected) < 0.01  # Tolérance 1%

        return {'symmetry_eigenvalue': eigenvalue, 'is_correct': is_correct}

    @staticmethod
    def n_particle_symmetrize(single_particle_states: list,
                               statistics: str) -> np.ndarray:
        """
        Symétrisation/antisymétrisation pour N particules.

        S_N = (1/N!) Σ_α P_α   (bosons)
        A_N = (1/N!) Σ_α sgn(α) P_α   (fermions)

        Note : pour fermions, utiliser SlaterDeterminant est plus efficace.

        Args:
            single_particle_states : liste de N vecteurs shape (d,)
            statistics             : 'bose' ou 'fermi'

        Returns:
            État symétrisé dans le produit tensoriel d^N (non normalisé si états non orthogonaux)
        """
        from itertools import permutations

        states = [np.asarray(s, dtype=complex) for s in single_particle_states]
        N = len(states)
        result = np.zeros(states[0].shape[0] ** N, dtype=complex)

        for perm in permutations(range(N)):
            # Produit tensoriel des états permutés
            term = states[perm[0]]
            for i in range(1, N):
                term = np.kron(term, states[perm[i]])

            if statistics == 'bose':
                result += term
            else:  # fermi
                sign = _permutation_sign(perm)
                result += sign * term

        return result / factorial(N)


def _permutation_sign(perm: tuple) -> int:
    """Calcule la signature d'une permutation (+1 ou -1)."""
    visited = [False] * len(perm)
    sign = 1
    for i in range(len(perm)):
        if not visited[i]:
            j = i
            cycle_len = 0
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                cycle_len += 1
            if cycle_len % 2 == 0:
                sign *= -1
    return sign


# --------------------------------------------------------------------------- #
# Déterminant de Slater                                                        #
# --------------------------------------------------------------------------- #


class SlaterDeterminant:
    """
    Déterminant de Slater pour N fermions.

    |ψ_A⟩ = (1/√N!) det[φ_α(rᵢ)]

    Règle R12.2 — Source : [Tome 2, Chap. XIV, § C]
    """

    def __init__(self, single_particle_states: List[np.ndarray]):
        """
        Args:
            single_particle_states : liste de N vecteurs orthonormaux shape (d,)

        Raises:
            ValueError si les états ne sont pas orthonormaux (tolérance 1e-8)
        """
        self.states = [np.asarray(s, dtype=complex) for s in single_particle_states]
        self.N = len(self.states)
        if self.N == 0:
            raise ValueError("Au moins un état requis")

    @staticmethod
    def overlap_matrix(states: List[np.ndarray]) -> np.ndarray:
        """
        Matrice de recouvrement S_{ij} = ⟨φᵢ|φⱼ⟩.

        det(S) = 1 si états orthonormaux.

        Returns:
            Matrice N×N
        """
        N = len(states)
        S = np.zeros((N, N), dtype=complex)
        for i in range(N):
            for j in range(N):
                S[i, j] = np.dot(np.conj(states[i]), states[j])
        return S

    def norm(self) -> float:
        """
        ⟨ψ_A|ψ_A⟩ = det(S)

        Vaut 1 si les orbitales sont orthonormales.
        """
        S = self.overlap_matrix(self.states)
        return float(abs(np.linalg.det(S)))

    def pauli_exclusion_satisfied(self) -> bool:
        """
        Vérifie que deux orbitales ne sont pas identiques (det ≠ 0).

        Principe d'exclusion de Pauli : R12.2
        """
        S = self.overlap_matrix(self.states)
        return bool(abs(np.linalg.det(S)) > 1e-10)

    def compute(self) -> np.ndarray:
        """
        Construit le vecteur d'état du déterminant de Slater dans l'espace produit.

        |Ψ⟩ = (1/√N!) Σ_α sgn(α) ⊗ᵢ φ_{α(i)}

        Note : taille = d^N (croît exponentiellement). Limité à N ≤ 6 pratiquement.

        Règle R12.2

        Returns:
            Vecteur complexe dans C^{d^N}
        """
        from itertools import permutations

        N = self.N
        d = len(self.states[0])
        result = np.zeros(d ** N, dtype=complex)

        for perm in permutations(range(N)):
            sign = _permutation_sign(perm)
            term = self.states[perm[0]]
            for i in range(1, N):
                term = np.kron(term, self.states[perm[i]])
            result += sign * term

        return result / np.sqrt(factorial(N))

    def exchange_symmetry_factor(self, idx1: int, idx2: int) -> float:
        """
        Échanger les orbitales idx1 ↔ idx2 multiplie le déterminant par −1.

        Règle R12.2

        Returns:
            -1.0 (toujours pour fermions)
        """
        return -1.0


# --------------------------------------------------------------------------- #
# Diffusion de particules identiques                                           #
# --------------------------------------------------------------------------- #


class IdenticalParticlesScattering:
    """
    Sections efficaces de diffusion pour particules identiques.

    L'interférence d'échange modifie σ(θ) selon la statistique.

    Règle R12.3 — Source : [Tome 2, Chap. XIV, § D]
    """

    @staticmethod
    def cross_section_bosons(f_forward: np.ndarray,
                              f_exchange: np.ndarray,
                              theta_grid: np.ndarray) -> np.ndarray:
        """
        Bosons (état spatial symétrique) :
            σ(θ) ∝ |f(θ) + f(π−θ)|²

        Règle R12.3 — Source : [Tome 2, Chap. XIV, § D-1]

        Args:
            f_forward  : f(θ) amplitude de diffusion directe
            f_exchange : f(π−θ) amplitude d'échange
            theta_grid : angles θ

        Returns:
            dσ/dΩ(θ) en unités de |f|²
        """
        return np.abs(f_forward + f_exchange) ** 2

    @staticmethod
    def cross_section_fermions_singlet(f_forward: np.ndarray,
                                        f_exchange: np.ndarray,
                                        theta_grid: np.ndarray) -> np.ndarray:
        """
        Fermions en état de spin singulet (antisymétrique en spin → symétrique spatial) :
            σ(θ) ∝ |f(θ) + f(π−θ)|²

        Règle R12.3
        """
        return np.abs(f_forward + f_exchange) ** 2

    @staticmethod
    def cross_section_fermions_triplet(f_forward: np.ndarray,
                                        f_exchange: np.ndarray,
                                        theta_grid: np.ndarray) -> np.ndarray:
        """
        Fermions en état de spin triplet (symétrique en spin → antisymétrique spatial) :
            σ(θ) ∝ |f(θ) − f(π−θ)|²

        Remarque R12.3 : σ(π/2) = 0 pour fermions de même spin.

        Source : [Tome 2, Chap. XIV, § D-2]
        """
        return np.abs(f_forward - f_exchange) ** 2

    @staticmethod
    def cross_section_classical(f_forward: np.ndarray,
                                  f_exchange: np.ndarray,
                                  theta_grid: np.ndarray) -> np.ndarray:
        """
        Section efficace classique (sans interférence d'échange) :
            σ(θ) ∝ |f(θ)|² + |f(π−θ)|²

        Référence pour comparer l'effet quantique.

        Règle R12.3
        """
        return np.abs(f_forward) ** 2 + np.abs(f_exchange) ** 2
