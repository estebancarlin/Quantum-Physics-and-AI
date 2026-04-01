"""
Théorie des perturbations stationnaires (1er et 2ème ordre, cas dégénéré).

Règles implémentées :
    R9.1 - Correction d'énergie au 1er ordre : E_n^(1) = ⟨φ_n|W|φ_n⟩
    R9.2 - Correction d'énergie au 2ème ordre
    R9.3 - Correction d'état au 1er ordre
    R9.4 - Perturbation dégénérée : diagonalisation dans le sous-espace
    R9.5 - Méthode variationnelle : E_var ≥ E_0

Sources : Cohen-Tannoudji, Tome 2, Chapitre XI
"""

import numpy as np
from typing import Callable, Optional
from scipy import optimize


class StationaryPerturbation:
    """
    Théorie des perturbations stationnaires : H = H₀ + λW.

    Opère entièrement dans l'espace des valeurs/vecteurs propres de H₀
    (indépendant du type d'état : spatial, spin, etc.).

    Règles R9.1–R9.4 — Source : [Tome 2, Chap. XI]
    """

    def __init__(self, H0_energies: np.ndarray, H0_states: np.ndarray,
                 W_matrix: np.ndarray, hbar: float = 1.054571817e-34):
        """
        Args:
            H0_energies : shape (N,), valeurs propres de H₀ triées par ordre croissant
            H0_states   : shape (N, N), colonnes = vecteurs propres de H₀
                          H0_states[:, n] = |φ_n⟩
            W_matrix    : shape (N, N), matrice de la perturbation W dans la base
                          de la base position (sera convertie en base propre de H₀)
            hbar        : ℏ (conservé pour usage futur)
        """
        H0_energies = np.asarray(H0_energies, dtype=float)
        H0_states = np.asarray(H0_states, dtype=complex)
        W_matrix = np.asarray(W_matrix, dtype=complex)

        N = len(H0_energies)
        if H0_states.shape != (N, N):
            raise ValueError(f"H0_states doit être ({N},{N}), reçu {H0_states.shape}")
        if W_matrix.shape != (N, N):
            raise ValueError(f"W_matrix doit être ({N},{N}), reçu {W_matrix.shape}")

        # Vérification hermiticité de W
        if np.max(np.abs(W_matrix - W_matrix.conj().T)) > 1e-8 * np.max(np.abs(W_matrix) + 1e-30):
            import warnings
            warnings.warn("W_matrix non hermitique (tolérance 1e-8*||W||)")

        self.H0_energies = H0_energies
        self.H0_states = H0_states
        self.hbar = hbar
        self.N = N

        # Matrice W dans la base propre de H₀ : W_nm = ⟨φ_n|W|φ_m⟩
        self.W_eigenbasis = H0_states.conj().T @ W_matrix @ H0_states

    # ------------------------------------------------------------------ #
    # Corrections d'énergie                                               #
    # ------------------------------------------------------------------ #

    def energy_correction_first_order(self, n: int) -> float:
        """
        E_n^(1) = ⟨φ_n|W|φ_n⟩ = W_nn.

        Règle R9.1 — Source : [Tome 2, Chap. XI, § B-1]
        """
        return float(self.W_eigenbasis[n, n].real)

    def energy_correction_second_order(self, n: int,
                                        n_states: Optional[int] = None,
                                        degeneracy_threshold: float = 1e-30) -> float:
        """
        E_n^(2) = Σ_{p≠n} |W_pn|² / (E_n^0 − E_p^0)

        Règle R9.2 — Source : [Tome 2, Chap. XI, § B-2]

        Args:
            n                    : indice de l'état
            n_states             : nombre d'états dans la somme (None = tous)
            degeneracy_threshold : seuil de quasi-dégénérescence (lève ValueError)

        Returns:
            Correction d'énergie au 2ème ordre (J)
        """
        N = n_states if n_states is not None else self.N
        E_n = self.H0_energies[n]
        result = 0.0
        for p in range(min(N, self.N)):
            if p == n:
                continue
            denom = E_n - self.H0_energies[p]
            if abs(denom) < degeneracy_threshold:
                raise ValueError(
                    f"États n={n} et p={p} quasi-dégénérés : |ΔE| = {abs(denom):.2e} < {degeneracy_threshold}. "
                    "Utiliser degenerate_subspace_correction()."
                )
            result += abs(self.W_eigenbasis[p, n]) ** 2 / denom
        return float(result)

    def corrected_energy(self, n: int, order: int = 2,
                          degeneracy_threshold: float = 1e-30) -> float:
        """
        E_n^corr = E_n^0 + E_n^(1) + E_n^(2)   [order = 2]
                  E_n^corr = E_n^0 + E_n^(1)     [order = 1]
        """
        E = self.H0_energies[n] + self.energy_correction_first_order(n)
        if order >= 2:
            E += self.energy_correction_second_order(n, degeneracy_threshold=degeneracy_threshold)
        return float(E)

    # ------------------------------------------------------------------ #
    # Corrections d'état                                                  #
    # ------------------------------------------------------------------ #

    def state_correction_first_order(self, n: int,
                                      degeneracy_threshold: float = 1e-30) -> np.ndarray:
        """
        |ψ_n^(1)⟩ = Σ_{p≠n} [W_pn / (E_n^0 − E_p^0)] |φ_p⟩

        Retourne les coefficients dans la base de H₀.

        Règle R9.3 — Source : [Tome 2, Chap. XI, § B-3]
        """
        E_n = self.H0_energies[n]
        coeffs = np.zeros(self.N, dtype=complex)
        for p in range(self.N):
            if p == n:
                continue
            denom = E_n - self.H0_energies[p]
            if abs(denom) < degeneracy_threshold:
                continue  # Ignorer états quasi-dégénérés dans la correction d'état
            coeffs[p] = self.W_eigenbasis[p, n] / denom
        return coeffs

    def corrected_state(self, n: int,
                        degeneracy_threshold: float = 1e-30) -> np.ndarray:
        """
        |ψ_n⟩ = |φ_n⟩ + |ψ_n^(1)⟩, renormalisé.

        Retourne les coefficients dans la base de H₀.
        """
        coeffs = np.zeros(self.N, dtype=complex)
        coeffs[n] = 1.0
        coeffs += self.state_correction_first_order(n, degeneracy_threshold)
        norm = np.sqrt(np.sum(np.abs(coeffs) ** 2))
        return coeffs / norm

    # ------------------------------------------------------------------ #
    # Perturbation dégénérée                                              #
    # ------------------------------------------------------------------ #

    def degenerate_subspace_correction(self, degeneracy_indices: list) -> tuple:
        """
        Diagonalise W dans le sous-espace dégénéré.

        W_ij = ⟨φ_n^i|W|φ_n^j⟩ pour i,j dans degeneracy_indices.

        Règle R9.4 — Source : [Tome 2, Chap. XI, § C]

        Args:
            degeneracy_indices : liste des indices des états dégénérés

        Returns:
            (energy_corrections: ndarray, good_eigenstates: ndarray(g, g))
            good_eigenstates[:, k] = k-ème état propre dans le sous-espace
        """
        idx = list(degeneracy_indices)
        g = len(idx)
        W_sub = np.zeros((g, g), dtype=complex)
        for i, ni in enumerate(idx):
            for j, nj in enumerate(idx):
                W_sub[i, j] = self.W_eigenbasis[ni, nj]

        # Diagonalisation hermitiaque
        energy_corrections, good_eigenstates = np.linalg.eigh(W_sub)
        return energy_corrections.real, good_eigenstates

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #

    def validate_perturbative_regime(self, n: int,
                                      threshold: float = 0.1) -> dict:
        """
        Vérifie |W_pn| / |E_n^0 − E_p^0| ≪ 1 pour tout p ≠ n.

        Règle R9.2 — Source : [Tome 2, Chap. XI, § B-2-a]

        Returns:
            {'is_valid': bool, 'max_ratio': float, 'worst_p': int}
        """
        E_n = self.H0_energies[n]
        max_ratio = 0.0
        worst_p = -1
        for p in range(self.N):
            if p == n:
                continue
            denom = abs(E_n - self.H0_energies[p])
            if denom < 1e-30:
                continue
            ratio = abs(self.W_eigenbasis[p, n]) / denom
            if ratio > max_ratio:
                max_ratio = ratio
                worst_p = p
        return {
            'is_valid': max_ratio < threshold,
            'max_ratio': float(max_ratio),
            'worst_p': worst_p,
        }


# --------------------------------------------------------------------------- #
# Méthode variationnelle                                                       #
# --------------------------------------------------------------------------- #


class VariationalMethod:
    """
    Méthode variationnelle : minimise ⟨ψ(α)|H|ψ(α)⟩ pour obtenir une borne
    supérieure de l'énergie de l'état fondamental.

    E_var(α) ≥ E_0  pour tout α.

    Règle R9.5 — Source : [Tome 2, Chap. XI, § D]
    """

    def __init__(self, hamiltonian_expectation_fn: Callable[[np.ndarray], float],
                 n_params: int):
        """
        Args:
            hamiltonian_expectation_fn : callable(params) → ⟨ψ(params)|H|ψ(params)⟩
                                         Doit retourner un float réel
            n_params                   : nombre de paramètres variationnels
        """
        self.hamiltonian_fn = hamiltonian_expectation_fn
        self.n_params = n_params

    def energy_upper_bound(self, params: np.ndarray) -> float:
        """
        ⟨ψ(params)|H|ψ(params)⟩ — borne supérieure de E₀.

        Règle R9.5
        """
        return float(self.hamiltonian_fn(np.asarray(params)))

    def minimize(self, initial_params: np.ndarray,
                 method: str = 'Nelder-Mead',
                 bounds: Optional[list] = None,
                 max_iter: int = 1000) -> dict:
        """
        Minimise l'énergie variationnelle.

        Args:
            initial_params : paramètres initiaux shape (n_params,)
            method         : méthode scipy.optimize.minimize
            bounds         : bornes sur les paramètres (pour méthodes bornées)
            max_iter       : nombre maximum d'itérations

        Returns:
            dict avec 'optimal_params', 'ground_state_energy', 'n_iterations', 'converged'
        """
        options = {'maxiter': max_iter}
        result = optimize.minimize(
            self.hamiltonian_fn,
            x0=np.asarray(initial_params, dtype=float),
            method=method,
            bounds=bounds,
            options=options,
        )
        return {
            'optimal_params': result.x,
            'ground_state_energy': float(result.fun),
            'n_iterations': result.nit,
            'converged': bool(result.success),
            'message': result.message,
        }
