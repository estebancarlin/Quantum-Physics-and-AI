"""
Validateurs d'invariants physiques pour les modules du Tome 2.

Six validateurs couvrant :
    - ScatteringValidator     : théorème optique, borne d'unitarité, régime Born
    - SpinValidator           : anticommutation Pauli, S², densité, Bloch
    - ClebschGordanValidator  : unitarité, règle M, règle triangle, valeurs analytiques
    - PerturbationValidator   : corrections réelles, signe E⁽²⁾₀, borne variationnelle
    - TimeDependentValidator  : P(t) ∈ [0,1], Γ ≥ 0, amplitude Rabi
    - SymmetrizationValidator : symétrie, normalisation Slater, exclusion Pauli, σ(π/2)=0

Sources : Cohen-Tannoudji, Tome 2
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantum_simulation.core.spin import SpinHalf, SpinOperators, SpinDensityMatrix
    from quantum_simulation.dynamics.scattering import CrossSection
    from quantum_simulation.systems.identical_particles import SlaterDeterminant


# --------------------------------------------------------------------------- #
# Diffusion                                                                    #
# --------------------------------------------------------------------------- #


class ScatteringValidator:
    """Invariants physiques pour les modules de diffusion (Chapitre VIII)."""

    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance

    def optical_theorem(self, cross_section, f_forward: complex,
                         k: float) -> dict:
        """
        Vérifie σ_tot = (4π/k) Im[f_k(0)].

        Invariant R6.1 — Source : [Tome 2, Chap. VIII, § B-2-b]

        Returns:
            dict avec sigma_tot, optical_estimate, relative_error, is_valid
        """
        sigma_tot = cross_section.total_cross_section()
        optical_estimate = (4 * np.pi / k) * f_forward.imag
        rel_error = (abs(sigma_tot - optical_estimate) / abs(sigma_tot)
                     if abs(sigma_tot) > 1e-30 else float('inf'))
        return {
            'sigma_tot': sigma_tot,
            'optical_estimate': optical_estimate,
            'relative_error': rel_error,
            'is_valid': rel_error < self.tolerance,
        }

    def unitarity_bound(self, phase_shifts: np.ndarray) -> bool:
        """
        0 ≤ sin²(δₗ) ≤ 1 pour tout l.

        Invariant R6.1
        """
        sin2 = np.sin(phase_shifts) ** 2
        return bool(np.all(sin2 >= -1e-10) and np.all(sin2 <= 1 + 1e-10))

    def differential_cs_nonnegative(self, sigma_diff: np.ndarray) -> bool:
        """σ(θ) ≥ 0 partout."""
        return bool(np.all(sigma_diff >= -1e-12))

    def born_approximation_regime(self, V0: float, energy: float,
                                   range_a: float, mass: float,
                                   hbar: float = 1.054571817e-34) -> dict:
        """
        Vérifie la validité de Born : (m |V₀| a²/ℏ²) ≪ 1.

        Returns:
            dict avec expansion_parameter, is_valid
        """
        eps = mass * abs(V0) * range_a ** 2 / hbar ** 2
        return {'expansion_parameter': float(eps), 'is_valid': bool(eps < 1.0)}


# --------------------------------------------------------------------------- #
# Spin                                                                         #
# --------------------------------------------------------------------------- #


class SpinValidator:
    """Invariants physiques pour le spin-1/2 (Chapitre IX)."""

    def __init__(self, hbar: float = 1.054571817e-34, tolerance: float = 1e-12):
        self.hbar = hbar
        self.tolerance = tolerance

    def pauli_anticommutation(self, spin_ops) -> bool:
        """
        {σ_i, σ_j} = 2δ_{ij}I pour toutes paires.

        Invariant R7.1
        """
        return bool(spin_ops.validate_pauli_anticommutation(self.tolerance))

    def spin_squared_eigenvalue(self, state) -> bool:
        """
        ⟨S²⟩ = (3/4)ℏ² pour tout état de spin-1/2.

        Invariant R7.1
        """
        S_sq = (3 / 4) * self.hbar ** 2 * np.eye(2, dtype=complex)
        actual = state.expectation_value_matrix(S_sq).real
        expected = 0.75 * self.hbar ** 2
        return bool(abs(actual - expected) < self.tolerance * abs(expected + 1e-40))

    def density_matrix_valid(self, rho) -> dict:
        """
        Vérifie Tr(ρ)=1, ρ†=ρ, valeurs propres ≥ 0, |P| ≤ 1.

        Invariant R7.2
        """
        result = rho.validate()
        P = rho.bloch_vector()
        bloch_valid = bool(np.linalg.norm(P) <= 1.0 + 1e-10)
        result['bloch_valid'] = bloch_valid
        result['all_valid'] = all(result.values())
        return result

    def bloch_vector_bound(self, state) -> bool:
        """|n|² ≤ 1 pour tout état pur."""
        n = state.to_bloch_vector()
        return bool(np.linalg.norm(n) <= 1.0 + 1e-10)


# --------------------------------------------------------------------------- #
# Clebsch-Gordan                                                               #
# --------------------------------------------------------------------------- #


class ClebschGordanValidator:
    """Invariants pour les coefficients de Clebsch-Gordan (Chapitre X)."""

    def unitarity(self, j1: float, j2: float,
                   tolerance: float = 1e-10) -> dict:
        """
        La table CG est orthogonale réelle : U^T U = I.

        Invariant R8.2
        """
        from quantum_simulation.core.angular_momentum import ClebschGordan
        U = ClebschGordan.table(j1, j2)
        product = U.T @ U
        I = np.eye(product.shape[0])
        max_error = float(np.max(np.abs(product - I)))
        return {'is_unitary': max_error < tolerance, 'max_error': max_error}

    def selection_rule_M(self, j1: float, j2: float) -> bool:
        """
        CG = 0 quand M ≠ m₁ + m₂.

        Règle de sélection R8.2
        """
        from quantum_simulation.core.angular_momentum import ClebschGordan
        j1_2 = int(round(2 * j1))
        j2_2 = int(round(2 * j2))
        J_min_2 = abs(j1_2 - j2_2)
        J_max_2 = j1_2 + j2_2
        for J_2 in range(J_min_2, J_max_2 + 2, 2):
            for M_2 in range(-J_2, J_2 + 2, 2):
                for m1_2 in range(-j1_2, j1_2 + 2, 2):
                    for m2_2 in range(-j2_2, j2_2 + 2, 2):
                        if abs((m1_2 + m2_2) - M_2) > 1:  # M ≠ m1 + m2
                            c = ClebschGordan.coefficient(
                                j1_2 / 2, m1_2 / 2, j2_2 / 2, m2_2 / 2,
                                J_2 / 2, M_2 / 2
                            )
                            if abs(c) > 1e-12:
                                return False
        return True

    def triangle_rule(self, j1: float, j2: float, J: float) -> bool:
        """|j₁ − j₂| ≤ J ≤ j₁ + j₂."""
        from quantum_simulation.core.angular_momentum import ClebschGordan
        return ClebschGordan._triangle_rule(j1, j2, J)

    def known_two_spin_half(self, tolerance: float = 1e-10) -> dict:
        """
        Vérifie les coefficients CG analytiques pour j₁ = j₂ = 1/2.

        Valeurs de référence (convention Condon-Shortley) :
            ⟨+1/2,+1/2|1,1⟩ = 1
            ⟨+1/2,-1/2|1,0⟩ = 1/√2
            ⟨-1/2,+1/2|1,0⟩ = 1/√2
            ⟨-1/2,-1/2|1,-1⟩ = 1
            ⟨+1/2,-1/2|0,0⟩ = 1/√2
            ⟨-1/2,+1/2|0,0⟩ = -1/√2
        """
        from quantum_simulation.core.angular_momentum import ClebschGordan
        s2 = 1.0 / np.sqrt(2)
        reference = {
            (0.5, 0.5, 1.0, 1.0): 1.0,
            (0.5, -0.5, 1.0, 0.0): s2,
            (-0.5, 0.5, 1.0, 0.0): s2,
            (-0.5, -0.5, 1.0, -1.0): 1.0,
            (0.5, -0.5, 0.0, 0.0): s2,
            (-0.5, 0.5, 0.0, 0.0): -s2,
        }
        errors = {}
        for (m1, m2, J, M), expected in reference.items():
            computed = ClebschGordan.coefficient(0.5, m1, 0.5, m2, J, M)
            errors[(m1, m2, J, M)] = abs(computed - expected)

        max_error = max(errors.values())
        return {
            'all_correct': max_error < tolerance,
            'max_error': max_error,
            'errors': errors,
        }


# --------------------------------------------------------------------------- #
# Perturbations stationnaires                                                  #
# --------------------------------------------------------------------------- #


class PerturbationValidator:
    """Invariants pour la théorie des perturbations stationnaires (Chapitre XI)."""

    def energy_corrections_real(self, corrections: np.ndarray) -> bool:
        """Toutes les corrections d'énergie doivent être réelles (W hermitique)."""
        return bool(np.all(np.abs(np.imag(corrections)) < 1e-10))

    def second_order_ground_state_negative(self, e2: float) -> bool:
        """
        E⁽²⁾₀ ≤ 0 pour l'état fondamental.

        Règle R9.2 : la correction du 2ème ordre de l'état fondamental est
        toujours négative (tous les dénominateurs E₀ − E_p < 0).
        """
        return bool(e2 <= 1e-10)

    def variational_bound(self, variational_energy: float,
                           true_ground_energy: float,
                           tolerance: float = 1e-8) -> bool:
        """
        E_var ≥ E₀ (avec tolérance numérique).

        Règle R9.5
        """
        return bool(variational_energy >= true_ground_energy - tolerance)

    def perturbative_regime(self, first_order: float,
                             unperturbed: float,
                             threshold: float = 0.1) -> dict:
        """
        Vérifie |E⁽¹⁾| / |E⁽⁰⁾| < threshold.

        Returns:
            {'ratio': float, 'is_valid': bool}
        """
        if abs(unperturbed) < 1e-30:
            return {'ratio': float('inf'), 'is_valid': False}
        ratio = abs(first_order) / abs(unperturbed)
        return {'ratio': float(ratio), 'is_valid': bool(ratio < threshold)}


# --------------------------------------------------------------------------- #
# Perturbations dépendantes du temps                                           #
# --------------------------------------------------------------------------- #


class TimeDependentValidator:
    """Invariants pour les perturbations dépendantes du temps (Chapitre XIII)."""

    def probability_bounds(self, P_values: np.ndarray) -> bool:
        """
        0 ≤ P(t) ≤ 1 pour toutes valeurs.

        Invariant R11.1
        """
        P = np.asarray(P_values)
        return bool(np.all(P >= -1e-8) and np.all(P <= 1.0 + 1e-8))

    def fermi_rate_nonnegative(self, rate: float) -> bool:
        """Γ ≥ 0. Invariant R11.2"""
        return bool(rate >= 0.0)

    def rabi_oscillation_amplitude(self, P_excited: np.ndarray,
                                    omega_rabi: float,
                                    omega_R: float) -> bool:
        """
        Max(P₂) = (Ω_R/Ω)² ≤ 1.

        À résonance (Ω = Ω_R) : max = 1.

        Invariant R11.3
        """
        if omega_R < 1e-20:
            return True
        expected_max = (omega_rabi / omega_R) ** 2
        actual_max = float(np.max(P_excited))
        return bool(abs(actual_max - expected_max) < 0.01 + 0.01 * expected_max)

    def linear_regime_check(self, rate: float, t_max: float,
                              threshold: float = 0.1) -> bool:
        """
        Γ × t_max < threshold (régime linéaire de Fermi valide).

        Invariant R11.2
        """
        return bool(rate * t_max < threshold)


# --------------------------------------------------------------------------- #
# Symétrisation                                                                #
# --------------------------------------------------------------------------- #


class SymmetrizationValidator:
    """Invariants pour les particules identiques (Chapitre XIV)."""

    def symmetry_check(self, state_vector: np.ndarray, n_particles: int,
                        single_particle_dim: int, expected: str) -> dict:
        """
        Vérifie P₁₂ |ψ⟩ = +|ψ⟩ (bosons) ou −|ψ⟩ (fermions).

        Invariant R12.1

        Returns:
            {'symmetry_eigenvalue': float, 'is_correct': bool}
        """
        from quantum_simulation.systems.identical_particles import Symmetrizer
        return Symmetrizer.verify_symmetry(state_vector, n_particles,
                                            single_particle_dim, expected)

    def slater_normalization(self, slater_det) -> bool:
        """
        ⟨ψ_A|ψ_A⟩ = 1 pour orbitales orthonormales.

        Invariant R12.2
        """
        return bool(abs(slater_det.norm() - 1.0) < 1e-8)

    def pauli_exclusion(self, slater_det) -> bool:
        """
        Pas deux orbitales identiques (det ≠ 0).

        Invariant R12.2 — Principe d'exclusion de Pauli
        """
        return bool(slater_det.pauli_exclusion_satisfied())

    def fermion_triplet_zero_at_pi_half(self, sigma_triplet: np.ndarray,
                                         theta_grid: np.ndarray,
                                         tolerance: float = 0.01) -> dict:
        """
        σ_fermions_triplet(θ=π/2) ≈ 0.

        Invariant R12.3 — Source : [Tome 2, Chap. XIV, § D-2]

        Returns:
            {'pi_half_value': float, 'is_zero': bool}
        """
        # Trouver l'indice le plus proche de π/2
        pi_half_idx = np.argmin(np.abs(theta_grid - np.pi / 2))
        val = float(sigma_triplet[pi_half_idx])
        return {'pi_half_value': val, 'is_zero': abs(val) < tolerance * np.max(sigma_triplet + 1e-30)}
