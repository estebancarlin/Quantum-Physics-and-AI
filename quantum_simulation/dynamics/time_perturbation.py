"""
Perturbations dépendantes du temps : amplitudes de transition, règle d'or de Fermi,
oscillations de Rabi.

Règles implémentées :
    R11.1 - Probabilité de transition au 1er ordre
    R11.2 - Règle d'or de Fermi : Γ = (2π/ℏ)|W_fi|²ρ(E_f)
    R11.3 - Oscillations de Rabi et équations de Bloch

Sources : Cohen-Tannoudji, Tome 2, Chapitre XIII
"""

import numpy as np
from typing import Callable
from scipy import integrate


# --------------------------------------------------------------------------- #
# Perturbations dépendantes du temps                                           #
# --------------------------------------------------------------------------- #


class TimeDependentPerturbation:
    """
    Théorie des perturbations dépendantes du temps au 1er ordre.

    H(t) = H₀ + W(t)

    Amplitude de transition :
        c_f(t) = (1/iℏ) ∫₀^t e^{iω_{fi}t'} W_{fi}(t') dt'

    P_{i→f}(t) = |c_f(t)|²

    Règle R11.1 — Source : [Tome 2, Chap. XIII, § B]
    """

    def __init__(self, H0_energies: np.ndarray, hbar: float = 1.054571817e-34):
        """
        Args:
            H0_energies : valeurs propres de H₀, shape (N,) en Joules
            hbar        : ℏ (J·s)
        """
        self.H0_energies = np.asarray(H0_energies, dtype=float)
        self.hbar = hbar
        N = len(H0_energies)
        # Matrice des fréquences : omega_fi[f, i] = (E_f - E_i)/hbar
        self.omega_fi = np.subtract.outer(H0_energies, H0_energies) / hbar

    def transition_amplitude(self, initial_state_idx: int, final_state_idx: int,
                              t_final: float, W_fi_t: Callable[[float], complex],
                              n_points: int = 1000) -> complex:
        """
        c_f(t) = (1/iℏ) ∫₀^t e^{iω_{fi}t'} W_{fi}(t') dt'

        Règle R11.1

        Args:
            initial_state_idx : indice état initial i
            final_state_idx   : indice état final f
            t_final           : durée d'évolution (s)
            W_fi_t            : callable(t) → ⟨f|W(t)|i⟩ complexe
            n_points          : nombre de points pour l'intégration numérique

        Returns:
            amplitude complexe c_f(t_final)
        """
        omega = self.omega_fi[final_state_idx, initial_state_idx]
        t_vals = np.linspace(0, t_final, n_points)

        integrand_re = np.zeros(n_points)
        integrand_im = np.zeros(n_points)
        for i, t in enumerate(t_vals):
            phase = np.exp(1j * omega * t)
            W_val = W_fi_t(t)
            val = phase * W_val
            integrand_re[i] = val.real
            integrand_im[i] = val.imag

        integral_re = np.trapz(integrand_re, t_vals)
        integral_im = np.trapz(integrand_im, t_vals)
        return complex(integral_re + 1j * integral_im) / (1j * self.hbar)

    def transition_probability_first_order(self, initial_state_idx: int,
                                            final_state_idx: int,
                                            t_final: float,
                                            W_fi_t: Callable[[float], complex],
                                            n_points: int = 1000) -> float:
        """
        P_{i→f}(t) = |c_f(t)|²

        Règle R11.1 — Source : [Tome 2, Chap. XIII, § B-2]

        Returns:
            P ∈ [0, 1] (avertissement si P > 0.3 : perturbation peu valide)
        """
        c_f = self.transition_amplitude(initial_state_idx, final_state_idx,
                                         t_final, W_fi_t, n_points)
        P = float(abs(c_f) ** 2)
        if P > 0.3:
            import warnings
            warnings.warn(f"P = {P:.2f} > 0.3 : approximation perturbative de moins en moins valide")
        return min(P, 1.0)  # Borne physique

    def transition_probability_sinusoidal(self, initial_state_idx: int,
                                           final_state_idx: int,
                                           W_fi_amplitude: float,
                                           omega_perturbation: float,
                                           t_values: np.ndarray) -> np.ndarray:
        """
        Perturbation harmonique W(t) = W₀ e^{±iωt}.

        P(t) = (|W_{fi}|/ℏ)² × F(t, ω_{fi} − ω)
        où F(t, Ω) = sin²(Ωt/2) / (Ω/2)²

        Règle R11.1 — Source : [Tome 2, Chap. XIII, § C-1]

        Returns:
            array de probabilités aux instants t_values
        """
        omega_fi = self.omega_fi[final_state_idx, initial_state_idx]
        Omega = omega_fi - omega_perturbation  # désaccord

        def resonance_fn(t):
            if abs(Omega) < 1e-15:
                return t ** 2
            return (np.sin(Omega * t / 2) / (Omega / 2)) ** 2

        prefactor = (W_fi_amplitude / self.hbar) ** 2
        return prefactor * np.array([resonance_fn(t) for t in t_values])

    def resonance_function(self, Omega: float, t: float) -> float:
        """
        F(t, Ω) = sin²(Ωt/2) / (Ω/2)² → t² quand Ω→0.

        Source : [Tome 2, Chap. XIII, § C-1-b]
        """
        if abs(Omega) < 1e-15:
            return float(t ** 2)
        return float((np.sin(Omega * t / 2) / (Omega / 2)) ** 2)


# --------------------------------------------------------------------------- #
# Règle d'or de Fermi                                                          #
# --------------------------------------------------------------------------- #


class FermiGoldenRule:
    """
    Taux de transition constant dans le régime de longue durée.

    Γ = (2π/ℏ) |W_{fi}|² ρ(E_f)

    Règle R11.2 — Source : [Tome 2, Chap. XIII, § C-2]
    """

    def __init__(self, hbar: float = 1.054571817e-34):
        self.hbar = hbar

    def transition_rate(self, W_fi_squared: float,
                         density_of_states: float) -> float:
        """
        Γ = (2π/ℏ) |W_{fi}|² ρ(E_f)

        Règle R11.2

        Args:
            W_fi_squared      : |⟨f|W|i⟩|² (J²)
            density_of_states : ρ(E_f) (J⁻¹)

        Returns:
            Taux de transition Γ ≥ 0 (s⁻¹)
        """
        rate = (2 * np.pi / self.hbar) * W_fi_squared * density_of_states
        return float(max(rate, 0.0))

    def validate_perturbative_time(self, rate: float, t: float,
                                    threshold: float = 0.1) -> bool:
        """
        Vérifie Γ·t ≪ 1 (régime linéaire P = Γt valide).

        Règle R11.2
        """
        return bool(rate * t < threshold)

    def density_of_states_1d(self, energy: float, mass: float,
                               hbar: float = None, volume: float = 1.0) -> float:
        """
        ρ(E) = (V/π) √(m/(2E)) / ℏ  pour particule libre 1D.

        Returns:
            ρ(E) en J⁻¹ m⁻¹
        """
        if hbar is None:
            hbar = self.hbar
        if energy <= 0:
            return 0.0
        return float(volume / np.pi * np.sqrt(mass / (2 * energy)) / hbar)

    def density_of_states_3d(self, energy: float, mass: float,
                               hbar: float = None, volume: float = 1.0) -> float:
        """
        ρ(E) = (V/2π²) (2m/ℏ²)^{3/2} √E  pour particule libre 3D.

        Returns:
            ρ(E) en J⁻¹ m⁻³ × volume
        """
        if hbar is None:
            hbar = self.hbar
        if energy <= 0:
            return 0.0
        return float(volume / (2 * np.pi ** 2) * (2 * mass / hbar ** 2) ** 1.5 * np.sqrt(energy))

    def compute_spontaneous_emission_rate(self, dipole_matrix_element: complex,
                                           omega_fi: float,
                                           epsilon_0: float = 8.854187817e-12,
                                           c: float = 2.99792458e8) -> float:
        """
        Taux d'émission spontanée (coefficient A d'Einstein) :

        A_{fi} = ω_{fi}³ |d_{fi}|² / (3π ε₀ ℏ c³)

        Application de la règle d'or de Fermi au couplage dipolaire électrique.

        Args:
            dipole_matrix_element : ⟨f|d|i⟩ en C·m
            omega_fi              : fréquence de transition (rad/s)
            epsilon_0             : permittivité du vide (F/m)
            c                     : vitesse de la lumière (m/s)

        Returns:
            A_{fi} en s⁻¹
        """
        d_sq = abs(dipole_matrix_element) ** 2
        return float(omega_fi ** 3 * d_sq / (3 * np.pi * epsilon_0 * self.hbar * c ** 3))


# --------------------------------------------------------------------------- #
# Oscillations de Rabi                                                         #
# --------------------------------------------------------------------------- #


class RabiOscillations:
    """
    Système à deux niveaux dans un champ oscillant (approximation tournante).

    H_eff = (ℏ/2) (δ σ_z − Ω_R σ_x)    [cadre tournant, RWA]

    P₂(t) = (Ω_R/Ω)² sin²(Ω t/2)
    avec Ω = √(Ω_R² + δ²)  (fréquence de Rabi généralisée)

    Règle R11.3 — Source : [Tome 2, Chap. XIII, § D]
    """

    def __init__(self, omega_0: float, omega_rabi: float,
                 hbar: float = 1.054571817e-34, detuning: float = 0.0):
        """
        Args:
            omega_0      : fréquence de transition ω₀ = (E₂−E₁)/ℏ (rad/s)
            omega_rabi   : fréquence de Rabi Ω_R = |W₁₂|/ℏ (rad/s)
            hbar         : ℏ (J·s)
            detuning     : désaccord δ = ω_drive − ω₀ (rad/s)
        """
        self.omega_0 = omega_0
        self.omega_rabi = omega_rabi
        self.hbar = hbar
        self.detuning = detuning
        # Fréquence de Rabi généralisée
        self.omega_generalized = np.sqrt(omega_rabi ** 2 + detuning ** 2)

    def rabi_frequency_generalized(self) -> float:
        """Ω = √(Ω_R² + δ²)"""
        return float(self.omega_generalized)

    def pi_pulse_time(self) -> float:
        """
        Temps de l'impulsion π : T_π = π / Ω_R  [à résonance δ=0].

        Source : [Tome 2, Chap. XIII, § D-2]
        """
        if self.omega_rabi < 1e-20:
            return float('inf')
        return float(np.pi / self.omega_rabi)

    def population_excited(self, t_values: np.ndarray) -> np.ndarray:
        """
        Population de l'état excité :
            P₂(t) = (Ω_R/Ω)² sin²(Ω t/2)

        À résonance (δ=0) : P₂(t) = sin²(Ω_R t/2), max = 1.

        Règle R11.3 — Source : [Tome 2, Chap. XIII, § D-1]

        Returns:
            array P₂(t) ∈ [0, 1]
        """
        t = np.asarray(t_values)
        if self.omega_generalized < 1e-20:
            return np.zeros_like(t)
        amplitude = (self.omega_rabi / self.omega_generalized) ** 2
        return amplitude * np.sin(self.omega_generalized * t / 2) ** 2

    def bloch_vector_evolution(self, initial_bloch: np.ndarray,
                                t_values: np.ndarray) -> np.ndarray:
        """
        Résout les équations de Bloch dans le cadre tournant :

        dR/dt = R × ω_eff
        ω_eff = (Ω_R, 0, −δ)   [RWA]

        Args:
            initial_bloch : vecteur de Bloch initial (3,) = (u, v, w)
                            État fondamental pur : (0, 0, -1)
                            État excité pur      : (0, 0, +1)
            t_values      : instants d'évaluation

        Returns:
            array shape (len(t_values), 3) — trajectoire sur la sphère de Bloch
        """
        omega_eff = np.array([self.omega_rabi, 0.0, -self.detuning])

        def bloch_rhs(t, R):
            return np.cross(R, omega_eff)

        sol = integrate.solve_ivp(
            bloch_rhs,
            t_span=(float(t_values[0]), float(t_values[-1])),
            y0=np.asarray(initial_bloch, dtype=float),
            t_eval=np.asarray(t_values, dtype=float),
            method='RK45',
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success:
            raise RuntimeError(f"Intégration équations de Bloch échouée : {sol.message}")

        return sol.y.T  # shape (n_t, 3)

    def population_from_bloch(self, bloch_trajectory: np.ndarray) -> np.ndarray:
        """
        Extrait P₂(t) depuis la trajectoire de Bloch.

        P₂ = (1 + w) / 2  où w est la composante z du vecteur de Bloch.

        Returns:
            array P₂(t) ∈ [0, 1]
        """
        w = bloch_trajectory[:, 2]
        return (1.0 + w) / 2.0
