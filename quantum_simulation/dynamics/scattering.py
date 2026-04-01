"""
Diffusion élastique : déphasages, approximation de Born, sections efficaces.

Règles implémentées :
    R6.1 - Section efficace totale et différentielle (ondes partielles)
    R6.2 - Approximation de Born
    R6.3 - Amplitude de diffusion par ondes partielles
    R6.4 - Déphasages δₗ par intégration radiale

Sources : Cohen-Tannoudji, Tome 2, Chapitre VIII
"""

import numpy as np
from typing import Callable
from scipy import integrate, special


# --------------------------------------------------------------------------- #
# Solveur de déphasages                                                        #
# --------------------------------------------------------------------------- #


class PhaseShiftSolver:
    """
    Résout l'équation de Schrödinger radiale et extrait les déphasages δₗ.

    Équation radiale : u_l'' + [k² − l(l+1)/r² − U(r)] u_l = 0
    avec U(r) = 2m V(r)/ℏ².

    Matching asymptotique :
        u_l(r) ~ A_l [j_l(kr) cos δ_l − n_l(kr) sin δ_l]

    Règle R6.4 — Source : [Tome 2, Chap. VIII, § B-3]
    """

    def __init__(self, mass: float, hbar: float, energy: float,
                 potential: Callable[[float], float],
                 r_grid: np.ndarray):
        """
        Args:
            mass      : masse particule (kg)
            hbar      : constante de Planck réduite (J·s)
            energy    : énergie incidente (J), doit être > 0
            potential : V(r) scalaire, ex. Yukawa -V0*exp(-r/a)/r
            r_grid    : grille radiale [r_min, r_max], r_min > 0
        """
        if energy <= 0:
            raise ValueError("Énergie incidente doit être > 0")
        self.mass = mass
        self.hbar = hbar
        self.energy = energy
        self.potential = potential
        self.r_grid = r_grid
        self.k = np.sqrt(2 * mass * energy) / hbar  # vecteur d'onde (m⁻¹)
        self._U_cache: dict = {}

    def _U(self, r: float) -> float:
        """Potentiel réduit U(r) = 2m V(r)/ℏ²"""
        return 2 * self.mass * self.potential(r) / self.hbar ** 2

    def _radial_ode(self, r: float, y: np.ndarray, l: int) -> np.ndarray:
        """
        Système du premier ordre pour [u_l, u_l'].

        u_l'' = [l(l+1)/r² + U(r) − k²] u_l
        """
        u, du = y
        if r < 1e-20:
            r = 1e-20
        d2u = (l * (l + 1) / r ** 2 + self._U(r) - self.k ** 2) * u
        return [du, d2u]

    def compute_phase_shift(self, l: int) -> float:
        """
        Calcule le déphasage δₗ pour le moment orbital l.

        Méthode :
            1. Intègre l'ODE radiale de r_min à r_max
            2. Extrait δₗ via le Wronskien avec j_l, n_l à grande distance

        Règle R6.4

        Returns:
            δₗ en radians ∈ (−π/2, π/2)
        """
        r_min = self.r_grid[0]
        r_max = self.r_grid[-1]

        # Conditions initiales : u_l ~ r^{l+1}, normalisées à u=1 pour éviter l'underflow
        # (le déphasage ne dépend que du rapport u'/u, l'amplitude est irrelevante)
        u0 = 1.0
        du0 = (l + 1) / r_min

        sol = integrate.solve_ivp(
            self._radial_ode,
            t_span=(r_min, r_max),
            y0=[u0, du0],
            args=(l,),
            method='LSODA',
            rtol=1e-6,
            atol=1e-8,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Intégration ODE radiale échouée pour l={l}: {sol.message}")

        # Méthode de la dérivée logarithmique (plus stable que le Wronskien 2-points)
        # f = u'/u  à r = r_max
        r = sol.t[-1]
        u = sol.y[0, -1]
        du = sol.y[1, -1]
        if abs(u) < 1e-40:
            return 0.0
        f = du / u   # dérivée logarithmique

        kr = self.k * r
        jl  = special.spherical_jn(l, kr)
        jlp = special.spherical_jn(l, kr, derivative=True)   # d/d(kr) j_l(kr)
        nl  = special.spherical_yn(l, kr)
        nlp = special.spherical_yn(l, kr, derivative=True)

        # (r·j_l)' = j_l + r·k·j_l'
        rj_prime = jl + r * self.k * jlp
        rn_prime = nl + r * self.k * nlp

        # tan δ = [f·(r·j_l) − (r·j_l)'] / [f·(r·n_l) − (r·n_l)']
        numer = f * (r * jl) - rj_prime
        denom = f * (r * nl) - rn_prime

        if abs(denom) < 1e-30:
            return 0.0

        delta_l = np.arctan2(numer, denom)
        return float(delta_l)

    def compute_all_phase_shifts(self, l_max: int,
                                 convergence_threshold: float = 1e-6) -> np.ndarray:
        """
        Calcule δₗ pour l = 0, 1, ..., l_max.

        S'arrête tôt si |δₗ| < convergence_threshold.

        Returns:
            array shape (l_max+1,)
        """
        deltas = np.zeros(l_max + 1)
        for l in range(l_max + 1):
            deltas[l] = self.compute_phase_shift(l)
            if l > 0 and abs(deltas[l]) < convergence_threshold:
                break  # Convergence atteinte
        return deltas


# --------------------------------------------------------------------------- #
# Approximation de Born                                                        #
# --------------------------------------------------------------------------- #


class BornApproximation:
    """
    Amplitude de diffusion et section efficace en approximation de Born.

    f_Born(θ) = −(m/2πℏ²) ∫ e^{iK·r} V(r) d³r

    Pour V sphériquement symétrique :
        f_Born(θ) = −(2m/ℏ²) ∫₀^∞ V(r) sin(Kr)/(Kr) r² dr
    avec |K| = 2k sin(θ/2).

    Règle R6.2 — Source : [Tome 2, Chap. VIII, § C-2]
    """

    def __init__(self, mass: float, hbar: float, energy: float,
                 potential_radial: Callable[[np.ndarray], np.ndarray]):
        """
        Args:
            mass             : masse (kg)
            hbar             : ℏ (J·s)
            energy           : énergie (J)
            potential_radial : V(r) où r est tableau de valeurs radiales
        """
        self.mass = mass
        self.hbar = hbar
        self.energy = energy
        self.k = np.sqrt(2 * mass * energy) / hbar
        self.potential_radial = potential_radial
        self._prefactor = -self.mass / (2 * np.pi * hbar ** 2)

    def _fourier_transform(self, K: float, r_max: float = 1e-8,
                            n_points: int = 5000) -> float:
        """
        FT sphérique : ∫₀^∞ V(r) sin(Kr)/(Kr) r² dr via intégration numérique.
        """
        r = np.linspace(1e-15, r_max, n_points)
        V = self.potential_radial(r)
        if K < 1e-15:
            # Limite K→0 : sin(Kr)/(Kr) → 1
            integrand = V * r ** 2
        else:
            integrand = V * r ** 2 * np.sinc(K * r / np.pi)  # sinc(x) = sin(πx)/(πx)
            # Correction : np.sinc(x) = sin(πx)/(πx), on veut sin(Kr)/(Kr)
            # → utiliser directement
            integrand = V * r ** 2 * np.where(K * r < 1e-15, 1.0, np.sin(K * r) / (K * r))
        return float(np.trapz(integrand, r))

    def scattering_amplitude(self, theta_grid: np.ndarray,
                              r_max: float = 1e-8) -> np.ndarray:
        """
        f_Born(θ) = −(2m/ℏ²) ∫₀^∞ V(r) sin(Kr)/(Kr) r² dr
        avec K = 2k sin(θ/2).

        Args:
            theta_grid : angles de diffusion ∈ (0, π)

        Returns:
            array complexe de même taille que theta_grid (réel pour V réel)
        """
        prefactor = -self.mass / (2 * np.pi * self.hbar ** 2) * 4 * np.pi
        # f_Born = -(m/2πℏ²) * 4π ∫ V(r) sin(Kr)/(K) r dr
        # Équivalent : f_Born = -(2m/ℏ²) ∫ V(r) sin(Kr)/K r² dr
        amplitudes = np.zeros(len(theta_grid))
        for i, theta in enumerate(theta_grid):
            K = 2 * self.k * np.sin(theta / 2)
            ft = self._fourier_transform(K, r_max)
            amplitudes[i] = -(2 * self.mass / self.hbar ** 2) * ft
        return amplitudes.astype(complex)

    def differential_cross_section(self, theta_grid: np.ndarray,
                                    r_max: float = 1e-8) -> np.ndarray:
        """
        dσ/dΩ (θ) = |f_Born(θ)|²

        Règle R6.2

        Returns:
            array réel ≥ 0
        """
        f = self.scattering_amplitude(theta_grid, r_max)
        return np.abs(f) ** 2

    def total_cross_section(self, n_theta: int = 500,
                             r_max: float = 1e-8) -> float:
        """
        σ_tot = 2π ∫₀^π (dσ/dΩ)(θ) sin θ dθ

        Returns:
            Section efficace totale (m²)
        """
        theta = np.linspace(1e-4, np.pi - 1e-4, n_theta)
        dsigma = self.differential_cross_section(theta, r_max)
        integrand = dsigma * np.sin(theta)
        return float(2 * np.pi * np.trapz(integrand, theta))

    def optical_theorem_check(self, r_max: float = 1e-8,
                               tolerance: float = 0.05) -> dict:
        """
        Vérifie le théorème optique : σ_tot = (4π/k) Im[f(0)].

        Note : en approximation de Born pure, f est réel → violation attendue.

        Returns:
            dict avec sigma_tot, optical_theorem_value, relative_error
        """
        sigma_tot = self.total_cross_section(r_max=r_max)
        f0 = self.scattering_amplitude(np.array([1e-6]), r_max)[0]
        optical_value = (4 * np.pi / self.k) * f0.imag
        if abs(sigma_tot) < 1e-30:
            rel_error = float('inf')
        else:
            rel_error = abs(sigma_tot - optical_value) / abs(sigma_tot)
        return {
            'sigma_tot': sigma_tot,
            'optical_theorem_value': optical_value,
            'relative_error': rel_error,
            'satisfies_theorem': rel_error < tolerance,
        }


# --------------------------------------------------------------------------- #
# Section efficace par ondes partielles                                        #
# --------------------------------------------------------------------------- #


class CrossSection:
    """
    Section efficace exacte par développement en ondes partielles.

    Amplitude de diffusion :
        f_k(θ) = (1/k) Σₗ (2l+1) e^{iδₗ} sin(δₗ) Pₗ(cos θ)

    Section efficace :
        σₗ = (4π/k²)(2l+1) sin²(δₗ)
        σ_tot = Σₗ σₗ

    Règles R6.1, R6.3 — Source : [Tome 2, Chap. VIII, § B]
    """

    def __init__(self, k: float, phase_shifts: np.ndarray,
                 hbar: float = 1.054571817e-34):
        """
        Args:
            k            : vecteur d'onde (m⁻¹)
            phase_shifts : δₗ pour l = 0, 1, ..., (radians)
            hbar         : ℏ (non utilisé directement ici, conservé pour compatibilité)
        """
        if k <= 0:
            raise ValueError("k doit être > 0")
        self.k = k
        self.phase_shifts = np.asarray(phase_shifts, dtype=float)
        self.hbar = hbar
        self.l_max = len(phase_shifts) - 1

    def partial_wave_cross_sections(self) -> np.ndarray:
        """
        σₗ = (4π/k²)(2l+1) sin²(δₗ) pour l = 0, ..., l_max.

        Règle R6.1

        Returns:
            array shape (l_max+1,)
        """
        l_vals = np.arange(len(self.phase_shifts))
        return (4 * np.pi / self.k ** 2) * (2 * l_vals + 1) * np.sin(self.phase_shifts) ** 2

    def total_cross_section(self) -> float:
        """
        σ_tot = Σₗ σₗ

        Règle R6.1
        """
        return float(np.sum(self.partial_wave_cross_sections()))

    def scattering_amplitude(self, theta_grid: np.ndarray) -> np.ndarray:
        """
        f_k(θ) = (1/k) Σₗ (2l+1) e^{iδₗ} sin(δₗ) Pₗ(cos θ)

        Règle R6.3

        Returns:
            array complexe, même taille que theta_grid
        """
        cos_theta = np.cos(theta_grid)
        f = np.zeros(len(theta_grid), dtype=complex)
        for l, delta_l in enumerate(self.phase_shifts):
            Pl = special.eval_legendre(l, cos_theta)
            f += (2 * l + 1) * np.exp(1j * delta_l) * np.sin(delta_l) * Pl
        f /= self.k
        return f

    def differential_cross_section(self, theta_grid: np.ndarray) -> np.ndarray:
        """
        dσ/dΩ(θ) = |f_k(θ)|²

        Returns:
            array réel ≥ 0
        """
        return np.abs(self.scattering_amplitude(theta_grid)) ** 2

    def optical_theorem_check(self, tolerance: float = 1e-6) -> dict:
        """
        Vérifie σ_tot = (4π/k) Im[f_k(0)].

        Pour les ondes partielles, ce théorème est exact.

        Invariant R6.1 — Source : [Tome 2, Chap. VIII, § B-2-b]

        Returns:
            dict avec sigma_tot, optical_theorem_value, relative_error, is_valid
        """
        sigma_tot = self.total_cross_section()
        f0 = self.scattering_amplitude(np.array([0.0]))[0]
        optical_value = (4 * np.pi / self.k) * f0.imag
        if abs(sigma_tot) < 1e-30:
            rel_error = float('inf')
        else:
            rel_error = abs(sigma_tot - optical_value) / abs(sigma_tot)
        return {
            'sigma_tot': sigma_tot,
            'optical_theorem_value': optical_value,
            'relative_error': rel_error,
            'is_valid': rel_error < tolerance,
        }
