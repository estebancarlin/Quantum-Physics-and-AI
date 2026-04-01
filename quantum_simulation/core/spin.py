"""
Spin-1/2 : matrices de Pauli, état spineur, matrice densité de spin.

Règles implémentées :
    R7.1 - Matrices de Pauli et opérateurs de spin
    R7.2 - Matrice densité de spin et vecteur de Bloch

Sources : Cohen-Tannoudji, Tome 2, Chapitre IX
"""

import numpy as np
from quantum_simulation.core.state import QuantumState
from quantum_simulation.core.operators import Observable


class SpinHalf(QuantumState):
    """
    État de spin-1/2 : |χ⟩ = α|+⟩ + β|−⟩.

    Règle R7.1 — Source : [Tome 2, Chap. IX, § A]

    Attributs :
        coefficients : np.ndarray shape (2,) complexe128
                       Index 0 = amplitude spin haut |+⟩
                       Index 1 = amplitude spin bas |−⟩
        hbar : constante de Planck réduite (J·s)
    """

    def __init__(self, coefficients: np.ndarray, hbar: float = 1.054571817e-34):
        coefficients = np.asarray(coefficients, dtype=complex)
        if coefficients.shape != (2,):
            raise ValueError(f"SpinHalf requiert coefficients shape (2,), reçu {coefficients.shape}")
        self.coefficients = coefficients
        self.hbar = hbar

    # ------------------------------------------------------------------ #
    # Contrat QuantumState                                                 #
    # ------------------------------------------------------------------ #

    def norm(self) -> float:
        """√(|α|² + |β|²)"""
        return float(np.sqrt(np.sum(np.abs(self.coefficients) ** 2)))

    def normalize(self) -> "SpinHalf":
        """Retourne nouvel état normé."""
        n = self.norm()
        if n < 1e-15:
            raise ValueError("État spin nul, normalisation impossible")
        return SpinHalf(self.coefficients / n, self.hbar)

    def inner_product(self, other: "SpinHalf") -> complex:
        """⟨other|self⟩ = conj(other.coeff) · self.coeff"""
        return complex(np.conj(other.coefficients) @ self.coefficients)

    # ------------------------------------------------------------------ #
    # Méthodes spin spécifiques                                            #
    # ------------------------------------------------------------------ #

    def expectation_value_matrix(self, matrix_2x2: np.ndarray) -> complex:
        """⟨χ|M|χ⟩ = coeff† M coeff"""
        return complex(np.conj(self.coefficients) @ matrix_2x2 @ self.coefficients)

    def to_bloch_vector(self) -> np.ndarray:
        """
        Vecteur de Bloch n = (⟨σₓ⟩, ⟨σᵧ⟩, ⟨σ_z⟩).

        Pour état pur : |n| = 1.
        Règle R7.2 — Source : [Tome 2, Chap. IX, § B]
        """
        SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
        SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
        nx = self.expectation_value_matrix(SIGMA_X).real
        ny = self.expectation_value_matrix(SIGMA_Y).real
        nz = self.expectation_value_matrix(SIGMA_Z).real
        return np.array([nx, ny, nz])

    # ------------------------------------------------------------------ #
    # Constructeurs alternatifs                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def spin_up(cls, hbar: float = 1.054571817e-34) -> "SpinHalf":
        """État propre |+⟩ de S_z."""
        return cls(np.array([1.0, 0.0], dtype=complex), hbar)

    @classmethod
    def spin_down(cls, hbar: float = 1.054571817e-34) -> "SpinHalf":
        """État propre |−⟩ de S_z."""
        return cls(np.array([0.0, 1.0], dtype=complex), hbar)

    @classmethod
    def from_bloch_angles(cls, theta: float, phi: float,
                          hbar: float = 1.054571817e-34) -> "SpinHalf":
        """
        |χ⟩ = cos(θ/2)|+⟩ + e^{iφ} sin(θ/2)|−⟩

        Args:
            theta : angle polaire ∈ [0, π]
            phi   : angle azimutal ∈ [0, 2π]
        """
        alpha = np.cos(theta / 2)
        beta = np.exp(1j * phi) * np.sin(theta / 2)
        return cls(np.array([alpha, beta], dtype=complex), hbar)

    def __repr__(self) -> str:
        a, b = self.coefficients
        return f"SpinHalf([{a:.4f}, {b:.4f}])"


# --------------------------------------------------------------------------- #
# Opérateurs de spin                                                           #
# --------------------------------------------------------------------------- #


class SpinOperators(Observable):
    """
    Opérateurs de spin-1/2 : S_x, S_y, S_z, S², S±.

    Règle R7.1 — Source : [Tome 2, Chap. IX, § A-2]

    Matrices de Pauli (attributs de classe) :
        SIGMA_X = [[0,1],[1,0]]
        SIGMA_Y = [[0,-i],[i,0]]
        SIGMA_Z = [[1,0],[0,-1]]
    """

    SIGMA_X: np.ndarray = np.array([[0, 1], [1, 0]], dtype=complex)
    SIGMA_Y: np.ndarray = np.array([[0, -1j], [1j, 0]], dtype=complex)
    SIGMA_Z: np.ndarray = np.array([[1, 0], [0, -1]], dtype=complex)
    SIGMA_PLUS: np.ndarray = np.array([[0, 1], [0, 0]], dtype=complex)   # (σx + iσy)/2
    SIGMA_MINUS: np.ndarray = np.array([[0, 0], [1, 0]], dtype=complex)  # (σx - iσy)/2

    def __init__(self, hbar: float = 1.054571817e-34):
        self.hbar = hbar

    # ------------------------------------------------------------------ #
    # Matrices S_i = (ℏ/2) σ_i                                           #
    # ------------------------------------------------------------------ #

    @property
    def S_x(self) -> np.ndarray:
        return (self.hbar / 2) * self.SIGMA_X

    @property
    def S_y(self) -> np.ndarray:
        return (self.hbar / 2) * self.SIGMA_Y

    @property
    def S_z(self) -> np.ndarray:
        return (self.hbar / 2) * self.SIGMA_Z

    @property
    def S_squared(self) -> np.ndarray:
        """S² = (3/4)ℏ² I pour spin-1/2"""
        return (3 / 4) * self.hbar ** 2 * np.eye(2, dtype=complex)

    @property
    def S_plus(self) -> np.ndarray:
        return self.hbar * self.SIGMA_PLUS

    @property
    def S_minus(self) -> np.ndarray:
        return self.hbar * self.SIGMA_MINUS

    def _matrix_for(self, component: str) -> np.ndarray:
        mapping = {
            'x': self.S_x, 'y': self.S_y, 'z': self.S_z,
            'plus': self.S_plus, 'minus': self.S_minus,
            'squared': self.S_squared,
        }
        if component not in mapping:
            raise ValueError(f"Composante '{component}' inconnue. Choisir parmi {list(mapping)}")
        return mapping[component]

    # ------------------------------------------------------------------ #
    # Contrat Observable                                                   #
    # ------------------------------------------------------------------ #

    def apply(self, state: SpinHalf, component: str = 'z') -> SpinHalf:
        """Applique S_component |χ⟩"""
        M = self._matrix_for(component)
        return SpinHalf(M @ state.coefficients, state.hbar)

    def expectation_value(self, state: SpinHalf, component: str = 'z') -> float:
        """⟨S_i⟩ = ⟨χ|S_i|χ⟩"""
        M = self._matrix_for(component)
        return state.expectation_value_matrix(M).real

    def uncertainty(self, state: SpinHalf, component: str = 'z') -> float:
        """ΔS_i = √(⟨S_i²⟩ − ⟨S_i⟩²)"""
        M = self._matrix_for(component)
        exp_S = state.expectation_value_matrix(M).real
        exp_S2 = state.expectation_value_matrix(M @ M).real
        variance = max(0.0, exp_S2 - exp_S ** 2)
        return float(np.sqrt(variance))

    def eigensystem(self, component: str = 'z') -> tuple:
        """
        Valeurs et vecteurs propres de S_component.

        Returns:
            (eigenvalues: ndarray(2,), eigenstates: list[SpinHalf])
        """
        M = self._matrix_for(component)
        vals, vecs = np.linalg.eigh(M)
        states = [SpinHalf(vecs[:, i], self.hbar) for i in range(2)]
        return vals, states

    # ------------------------------------------------------------------ #
    # Validation physique                                                  #
    # ------------------------------------------------------------------ #

    def validate_pauli_anticommutation(self, tolerance: float = 1e-14) -> bool:
        """
        Vérifie {σ_i, σ_j} = 2δ_{ij}I pour toutes paires (i,j).

        Invariant R7.1 — Source : [Tome 2, Chap. IX, § A-2-b]
        """
        matrices = [self.SIGMA_X, self.SIGMA_Y, self.SIGMA_Z]
        I2 = np.eye(2, dtype=complex)
        for i, si in enumerate(matrices):
            for j, sj in enumerate(matrices):
                anticomm = si @ sj + sj @ si
                expected = 2 * (1.0 if i == j else 0.0) * I2
                if np.max(np.abs(anticomm - expected)) > tolerance:
                    return False
        return True

    def validate_commutation_relations(self, tolerance: float = 1e-28) -> bool:
        """
        Vérifie [S_x, S_y] = iℏS_z et permutations cycliques.

        Invariant R7.1 — Source : [Tome 2, Chap. IX, § A-3]
        """
        Sx, Sy, Sz = self.S_x, self.S_y, self.S_z
        checks = [
            (Sx @ Sy - Sy @ Sx, 1j * self.hbar * Sz),
            (Sy @ Sz - Sz @ Sy, 1j * self.hbar * Sx),
            (Sz @ Sx - Sx @ Sz, 1j * self.hbar * Sy),
        ]
        for comm, expected in checks:
            if np.max(np.abs(comm - expected)) > tolerance:
                return False
        return True


# --------------------------------------------------------------------------- #
# Matrice densité de spin                                                      #
# --------------------------------------------------------------------------- #


class SpinDensityMatrix:
    """
    Matrice densité pour état de spin-1/2 (pur ou mélange).

    ρ = (1/2)(I + P·σ) avec |P| ≤ 1.

    Règle R7.2 — Source : [Tome 2, Chap. IX, § C]
    """

    def __init__(self, density_matrix: np.ndarray):
        dm = np.asarray(density_matrix, dtype=complex)
        if dm.shape != (2, 2):
            raise ValueError(f"Matrice densité spin doit être 2×2, reçu {dm.shape}")
        self.density_matrix = dm
        # Validation légère à la construction
        err = self.validate()
        if not all(err.values()):
            import warnings
            warnings.warn(f"Matrice densité physiquement invalide : {err}")

    @classmethod
    def from_pure_state(cls, state: SpinHalf) -> "SpinDensityMatrix":
        """ρ = |χ⟩⟨χ|"""
        psi = state.normalize().coefficients
        dm = np.outer(psi, np.conj(psi))
        return cls(dm)

    @classmethod
    def from_bloch_vector(cls, polarization: np.ndarray) -> "SpinDensityMatrix":
        """
        ρ = (I + P·σ)/2  avec |P| ≤ 1.

        Args:
            polarization : vecteur (Px, Py, Pz), |P| ≤ 1
        """
        P = np.asarray(polarization, dtype=float)
        if P.shape != (3,):
            raise ValueError("Vecteur de polarisation doit être (3,)")
        if np.linalg.norm(P) > 1.0 + 1e-10:
            raise ValueError(f"|P| = {np.linalg.norm(P):.4f} > 1 interdit")
        I2 = np.eye(2, dtype=complex)
        SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
        SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
        dm = 0.5 * (I2 + P[0] * SIGMA_X + P[1] * SIGMA_Y + P[2] * SIGMA_Z)
        return cls(dm)

    def bloch_vector(self) -> np.ndarray:
        """P_i = Tr(ρ σ_i)"""
        SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
        SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
        Px = np.trace(self.density_matrix @ SIGMA_X).real
        Py = np.trace(self.density_matrix @ SIGMA_Y).real
        Pz = np.trace(self.density_matrix @ SIGMA_Z).real
        return np.array([Px, Py, Pz])

    def purity(self) -> float:
        """Tr(ρ²) ∈ [1/2, 1]. Vaut 1 pour état pur."""
        return float(np.trace(self.density_matrix @ self.density_matrix).real)

    def expectation_value(self, matrix_2x2: np.ndarray) -> float:
        """⟨A⟩ = Tr(ρ A)"""
        return float(np.trace(self.density_matrix @ matrix_2x2).real)

    def validate(self, tolerance: float = 1e-10) -> dict:
        """
        Vérifie les conditions physiques sur ρ.

        Returns:
            dict avec clés 'is_hermitian', 'trace_one', 'positive_semidefinite'
        """
        dm = self.density_matrix
        # Hermiticité
        is_hermitian = bool(np.max(np.abs(dm - np.conj(dm.T))) < tolerance)
        # Trace = 1
        trace_one = bool(abs(np.trace(dm).real - 1.0) < tolerance)
        # Valeurs propres ≥ 0
        eigenvalues = np.linalg.eigvalsh(dm)
        positive_semidefinite = bool(np.all(eigenvalues >= -tolerance))
        return {
            'is_hermitian': is_hermitian,
            'trace_one': trace_one,
            'positive_semidefinite': positive_semidefinite,
        }
