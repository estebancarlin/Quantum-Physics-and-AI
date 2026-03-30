"""
Oscillateur harmonique quantique 1D.

Implémentation en base de Fock {|n⟩} avec opérateurs échelle.

Sources théoriques :
- [file:1, Chapitre V, § B] : Spectre et algèbre
- Document de référence, Section 1.2.3, lignes 259-292
"""

import math
import numpy as np
from typing import Optional
from quantum_simulation.core.state import WaveFunctionState


class HarmonicOscillator:
    """
    Oscillateur harmonique quantique 1D.
    
    Hamiltonien : H = ℏω(a†a + 1/2) = ℏω(N + 1/2)
    où a, a† sont opérateurs annihilation/création.
    
    Règles physiques implémentées :
    - R6.1 : Spectre Eₙ = ℏω(n + 1/2)
    - R6.2 : Algèbre [a, a†] = 1
    - R6.3 : Actions a|n⟩ = √n|n-1⟩, a†|n⟩ = √(n+1)|n+1⟩
    
    Source : [file:1, Chapitre V, § B]
    """
    
    def __init__(self, mass: float, omega: float, hbar: float, n_max: int):
        """
        Args:
            mass: Masse particule (kg)
            omega: Pulsation oscillateur ω (rad/s)
            hbar: Constante Planck réduite (J·s)
            n_max: Troncature base Fock (nombre max états)
        """
        self.mass = mass
        self.omega = omega
        self.hbar = hbar
        self.n_max = n_max
        
        # Précalcul matrices (optimisation)
        self._a_matrix = None
        self._a_dag_matrix = None
        self._H_matrix = None
        
    def energy_eigenvalue(self, n: int) -> float:
        """
        Calcule énergie état propre |n⟩.
        
        Règle R6.1 : Eₙ = ℏω(n + 1/2)
        Source : [file:1, Chapitre V, § B-2]
        
        Args:
            n: Nombre quantique (n ≥ 0)
            
        Returns:
            Énergie (J)
            
        Raises:
            ValueError: Si n hors limites [0, n_max]
        """
        if n < 0 or n > self.n_max:
            raise ValueError(f"n={n} hors limites [0, {self.n_max}]")
        
        return self.hbar * self.omega * (n + 0.5)
    
    def annihilation_matrix(self) -> np.ndarray:
        """
        Construit matrice opérateur annihilation a en base {|0⟩, |1⟩, ..., |n_max⟩}.
        
        Règle R6.3 : a|n⟩ = √n|n-1⟩
        
        Éléments matrice : ⟨m|a|n⟩ = √n δ_{m,n-1}
        
        Returns:
            Matrice (n_max+1) × (n_max+1)
        """
        if self._a_matrix is not None:
            return self._a_matrix
        
        dim = self.n_max + 1
        a = np.zeros((dim, dim), dtype=complex)
        
        # Remplir sur-diagonale : a_{m,n} = √n si m = n-1
        for n in range(1, dim):
            a[n-1, n] = np.sqrt(n)
        
        self._a_matrix = a
        return a
    
    def creation_matrix(self) -> np.ndarray:
        """
        Construit matrice opérateur création a† en base de Fock.
        
        Règle R6.3 : a†|n⟩ = √(n+1)|n+1⟩
        
        Éléments matrice : ⟨m|a†|n⟩ = √(n+1) δ_{m,n+1}
        
        Returns:
            Matrice (n_max+1) × (n_max+1)
            
        Note:
            a† = (a)† (conjugué hermitien de a)
        """
        if self._a_dag_matrix is not None:
            return self._a_dag_matrix
        
        # a† est hermitien conjugué de a
        a = self.annihilation_matrix()
        self._a_dag_matrix = np.conj(a.T)
        
        return self._a_dag_matrix
    
    def validate_algebra(self, tolerance: float = 1e-10) -> bool:
        """
        Vérifie algèbre opérateurs échelle.
        
        Règle R6.2 : [a, a†] = aa† - a†a = 𝟙
        Source : [file:1, Chapitre V, § B-1-b]
        
        Args:
            tolerance: Tolérance numérique
            
        Returns:
            True si commutateur vérifié pour n ∈ [0, n_max-1]
            
        Note importante sur troncature :
            En espace Hilbert complet (dimension ∞), [a,a†] = 𝟙 exactement.
            
            En base tronquée {|0⟩,...,|n_max⟩}, pour n < n_max :
                [a,a†]|n⟩ = |n⟩  ✓ (correct)
            
            Mais pour n = n_max :
                a†|n_max⟩ = √(n_max+1)|n_max+1⟩  (hors base)
                aa†|n_max⟩ = 0                    (projection sur base)
                a†a|n_max⟩ = n_max|n_max⟩
                [a,a†]|n_max⟩ = -n_max|n_max⟩  ✗ (erreur O(n_max))
            
            Cette méthode valide donc l'algèbre sur le sous-espace
            physiquement pertinent (états d'énergie < E_max).
        """
        a = self.annihilation_matrix()
        a_dag = self.creation_matrix()
        
        # Commutateur [a, a†]
        commutator = a @ a_dag - a_dag @ a
        
        # Vérifier identité SAUF sur dernier état
        for n in range(self.n_max):  # Exclure n_max
            # Diagonale : doit être 1
            if abs(commutator[n, n] - 1.0) > tolerance:
                return False
            
            # Hors-diagonale ligne n : doit être 0
            for m in range(self.n_max + 1):
                if m != n and abs(commutator[n, m]) > tolerance:
                    return False
        
        return True
    
    def hamiltonian_matrix(self) -> np.ndarray:
        """
        Construit matrice hamiltonien H = ℏω(a†a + 1/2).
        
        En base propre {|n⟩}, H est diagonal :
            H|n⟩ = Eₙ|n⟩ avec Eₙ = ℏω(n + 1/2)
        
        Returns:
            Matrice hamiltonien (n_max+1) × (n_max+1)
        """
        if self._H_matrix is not None:
            return self._H_matrix
        
        dim = self.n_max + 1
        
        # Méthode 1 : Construction directe (diagonal)
        H = np.zeros((dim, dim), dtype=complex)
        for n in range(dim):
            H[n, n] = self.energy_eigenvalue(n)
        
        self._H_matrix = H
        return H
    
    def number_operator_matrix(self) -> np.ndarray:
        """
        Construit matrice opérateur nombre N = a†a.
        
        Valeurs propres : N|n⟩ = n|n⟩
        
        Returns:
            Matrice N (n_max+1) × (n_max+1)
        """
        a = self.annihilation_matrix()
        a_dag = self.creation_matrix()
        
        return a_dag @ a
    
    def position_operator_matrix(self) -> np.ndarray:
        """
        Construit matrice opérateur position X en base de Fock.
        
        X = √(ℏ/(2mω)) (a + a†)
        
        Returns:
            Matrice X (n_max+1) × (n_max+1)
        """
        a = self.annihilation_matrix()
        a_dag = self.creation_matrix()
        
        x_0 = np.sqrt(self.hbar / (2 * self.mass * self.omega))
        
        return x_0 * (a + a_dag)
    
    def momentum_operator_matrix(self) -> np.ndarray:
        """
        Construit matrice opérateur impulsion P en base de Fock.
        
        P = i√(mℏω/2) (a† - a)
        
        Returns:
            Matrice P (n_max+1) × (n_max+1)
        """
        a = self.annihilation_matrix()
        a_dag = self.creation_matrix()
        
        p_0 = np.sqrt(self.mass * self.hbar * self.omega / 2)
        
        return 1j * p_0 * (a_dag - a)
    
    def eigenstate_in_fock_basis(self, n: int) -> np.ndarray:
        """
        Retourne vecteur état |n⟩ en base de Fock.
        
        |n⟩ = (0, 0, ..., 1, ..., 0)^T  (1 en position n)
        
        Args:
            n: Nombre quantique
            
        Returns:
            Vecteur colonne (n_max+1) × 1
        """
        if n < 0 or n > self.n_max:
            raise ValueError(f"n={n} hors limites")
        
        state = np.zeros(self.n_max + 1, dtype=complex)
        state[n] = 1.0
        
        return state
    
    def coherent_state(self, alpha: complex) -> np.ndarray:
        """
        Construit état cohérent |α⟩ en base de Fock.
        
        |α⟩ = exp(-|α|²/2) Σₙ (αⁿ/√n!) |n⟩
        
        États propres opérateur annihilation : a|α⟩ = α|α⟩
        
        Args:
            alpha: Paramètre complexe état cohérent
            
        Returns:
            Vecteur état cohérent (n_max+1) × 1
            
        Note:
            Troncature à n_max introduit petite erreur normalisation.
        """
        dim = self.n_max + 1
        state = np.zeros(dim, dtype=complex)
        
        # Coefficients cₙ = exp(-|α|²/2) αⁿ/√n!
        prefactor = np.exp(-0.5 * np.abs(alpha)**2)
        
        for n in range(dim):
            state[n] = prefactor * (alpha**n) / math.sqrt(math.factorial(n))
        
        # Renormalisation (compenser troncature)
        norm = np.linalg.norm(state)
        state /= norm
        
        return state
    
    def thermal_state_density_matrix(self, temperature: float, 
                                    boltzmann_k: float = 1.380649e-23) -> np.ndarray:
        """
        Construit matrice densité état thermique à température T.
        
        ρ_thermal = Σₙ pₙ |n⟩⟨n|
        où pₙ = (1 - exp(-βℏω)) exp(-nβℏω)
        avec β = 1/(k_B T)
        
        Args:
            temperature: Température (K)
            boltzmann_k: Constante Boltzmann (J/K)
            
        Returns:
            Matrice densité (n_max+1) × (n_max+1)
        """
        beta = 1.0 / (boltzmann_k * temperature)
        exp_factor = np.exp(-beta * self.hbar * self.omega)
        
        dim = self.n_max + 1
        rho = np.zeros((dim, dim), dtype=complex)
        
        # Normalisation
        Z = 1.0 / (1.0 - exp_factor)  # Fonction partition
        
        # Populations diagonales
        for n in range(dim):
            p_n = Z * (exp_factor**n) * (1 - exp_factor)
            rho[n, n] = p_n
        
        return rho
    
    def mean_occupation_thermal(self, temperature: float,
                                boltzmann_k: float = 1.380649e-23) -> float:
        """
        Calcule nombre moyen occupation thermique ⟨N⟩.
        
        ⟨N⟩ = 1/(exp(βℏω) - 1)
        
        Args:
            temperature: Température (K)
            boltzmann_k: Constante Boltzmann (J/K)
            
        Returns:
            Nombre moyen occupation
        """
        beta = 1.0 / (boltzmann_k * temperature)
        
        return 1.0 / (np.exp(beta * self.hbar * self.omega) - 1.0)
    
    def wavefunction_position(self, n: int, x_grid: np.ndarray) -> WaveFunctionState:
        """
        Fonction d'onde ψₙ(x) en représentation position.
        
        Formule (Complément BV, hors extraits fournis) :
            ψₙ(x) = (mω/πℏ)^(1/4) · 1/√(2ⁿn!) · Hₙ(ξ) · exp(-ξ²/2)
            où ξ = √(mω/ℏ) x
            Hₙ = polynômes Hermite (physicien)
        
        Args:
            n: Niveau quantique (n ≥ 0)
            x_grid: Grille spatiale (m)
            
        Returns:
            État normalisé WaveFunctionState
            
        Note:
            EXTENSION : Utilise scipy.special.eval_hermite.
            Source externe au cours (polynômes Hermite mentionnés mais non détaillés).
            
        Raises:
            ImportError: Si scipy non disponible
            ValueError: Si n < 0
            
        References:
            - Décision D4 : Extension Option 2 (Hermite)
            - Chapitre V, § B : Spectre HO
        """
        from scipy.special import eval_hermite
        from math import factorial
        
        if n < 0:
            raise ValueError(f"Niveau quantique négatif : n={n}")
        
        # Longueur caractéristique
        x0 = np.sqrt(self.hbar / (self.mass * self.omega))
        xi = x_grid / x0  # Variable adimensionnée
        
        # Normalisation
        norm_factor = (self.mass * self.omega / (np.pi * self.hbar))**0.25
        norm_factor /= math.sqrt(2**n * factorial(n))
        
        # Polynôme Hermite (physicien) + gaussienne
        Hn = eval_hermite(n, xi)
        psi = norm_factor * Hn * np.exp(-0.5 * xi**2)
        
        return WaveFunctionState(x_grid, psi.astype(complex))