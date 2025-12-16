"""
Systèmes quantiques avec potentiels classiques.

Implémente :
- Puits de potentiel infini
- Puits de potentiel fini (estimation états liés)
- Barrière de potentiel (effet tunnel)

Sources théoriques :
- [file:1, Chapitre I, § D] : puits infini mentionné
- Document de référence, Section 1.2.2
"""

import numpy as np
from typing import Callable
from quantum_simulation.core.operators import Hamiltonian
from quantum_simulation.core.state import WaveFunctionState


class InfiniteWell:
    """
    Puits de potentiel infini 1D.
    
    V(x) = 0     pour 0 < x < L
    V(x) = ∞     ailleurs
    
    États propres analytiques :
        ψₙ(x) = √(2/L) sin(nπx/L)
        Eₙ = n²π²ℏ²/(2mL²)
    
    Source théorique : [file:1, Chapitre I, § D] (mentionné)
    Formules standards mécanique quantique.
    """
    
    def __init__(self, width: float, mass: float, hbar: float):
        """
        Args:
            width: Largeur puits L (m)
            mass: Masse particule (kg)
            hbar: Constante Planck réduite (J·s)
        """
        self.width = width
        self.mass = mass
        self.hbar = hbar
        
    def energy_eigenvalue(self, n: int) -> float:
        """
        Calcule énergie niveau n.
        
        Eₙ = n²π²ℏ²/(2mL²)
        
        Args:
            n: Nombre quantique (n ≥ 1)
            
        Returns:
            Énergie (J)
            
        Raises:
            ValueError: Si n < 1
        """
        if n < 1:
            raise ValueError("Nombre quantique n doit être ≥ 1 pour puits infini")
        
        return (n**2 * np.pi**2 * self.hbar**2) / (2 * self.mass * self.width**2)
    
    def eigenstate_wavefunction(self, n: int, x: np.ndarray) -> WaveFunctionState:
        """
        Construit fonction d'onde état propre ψₙ(x).
        
        ψₙ(x) = √(2/L) sin(nπx/L)  pour x ∈ [0, L]
        ψₙ(x) = 0                   ailleurs
        
        Args:
            n: Nombre quantique (n ≥ 1)
            x: Grille spatiale
            
        Returns:
            État quantique |n⟩ normalisé numériquement
            
        Note:
            Conditions aux bords : ψ(0) = ψ(L) = 0
            
            Normalisation analytique √(2/L) suppose intégrale continue.
            Sur grille discrète, on renormalise numériquement pour garantir ∫|ψ|²dx = 1.
        """
        if n < 1:
            raise ValueError("n doit être ≥ 1")
        
        psi = np.zeros_like(x, dtype=complex)
        
        # Masque strict : x dans ]0, L[ (exclut bords)
        mask = (x > 0) & (x < self.width)
        
        norm_factor = np.sqrt(2.0 / self.width)
        psi[mask] = norm_factor * np.sin(n * np.pi * x[mask] / self.width)
        
        state = WaveFunctionState(x, psi)
        
        # Renormalisation numérique si nécessaire
        if not state.is_normalized(tolerance=1e-5):
            state = state.normalize()
        
        return state
    
    def potential(self, x: np.ndarray) -> np.ndarray:
        """
        Potentiel V(x).
        
        V = 0 dans [0, L], ∞ ailleurs (implémenté comme grand nombre).
        
        Returns:
            Valeurs potentiel (J)
        """
        V = np.zeros_like(x)
        
        # Murs infinis (valeur très grande en pratique)
        V_infinity = 1e50  # "Infini" numérique
        V[x <= 0] = V_infinity
        V[x >= self.width] = V_infinity
        
        return V
    
    def hamiltonian(self, x: np.ndarray) -> Hamiltonian:
        """
        Construit hamiltonien H = P²/2m + V(x).
        
        Returns:
            Opérateur hamiltonien
        """
        return Hamiltonian(self.mass, self.potential, self.hbar)
    
    def validate_orthonormality(self, n_states: int, x: np.ndarray, 
                                tolerance: float = 1e-8) -> bool:
        """
        Vérifie orthonormalité états propres : ⟨ψₙ|ψₘ⟩ = δₙₘ
        
        Args:
            n_states: Nombre d'états à tester
            x: Grille spatiale
            tolerance: Tolérance numérique
            
        Returns:
            True si orthonormalité vérifiée
        """
        states = [self.eigenstate_wavefunction(n, x) for n in range(1, n_states + 1)]
        
        for i, psi_i in enumerate(states):
            for j, psi_j in enumerate(states):
                overlap = psi_i.inner_product(psi_j)
                expected = 1.0 if i == j else 0.0
                
                if abs(overlap - expected) > tolerance:
                    return False
        
        return True


class FiniteWell:
    """
    Puits de potentiel fini 1D.
    
    V(x) = -V₀  pour -L/2 < x < L/2
    V(x) = 0    ailleurs
    
    LIMITE : Résolution complète nécessite équation transcendante numérique.
    Cette implémentation fournit :
    - Potentiel V(x)
    - Hamiltonien
    - Estimation nombre états liés
    
    Note Document de référence, Section 8.2-N3 :
        "Calcul valeurs/vecteurs propres : np.linalg.eigh() pour matrices hermitiennes"
        → Diagonalisation possible mais coûteuse pour grandes grilles
    """
    
    def __init__(self, width: float, depth: float, mass: float, hbar: float):
        """
        Args:
            width: Largeur puits (m)
            depth: Profondeur puits V₀ > 0 (J)
            mass: Masse particule (kg)
            hbar: Constante Planck réduite (J·s)
        """
        self.width = width
        self.depth = depth
        self.mass = mass
        self.hbar = hbar
        
    def potential(self, x: np.ndarray) -> np.ndarray:
        """
        Potentiel V(x).
        
        V(x) = -V₀ dans [-L/2, L/2], 0 ailleurs
        
        Returns:
            Valeurs potentiel (J)
        """
        V = np.zeros_like(x)
        mask = (x >= -self.width/2) & (x <= self.width/2)
        V[mask] = -self.depth
        return V
    
    def hamiltonian(self, x: np.ndarray) -> Hamiltonian:
        """
        Construit hamiltonien H = P²/2m + V(x).
        
        Returns:
            Opérateur hamiltonien
        """
        return Hamiltonian(self.mass, self.potential, self.hbar)
    
    def estimate_bound_states_count(self) -> int:
        """
        Estime nombre d'états liés.
        
        Formule approximative basée sur argument dimensionnel :
        ξ₀ = √(2mV₀)L/(2ℏ)
        N ≈ floor(ξ₀/(π/2)) + 1
        
        Cette estimation suppose puits symétrique et états pairs/impairs.
        
        Returns:
            Nombre estimé états liés (au moins 1 pour puits fini attractif)
            
        Note:
            Calcul exact nécessiterait résolution numérique équation transcendante :
            - États pairs : ξ tan(ξ) = √(ξ₀² - ξ²)
            - États impairs : -ξ cot(ξ) = √(ξ₀² - ξ²)
        """
        xi_0 = np.sqrt(2 * self.mass * self.depth) * self.width / (2 * self.hbar)
        n_approx = int(np.floor(xi_0 / (np.pi / 2))) + 1
        return max(1, n_approx)


class PotentialBarrier:
    """
    Barrière de potentiel rectangulaire 1D.
    
    V(x) = V₀   pour -L/2 < x < L/2
    V(x) = 0    ailleurs
    
    Phénomène physique : effet tunnel quantique
    - Transmission classique impossible si E < V₀
    - Transmission quantique non nulle (probabilité exponentielle)
    
    Cette implémentation fournit :
    - Potentiel V(x)
    - Hamiltonien
    - Coefficient transmission approximatif
    
    Note:
        Calcul exact transmission/réflexion nécessite conditions raccordement
        des fonctions d'onde aux interfaces (méthode transfert de matrices).
    """
    
    def __init__(self, width: float, height: float, mass: float, hbar: float):
        """
        Args:
            width: Largeur barrière (m)
            height: Hauteur barrière V₀ (J)
            mass: Masse particule (kg)
            hbar: Constante Planck réduite (J·s)
        """
        self.width = width
        self.height = height
        self.mass = mass
        self.hbar = hbar
        
    def potential(self, x: np.ndarray) -> np.ndarray:
        """
        Potentiel V(x).
        
        V(x) = V₀ dans [-L/2, L/2], 0 ailleurs
        
        Returns:
            Valeurs potentiel (J)
        """
        V = np.zeros_like(x)
        mask = (x >= -self.width/2) & (x <= self.width/2)
        V[mask] = self.height
        return V
    
    def hamiltonian(self, x: np.ndarray) -> Hamiltonian:
        """
        Construit hamiltonien H = P²/2m + V(x).
        
        Returns:
            Opérateur hamiltonien
        """
        return Hamiltonian(self.mass, self.potential, self.hbar)
    
    def transmission_coefficient_approx(self, energy: float) -> float:
        """
        Coefficient transmission approximatif selon énergie incidente.
        
        Deux régimes :
        1. E > V₀ (classiquement autorisé) : T ≈ 1
        2. E < V₀ (effet tunnel) : T ≈ 16(E/V₀)(1 - E/V₀) exp(-2κL)
           où κ = √(2m(V₀-E))/ℏ
        
        Args:
            energy: Énergie incidente particule (J)
            
        Returns:
            Coefficient transmission T ∈ [0, 1]
            
        Note:
            Formule approximative valide pour barrière opaque (κL >> 1).
            Pour barrière fine ou énergie proche V₀, erreur significative.
            
            Formule exacte (WKB ou matrice transfert) nécessiterait résolution
            équations couplées aux interfaces.
        """
        if energy >= self.height:
            # Transmission classique (sur-barrière)
            # En réalité : oscillations autour de 1 dues interférences quantiques
            # Approximation : T ≈ 1 si E >> V₀
            return 1.0
        
        # Sous-barrière : effet tunnel
        kappa = np.sqrt(2 * self.mass * (self.height - energy)) / self.hbar
        
        # Formule approx WKB
        ratio = energy / self.height
        T_approx = 16 * ratio * (1 - ratio) * np.exp(-2 * kappa * self.width)
        
        # Assurer T ≤ 1 (limitation formule approximative)
        return min(1.0, T_approx)


# Alias pour compatibilité avec ancien code
InfinitePotentialWell = InfiniteWell