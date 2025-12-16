from abc import ABC, abstractmethod
import numpy as np
from quantum_simulation.utils.numerical import integrate_1d

class QuantumState(ABC):
    """
    État quantique |ψ⟩ d'un système.
    Sources : 
    - [file:1, Chapitre III, § B-1] : "vecteur d'état"
    - [file:1, Chapitre II, § B-2-a] : notations de Dirac
    
    Invariants physiques :
    - Normalisation : ⟨ψ|ψ⟩ = 1 (si normalisé)
    - Linéarité de l'espace des états
    """
    
    @abstractmethod
    def norm(self) -> float:
        """Calcule √⟨ψ|ψ⟩"""
        pass
        
    @abstractmethod
    def normalize(self) -> 'QuantumState':
        """Retourne état normé"""
        pass
        
    @abstractmethod
    def inner_product(self, other: 'QuantumState') -> complex:
        """
        Produit scalaire ⟨φ|ψ⟩
        Source : [file:1, Chapitre II, § B-2-c]
        """
        pass
        
    def is_normalized(self, tolerance: float = 1e-10) -> bool:
        """
        Vérifie si ⟨ψ|ψ⟩ ≈ 1
        
        Args:
            tolerance: Tolérance numérique
            
        Returns:
            True si état normalisé
        """
        norm_value = self.norm()
        return abs(norm_value - 1.0) < tolerance


class WaveFunctionState(QuantumState):
    """
    Implémentation Règle R2.1 : ρ(r) = |ψ(r)|²
    Source : [file:1, Chapitre I, § B-2]
    """
    
    def __init__(self, spatial_grid: np.ndarray, wavefunction: np.ndarray):
        self.spatial_grid = np.asarray(spatial_grid, dtype=float)
        self.wavefunction = np.asarray(wavefunction, dtype=complex)
        
        # Vérification cohérence tailles
        if self.spatial_grid.shape != self.wavefunction.shape:
            raise ValueError(
                f"Taille grille ({self.spatial_grid.shape}) != "
                f"taille wavefunction ({self.wavefunction.shape})"
            )
        
        self.dx = spatial_grid[1] - spatial_grid[0]  # Grille uniforme
        
    def norm(self) -> float:
        """
        Calcule √⟨ψ|ψ⟩ via intégration numérique.
        Implémente Règle R5.1 : ⟨ψ|ψ⟩ doit rester = 1
        """
        rho = np.abs(self.wavefunction)**2
        return np.sqrt(integrate_1d(rho, self.dx))
        
    def normalize(self) -> 'WaveFunctionState':
        """Retourne état normé : |ψ⟩ / √⟨ψ|ψ⟩"""
        norm_value = self.norm()
        if norm_value < 1e-12:
            raise ValueError("État nul, normalisation impossible")
        return WaveFunctionState(
            self.spatial_grid,
            self.wavefunction / norm_value
        )
        
    def inner_product(self, other: 'WaveFunctionState') -> complex:
        """
        Calcule ⟨φ|ψ⟩ = ∫ φ*(x)ψ(x) dx
        Source : [file:1, Chapitre II, § B-2-c]
        """
        # Vérification 1 : Tailles compatibles
        if len(self.spatial_grid) != len(other.spatial_grid):
            raise ValueError(
                f"Grilles spatiales incompatibles: "
                f"tailles différentes {len(self.spatial_grid)} vs {len(other.spatial_grid)}"
            )
        
        # Vérification 2 : Valeurs similaires (après vérif tailles)
        if not np.allclose(self.spatial_grid, other.spatial_grid):
            raise ValueError(
                f"Grilles spatiales incompatibles: "
                f"self [{self.spatial_grid[0]:.2e}, {self.spatial_grid[-1]:.2e}] "
                f"vs other [{other.spatial_grid[0]:.2e}, {other.spatial_grid[-1]:.2e}]"
            )
        
        integrand = np.conj(other.wavefunction) * self.wavefunction
        return integrate_1d(integrand, self.dx)
        
    def probability_density(self) -> np.ndarray:
        """
        Calcule densité de probabilité ρ(x) = |ψ(x)|².
        
        Règle R2.1
        Source : [file:1, Chapitre I, § B-2]
        
        Returns:
            Densité de probabilité sur grille spatiale
        """
        return np.abs(self.wavefunction)**2
    
    def probability_in_volume(self, x_min: float, x_max: float) -> float:
        """
        Calcule probabilité de trouver particule dans intervalle [x_min, x_max].
        
        P([x_min, x_max]) = ∫_{x_min}^{x_max} |ψ(x)|² dx
        
        Args:
            x_min: Borne inférieure
            x_max: Borne supérieure
            
        Returns:
            Probabilité ∈ [0, 1]
            
        Raises:
            ValueError: Si bornes hors grille
        """
        # Masque : points dans intervalle
        mask = (self.spatial_grid >= x_min) & (self.spatial_grid <= x_max)
        
        if not np.any(mask):
            raise ValueError(
                f"Intervalle [{x_min:.2e}, {x_max:.2e}] hors grille "
                f"[{self.spatial_grid[0]:.2e}, {self.spatial_grid[-1]:.2e}]"
            )
        
        # Intégration sur sous-intervalle
        rho = self.probability_density()
        rho_interval = rho[mask]
        
        # Utiliser integrate_1d nécessite grille complète → trapèzes manuel
        prob = np.trapz(rho_interval, self.spatial_grid[mask])
        
        return prob


class EigenStateBasis(QuantumState):
    """
    État décomposé sur base d'états propres {|un⟩}.
    Source : [file:1, Chapitre III, § D-2-a] : décomposition ψ = Σ cn|un⟩
    
    Attributs :
    - eigenstates : list[QuantumState]  # Base orthonormée
    - coefficients : np.ndarray          # Coefficients cn complexes
    - eigenvalues : np.ndarray           # Valeurs propres an associées
    """
    
    def __init__(self, eigenstates: list, coefficients: np.ndarray, eigenvalues: np.ndarray):
        """
        Args:
            eigenstates: Liste états propres |uₙ⟩ (base orthonormée)
            coefficients: Coefficients cₙ complexes
            eigenvalues: Valeurs propres aₙ correspondantes
            
        Hypothèses :
        - Base orthonormée : ⟨uᵢ|uⱼ⟩ = δᵢⱼ (Règle depuis [file:1, Chapitre II, § C-2-a])
        """
        if len(eigenstates) != len(coefficients) or len(eigenstates) != len(eigenvalues):
            raise ValueError("Tailles eigenstates, coefficients, eigenvalues incompatibles")
        
        self.eigenstates = eigenstates
        self.coefficients = coefficients
        self.eigenvalues = eigenvalues
    
    def norm(self) -> float:
        """
        Calcule √⟨ψ|ψ⟩ = √(Σ |cₙ|²)
        
        En base orthonormée : ⟨ψ|ψ⟩ = Σₙ |cₙ|²
        """
        return np.sqrt(np.sum(np.abs(self.coefficients)**2))
    
    def normalize(self) -> 'EigenStateBasis':
        """Retourne état normé"""
        norm_value = self.norm()
        if norm_value < 1e-12:
            raise ValueError("État nul, normalisation impossible")
        return EigenStateBasis(
            self.eigenstates,
            self.coefficients / norm_value,
            self.eigenvalues
        )
    
    def inner_product(self, other: 'EigenStateBasis') -> complex:
        """
        Calcule ⟨φ|ψ⟩ en base états propres.
        
        Si même base : ⟨φ|ψ⟩ = Σₙ cₙ*(φ) cₙ(ψ)
        """
        if len(self.eigenstates) != len(other.eigenstates):
            raise ValueError("Bases de tailles différentes")
        
        # Hypothèse : même base (vérification stricte omise pour performance)
        return np.sum(np.conj(other.coefficients) * self.coefficients)
    
    def validate_orthonormality(self, tolerance: float = 1e-8) -> bool:
        """
        Vérifie ⟨uᵢ|uⱼ⟩ = δᵢⱼ pour tous états base.
        
        Args:
            tolerance: Tolérance numérique
            
        Returns:
            True si base orthonormée
        """
        n = len(self.eigenstates)
        for i in range(n):
            for j in range(n):
                overlap = self.eigenstates[i].inner_product(self.eigenstates[j])
                expected = 1.0 if i == j else 0.0
                
                if abs(overlap - expected) > tolerance:
                    return False
        
        return True