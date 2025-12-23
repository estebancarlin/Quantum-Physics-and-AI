from abc import ABC, abstractmethod
import numpy as np
from quantum_simulation.utils.numerical import integrate_1d
from typing import Union, Tuple

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

class WaveFunctionState2D(QuantumState):
    """
    État quantique en représentation position 2D.
    
    Fonction d'onde ψ(x,y) sur grille cartésienne.
    
    Attributes:
        x_grid: Grille coordonnées X (1D array)
        y_grid: Grille coordonnées Y (1D array)
        wavefunction: ψ(x,y) array (nx, ny) complexe
        dx: Pas spatial X
        dy: Pas spatial Y
    
    Règles implémentées:
        - R2.1 : Normalisation ∫∫|ψ|² dxdy = 1
        - R2.4 : Densité probabilité ρ(x,y) = |ψ(x,y)|²
    """
    
    def __init__(self, x_grid: np.ndarray, y_grid: np.ndarray, wavefunction: np.ndarray):
        """
        Args:
            x_grid: Points grille X (nx,)
            y_grid: Points grille Y (ny,)
            wavefunction: ψ(x,y) array (nx, ny) complexe
        """
        if wavefunction.shape != (len(x_grid), len(y_grid)):
            raise ValueError(
                f"Forme wavefunction {wavefunction.shape} incompatible avec grille "
                f"({len(x_grid)}, {len(y_grid)})"
            )
        
        self.x_grid = np.array(x_grid, dtype=float)
        self.y_grid = np.array(y_grid, dtype=float)
        self.wavefunction = np.array(wavefunction, dtype=complex)
        
        # Pas spatiaux
        self.dx = self.x_grid[1] - self.x_grid[0] if len(self.x_grid) > 1 else 1.0
        self.dy = self.y_grid[1] - self.y_grid[0] if len(self.y_grid) > 1 else 1.0
        
        # Dimensions
        self.nx = len(self.x_grid)
        self.ny = len(self.y_grid)
    
    def norm(self) -> float:
        """
        Norme L² : ||ψ|| = √(∫∫|ψ(x,y)|² dxdy)
        
        Utilise intégration trapézoïdale 2D.
        """
        rho = np.abs(self.wavefunction)**2
        norm_squared = np.sum(rho) * self.dx * self.dy
        return np.sqrt(norm_squared)
    
    def normalize(self) -> 'WaveFunctionState2D':
        """
        Renormalise état : ||ψ|| = 1
        
        Returns:
            Nouvel état normalisé
        """
        current_norm = self.norm()
        if current_norm == 0:
            raise ValueError("État nul ne peut être normalisé")
        
        normalized_wavefunction = self.wavefunction / current_norm
        
        return WaveFunctionState2D(
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            wavefunction=normalized_wavefunction
        )
    
    def inner_product(self, other: 'WaveFunctionState2D') -> complex:
        """
        Produit scalaire ⟨self|other⟩ = ∫∫ ψ₁*(x,y) ψ₂(x,y) dxdy
        
        Args:
            other: Autre état 2D
            
        Returns:
            Produit scalaire (complexe)
        """
        if not np.allclose(self.x_grid, other.x_grid) or \
            not np.allclose(self.y_grid, other.y_grid):
            raise ValueError("Grilles spatiales incompatibles")
        
        integrand = np.conj(self.wavefunction) * other.wavefunction
        result = np.sum(integrand) * self.dx * self.dy
        
        return result
    
    def probability_density(self) -> np.ndarray:
        """
        Densité probabilité ρ(x,y) = |ψ(x,y)|²
        
        Returns:
            Array (nx, ny) réel positif
        """
        return np.abs(self.wavefunction)**2
    
    def probability_in_region(self, x_range: tuple, y_range: tuple) -> float:
        """
        Probabilité présence dans région [x₁,x₂] × [y₁,y₂]
        
        Args:
            x_range: (x_min, x_max)
            y_range: (y_min, y_max)
            
        Returns:
            Probabilité ∈ [0,1]
        """
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        # Masques indices
        x_mask = (self.x_grid >= x_min) & (self.x_grid <= x_max)
        y_mask = (self.y_grid >= y_min) & (self.y_grid <= y_max)
        
        # Densité dans région
        rho = self.probability_density()
        rho_region = rho[np.ix_(x_mask, y_mask)]
        
        # Intégration
        prob = np.sum(rho_region) * self.dx * self.dy
        
        return float(prob)
    
    def marginal_x(self) -> WaveFunctionState:
        """
        Distribution marginale en X : ρₓ(x) = ∫|ψ(x,y)|² dy
        
        Returns:
            État 1D intégré sur Y
        """
        rho_2d = self.probability_density()
        rho_x = np.sum(rho_2d, axis=1) * self.dy  # Intégration sur y
        
        # Normalisation
        norm_x = np.sqrt(np.sum(rho_x) * self.dx)
        psi_x = np.sqrt(rho_x) / norm_x if norm_x > 0 else np.sqrt(rho_x)
        
        return WaveFunctionState(
            spatial_grid=self.x_grid,
            wavefunction=psi_x.astype(complex)
        )
    
    def marginal_y(self) -> WaveFunctionState:
        """
        Distribution marginale en Y : ρᵧ(y) = ∫|ψ(x,y)|² dx
        
        Returns:
            État 1D intégré sur X
        """
        rho_2d = self.probability_density()
        rho_y = np.sum(rho_2d, axis=0) * self.dx  # Intégration sur x
        
        # Normalisation
        norm_y = np.sqrt(np.sum(rho_y) * self.dy)
        psi_y = np.sqrt(rho_y) / norm_y if norm_y > 0 else np.sqrt(rho_y)
        
        return WaveFunctionState(
            spatial_grid=self.y_grid,
            wavefunction=psi_y.astype(complex)
        )

class WaveFunctionStateND:
    """
    État fonction d'onde N-dimensionnel (1D/2D/3D).
    
    Généralise WaveFunctionState actuel.
    
    Attributs:
        - spatial_grid: tuple[np.ndarray] (grille 1D, meshgrid 2D, ou meshgrid 3D)
        - wavefunction: np.ndarray (shape = grid.shape)
        - dimension: int (1, 2 ou 3)
    """
    
    def __init__(self, spatial_grid: Union[np.ndarray, Tuple[np.ndarray]],
                wavefunction: np.ndarray):
        """
        Args:
            spatial_grid: 
                - 1D: np.ndarray (x,)
                - 2D: (X, Y) avec X, Y = np.meshgrid(x, y)
                - 3D: (X, Y, Z) avec X, Y, Z = np.meshgrid(x, y, z)
            wavefunction: np.ndarray avec shape cohérente
        """
        # Détection automatique dimension
        if isinstance(spatial_grid, np.ndarray):
            self.dimension = 1
            self.grid = (spatial_grid,)
        else:
            self.dimension = len(spatial_grid)
            self.grid = spatial_grid
            
        self.wavefunction = wavefunction
        self._validate_shape()
        
    def norm(self) -> float:
        """
        Norme ∫...∫ |ψ|² d^N r.
        
        Utilise intégration multi-dimensionnelle (scipy.integrate.nquad).
        """
        
    def probability_density(self) -> np.ndarray:
        """ρ = |ψ|² (toutes dimensions)."""
        return np.abs(self.wavefunction)**2
        
    def inner_product(self, other: 'WaveFunctionStateND') -> complex:
        """⟨ψ|φ⟩ = ∫...∫ ψ*φ d^N r."""


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