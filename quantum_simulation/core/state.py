from abc import ABC, abstractmethod

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
        """Calcule ⟨ψ|ψ⟩"""
        
    @abstractmethod
    def normalize(self) -> 'QuantumState':
        """Retourne état normé"""
        
    @abstractmethod
    def inner_product(self, other: 'QuantumState') -> complex:
        """
        Produit scalaire ⟨φ|ψ⟩
        Source : [file:1, Chapitre II, § B-2-c]
        """
        
    def is_normalized(self, tolerance: float = 1e-10) -> bool:
        """Vérifie si ⟨ψ|ψ⟩ ≈ 1"""

class WaveFunctionState(QuantumState):
    """
    État représenté par fonction d'onde ψ(r,t) en représentation position.
    Source : [file:1, Chapitre I, § B-2] : "fonction d'onde ψ(r)"
    
    Attributs :
    - spatial_grid : grille de discrétisation spatiale (r)
    - wavefunction : np.ndarray (valeurs complexes de ψ aux points de grille)
    - dimension : int (1D, 2D ou 3D)
    """
    
    def __init__(self, spatial_grid, wavefunction: np.ndarray):
        """
        Hypothèses :
        - Discrétisation spatiale nécessaire (méthode numérique non spécifiée dans synthèse)
        """
        
    def probability_density(self) -> np.ndarray:
        """
        Calcule ρ(r) = |ψ(r)|²
        Source : [file:1, Chapitre I, § B-2]
        """
        
    def probability_in_volume(self, volume_indices) -> float:
        """
        Intègre |ψ(r)|² sur un volume
        Approximation : somme discrète (intégration non détaillée dans synthèse)
        """

class EigenStateBasis(QuantumState):
    """
    État décomposé sur base d'états propres {|un⟩}.
    Source : [file:1, Chapitre III, § D-2-a] : décomposition ψ = Σ cn|un⟩
    
    Attributs :
    - eigenstates : list[QuantumState]  # Base orthonormée
    - coefficients : np.ndarray          # Coefficients cn complexes
    - eigenvalues : np.ndarray           # Valeurs propres an associées
    """
    
    def __init__(self, eigenstates, coefficients, eigenvalues):
        """
        Hypothèses :
        - Base orthonormée : ⟨ui|uj⟩ = δij (Règle depuis [file:1, Chapitre II, § C-2-a])
        """
        
    def validate_orthonormality(self, tolerance: float = 1e-8) -> bool:
        """Vérifie ⟨ui|uj⟩ = δij"""
