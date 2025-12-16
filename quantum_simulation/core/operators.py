import numpy as np
from typing import Callable

from abc import ABC, abstractmethod

from quantum_simulation.core.state import QuantumState
from quantum_simulation.core.state import WaveFunctionState

class Observable(ABC):
    """
    Observable quantique (opérateur hermitique).
    Sources :
    - [file:1, Chapitre II, § D-1] : "observable = opérateur hermitique"
    - [file:1, Chapitre II, § D-2] : valeurs propres réelles
    
    Invariants :
    - Hermiticité : A† = A
    - Valeurs propres réelles
    - Vecteurs propres forment base orthonormée
    """
    
    @abstractmethod
    def apply(self, state: QuantumState) -> QuantumState:
        """Application A|ψ⟩"""
        
    @abstractmethod
    def expectation_value(self, state: QuantumState) -> float:
        """
        Calcule ⟨A⟩ = ⟨ψ|A|ψ⟩
        Source : [file:1, Chapitre III, § C-4] (Règle 1.4.1)
        """
        
    @abstractmethod
    def uncertainty(self, state: QuantumState) -> float:
        """
        Calcule ΔA = √(⟨A²⟩ - ⟨A⟩²)
        Source : [file:1, Chapitre III, § C-5] (Règle 1.4.2)
        """
        
    @abstractmethod
    def eigensystem(self) -> tuple[np.ndarray, list[QuantumState]]:
        """
        Retourne (valeurs_propres, vecteurs_propres)
        
        Limite : Méthode de diagonalisation non spécifiée dans synthèse.
        Implémentation numérique laissée libre (numpy.linalg, etc.)
        """
        
    def is_hermitian(self, tolerance: float = 1e-10) -> bool:
        """Vérifie A† = A"""
        
    def commutator(self, other: 'Observable') -> 'Observable':
        """
        Calcule [A,B] = AB - BA
        Source : [file:1, Chapitre III, § C-6-a] : compatibilité si [A,B]=0
        """

class PositionOperator(Observable):
    """
    Observable position R.
    Source : [file:1, Chapitre II, § E] : observable R(X,Y,Z)
    
    En représentation position : multiplication par r
    """
    
    def apply(self, state: WaveFunctionState) -> WaveFunctionState:
        """Applique R|ψ⟩ : multiplication r·ψ(r)"""

class MomentumOperator(Observable):
    """
    Observable impulsion P.
    Sources :
    - [file:1, Chapitre II, § E] : observable P(Px,Py,Pz)
    - [file:1, Chapitre II, § E-2] : P = -iℏ∇ en représentation position (Règle 1.7.1)
    
    Relations de commutation canoniques :
    - [file:1, Chapitre III, § B-5-a] : [Ri,Pj] = iℏδij (Règle 1.1.3)
    """
    
    def apply(self, state: WaveFunctionState) -> WaveFunctionState:
        """
        Applique P|ψ⟩ = -iℏ∇ψ
        
        Limite : Implémentation de la dérivée numérique non spécifiée.
        Choix laissé libre (différences finies, FFT, etc.)
        """
        
    def validate_canonical_commutation(self, R: PositionOperator, 
                                       test_states: list) -> bool:
        """
        Vérifie [Ri,Pj] = iℏδij sur états tests
        Source : Règle 1.1.3
        """

class Hamiltonian(Observable):
    """
    Opérateur hamiltonien H (énergie totale).
    Sources :
    - [file:1, Chapitre III, § B-4] : "observable associée à l'énergie totale"
    - [file:1, Chapitre III, § B-5-b] : H = P²/2m + V(R) (Règle 1.7.2)
    """
    
    def __init__(self, mass: float, potential: Callable):
        """
        Attributs :
        - mass : m (masse de la particule)
        - potential : V(r) fonction scalaire
        
        Forme en représentation position :
        H = -ℏ²/2m Δ + V(r)
        """
        
    def apply(self, state: WaveFunctionState) -> WaveFunctionState:
        """
        Applique H|ψ⟩ = [-ℏ²/2m Δ + V(r)]ψ(r)
        
        Limite : Implémentation du laplacien numérique non spécifiée.
        """
