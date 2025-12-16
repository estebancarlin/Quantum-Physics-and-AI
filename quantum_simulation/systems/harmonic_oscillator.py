from quantum_simulation.core.state import EigenStateBasis
from quantum_simulation.core.operators import LadderOperator

class HarmonicOscillator:
    """
    Système de l'oscillateur harmonique à 1D.
    Sources : [file:1, Chapitre V] (Règles 1.6.x)
    """
    
    def __init__(self, mass: float, omega: float, hbar: float):
        """
        Paramètres : m, ω
        Hamiltonien : H = P²/2m + (1/2)mω²X²
        Source : [file:1, Chapitre V, § A] (Règle 1.6.1)
        """
        self.mass = mass
        self.omega = omega
        self.hbar = hbar
        
    def energy_eigenvalue(self, n: int) -> float:
        """
        Retourne En = ℏω(n + 1/2)
        Source : [file:1, Chapitre V, § B] (Règle 1.6.3)
        """
        return self.hbar * self.omega * (n + 0.5)
        
    def creation_operator(self) -> 'LadderOperator':
        """
        Construit a† = √(mω/2ℏ)(X - i/(mω)P)
        Source : [file:1, Chapitre V, § B] (Règle 1.6.2)
        """
        
    def annihilation_operator(self) -> 'LadderOperator':
        """
        Construit a = √(mω/2ℏ)(X + i/(mω)P)
        Source : [file:1, Chapitre V, § B] (Règle 1.6.2)
        
        Vérifie : [a, a†] = 1
        """
        
    def construct_eigenstate(self, n: int, n_max_basis: int) -> EigenStateBasis:
        """
        Construit |n⟩ par récurrence :
        - |0⟩ défini par a|0⟩ = 0
        - |n⟩ = (a†)^n / √(n!) |0⟩
        
        Relations d'action :
        - a|n⟩ = √n|n-1⟩
        - a†|n⟩ = √(n+1)|n+1⟩
        Source : [file:1, Chapitre V, § C] (Règle 1.6.4)
        
        LIMITE : Construction de |0⟩ en représentation position nécessiterait
        la fonction d'onde explicite, non donnée dans l'extrait de synthèse fourni.
        (Mentionné au Complément BV mais détails absents de l'extrait).
        
        Implémentation : travailler en base de Fock {|n⟩} abstraite.
        """
