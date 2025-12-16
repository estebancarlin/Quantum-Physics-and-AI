from quantum_simulation.core.state import QuantumState
from quantum_simulation.core.operators import Hamiltonian

class ConservationValidator:
    """
    Validation des lois de conservation quantiques.
    """
    
    def validate_probability_conservation(self, states: list[QuantumState]) -> bool:
        """
        Vérifie ⟨ψ(t)|ψ(t)⟩ = cste au cours de l'évolution.
        Source : [file:1, Chapitre III, § D-1-c]
        """
        
    def validate_continuity_equation(self, rho_field, J_field, dt, dx) -> bool:
        """
        Vérifie ∂ρ/∂t + ∇·J = 0 numériquement.
        Source : [file:1, Chapitre III, § D-1-c] (Règle 1.5.1)
        
        Entrées :
        - rho_field : ρ(r,t) calculé depuis |ψ|²
        - J_field : J(r,t) calculé selon Règle 1.5.2
        
        LIMITE : Discrétisation spatiotemporelle (dt, dx) non spécifiée dans synthèse.
        Choix laissé à l'implémentation numérique.
        """
        
    def validate_energy_conservation(self, state: QuantumState, 
                                    hamiltonian: Hamiltonian, 
                                    times: list[float]) -> bool:
        """
        Pour système conservatif : si état propre de H, ⟨H⟩ = cste.
        Source : [file:1, Chapitre III, § D-2-b]
        """
