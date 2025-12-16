from abc import ABC, abstractmethod
from quantum_simulation.core.state import QuantumState, WaveFunctionState
from quantum_simulation.core.operators import Hamiltonian

class Experiment(ABC):
    """
    Structure générique d'une expérience quantique.
    
    Cycle standard : préparation → évolution → mesure → analyse
    Configuration via parameters.yaml
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.results = {}
        
    @abstractmethod
    def prepare_initial_state(self) -> QuantumState:
        """Étape 1 : Préparation de l'état initial |ψ(t0)⟩"""
        
    @abstractmethod
    def define_hamiltonian(self) -> Hamiltonian:
        """Étape 2 : Définition du hamiltonien du système"""
        
    @abstractmethod
    def evolve_state(self, initial_state: QuantumState, 
                    hamiltonian: Hamiltonian) -> QuantumState:
        """Étape 3 : Évolution temporelle (ou état stationnaire)"""
        
    @abstractmethod
    def perform_measurements(self, state: QuantumState) -> dict:
        """Étape 4 : Mesures et statistiques"""
        
    def run(self) -> dict:
        """
        Exécution complète de l'expérience.
        Retourne : résultats structurés (statistiques, validations, etc.)
        """
        state_init = self.prepare_initial_state()
        H = self.define_hamiltonian()
        state_evolved = self.evolve_state(state_init, H)
        self.results = self.perform_measurements(state_evolved)
        return self.results
        
    def validate_physics(self) -> dict:
        """
        Vérifications physiques post-expérience :
        - Normalisation maintenue
        - Relations d'incertitude respectées (Règle 1.4.3)
        - Conservation de probabilité (Règle 1.5.1)
        
        Retourne : {test_name: passed/failed}
        """

class WavePacketEvolution(Experiment):
    """
    Expérience : évolution d'un paquet d'ondes gaussien libre.
    
    Objectif :
    - Observer étalement temporel (mentionné Complément GI)
    - Vérifier relations de Heisenberg (Règle 1.4.3)
    - Vérifier théorème d'Ehrenfest (Règle 1.4.4)
    """
    
    def prepare_initial_state(self) -> WaveFunctionState:
        """
        Construit paquet gaussien initial (paramètres depuis config)
        
        LIMITE : Forme explicite du paquet gaussien mentionnée
        (Complément GI) mais formules détaillées absentes de l'extrait synthèse.
        Nécessiterait compléments du cours pour forme analytique complète.
        """
        
    def evolve_state(self, initial_state, hamiltonian) -> WaveFunctionState:
        """Évolution via TimeEvolution.evolve_wavefunction()"""
        
    def perform_measurements(self, state) -> dict:
        """
        Mesure ⟨X⟩, ⟨P⟩, ΔX, ΔP à différents temps.
        Vérification ΔX·ΔP ≥ ℏ/2
        """
