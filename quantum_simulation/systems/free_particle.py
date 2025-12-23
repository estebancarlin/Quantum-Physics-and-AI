"""
Système particule libre (V = 0).

Règle R3.2 avec V=0 : H = P²/2m = -ℏ²/2m Δ
Source : [file:1, Chapitre I, § C]
"""

import numpy as np
import warnings  # ← DÉPLACER ICI
from typing import Tuple
from quantum_simulation.core.state import WaveFunctionState
from quantum_simulation.core.operators import Hamiltonian


class FreeParticle:
    """
    Particule libre dans potentiel nul.
    
    Hamiltonien : H = P²/(2m) = -ℏ²/(2m) ∇²
    
    États propres : ondes planes exp(ikx) avec E = ℏ²k²/(2m)
    """
    
    def __init__(self, mass: float, hbar: float):
        """
        Args:
            mass: Masse particule (kg)
            hbar: Constante réduite de Planck (J·s)
        """
        self.mass = mass
        self.hbar = hbar
        
        # ✅ Hamiltonien particule libre (potentiel nul)
        from quantum_simulation.core.operators import Hamiltonian
        self.hamiltonian = Hamiltonian(mass, hbar, potential=None)
    
    def hamiltonian_matrix(self, spatial_grid: np.ndarray) -> np.ndarray:
        """
        Construit matrice hamiltonien particule libre sur grille.
        
        ANCIENNEMENT : hamiltonian() (renommé pour éviter conflit avec attribut)
        """
        n = len(spatial_grid)
        dx = spatial_grid[1] - spatial_grid[0]
        
        # Matrice laplacien tridiagonale
        diag = -2.0 * np.ones(n)
        off_diag = np.ones(n-1)
        
        laplacian = (
            np.diag(diag) + 
            np.diag(off_diag, k=1) + 
            np.diag(off_diag, k=-1)
        ) / dx**2
        
        # H = -ℏ²/(2m) Δ
        return -(self.hbar**2 / (2 * self.mass)) * laplacian
    
    def create_gaussian_wavepacket(self, spatial_grid: np.ndarray,
                                    x0: float, sigma_x: float, k0: float) -> WaveFunctionState:
        """
        Crée paquet d'ondes gaussien.
        
        ψ(x) = (2πσₓ²)^(-1/4) exp[-(x-x₀)²/(4σₓ²) + ik₀x]
        
        Args:
            spatial_grid: Grille spatiale (ndarray)
            x0: Position centrale (m)
            sigma_x: Largeur gaussienne (m)
            k0: Impulsion moyenne ℏk₀ (m⁻¹)
            
        Returns:
            État gaussien normalisé
            
        Propriétés:
            - ⟨X⟩ = x0
            - ΔX = σx
            - ⟨P⟩ = ℏk0
            - ΔP = ℏ/(2σx)
            - ΔX·ΔP = ℏ/2 (état minimum incertitude)
        """
        # # DEBUG CRITIQUE
        # print(f"      DEBUG create_gaussian ENTRÉE: type(spatial_grid) = {type(spatial_grid)}")
        # print(f"      DEBUG create_gaussian ENTRÉE: spatial_grid.dtype = {spatial_grid.dtype if isinstance(spatial_grid, np.ndarray) else 'N/A'}")
        
        # Conversion robuste en ndarray
        x = np.asarray(spatial_grid, dtype=np.float64)
        
        # # DEBUG après conversion
        # print(f"      DEBUG create_gaussian APRÈS asarray: type(x) = {type(x)}")
        # print(f"      DEBUG create_gaussian APRÈS asarray: x.dtype = {x.dtype}")
        # print(f"      DEBUG create_gaussian APRÈS asarray: x[:3] = {x[:3]}")
        
        # Vérification type final
        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"Conversion échouée : spatial_grid est {type(spatial_grid)}, "
                f"x converti est {type(x)} au lieu de ndarray"
            )
        
        # Vérification couverture grille
        x_min_needed = x0 - 5 * sigma_x
        x_max_needed = x0 + 5 * sigma_x
        
        if x[0] > x_min_needed or x[-1] < x_max_needed:
            warnings.warn(
                f"Grille [{x[0]:.2e}, {x[-1]:.2e}] ne couvre pas ±5σ "
                f"[{x_min_needed:.2e}, {x_max_needed:.2e}]. "
                f"Normalisation et incertitudes seront imprécises.",
                f"Suggestion: x_min={x_min_needed:.2e}, x_max={x_max_needed:.2e}"
            )
        
        # Construction paquet gaussien
        normalization = (2 * np.pi * sigma_x**2)**(-0.25)
        envelope = np.exp(-(x - x0)**2 / (4 * sigma_x**2))
        
        # # DEBUG avant phase
        # print(f"      DEBUG create_gaussian: Calcul phase avec k0={k0}, type(x)={type(x)}")
        phase = np.exp(1j * np.float64(k0) * x, dtype=np.complex128)
        
        psi = normalization * envelope * phase
        
        # Vérification normalisation sur grille discrète
        state = WaveFunctionState(spatial_grid, psi)
        if not state.is_normalized(tolerance=1e-3):  # ← Tolérance relâchée
            state = state.normalize()
        
        # ✅ VALIDATION FINALE stricte
        final_norm = state.norm()
        if abs(final_norm - 1.0) > 1e-9:
            # Dernière normalisation forcée si nécessaire
            state = WaveFunctionState(spatial_grid, state.wavefunction / final_norm)
            
        return state
    
    def create_plane_wave(self, spatial_grid: np.ndarray, k: float) -> WaveFunctionState:
        """
        Crée onde plane exp(ikx).
        
        ψ(x) = A exp(ikx)  où A choisi pour normaliser sur grille finie
        
        Args:
            spatial_grid: Grille spatiale (ndarray)
            k: Nombre d'onde (m⁻¹)
            
        Returns:
            État onde plane normalisé numériquement
            
        Propriétés:
            - État propre H avec E = ℏ²k²/(2m)
            - ⟨P⟩ = ℏk (impulsion définie)
            - ΔX → ∞ (position indéfinie sur grille infinie)
            
        Note:
            Sur grille finie, onde plane tronquée n'est pas strictement
            état propre (effets de bord). Normalisation discrète nécessaire.
        """
        # Conversion robuste en ndarray
        x = np.asarray(spatial_grid, dtype=np.float64)
        
        # Onde plane complexe
        psi = np.exp(1j * k * x)
        
        # Normalisation sur grille discrète
        state = WaveFunctionState(spatial_grid, psi)
        
        # Forcer normalisation (onde plane sur grille finie non normée analytiquement)
        if not state.is_normalized(tolerance=1e-6):
            state = state.normalize()
        
        return state
    
    def expected_position(self, t: float, x0: float, k0: float) -> float:
        """
        Position moyenne paquet gaussien libre au temps t.
        
        Pour particule libre : ⟨X⟩(t) = x₀ + (ℏk₀/m)t
        
        Args:
            t: Temps (s)
            x0: Position initiale (m)
            k0: Nombre d'onde initial (m⁻¹)
            
        Returns:
            Position moyenne ⟨X⟩(t) (m)
        """
        return x0 + (self.hbar * k0 / self.mass) * t
    
    def expected_momentum(self, k0: float) -> float:
        """
        Impulsion moyenne paquet gaussien libre.
        
        Pour particule libre : ⟨P⟩ = ℏk₀ (conservée)
        
        Args:
            k0: Nombre d'onde initial (m⁻¹)
            
        Returns:
            Impulsion moyenne ⟨P⟩ (kg·m/s)
        """
        return self.hbar * k0
    
    def position_uncertainty(self, t: float, sigma_x: float) -> float:
        """
        Incertitude position paquet gaussien libre au temps t.
        
        Pour particule libre : ΔX(t) = σₓ√[1 + (ℏt/(2mσₓ²))²]
        
        Args:
            t: Temps (s)
            sigma_x: Largeur initiale (m)
            
        Returns:
            Incertitude ΔX(t) (m)
        """
        spreading_factor = 1 + (self.hbar * t / (2 * self.mass * sigma_x**2))**2
        return sigma_x * np.sqrt(spreading_factor)
    
    def momentum_uncertainty(self, sigma_x: float) -> float:
        """
        Incertitude impulsion paquet gaussien libre.
        
        Pour particule libre : ΔP = ℏ/(2σₓ) (conservée)
        
        Args:
            sigma_x: Largeur spatiale (m)
            
        Returns:
            Incertitude ΔP (kg·m/s)
        """
        return self.hbar / (2 * sigma_x)