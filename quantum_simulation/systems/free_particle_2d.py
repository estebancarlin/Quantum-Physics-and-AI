# quantum_simulation/systems/free_particle_2d.py
"""
Particule libre en 2D.

Implémente :
- Paquets gaussiens 2D
- Ondes planes 2D
- Hamiltonien H = (Pₓ² + Pᵧ²)/2m

Sources théoriques :
- Extension 2D particule libre 1D [Document de référence, E1]
- Règle R4.1 : H particule libre
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Tuple, Optional
from quantum_simulation.core.state import WaveFunctionState2D
from quantum_simulation.core.operators import Hamiltonian


class FreeParticle2D:
    """
    Système particule libre 2D.
    
    Hamiltonien : H = -ℏ²/(2m) (∂²/∂x² + ∂²/∂y²)
    
    Usage:
        fp2d = FreeParticle2D(mass, hbar)
        state = fp2d.create_gaussian_packet_2d(x0, y0, sigma_x, sigma_y, kx0, ky0)
    """
    
    def __init__(self, mass: float, hbar: float):
        """
        Args:
            mass: Masse particule (kg)
            hbar: Constante Planck réduite (J·s)
        """
        self.mass = mass
        self.hbar = hbar
        
    def create_gaussian_packet_2d(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        x0: float = 0.0,
        y0: float = 0.0,
        sigma_x: float = 1e-9,
        sigma_y: float = 1e-9,
        kx0: float = 0.0,
        ky0: float = 0.0
    ) -> WaveFunctionState2D:
        """
        Crée paquet gaussien 2D.
        
        ψ(x,y) = N exp[-(x-x₀)²/4σₓ² - (y-y₀)²/4σᵧ²] exp[i(kₓ₀x + kᵧ₀y)]
        
        Args:
            x_grid: Grille X
            y_grid: Grille Y
            x0, y0: Positions centrales (m)
            sigma_x, sigma_y: Largeurs spatiales (m)
            kx0, ky0: Vecteur d'onde initial (rad/m)
            
        Returns:
            État 2D normalisé
        """
        # Grilles 2D
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        # Enveloppe gaussienne
        gaussian_x = np.exp(-(X - x0)**2 / (4 * sigma_x**2))
        gaussian_y = np.exp(-(Y - y0)**2 / (4 * sigma_y**2))
        
        # Phase onde plane
        phase = np.exp(1j * (kx0 * X + ky0 * Y))
        
        # Fonction d'onde brute
        psi_raw = gaussian_x * gaussian_y * phase
        
        # Normalisation
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        norm = np.sqrt(np.sum(np.abs(psi_raw)**2) * dx * dy)
        
        psi_normalized = psi_raw / norm
        
        state = WaveFunctionState2D(x_grid, y_grid, psi_normalized)
        
        # Vérification normalisation
        final_norm = state.norm()
        if abs(final_norm - 1.0) > 1e-6:
            import warnings
            warnings.warn(
                f"Normalisation imparfaite: ||ψ|| = {final_norm:.10f}. "
                f"Grille [{x_grid[0]:.2e}, {x_grid[-1]:.2e}] × "
                f"[{y_grid[0]:.2e}, {y_grid[-1]:.2e}] "
                f"devrait couvrir ±5σ."
            )
        
        return state
    
    def create_plane_wave_2d(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        kx: float,
        ky: float
    ) -> WaveFunctionState2D:
        """
        Crée onde plane 2D.
        
        ψ(x,y) = N exp[i(kₓx + kᵧy)]
        
        Args:
            x_grid: Grille X
            y_grid: Grille Y
            kx, ky: Vecteur d'onde (rad/m)
            
        Returns:
            État 2D normalisé sur grille finie
        """
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        psi_raw = np.exp(1j * (kx * X + ky * Y))
        
        # Normalisation discrète
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        norm = np.sqrt(np.sum(np.abs(psi_raw)**2) * dx * dy)
        
        psi_normalized = psi_raw / norm
        
        return WaveFunctionState2D(x_grid, y_grid, psi_normalized)
    
    def create_hamiltonian_2d(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray
    ) -> Hamiltonian:
        """
        Construit Hamiltonien particule libre 2D.
        
        H = -ℏ²/(2m) ∇²  avec ∇² = ∂²/∂x² + ∂²/∂y²
        
        Args:
            x_grid: Grille X
            y_grid: Grille Y
            
        Returns:
            Hamiltonien 2D (potentiel V=0)
        """
        # Potentiel nul
        def potential_2d(X, Y):
            return np.zeros_like(X)
        
        # Stocker grilles pour opérations
        self.x_grid_stored = x_grid
        self.y_grid_stored = y_grid
        
        # Hamiltonien avec méta-données 2D
        hamiltonian = Hamiltonian(
            mass=self.mass,
            potential=potential_2d,  # V(x,y) = 0
            hbar=self.hbar
        )
        
        # Attributs spécifiques 2D
        hamiltonian.dimension = 2
        hamiltonian.x_grid = x_grid
        hamiltonian.y_grid = y_grid
        
        return hamiltonian
    
    def energy_eigenvalue_2d(self, kx: float, ky: float) -> float:
        """
        Énergie onde plane 2D.
        
        E = ℏ²(kₓ² + kᵧ²)/(2m)
        
        Args:
            kx, ky: Composantes vecteur d'onde (rad/m)
            
        Returns:
            Énergie (J)
        """
        k_squared = kx**2 + ky**2
        return (self.hbar**2 * k_squared) / (2 * self.mass)
    
    def classical_trajectory_2d(
        self,
        x0: float,
        y0: float,
        vx0: float,
        vy0: float,
        times: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trajectoire classique particule libre 2D.
        
        x(t) = x₀ + vₓ₀·t
        y(t) = y₀ + vᵧ₀·t
        
        Args:
            x0, y0: Positions initiales (m)
            vx0, vy0: Vitesses initiales (m/s)
            times: Temps échantillonnage (s)
            
        Returns:
            (x_traj, y_traj) : Trajectoires (arrays)
        """
        x_traj = x0 + vx0 * times
        y_traj = y0 + vy0 * times
        
        return x_traj, y_traj


if __name__ == "__main__":
    # Test création paquet gaussien 2D
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    
    x = np.linspace(-2e-8, 2e-8, 256)
    y = np.linspace(-2e-8, 2e-8, 256)
    
    fp2d = FreeParticle2D(mass, hbar)
    
    state = fp2d.create_gaussian_packet_2d(
        x, y,
        x0=0.0, y0=0.0,
        sigma_x=2e-9, sigma_y=2e-9,
        kx0=5e9, ky0=3e9
    )
    
    print(f"✓ Paquet gaussien 2D créé")
    print(f"  Norme: {state.norm():.10f}")
    print(f"  Grille: ({state.nx}, {state.ny})")
    
    # Énergie
    E = fp2d.energy_eigenvalue_2d(5e9, 3e9)
    print(f"  Énergie: {E:.6e} J")