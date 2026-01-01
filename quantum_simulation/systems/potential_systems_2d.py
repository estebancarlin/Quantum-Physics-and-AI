# quantum_simulation/systems/potential_systems_2d.py
"""
Systèmes potentiels 2D.

Implémente :
- Puits infini rectangulaire 2D
- Puits fini 2D
- Barrières 2D (double-slit, etc.)
- Boîtes quantiques (dots)

Sources théoriques :
- Extension 2D puits 1D [Document de référence, Section 1.2.2]
- Séparation variables : ψ(x,y) = ψₓ(x)·ψᵧ(y)
"""

import numpy as np
from typing import Tuple, Optional
from quantum_simulation.core.state import WaveFunctionState2D
from quantum_simulation.core.operators import Hamiltonian


class InfiniteWell2D:
    """
    Puits de potentiel infini rectangulaire 2D.
    
    V(x,y) = 0      pour 0 < x < Lₓ et 0 < y < Lᵧ
    V(x,y) = ∞      ailleurs
    
    États propres analytiques :
        ψₙₓ,ₙᵧ(x,y) = (2/√(LₓLᵧ)) sin(nₓπx/Lₓ) sin(nᵧπy/Lᵧ)
        Eₙₓ,ₙᵧ = π²ℏ²/(2m) [(nₓ/Lₓ)² + (nᵧ/Lᵧ)²]
    
    Applications physiques :
        - Modèle particule confinée (atome dans cavité)
        - Électron dans boîte quantique rectangulaire
        - Photon dans cavité résonnante
    
    Usage:
        well = InfiniteWell2D(Lx=2e-9, Ly=2e-9, mass=m_e, hbar=ℏ)
        state = well.eigenstate_wavefunction_2d(nx=3, ny=2, x_grid, y_grid)
        E = well.energy_eigenvalue_2d(nx=3, ny=2)
    """
    
    def __init__(self, width_x: float, width_y: float, mass: float, hbar: float):
        """
        Args:
            width_x: Largeur x (m)
            width_y: Largeur y (m)
            mass: Masse particule (kg)
            hbar: Constante Planck réduite (J·s)
        """
        self.Lx = width_x
        self.Ly = width_y
        self.mass = mass
        self.hbar = hbar
    
    def energy_eigenvalue_2d(self, nx: int, ny: int) -> float:
        """
        Énergie état propre (nx, ny).
        
        E_{nx,ny} = π²ℏ²/(2m) [(nx/Lx)² + (ny/Ly)²]
        
        Args:
            nx, ny: Nombres quantiques (nx, ny ≥ 1)
            
        Returns:
            Énergie (J)
        """
        if nx < 1 or ny < 1:
            raise ValueError(f"Nombres quantiques doivent être ≥ 1 : nx={nx}, ny={ny}")
        
        factor = (np.pi**2 * self.hbar**2) / (2 * self.mass)
        kx_squared = (nx / self.Lx)**2
        ky_squared = (ny / self.Ly)**2
        
        return factor * (kx_squared + ky_squared)
    
    def eigenstate_wavefunction_2d(
        self,
        nx: int,
        ny: int,
        x_grid: np.ndarray,
        y_grid: np.ndarray
    ) -> WaveFunctionState2D:
        """
        Fonction d'onde état propre (nx, ny).
        
        ψ_{nx,ny}(x,y) = (2/√(LxLy)) sin(nxπx/Lx) sin(nyπy/Ly)
        
        Args:
            nx, ny: Nombres quantiques
            x_grid, y_grid: Grilles spatiales (m)
            
        Returns:
            État 2D normalisé
        """
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        # Normalisation
        norm_factor = 2.0 / np.sqrt(self.Lx * self.Ly)
        
        # Fonction d'onde
        psi_x = np.sin(nx * np.pi * X / self.Lx)
        psi_y = np.sin(ny * np.pi * Y / self.Ly)
        
        psi = norm_factor * psi_x * psi_y
        
        # Conditions bord : ψ = 0 hors puits
        mask_x = (X < 0) | (X > self.Lx)
        mask_y = (Y < 0) | (Y > self.Ly)
        psi[mask_x | mask_y] = 0.0
        
        return WaveFunctionState2D(
            wavefunction=psi,
            x_grid=x_grid,
            y_grid=y_grid
        )
    
    def potential_2d(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Potentiel V(x,y).
        
        Args:
            X, Y: Grilles meshgrid
            
        Returns:
            V(x,y) : 0 dans puits, ∞ ailleurs
        """
        V = np.full_like(X, np.inf)
        
        mask_inside = (X >= 0) & (X <= self.Lx) & (Y >= 0) & (Y <= self.Ly)
        V[mask_inside] = 0.0
        
        return V
    
    def create_hamiltonian_2d(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray
    ) -> Hamiltonian:
        """
        Hamiltonien puits infini 2D.
        
        H = -ℏ²/(2m) ∇² avec ψ = 0 aux bords
        
        Args:
            x_grid, y_grid: Grilles spatiales
            
        Returns:
            Hamiltonien 2D
        """
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        hamiltonian = Hamiltonian(
            mass=self.mass,
            potential=lambda X, Y: self.potential_2d(X, Y),
            hbar=self.hbar
        )
        
        hamiltonian.dimension = 2
        hamiltonian.x_grid = x_grid
        hamiltonian.y_grid = y_grid
        
        return hamiltonian
    
    def degeneracy_count(self, n_max: int = 10) -> dict:
        """
        Calcule dégénérescence niveaux énergie.
        
        Puits carré (Lx = Ly) : états (nx, ny) et (ny, nx) dégénérés si nx ≠ ny
        
        Args:
            n_max: Nombres quantiques max à considérer
            
        Returns:
            Dict {E: [(nx1, ny1), (nx2, ny2), ...]}
        """
        from collections import defaultdict
        
        degeneracies = defaultdict(list)
        
        for nx in range(1, n_max + 1):
            for ny in range(1, n_max + 1):
                E = self.energy_eigenvalue_2d(nx, ny)
                degeneracies[E].append((nx, ny))
        
        # Grouper énergies équivalentes (tolérance numérique)
        grouped = {}
        for E, states in degeneracies.items():
            # Arrondir pour grouper énergies proches
            E_rounded = round(E, 25)
            if E_rounded not in grouped:
                grouped[E_rounded] = []
            grouped[E_rounded].extend(states)
        
        return {E: list(set(states)) for E, states in grouped.items()}


class QuantumDot2D:
    """
    Boîte quantique 2D (dot circulaire).
    
    Potentiel confinement parabolique :
        V(r) = ½mω²r²  avec r = √(x² + y²)
    
    Approximation harmonique pour dot réaliste.
    
    États propres (coordonnées polaires) :
        ψₙ,ₘ(r,θ) = R_{n,|m|}(r) exp(imθ)
        E_{n,m} = ℏω(2n + |m| + 1)
    
    Applications :
        - Électron confiné dans quantum dot semiconducteur
        - Atome piégé optiquement (2D)
        - Exciton dans nanocristal
    """
    
    def __init__(self, omega: float, mass: float, hbar: float):
        """
        Args:
            omega: Pulsation confinement (rad/s)
            mass: Masse particule (kg)
            hbar: Constante Planck réduite (J·s)
        """
        self.omega = omega
        self.mass = mass
        self.hbar = hbar
    
    def energy_eigenvalue(self, n: int, m: int) -> float:
        """
        Énergie état (n, m).
        
        E_{n,m} = ℏω(2n + |m| + 1)
        
        Args:
            n: Nombre quantique radial (n ≥ 0)
            m: Nombre quantique angulaire (m entier)
            
        Returns:
            Énergie (J)
        """
        if n < 0:
            raise ValueError(f"n doit être ≥ 0 : n={n}")
        
        return self.hbar * self.omega * (2*n + abs(m) + 1)
    
    def potential_2d(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Potentiel parabolique V(r) = ½mω²r².
        
        Args:
            X, Y: Grilles meshgrid
            
        Returns:
            V(x,y) (J)
        """
        r_squared = X**2 + Y**2
        return 0.5 * self.mass * self.omega**2 * r_squared
    
    def create_gaussian_state_2d(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        x0: float = 0.0,
        y0: float = 0.0
    ) -> WaveFunctionState2D:
        """
        État fondamental gaussien (approximation n=0, m=0).
        
        ψ₀,₀(x,y) = (mω/πℏ)^{1/2} exp[-mω(x²+y²)/(2ℏ)]
        
        Args:
            x_grid, y_grid: Grilles spatiales
            x0, y0: Centre état (m)
            
        Returns:
            État fondamental 2D normalisé
        """
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        # Longueur caractéristique
        a0 = np.sqrt(self.hbar / (self.mass * self.omega))
        
        # Normalisation
        norm_factor = 1.0 / (np.sqrt(np.pi) * a0)
        
        # Gaussienne
        r_squared = (X - x0)**2 + (Y - y0)**2
        psi = norm_factor * np.exp(-r_squared / (2 * a0**2))
        
        return WaveFunctionState2D(
            wavefunction=psi.astype(complex),
            x_grid=x_grid,
            y_grid=y_grid
        )


class DoubleSlit2D:
    """
    Barrière avec deux fentes (expérience Young 2D).
    
    Potentiel :
        V(x,y) = V₀  dans barrière (x ∈ [x_barr - δ, x_barr + δ])
        V(x,y) = 0   dans fentes (y ∈ fentes)
        V(x,y) = 0   ailleurs
    
    Configuration :
        - Fente 1 : y ∈ [y_center - d/2 - w/2, y_center - d/2 + w/2]
        - Fente 2 : y ∈ [y_center + d/2 - w/2, y_center + d/2 + w/2]
        - d : séparation fentes
        - w : largeur fente
    
    Observable clé :
        - Interfrange : Δy = λD/d
        - λ = 2πℏ/p (de Broglie)
        - D : distance barrière-écran
    """
    
    def __init__(
        self,
        x_barrier: float,
        barrier_thickness: float,
        barrier_height: float,
        slit_separation: float,
        slit_width: float,
        mass: float,
        hbar: float
    ):
        """
        Args:
            x_barrier: Position barrière (m)
            barrier_thickness: Épaisseur barrière (m)
            barrier_height: Hauteur potentiel V₀ (J)
            slit_separation: Séparation fentes d (m)
            slit_width: Largeur fente w (m)
            mass: Masse particule (kg)
            hbar: Constante Planck réduite (J·s)
        """
        self.x_barr = x_barrier
        self.thickness = barrier_thickness
        self.V0 = barrier_height
        self.d = slit_separation
        self.w = slit_width
        self.mass = mass
        self.hbar = hbar
    
    def potential_2d(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Potentiel double-slit V(x,y).
        
        Args:
            X, Y: Grilles meshgrid
            
        Returns:
            V(x,y) (J)
        """
        V = np.zeros_like(X)
        
        # Zone barrière
        mask_barrier = (
            (X >= self.x_barr - self.thickness/2) &
            (X <= self.x_barr + self.thickness/2)
        )
        
        V[mask_barrier] = self.V0
        
        # Fentes (V = 0)
        y_center = 0.0  # Centrage fentes
        
        # Fente 1 (y < 0)
        slit1_min = y_center - self.d/2 - self.w/2
        slit1_max = y_center - self.d/2 + self.w/2
        mask_slit1 = (Y >= slit1_min) & (Y <= slit1_max) & mask_barrier
        
        # Fente 2 (y > 0)
        slit2_min = y_center + self.d/2 - self.w/2
        slit2_max = y_center + self.d/2 + self.w/2
        mask_slit2 = (Y >= slit2_min) & (Y <= slit2_max) & mask_barrier
        
        # Ouvrir fentes
        V[mask_slit1 | mask_slit2] = 0.0
        
        return V
    
    def expected_fringe_spacing(self, momentum: float, screen_distance: float) -> float:
        """
        Interfrange attendu (formule Young).
        
        Δy = λD/d  avec λ = 2πℏ/p
        
        Args:
            momentum: Impulsion particule p (kg·m/s)
            screen_distance: Distance barrière-écran D (m)
            
        Returns:
            Interfrange Δy (m)
        """
        lambda_deB = 2 * np.pi * self.hbar / momentum
        return lambda_deB * screen_distance / self.d


# Tests unitaires
if __name__ == "__main__":
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    
    print("="*60)
    print(" Tests Systèmes 2D")
    print("="*60)
    print()
    
    # Test puits infini 2D
    print("[1/3] Puits infini 2D...")
    well = InfiniteWell2D(width_x=2e-9, width_y=2e-9, mass=mass, hbar=hbar)
    
    # Énergies quelques états
    for nx, ny in [(1,1), (1,2), (2,1), (2,2)]:
        E = well.energy_eigenvalue_2d(nx, ny)
        print(f"  E({nx},{ny}) = {E:.6e} J")
    
    # Dégénérescences
    deg = well.degeneracy_count(n_max=5)
    print(f"  Niveaux dégénérés : {len([d for d in deg.values() if len(d) > 1])}")
    print()
    
    # Test quantum dot
    print("[2/3] Quantum dot 2D...")
    omega = 1e15  # rad/s
    dot = QuantumDot2D(omega, mass, hbar)
    
    for n, m in [(0,0), (0,1), (1,0), (0,-1)]:
        E = dot.energy_eigenvalue(n, m)
        print(f"  E({n},{m}) = {E:.6e} J")
    print()
    
    # Test double-slit
    print("[3/3] Double-slit 2D...")
    slit = DoubleSlit2D(
        x_barrier=0.0,
        barrier_thickness=1e-10,
        barrier_height=1e-18,
        slit_separation=1e-8,
        slit_width=2e-9,
        mass=mass,
        hbar=hbar
    )
    
    p = mass * 1e6  # Impulsion typique
    D = 1e-7  # Distance écran
    delta_y = slit.expected_fringe_spacing(p, D)
    print(f"  Interfrange attendu : {delta_y*1e9:.2f} nm")
    print()
    
    print("✓ Tests systèmes 2D complétés")