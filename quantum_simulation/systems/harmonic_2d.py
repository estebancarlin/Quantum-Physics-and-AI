# quantum_simulation/systems/harmonic_2d.py
"""
Oscillateur harmonique 2D.

Cas traités :
- Isotrope : ωx = ωy = ω
- Anisotrope : ωx ≠ ωy
- Couplé : V = ½m(ωx²x² + ωy²y² + 2κxy)

États propres (isotrope) :
    ψₙₓ,ₙᵧ(x,y) = ψₙₓ(x) · ψₙᵧ(y)
    E_{nx,ny} = ℏω(nx + ny + 1)
"""
import numpy as np
from quantum_simulation.core.state import WaveFunctionState2D

class HarmonicOscillator2D:
    """Oscillateur harmonique 2D anisotrope."""
    
    def __init__(self, omega_x: float, omega_y: float, mass: float, hbar: float):
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.mass = mass
        self.hbar = hbar
    
    def energy_eigenvalue(self, nx: int, ny: int) -> float:
        """Énergie E_{nx,ny} = ℏ(ωx(nx+½) + ωy(ny+½))."""
        Ex = self.hbar * self.omega_x * (nx + 0.5)
        Ey = self.hbar * self.omega_y * (ny + 0.5)
        return Ex + Ey
    
    def wavefunction_2d(
        self,
        nx: int,
        ny: int,
        x_grid: np.ndarray,
        y_grid: np.ndarray
    ) -> WaveFunctionState2D:
        """
        Fonction d'onde ψ_{nx,ny}(x,y) = ψ_{nx}(x) · ψ_{ny}(y).
        
        Utilise polynômes Hermite (scipy.special.eval_hermite).
        """
        from scipy.special import eval_hermite
        
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        # Longueurs caractéristiques
        ax = np.sqrt(self.hbar / (self.mass * self.omega_x))
        ay = np.sqrt(self.hbar / (self.mass * self.omega_y))
        
        # Variables adimensionnées
        xi_x = X / ax
        xi_y = Y / ay
        
        # Polynômes Hermite
        Hn_x = eval_hermite(nx, xi_x)
        Hn_y = eval_hermite(ny, xi_y)
        
        # Normalisations
        Nx = 1.0 / np.sqrt(2**nx * np.math.factorial(nx) * np.sqrt(np.pi) * ax)
        Ny = 1.0 / np.sqrt(2**ny * np.math.factorial(ny) * np.sqrt(np.pi) * ay)
        
        # Gaussiennes
        gauss_x = np.exp(-xi_x**2 / 2)
        gauss_y = np.exp(-xi_y**2 / 2)
        
        psi = Nx * Ny * Hn_x * Hn_y * gauss_x * gauss_y
        
        return WaveFunctionState2D(
            wavefunction=psi.astype(complex),
            x_grid=x_grid,
            y_grid=y_grid
        )
    
    def create_coherent_state_2d(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        alpha_x: complex,
        alpha_y: complex
    ) -> WaveFunctionState2D:
        """
        État cohérent 2D |αx⟩|αy⟩.
        
        Gaussienne déplacée dans espace phase.
        """
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        ax = np.sqrt(self.hbar / (self.mass * self.omega_x))
        ay = np.sqrt(self.hbar / (self.mass * self.omega_y))
        
        # Déplacements
        x0 = np.sqrt(2) * ax * np.real(alpha_x)
        y0 = np.sqrt(2) * ay * np.real(alpha_y)
        
        # Phase
        phi_x = np.sqrt(2) * np.imag(alpha_x)
        phi_y = np.sqrt(2) * np.imag(alpha_y)
        
        # Gaussienne
        psi = (
            (1.0 / (np.pi * ax * ay))**(0.25) *
            np.exp(-(X - x0)**2 / (2 * ax**2)) *
            np.exp(-(Y - y0)**2 / (2 * ay**2)) *
            np.exp(1j * (phi_x * X / ax + phi_y * Y / ay))
        )
        
        return WaveFunctionState2D(
            wavefunction=psi,
            x_grid=x_grid,
            y_grid=y_grid
        )


if __name__ == "__main__":
    # Test oscillateur 2D
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    omega = 1e15
    
    ho = HarmonicOscillator2D(omega, omega, mass, hbar)
    
    print("Oscillateur harmonique 2D isotrope:")
    for nx, ny in [(0,0), (1,0), (0,1), (1,1), (2,0)]:
        E = ho.energy_eigenvalue(nx, ny)
        print(f"  E({nx},{ny}) = {E:.6e} J  ({E/(hbar*omega):.1f}ℏω)")