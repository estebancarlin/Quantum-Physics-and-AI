# quantum_simulation/visualization/viz_2d.py
"""
Visualisations quantiques 2D.

Fonctionnalités:
    - Densités probabilité 2D (heatmaps)
    - Animations évolution temporelle
    - Lignes équipotentiel
    - Vecteurs courant probabilité

Dépendances:
    - matplotlib (plots 2D)
    - ffmpeg (animations MP4, optionnel)
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    import imageio_ffmpeg  # Fallback Python
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Optional, Callable
import warnings

from quantum_simulation.core.state import WaveFunctionState2D


class QuantumVisualizer2D:
    """
    Visualiseur états quantiques 2D.
    
    Usage:
        viz = QuantumVisualizer2D(output_dir='./results_2d/')
        viz.plot_density_2d(state, title='Densité t=0')
        viz.create_animation_2d(states, times, output_name='evolution.mp4')
    """
    
    def __init__(self, output_dir: str = "./results/2d/", dpi: int = 150):
        """
        Args:
            output_dir: Dossier sortie figures
            dpi: Résolution images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
    def plot_density_2d(
        self,
        state: WaveFunctionState2D,
        title: str = "Densité probabilité 2D",
        save_name: Optional[str] = None,
        show_colorbar: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ):
        """
        Plot densité probabilité ρ(x,y) = |ψ(x,y)|².
        
        Args:
            state: État 2D
            title: Titre figure
            save_name: Nom fichier (sans extension)
            show_colorbar: Afficher barre couleurs
            vmin, vmax: Limites échelle couleur
        """
        rho = state.probability_density()
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        # Grilles meshgrid pour pcolormesh
        X, Y = np.meshgrid(state.x_grid, state.y_grid, indexing='ij')
        
        # Heatmap
        im = ax.pcolormesh(
            X * 1e9,  # Conversion nm
            Y * 1e9,
            rho,
            cmap='viridis',
            shading='auto',
            vmin=vmin,
            vmax=vmax
        )
        
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('ρ(x,y) [m⁻²]', rotation=270, labelpad=20)
        
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / f"{save_name}.png"
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"  ✓ Figure sauvegardée: {filepath}")
        
        plt.close()
    
    def plot_wavefunction_3d(
        self,
        state: WaveFunctionState2D,
        component: str = 'real',
        title: str = "Fonction d'onde 3D",
        save_name: Optional[str] = None
    ):
        """
        Plot 3D partie réelle/imaginaire/module ψ(x,y).
        
        Args:
            state: État 2D
            component: 'real', 'imag', 'abs'
            title: Titre figure
            save_name: Nom fichier
        """
        X, Y = np.meshgrid(state.x_grid, state.y_grid, indexing='ij')
        
        if component == 'real':
            Z = np.real(state.wavefunction)
            zlabel = 'Re[ψ(x,y)]'
        elif component == 'imag':
            Z = np.imag(state.wavefunction)
            zlabel = 'Im[ψ(x,y)]'
        elif component == 'abs':
            Z = np.abs(state.wavefunction)
            zlabel = '|ψ(x,y)|'
        else:
            raise ValueError(f"Composante inconnue: {component}")
        
        fig = plt.figure(figsize=(12, 9), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Surface plot
        surf = ax.plot_surface(
            X * 1e9,
            Y * 1e9,
            Z,
            cmap='coolwarm',
            linewidth=0,
            antialiased=True,
            alpha=0.8
        )
        
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        
        if save_name:
            filepath = self.output_dir / f"{save_name}.png"
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"  ✓ Figure 3D sauvegardée: {filepath}")
        
        plt.close()
    
    def plot_marginal_distributions(
        self,
        state: WaveFunctionState2D,
        title: str = "Distributions marginales",
        save_name: Optional[str] = None
    ):
        """
        Plot distributions marginales ρₓ(x) et ρᵧ(y).
        
        Args:
            state: État 2D
            title: Titre figure
            save_name: Nom fichier
        """
        # Calcul marginales
        state_x = state.marginal_x()
        state_y = state.marginal_y()
        
        rho_x = state_x.probability_density()
        rho_y = state_y.probability_density()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)
        
        # Marginale X
        ax1.plot(state.x_grid * 1e9, rho_x, 'b-', linewidth=2)
        ax1.fill_between(state.x_grid * 1e9, 0, rho_x, alpha=0.3)
        ax1.set_xlabel('x (nm)')
        ax1.set_ylabel('ρₓ(x)')
        ax1.set_title('Distribution marginale X')
        ax1.grid(True, alpha=0.3)
        
        # Marginale Y
        ax2.plot(state.y_grid * 1e9, rho_y, 'r-', linewidth=2)
        ax2.fill_between(state.y_grid * 1e9, 0, rho_y, alpha=0.3)
        ax2.set_xlabel('y (nm)')
        ax2.set_ylabel('ρᵧ(y)')
        ax2.set_title('Distribution marginale Y')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / f"{save_name}.png"
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"  ✓ Marginales sauvegardées: {filepath}")
        
        plt.close()
    
    def create_animation_2d(
        self,
        states: List[WaveFunctionState2D],
        times: np.ndarray,
        output_name: str = "evolution_2d.mp4",
        interval: int = 100,
        fps: int = 10,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ):
        """
        Crée animation évolution temporelle densité 2D.
        
        Args:
            states: Liste états temporels
            times: Temps correspondants (s)
            output_name: Nom fichier MP4
            interval: Délai entre frames (ms)
            fps: Images par seconde
            vmin, vmax: Limites échelle couleur (fixes sur animation)
        
        Nécessite:
            ffmpeg installé sur système
        """
        if len(states) != len(times):
            raise ValueError(f"Longueurs incompatibles: {len(states)} états vs {len(times)} temps")
        
        # Vérifier ffmpeg disponible
        try:
            writer = FFMpegWriter(fps=fps, metadata={'artist': 'QuantumSimulation'})
        except RuntimeError:
            warnings.warn(
                "ffmpeg non trouvé. Animation sauvegardée comme GIF (qualité réduite).",
                ImportWarning
            )
            writer = 'pillow'  # Fallback GIF
            output_name = output_name.replace('.mp4', '.gif')
        
        # Déterminer limites fixes couleur
        if vmin is None or vmax is None:
            all_densities = [state.probability_density() for state in states]
            global_min = min(np.min(rho) for rho in all_densities)
            global_max = max(np.max(rho) for rho in all_densities)
            vmin = vmin or global_min
            vmax = vmax or global_max
        
        # Figure
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        X, Y = np.meshgrid(states[0].x_grid, states[0].y_grid, indexing='ij')
        
        # Frame initial
        rho0 = states[0].probability_density()
        im = ax.pcolormesh(
            X * 1e9,
            Y * 1e9,
            rho0,
            cmap='viridis',
            shading='auto',
            vmin=vmin,
            vmax=vmax
        )
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('ρ(x,y) [m⁻²]', rotation=270, labelpad=20)
        
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_aspect('equal')
        
        time_text = ax.text(
            0.02, 0.98,
            '',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        def update_frame(i):
            """Mise à jour frame i."""
            rho = states[i].probability_density()
            im.set_array(rho.ravel())
            
            time_text.set_text(f't = {times[i]*1e15:.2f} fs')
            
            return im, time_text
        
        # Création animation
        anim = FuncAnimation(
            fig,
            update_frame,
            frames=len(states),
            interval=interval,
            blit=True,
            repeat=True
        )
        
        # Sauvegarde
        filepath = self.output_dir / output_name
        
        if isinstance(writer, str):  # GIF fallback
            anim.save(filepath, writer=writer, fps=fps)
        else:  # MP4
            anim.save(filepath, writer=writer)
        
        print(f"  ✓ Animation sauvegardée: {filepath}")
        plt.close()
    
    def plot_probability_current_2d(
        self,
        state: WaveFunctionState2D,
        hbar: float,
        mass: float,
        title: str = "Courant probabilité 2D",
        save_name: Optional[str] = None,
        subsample: int = 4
    ):
        """
        Plot vecteurs courant probabilité J(x,y).
        
        J = (ℏ/m) Im[ψ* ∇ψ]
        
        Args:
            state: État 2D
            hbar: ℏ (J·s)
            mass: Masse (kg)
            title: Titre figure
            save_name: Nom fichier
            subsample: Sous-échantillonnage vecteurs (affichage)
        """
        psi = state.wavefunction
        dx = state.dx
        dy = state.dy
        
        # Gradients centrés
        grad_x = np.gradient(psi, dx, axis=0)
        grad_y = np.gradient(psi, dy, axis=1)
        
        # Composantes courant : J = (ℏ/m) Im[ψ* ∇ψ]
        Jx = (hbar / mass) * np.imag(np.conj(psi) * grad_x)
        Jy = (hbar / mass) * np.imag(np.conj(psi) * grad_y)
        
        # Densité probabilité
        rho = state.probability_density()
        
        # Grilles
        X, Y = np.meshgrid(state.x_grid, state.y_grid, indexing='ij')
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)
        
        # Fond: densité
        im = ax.pcolormesh(
            X * 1e9,
            Y * 1e9,
            rho,
            cmap='gray',
            alpha=0.5,
            shading='auto'
        )
        
        # Vecteurs courant (sous-échantillonnés)
        skip = subsample
        ax.quiver(
            X[::skip, ::skip] * 1e9,
            Y[::skip, ::skip] * 1e9,
            Jx[::skip, ::skip],
            Jy[::skip, ::skip],
            color='red',
            scale=None,
            scale_units='xy',
            angles='xy',
            width=0.003,
            headwidth=3,
            headlength=4
        )
        
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        plt.colorbar(im, ax=ax, label='ρ(x,y)')
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / f"{save_name}.png"
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"  ✓ Courant probabilité sauvegardé: {filepath}")
        
        plt.close()


if __name__ == "__main__":
    # Test visualisations 2D
    from quantum_simulation.systems.free_particle_2d import FreeParticle2D
    
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    
    x = np.linspace(-2e-8, 2e-8, 200)
    y = np.linspace(-2e-8, 2e-8, 200)
    
    fp2d = FreeParticle2D(mass, hbar)
    state = fp2d.create_gaussian_packet_2d(
        x, y,
        x0=0, y0=0,
        sigma_x=3e-9, sigma_y=3e-9,
        kx0=5e9, ky0=3e9
    )
    
    viz = QuantumVisualizer2D(output_dir='./test_viz_2d/')
    
    print("Test visualisations 2D:")
    viz.plot_density_2d(state, save_name='test_density')
    viz.plot_wavefunction_3d(state, component='abs', save_name='test_wavefunction_3d')
    viz.plot_marginal_distributions(state, save_name='test_marginals')
    viz.plot_probability_current_2d(state, hbar, mass, save_name='test_current')
    
    print("✓ Tests visualisations 2D complétés")