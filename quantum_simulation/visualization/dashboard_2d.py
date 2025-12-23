# quantum_simulation/visualization/dashboard_2d.py
"""
Dashboard visualisation évolution 2D avec multi-plots.

Affiche simultanément :
- Densité ρ(x,y,t)
- Observables ⟨X⟩, ⟨Y⟩, ⟨Pₓ⟩, ⟨Pᵧ⟩
- Marginales ρₓ(x,t), ρᵧ(y,t)
- Produit Heisenberg ΔX·ΔY
- Norme conservation
"""

from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Optional, Dict, Any

from quantum_simulation.core.state import WaveFunctionState2D
from quantum_simulation.core.operators import PositionOperator, MomentumOperator

try:
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    import imageio_ffmpeg  # Fallback Python
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False


class QuantumDashboard2D:
    """
    Dashboard évolution 2D avec 6 sous-plots synchronisés.
    
    Layout:
        [Densité 2D]  [Marginales]  [Observables]
        [Courant J]   [Heisenberg]  [Conservation]
    """
    
    def __init__(self, output_dir: str = "./results/dashboards/", dpi: int = 120):
        """
        Args:
            output_dir: Dossier sortie vidéos
            dpi: Résolution animations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
    
    def create_evolution_dashboard(
        self,
        states: List[WaveFunctionState2D],
        times: np.ndarray,
        hbar: float,
        mass: float,
        output_name: str = "dashboard_evolution.gif",
        fps: int = 10,
        observables: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Crée animation dashboard complet.
        
        Args:
            states: Séquence états temporels
            times: Temps correspondants (s)
            hbar, mass: Constantes physiques
            output_name: Nom fichier sortie (.gif ou .mp4)
            fps: Images/seconde
            observables: Dict pré-calculé (optionnel, sinon calcul automatique)
        """
        if len(states) != len(times):
            raise ValueError(f"Longueurs incompatibles: {len(states)} vs {len(times)}")
        
        # Calcul observables si non fournis
        if observables is None:
            print("  Calcul observables temporels...")
            observables = self._compute_observables_evolution(states, hbar, mass)
        
        # Configuration figure
        fig = plt.figure(figsize=(18, 12), dpi=self.dpi)
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Axes
        ax_density = fig.add_subplot(gs[0, 0])      # Densité 2D
        ax_marginal = fig.add_subplot(gs[0, 1])     # Marginales
        ax_observables = fig.add_subplot(gs[0, 2])  # ⟨X⟩, ⟨Y⟩
        ax_current = fig.add_subplot(gs[1, 0])      # Courant probabilité
        ax_heisenberg = fig.add_subplot(gs[1, 1])   # ΔX·ΔY
        ax_norm = fig.add_subplot(gs[1, 2])         # Conservation norme
        
        # Limites couleur fixes
        all_densities = [state.probability_density() for state in states]
        vmin = min(np.min(rho) for rho in all_densities)
        vmax = max(np.max(rho) for rho in all_densities)
        
        # Initialisation plots
        self._init_density_plot(ax_density, states[0], vmin, vmax)
        self._init_marginal_plot(ax_marginal, states[0])
        self._init_observables_plot(ax_observables, times, observables)
        self._init_current_plot(ax_current, states[0], hbar, mass)
        self._init_heisenberg_plot(ax_heisenberg, times, observables, hbar)
        self._init_norm_plot(ax_norm, times, observables)
        
        # Texte temps global
        time_text = fig.text(
            0.5, 0.98,
            '',
            ha='center',
            va='top',
            fontsize=16,
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8)
        )
        
        # Fonction update
        def update_frame(i):
            """Mise à jour frame i."""
            # Densité 2D
            self._update_density_plot(ax_density, states[i])
            
            # Marginales
            self._update_marginal_plot(ax_marginal, states[i])
            
            # Courant
            self._update_current_plot(ax_current, states[i], hbar, mass)
            
            # Marqueurs temporels (observables, heisenberg, norm)
            self._update_time_markers(
                ax_observables, ax_heisenberg, ax_norm, i, times
            )
            
            # Temps
            time_text.set_text(f't = {times[i]*1e15:.2f} fs')
            
            return time_text,
        
        # Animation
        print(f"  Création animation ({len(states)} frames)...")
        anim = FuncAnimation(
            fig,
            update_frame,
            frames=len(states),
            interval=1000//fps,
            blit=False,
            repeat=True
        )
        
        # Sauvegarde
        filepath = self.output_dir / output_name
        
        if output_name.endswith('.mp4'):
            try:
                writer = FFMpegWriter(fps=fps, metadata={'artist': 'QuantumSim'})
                anim.save(filepath, writer=writer)
            except Exception as e:
                print(f"    ⚠️ MP4 échec ({e}), fallback GIF...")
                filepath = filepath.with_suffix('.gif')
                writer = PillowWriter(fps=fps)
                anim.save(filepath, writer=writer)
        else:
            writer = PillowWriter(fps=fps)
            anim.save(filepath, writer=writer)
        
        print(f"  ✓ Dashboard sauvegardé: {filepath}")
        plt.close()
        
        return str(filepath)
    
    # ==================== Fonctions auxiliaires ====================
    
    def _compute_observables_evolution(
        self,
        states: List[WaveFunctionState2D],
        hbar: float,
        mass: float
    ) -> Dict[str, np.ndarray]:
        """Calcule observables pour tous temps."""
        n_times = len(states)
        
        X_op = PositionOperator()
        P_op = MomentumOperator(hbar)
        
        observables = {
            'mean_x': np.zeros(n_times),
            'mean_y': np.zeros(n_times),
            'mean_px': np.zeros(n_times),
            'mean_py': np.zeros(n_times),
            'delta_x': np.zeros(n_times),
            'delta_y': np.zeros(n_times),
            'delta_px': np.zeros(n_times),
            'delta_py': np.zeros(n_times),
            'norm': np.zeros(n_times)
        }
        
        for i, state in enumerate(states):
            # Marginales 1D
            state_x = state.marginal_x()
            state_y = state.marginal_y()
            
            # Observables X
            observables['mean_x'][i] = X_op.expectation_value(state_x)
            observables['delta_x'][i] = X_op.uncertainty(state_x)
            
            # Observables Y (adapter PositionOperator pour Y)
            # Simplification: calcul direct
            Y_grid = state.y_grid
            rho_y = state_y.probability_density()
            observables['mean_y'][i] = np.sum(Y_grid * rho_y) * state_y.dx
            Y2 = np.sum(Y_grid**2 * rho_y) * state_y.dx
            observables['delta_y'][i] = np.sqrt(Y2 - observables['mean_y'][i]**2)
            
            # Impulsions (via FFT marginales)
            # Simplification: estimer depuis ⟨P⟩ ≈ m·d⟨X⟩/dt (Ehrenfest)
            observables['mean_px'][i] = P_op.expectation_value(state_x)
            observables['delta_px'][i] = P_op.uncertainty(state_x)
            
            # Norme
            observables['norm'][i] = state.norm()
        
        # Impulsion Y (estimation différences finies)
        dt = 1e-17  # Approximation
        observables['mean_py'] = np.gradient(observables['mean_y'], dt) * mass
        observables['delta_py'] = observables['delta_px']  # Simplification
        
        return observables
    
    def _init_density_plot(self, ax, state, vmin, vmax):
        """Initialise subplot densité 2D."""
        rho = state.probability_density()
        X, Y = np.meshgrid(state.x_grid, state.y_grid, indexing='ij')
        
        im = ax.pcolormesh(
            X * 1e9, Y * 1e9, rho,
            cmap='viridis', shading='auto',
            vmin=vmin, vmax=vmax
        )
        
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_title('Densité ρ(x,y,t)', fontweight='bold')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='ρ [m⁻²]')
        
        # Stockage objet image pour update
        ax._im = im
    
    def _update_density_plot(self, ax, state):
        """Mise à jour densité."""
        rho = state.probability_density()
        ax._im.set_array(rho.ravel())
    
    def _init_marginal_plot(self, ax, state):
        """Initialise marginales X et Y."""
        state_x = state.marginal_x()
        state_y = state.marginal_y()
        
        rho_x = state_x.probability_density()
        rho_y = state_y.probability_density()
        
        line_x, = ax.plot(state.x_grid * 1e9, rho_x, 'b-', linewidth=2, label='ρₓ(x)')
        line_y, = ax.plot(state.y_grid * 1e9, rho_y, 'r-', linewidth=2, label='ρᵧ(y)')
        
        ax.set_xlabel('Position (nm)')
        ax.set_ylabel('Densité')
        ax.set_title('Marginales', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax._line_x = line_x
        ax._line_y = line_y
        ax._x_grid = state.x_grid
        ax._y_grid = state.y_grid
    
    def _update_marginal_plot(self, ax, state):
        """Mise à jour marginales."""
        state_x = state.marginal_x()
        state_y = state.marginal_y()
        
        rho_x = state_x.probability_density()
        rho_y = state_y.probability_density()
        
        ax._line_x.set_ydata(rho_x)
        ax._line_y.set_ydata(rho_y)
        
        # Réajuster limites Y si nécessaire
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
    
    def _init_observables_plot(self, ax, times, obs):
        """Initialise évolution ⟨X⟩, ⟨Y⟩."""
        ax.plot(times * 1e15, obs['mean_x'] * 1e9, 'b-', linewidth=2, label='⟨X⟩')
        ax.plot(times * 1e15, obs['mean_y'] * 1e9, 'r-', linewidth=2, label='⟨Y⟩')
        
        # Marqueur temps actuel
        marker, = ax.plot([], [], 'go', markersize=10, label='t actuel')
        
        ax.set_xlabel('Temps (fs)')
        ax.set_ylabel('Position (nm)')
        ax.set_title('Observables ⟨X⟩, ⟨Y⟩', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax._marker = marker
    
    def _init_current_plot(self, ax, state, hbar, mass):
        """Initialise courant probabilité."""
        psi = state.wavefunction
        dx, dy = state.dx, state.dy
        
        grad_x = np.gradient(psi, dx, axis=0)
        grad_y = np.gradient(psi, dy, axis=1)
        
        Jx = (hbar / mass) * np.imag(np.conj(psi) * grad_x)
        Jy = (hbar / mass) * np.imag(np.conj(psi) * grad_y)
        
        X, Y = np.meshgrid(state.x_grid, state.y_grid, indexing='ij')
        
        rho = state.probability_density()
        im = ax.pcolormesh(X * 1e9, Y * 1e9, rho, cmap='gray', alpha=0.5, shading='auto')
        
        skip = 8
        quiv = ax.quiver(
            X[::skip, ::skip] * 1e9,
            Y[::skip, ::skip] * 1e9,
            Jx[::skip, ::skip],
            Jy[::skip, ::skip],
            color='red', scale=None
        )
        
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_title('Courant J(x,y,t)', fontweight='bold')
        ax.set_aspect('equal')
        
        ax._im = im
        ax._quiv = quiv
        ax._skip = skip
    
    def _update_current_plot(self, ax, state, hbar, mass):
        """Mise à jour courant."""
        psi = state.wavefunction
        dx, dy = state.dx, state.dy
        
        grad_x = np.gradient(psi, dx, axis=0)
        grad_y = np.gradient(psi, dy, axis=1)
        
        Jx = (hbar / mass) * np.imag(np.conj(psi) * grad_x)
        Jy = (hbar / mass) * np.imag(np.conj(psi) * grad_y)
        
        rho = state.probability_density()
        ax._im.set_array(rho.ravel())
        
        skip = ax._skip
        ax._quiv.set_UVC(Jx[::skip, ::skip], Jy[::skip, ::skip])
    
    def _init_heisenberg_plot(self, ax, times, obs, hbar):
        """Initialise produit Heisenberg."""
        heisenberg_product = obs['delta_x'] * obs['delta_y']
        heisenberg_bound = hbar / 2.0
        
        ax.plot(times * 1e15, heisenberg_product / heisenberg_bound, 'b-', linewidth=2, label='ΔX·ΔY / (ℏ/2)')
        ax.axhline(1.0, color='r', linestyle='--', linewidth=2, label='Limite ℏ/2')
        
        marker, = ax.plot([], [], 'go', markersize=10, label='t actuel')
        
        ax.set_xlabel('Temps (fs)')
        ax.set_ylabel('ΔX·ΔY / (ℏ/2)')
        ax.set_title('Produit Heisenberg', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.9, 1.5)
        
        ax._marker = marker
    
    def _init_norm_plot(self, ax, times, obs):
        """Initialise conservation norme."""
        deviation = np.abs(obs['norm'] - 1.0)
        
        ax.semilogy(times * 1e15, deviation, 'b-', linewidth=2)
        ax.axhline(1e-10, color='r', linestyle='--', linewidth=2, label='Tolérance 10⁻¹⁰')
        
        marker, = ax.plot([], [], 'go', markersize=10, label='t actuel')
        
        ax.set_xlabel('Temps (fs)')
        ax.set_ylabel('|Norme - 1|')
        ax.set_title('Conservation Norme', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        ax._marker = marker
    
    def _update_time_markers(self, ax_obs, ax_heis, ax_norm, i, times):
        """Mise à jour marqueurs temps sur courbes."""
        t_current = times[i] * 1e15
        
        # Observables
        # (Récupérer données depuis plot existant)
        mean_x_data = ax_obs.lines[0].get_ydata()
        ax_obs._marker.set_data([t_current], [mean_x_data[i]])
        
        # Heisenberg
        heis_data = ax_heis.lines[0].get_ydata()
        ax_heis._marker.set_data([t_current], [heis_data[i]])
        
        # Norme
        norm_data = ax_norm.lines[0].get_ydata()
        ax_norm._marker.set_data([t_current], [norm_data[i]])


if __name__ == "__main__":
    # Test dashboard
    from quantum_simulation.systems.free_particle_2d import FreeParticle2D
    
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    
    x = np.linspace(-2e-8, 2e-8, 128)
    y = np.linspace(-2e-8, 2e-8, 128)
    
    fp2d = FreeParticle2D(mass, hbar)
    state0 = fp2d.create_gaussian_packet_2d(
        x, y, x0=0, y0=0,
        sigma_x=3e-9, sigma_y=3e-9,
        kx0=5e9, ky0=3e9
    )
    
    # États simulés (translation)
    times = np.linspace(0, 1e-15, 20)
    states = [state0] * len(times)  # Simplification test
    
    dashboard = QuantumDashboard2D(output_dir='quantum_simulation/results/test_dashboard/')
    dashboard.create_evolution_dashboard(
        states, times, hbar, mass,
        output_name='test_dashboard.mp4',
        fps=5
    )
    
    print("✓ Dashboard test créé")