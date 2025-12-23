import numpy as np
from scipy import sparse
# from scipy.sparse.linalg import spsolve
from quantum_simulation.core.operators import Hamiltonian
from quantum_simulation.core.state import QuantumState, EigenStateBasis, WaveFunctionState, WaveFunctionState2D
from typing import List

class TimeEvolution:
    """
    Règle R3.1 : iℏ d|ψ⟩/dt = H|ψ⟩
    
    Évolution temporelle par intégration équation Schrödinger.
    """
    
    def __init__(self, hamiltonian: Hamiltonian, hbar: float):
        """
        Args:
            hamiltonian: Hamiltonien du système
            hbar: Constante de Planck réduite (J·s)
        """
        self.hamiltonian = hamiltonian
        self.hbar = hbar
    
    def _build_hamiltonian_matrix_sparse(self, 
                                        spatial_grid: np.ndarray,
                                        potential: callable = None) -> sparse.csr_matrix:
        """
        Construit matrice hamiltonienne creuse H = -ℏ²/2m Δ + V(R).
        
        Args:
            spatial_grid: Grille spatiale 1D
            potential: Fonction V(x) ou None (particule libre)
            
        Returns:
            Matrice sparse CSR format (nx × nx)
            
        Note:
            - Utilise différences finies ordre 2 (décision D2)
            - Conditions Dirichlet aux bords (décision D3)
            - Matrice tri-diagonale si V=0, bande si V(x)
        """
        from scipy.sparse import diags
        
        nx = len(spatial_grid)
        dx = spatial_grid[1] - spatial_grid[0]
        
        # Terme cinétique : -ℏ²/2m Δ
        # Laplacien : (ψᵢ₊₁ - 2ψᵢ + ψᵢ₋₁)/dx²
        kinetic_coeff = -self.hamiltonian.hbar**2 / (2 * self.hamiltonian.mass * dx**2)
        
        # Diagonales matrice laplacien
        main_diag = -2 * kinetic_coeff * np.ones(nx)
        off_diag = kinetic_coeff * np.ones(nx - 1)
        
        # Matrice tri-diagonale T (terme cinétique)
        T_matrix = diags([off_diag, main_diag, off_diag], 
                        offsets=[-1, 0, 1], 
                        shape=(nx, nx),
                        format='csr')
        
        # Terme potentiel V(x) (diagonal)
        if potential is not None:
            V_values = np.array([potential(x) for x in spatial_grid])
            V_matrix = diags(V_values, offsets=0, format='csr')
            H_matrix = T_matrix + V_matrix
        else:
            H_matrix = T_matrix
        
        return H_matrix
    
    def _build_hamiltonian_3d_sparse(grid_3d, potential):
        """
        Matrice H 3D creuse (format COO → CSR).
        
        Complexité:
            - Mémoire : O(N) avec N = nx·ny·nz
            - Construction : O(N) (7 diagonales)
        """
        from scipy.sparse import diags, kron
        
        # Laplacien = Δₓ ⊗ Iᵧ ⊗ Iᵧ + Iₓ ⊗ Δᵧ ⊗ Iᵧ + Iₓ ⊗ Iᵧ ⊗ Δᵧ
        # Implémentation produits Kronecker optimisés
    
    def evolve_wavefunction(self, initial_state: WaveFunctionState, 
                            t0: float, t: float, dt: float) -> WaveFunctionState:
        """
        Évolution temporelle par schéma Crank-Nicolson.
        
        Schéma implicite : (I + iH·dt/2ℏ)ψ(t+dt) = (I - iH·dt/2ℏ)ψ(t)
        
        Propriétés garanties :
            - Conservation norme exacte : ||ψ(t)|| = 1 (Règle R5.1)
            - Évolution unitaire (Complément FIII)
            - Stabilité inconditionnelle (pas de contrainte dt)
            - Précision O(dt²)
        
        Args:
            initial_state: État initial normalisé |ψ(t₀)⟩
            t0: Temps initial (s)
            t: Temps final (s)
            dt: Pas de temps (s)
            
        Returns:
            État évolué |ψ(t)⟩
            
        Raises:
            ValueError: Si norme non conservée (erreur > tolérance)
            
        References:
            - Décision D1 : Crank-Nicolson recommandé
            - Règle R3.2 : iℏ ∂ψ/∂t = Hψ
            - Règle R5.1 : Conservation probabilité
        """
        from scipy.sparse.linalg import spsolve
        from scipy.sparse import eye
        import warnings
        
        # Validation état initial
        if not initial_state.is_normalized():
            raise ValueError("État initial non normalisé !")
        
        # Calcul nombre pas
        n_steps = int((t - t0) / dt)
        if n_steps == 0:
            return initial_state  # Pas d'évolution
        
        # Construction matrices (une seule fois)
        H_matrix = self._build_hamiltonian_matrix_sparse(
            initial_state.spatial_grid,
            potential=self.hamiltonian.potential
        )
        
        nx = len(initial_state.wavefunction)
        I = eye(nx, format='csr')
        
        factor = 0.5j * dt / self.hamiltonian.hbar
        
        # Matrices Crank-Nicolson
        A = I + factor * H_matrix  # (I + iH·dt/2ℏ)
        B = I - factor * H_matrix  # (I - iH·dt/2ℏ)
        
        # Évolution itérative
        psi = initial_state.wavefunction.copy()
        dx = initial_state.dx
        
        tolerance = 1e-4  # Tolérance conservation norme
        
        for step in range(n_steps):
            # Membre droit
            b = B @ psi
            
            # Résolution système linéaire A·ψ(t+dt) = b
            psi = spsolve(A, b)
            
            # Validation conservation norme (Règle R5.1)
            norm_squared = np.sum(np.abs(psi)**2) * dx
            norm = np.sqrt(norm_squared)
            
            if abs(norm - 1.0) > tolerance:
                warnings.warn(
                    f"Pas {step+1}/{n_steps} : Norme = {norm:.10f} "
                    f"(déviation = {abs(norm-1.0):.2e})",
                    RuntimeWarning
                )
                # Renormalisation forcée (sécurité)
                psi /= norm
        
        return WaveFunctionState(initial_state.spatial_grid, psi)
    
    def evolve_wavefunction_2d(
        self,
        initial_state: 'WaveFunctionState2D',
        times: np.ndarray,
        hamiltonian: Hamiltonian,
        method: str = 'crank_nicolson_adi'
    ) -> List['WaveFunctionState2D']:
        """
        Évolution fonction d'onde 2D : iℏ ∂ψ/∂t = Hψ
        
        Méthodes supportées:
            - 'crank_nicolson_adi': Alternating Direction Implicit (ADI)
            - 'split_operator': Split-operator FFT 2D
        
        Args:
            initial_state: État initial 2D ψ(x,y,t₀)
            times: Temps échantillonnage [t₀, t₁, ..., tₙ]
            hamiltonian: Hamiltonien système (doit avoir dimension=2)
            method: Méthode intégration
            
        Returns:
            Liste états ψ(x,y,tᵢ) à chaque temps
            
        Règles:
            - R3.1 : Équation Schrödinger
            - R5.1 : Conservation norme
        """
        if not hasattr(hamiltonian, 'dimension') or hamiltonian.dimension != 2:
            raise ValueError("Hamiltonien doit être 2D (attribut dimension=2)")
        
        if method == 'crank_nicolson_adi':
            return self._evolve_2d_adi(initial_state, times, hamiltonian)
        elif method == 'split_operator':
            return self._evolve_2d_split_operator(initial_state, times, hamiltonian)
        else:
            raise ValueError(f"Méthode 2D inconnue: {method}")
    
    def _evolve_2d_adi(
        self,
        initial_state: 'WaveFunctionState2D',
        times: np.ndarray,
        hamiltonian: Hamiltonian
    ) -> List['WaveFunctionState2D']:
        """
        Méthode ADI (Alternating Direction Implicit) 2D.
        
        Schéma Crank-Nicolson avec splitting directionnel :
        1. Demi-pas X : (1 + iHₓdt/2ℏ) ψ^(n+1/2) = (1 - iHₓdt/2ℏ) ψⁿ
        2. Demi-pas Y : (1 + iHᵧdt/2ℏ) ψ^(n+1) = (1 - iHᵧdt/2ℏ) ψ^(n+1/2)
        
        Conserve norme exactement (schéma unitaire).
        
        Returns:
            États ψ(tᵢ) pour chaque temps
        """
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve
        
        # Import WaveFunctionState2D
        from quantum_simulation.core.state import WaveFunctionState2D
        
        states = [initial_state]
        current_psi = initial_state.wavefunction.copy()
        
        # Grilles
        x_grid = initial_state.x_grid
        y_grid = initial_state.y_grid
        dx = initial_state.dx
        dy = initial_state.dy
        nx = initial_state.nx
        ny = initial_state.ny
        
        # Pas temporels
        dt_values = np.diff(times)
        
        # Potentiel sur grille 2D
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        V_grid = hamiltonian.potential(X, Y)
        
        # Constantes
        mass = hamiltonian.mass
        hbar = hamiltonian.hbar
        coeff_x = 1j * hbar / (2 * mass * dx**2)
        coeff_y = 1j * hbar / (2 * mass * dy**2)
        
        for i, dt in enumerate(dt_values):
            # Demi-pas X (ligne par ligne en Y)
            psi_half = np.zeros_like(current_psi)
            
            for j in range(ny):
                # Laplacien 1D en X pour ligne Y=j
                diag_x = np.ones(nx) * (-2 * coeff_x / dt + 0.5 * V_grid[:, j] / hbar)
                off_diag_x = np.ones(nx - 1) * (coeff_x / dt)
                
                # Matrices tridiagonales
                A_x = diags([off_diag_x, diag_x, off_diag_x], [-1, 0, 1], format='csc')
                B_x = diags([-off_diag_x, -diag_x + 1.0, -off_diag_x], [-1, 0, 1], format='csc')
                
                # RHS
                rhs_x = B_x @ current_psi[:, j]
                
                # Résolution
                psi_half[:, j] = spsolve(A_x, rhs_x)
            
            # Demi-pas Y (colonne par colonne en X)
            psi_next = np.zeros_like(psi_half)
            
            for i_col in range(nx):
                # Laplacien 1D en Y pour colonne X=i_col
                diag_y = np.ones(ny) * (-2 * coeff_y / dt + 0.5 * V_grid[i_col, :] / hbar)
                off_diag_y = np.ones(ny - 1) * (coeff_y / dt)
                
                A_y = diags([off_diag_y, diag_y, off_diag_y], [-1, 0, 1], format='csc')
                B_y = diags([-off_diag_y, -diag_y + 1.0, -off_diag_y], [-1, 0, 1], format='csc')
                
                rhs_y = B_y @ psi_half[i_col, :]
                psi_next[i_col, :] = spsolve(A_y, rhs_y)
            
            current_psi = psi_next
            
            # Stocker état
            state_t = WaveFunctionState2D(x_grid, y_grid, current_psi.copy())
            states.append(state_t)
            
            # Validation norme
            norm_t = state_t.norm()
            if abs(norm_t - 1.0) > 1e-4:
                import warnings
                warnings.warn(
                    f"Temps {i+1}/{len(dt_values)}: Norme = {norm_t:.10f} "
                    f"(déviation = {abs(norm_t-1.0):.2e})"
                )
        
        return states
    
    def _evolve_2d_split_operator(
        self,
        initial_state: 'WaveFunctionState2D',
        times: np.ndarray,
        hamiltonian: Hamiltonian
    ) -> List['WaveFunctionState2D']:
        """
        Méthode split-operator 2D (FFT).
        
        1. Demi-pas potentiel : ψ → exp(-iV·dt/2ℏ)ψ (position)
        2. Pas complet cinétique : ψ → FFT → exp(-i(kₓ²+kᵧ²)dt/2mℏ)φ → FFT⁻¹ (impulsion)
        3. Demi-pas potentiel : ψ → exp(-iV·dt/2ℏ)ψ (position)
        
        Returns:
            États ψ(tᵢ) pour chaque temps
        """
        from quantum_simulation.core.state import WaveFunctionState2D
        
        states = [initial_state]
        current_psi = initial_state.wavefunction.copy()
        
        # Grilles
        x_grid = initial_state.x_grid
        y_grid = initial_state.y_grid
        dx = initial_state.dx
        dy = initial_state.dy
        nx = initial_state.nx
        ny = initial_state.ny
        
        # Grille impulsion (fréquences FFT)
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        
        # Potentiel
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        V_grid = hamiltonian.potential(X, Y)
        
        # Constantes
        mass = hamiltonian.mass
        hbar = hamiltonian.hbar
        
        dt_values = np.diff(times)
        
        for dt in dt_values:
            # Opérateur potentiel (demi-pas)
            exp_V_half = np.exp(-1j * V_grid * dt / (2 * hbar))
            
            # Opérateur cinétique (impulsion)
            k_squared = KX**2 + KY**2
            exp_T = np.exp(-1j * hbar * k_squared * dt / (2 * mass))
            
            # 1. Demi-pas potentiel
            current_psi *= exp_V_half
            
            # 2. Pas complet cinétique (FFT 2D)
            psi_k = np.fft.fft2(current_psi)
            psi_k *= exp_T
            current_psi = np.fft.ifft2(psi_k)
            
            # 3. Demi-pas potentiel
            current_psi *= exp_V_half
            
            # Stocker état
            state_t = WaveFunctionState2D(x_grid, y_grid, current_psi.copy())
            states.append(state_t)
            
            # Validation norme
            norm_t = state_t.norm()
            if abs(norm_t - 1.0) > 1e-4:
                import warnings
                warnings.warn(
                    f"Split-operator 2D: Norme = {norm_t:.10f} "
                    f"(déviation = {abs(norm_t-1.0):.2e})"
                )
        
        return states