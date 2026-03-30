"""
Tests unitaires pour module core.operators.

Vérifie :
- R4.5 : Hermiticité observables (A† = A)
- R1.3 : Relations commutation [X,P] = iℏ
- R4.1, R4.2 : Valeurs moyennes et incertitudes
"""

import sys
from pathlib import Path
# HACK: Ajouter racine projet au path (temporaire)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from quantum_simulation.core.operators import (
    PositionOperator, MomentumOperator, Hamiltonian
)
from quantum_simulation.core.state import WaveFunctionState
from quantum_simulation.systems.free_particle import FreeParticle


def test_hamiltonian_hermiticity():
    """
    Règle R4.5 : H† = H (hamiltonien hermitique)
    Source : [file:1, Chapitre II, § D-1]
    
    Test : ⟨φ|H|ψ⟩* = ⟨ψ|H|φ⟩
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    
    x = np.linspace(-5e-8, 5e-8, 512)
    
    # Potentiel arbitraire
    def potential(x):
        return 0.5 * mass * (1e15)**2 * x**2  # Oscillateur harmonique
    
    H = Hamiltonian(mass, hbar, potential)
    
    # Deux états différents
    free_particle = FreeParticle(mass, hbar)
    psi = free_particle.create_gaussian_wavepacket(x, x0=-1e-8, sigma_x=1e-9, k0=3e9)
    phi = free_particle.create_gaussian_wavepacket(x, x0=1e-8, sigma_x=1.5e-9, k0=4e9)
    
    # Calcul ⟨φ|H|ψ⟩
    H_psi = H.apply(psi)
    phi_H_psi = phi.inner_product(H_psi)
    
    # Calcul ⟨ψ|H|φ⟩
    H_phi = H.apply(phi)
    psi_H_phi = psi.inner_product(H_phi)
    
    # Vérification hermiticité
    assert abs(np.conj(phi_H_psi) - psi_H_phi) < 1e-10, \
        "Hamiltonien doit être hermitique"


def test_momentum_expectation_gaussian():
    """
    Test ⟨P⟩ pour paquet gaussien avec impulsion k0.
    
    Théorie : ⟨P⟩ = ℏk0
    Règle R4.1
    
    Note:
        Erreur attendue ~4% due à :
        - Différences finies ordre 2 : O(dx²)
        - Effets bord grille finie
        - Troncature paquet gaussien
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    k0 = 5e9  # m^-1
    
    x = np.linspace(-10e-8, 10e-8, 2048)
    
    free_particle = FreeParticle(mass, hbar)
    state = free_particle.create_gaussian_wavepacket(x, x0=0, sigma_x=2e-9, k0=k0)
    
    P = MomentumOperator(hbar)
    mean_p = P.expectation_value(state)
    
    # Valeur théorique
    p_theory = hbar * k0
    
    # Tolérance 5% (justifiée par erreurs numériques)
    relative_error = abs(mean_p - p_theory) / abs(p_theory)
    assert relative_error < 0.05, \
        f"⟨P⟩ devrait être {p_theory:.2e}, obtenu {mean_p:.2e} " \
        f"(erreur relative {relative_error:.1%})"


def test_position_expectation_centered_gaussian():
    """
    Gaussienne centrée en x0 doit avoir ⟨X⟩ = x0.
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    x0 = 3e-9
    
    x = np.linspace(-10e-8, 10e-8, 1024)
    
    free_particle = FreeParticle(mass, hbar)
    state = free_particle.create_gaussian_wavepacket(x, x0=x0, sigma_x=2e-9, k0=0)
    
    X = PositionOperator()
    mean_x = X.expectation_value(state)
    
    assert abs(mean_x - x0) < 1e-10, \
        f"⟨X⟩ devrait être {x0:.2e}, obtenu {mean_x:.2e}"


def test_commutator_position_momentum():
    """
    Règle R1.3 : [X, P] = iℏ
    Source : [file:1, Chapitre III, § B-5-a]
    
    Test : ([X,P]ψ - iℏψ) doit être nul
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    
    x = np.linspace(-5e-8, 5e-8, 1024)
    
    # État test
    free_particle = FreeParticle(mass, hbar)
    state = free_particle.create_gaussian_wavepacket(x, x0=0, sigma_x=2e-9, k0=3e9)
    
    X = PositionOperator()
    P = MomentumOperator(hbar)
    
    # Calcul XP|ψ⟩
    P_psi = P.apply(state)
    XP_psi = X.apply(P_psi)
    
    # Calcul PX|ψ⟩
    X_psi = X.apply(state)
    PX_psi = P.apply(X_psi)
    
    # Commutateur [X,P]|ψ⟩
    commutator_psi = WaveFunctionState(
        state.spatial_grid,
        XP_psi.wavefunction - PX_psi.wavefunction
    )
    
    # iℏ|ψ⟩
    ihbar_psi = WaveFunctionState(
        state.spatial_grid,
        1j * hbar * state.wavefunction
    )
    
    # Différence
    diff = WaveFunctionState(
        state.spatial_grid,
        commutator_psi.wavefunction - ihbar_psi.wavefunction
    )
    
    # Norme différence doit être petite
    diff_norm = diff.norm()
    state_norm = state.norm()
    
    relative_diff = diff_norm / state_norm
    assert relative_diff < 0.01, \
        f"[X,P] devrait être iℏ, écart relatif {relative_diff:.2e}"


def test_uncertainty_minimum_gaussian():
    """
    Paquet gaussien k0=0 doit être proche état minimum incertitude.
    
    Pour gaussienne pure : ΔX·ΔP = ℏ/2
    Règle R4.2, R4.3
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    sigma_x = 2e-9
    
    x = np.linspace(-20e-8, 20e-8, 4096)
    
    free_particle = FreeParticle(mass, hbar)
    state = free_particle.create_gaussian_wavepacket(x, x0=0, sigma_x=sigma_x, k0=0)
    
    X = PositionOperator()
    P = MomentumOperator(hbar)
    
    delta_x = X.uncertainty(state)
    delta_p = P.uncertainty(state)
    
    product = delta_x * delta_p
    bound = hbar / 2.0
    
    # Vérifier proche minimum (tolérance 5% car discrétisation)
    relative_excess = (product - bound) / bound
    assert relative_excess < 0.05, \
        f"Gaussienne devrait avoir ΔX·ΔP ≈ ℏ/2, obtenu {relative_excess:.2%} excès"


def test_expectation_value_real():
    """
    Valeur moyenne observable hermitique doit être réelle.
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    
    x = np.linspace(-5e-8, 5e-8, 512)
    
    free_particle = FreeParticle(mass, hbar)
    state = free_particle.create_gaussian_wavepacket(x, x0=0, sigma_x=2e-9, k0=5e9)
    
    X = PositionOperator()
    P = MomentumOperator(hbar)
    
    mean_x = X.expectation_value(state)
    mean_p = P.expectation_value(state)
    
    # Pas de partie imaginaire
    assert isinstance(mean_x, (int, float)), "⟨X⟩ doit être réel"
    assert isinstance(mean_p, (int, float)), "⟨P⟩ doit être réel"


def test_uncertainty_positive():
    """
    ΔA ≥ 0 toujours (écart quadratique).
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    
    x = np.linspace(-5e-8, 5e-8, 512)
    
    free_particle = FreeParticle(mass, hbar)
    state = free_particle.create_gaussian_wavepacket(x, x0=0, sigma_x=2e-9, k0=3e9)
    
    X = PositionOperator()
    P = MomentumOperator(hbar)
    
    delta_x = X.uncertainty(state)
    delta_p = P.uncertainty(state)
    
    assert delta_x >= 0, "ΔX doit être positif"
    assert delta_p >= 0, "ΔP doit être positif"


def test_hamiltonian_free_particle_eigenvalue():
    """
    Onde plane doit être état propre H = P²/2m avec E = ℏ²k²/2m.
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    k = 5e9
    
    x = np.linspace(-5e-8, 5e-8, 1024)
    
    # Potentiel nul
    def potential(x):
        return np.zeros_like(x)
    
    H = Hamiltonian(mass, hbar, potential)
    
    # Onde plane (approximation sur grille finie)
    free_particle = FreeParticle(mass, hbar)
    state = free_particle.create_plane_wave(x, k)
    
    # Énergie théorique
    E_theory = (hbar**2 * k**2) / (2 * mass)
    
    # ⟨H⟩ pour onde plane devrait être E
    E_measured = H.expectation_value(state)
    
    # Tolérance large (onde plane tronquée)
    relative_error = abs(E_measured - E_theory) / E_theory
    assert relative_error < 0.1, \
        f"Énergie onde plane devrait être {E_theory:.2e}, obtenu {E_measured:.2e}"