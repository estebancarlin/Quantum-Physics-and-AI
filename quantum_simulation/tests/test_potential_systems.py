"""
Tests unitaires pour systèmes à potentiel.

Vérifie :
- Spectre énergétique puits infini : Eₙ = n²π²ℏ²/(2mL²)
- Normalisation états propres
- Conditions aux bords : ψ(0) = ψ(L) = 0
- Orthonormalité : ⟨ψₙ|ψₘ⟩ = δₙₘ
- Estimation états liés puits fini
- Coefficients transmission barrière
"""

import sys
from pathlib import Path
# HACK: Ajouter racine projet au path (temporaire)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from quantum_simulation.systems.potential_systems import (
    InfiniteWell, FiniteWell, PotentialBarrier
)


def test_infinite_well_energy_spectrum():
    """
    Énergies puits infini : Eₙ = n²π²ℏ²/(2mL²)
    
    Test formule analytique exacte.
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    width = 1e-9
    
    well = InfiniteWell(width, mass, hbar)
    
    for n in range(1, 10):
        E_n = well.energy_eigenvalue(n)
        E_theory = (n**2 * np.pi**2 * hbar**2) / (2 * mass * width**2)
        
        relative_error = abs(E_n - E_theory) / E_theory
        assert relative_error < 1e-15, \
            f"Énergie E_{n} incorrecte : {E_n:.3e} vs {E_theory:.3e}"


def test_infinite_well_wavefunction_normalization():
    """
    États propres puits infini doivent être normés.
    
    Note:
        Normalisation analytique √(2/L) donne erreur O(dx²) sur grille discrète.
        Pour n grand, oscillations rapides augmentent erreur d'intégration.
        Tolérance adaptée : 1e-6 (au lieu de 1e-8).
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    width = 1e-9
    
    x = np.linspace(-width, 2*width, 2048)
    
    well = InfiniteWell(width, mass, hbar)
    
    for n in [1, 2, 5]:
        state = well.eigenstate_wavefunction(n, x)
        norm = state.norm()
        
        # Tolérance 1e-6 : acceptable pour n ≤ 10 sur grille 2048 points
        assert abs(norm - 1.0) < 1e-6, \
            f"État n={n} non normé : norme={norm:.10f}"


def test_infinite_well_boundary_conditions():
    """
    ψ(0) = ψ(L) = 0 pour puits infini.
    
    Note:
        Grille doit contenir exactement x=0 et x=L pour tester conditions limites.
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    width = 1e-9
    
    # Grille alignée sur bords [0, L]
    x = np.linspace(0, width, 1024)
    
    well = InfiniteWell(width, mass, hbar)
    
    for n in [1, 3, 7]:
        state = well.eigenstate_wavefunction(n, x)
        psi = state.wavefunction
        
        # Vérifier bords
        assert abs(psi[0]) < 1e-10, \
            f"ψ_{n}(0) = {abs(psi[0]):.2e} devrait être nul"
        assert abs(psi[-1]) < 1e-10, \
            f"ψ_{n}(L) = {abs(psi[-1]):.2e} devrait être nul"


def test_infinite_well_orthonormality():
    """
    ⟨ψₙ|ψₘ⟩ = δₙₘ pour puits infini.
    
    Test propriété fondamentale base d'états propres.
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    width = 1e-9
    
    x = np.linspace(-width/2, 1.5*width, 2048)
    
    well = InfiniteWell(width, mass, hbar)
    
    is_orthonormal = well.validate_orthonormality(n_states=5, x=x, tolerance=1e-6)
    assert is_orthonormal, "États propres devraient être orthonormés"


def test_finite_well_bound_states_estimate():
    """
    Nombre états liés puits fini doit être cohérent.
    
    Formule approximative : N ≈ √(2mV₀L²/π²ℏ²)
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    width = 1e-9
    depth = 1e-18  # ~6 eV
    
    well = FiniteWell(width, depth, mass, hbar)
    
    n_bound = well.estimate_bound_states_count()
    
    # Doit être au moins 1
    assert n_bound >= 1, "Puits fini doit avoir au moins 1 état lié"
    
    # Doit être fini (pas trop grand)
    assert n_bound < 100, f"Nombre états liés suspects : {n_bound}"
    
    # Vérification ordre grandeur
    # ξ₀ ≈ 7.5 → N ≈ 4-5 états
    assert 3 <= n_bound <= 10, f"Nombre états liés hors attentes : {n_bound}"


def test_barrier_transmission_classical():
    """
    Transmission sur-barrière (E > V₀) devrait être ~1.
    
    Limite classique : particule passe sans obstacle.
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    width = 1e-9
    height = 1e-19
    
    barrier = PotentialBarrier(width, height, mass, hbar)
    
    # Énergie > hauteur
    E = 2 * height
    T = barrier.transmission_coefficient_approx(E)
    
    assert T > 0.9, f"Transmission classique devrait être ~1, obtenu {T:.2f}"


def test_barrier_transmission_tunnel():
    """
    Transmission sous-barrière (effet tunnel) : 0 < T < 1.
    
    Phénomène purement quantique : transmission partielle malgré E < V₀.
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    width = 1e-9
    height = 1e-18
    
    barrier = PotentialBarrier(width, height, mass, hbar)
    
    # Énergie < hauteur
    E = 0.5 * height
    T = barrier.transmission_coefficient_approx(E)
    
    assert 0 < T < 1, f"Transmission tunnel hors [0,1] : {T:.2e}"
    assert T < 0.5, f"Transmission barrière opaque devrait être faible, obtenu {T:.2e}"


def test_potential_continuity():
    """
    Potentiels doivent être continus par morceaux (pas de NaN/inf).
    
    Vérifie intégrité numérique potentiels.
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    width = 1e-9
    
    x = np.linspace(-2*width, 2*width, 1024)
    
    # Puits infini (autorise inf aux murs)
    well = InfiniteWell(width, mass, hbar)
    V_well = well.potential(x)
    assert not np.any(np.isnan(V_well)), "Potentiel puits contient NaN"
    
    # Puits fini
    finite_well = FiniteWell(width, 1e-18, mass, hbar)
    V_finite = finite_well.potential(x)
    assert not np.any(np.isnan(V_finite)), "Potentiel puits fini contient NaN"
    assert np.all(np.isfinite(V_finite)), "Potentiel puits fini contient inf"
    
    # Barrière
    barrier = PotentialBarrier(width, 1e-19, mass, hbar)
    V_barrier = barrier.potential(x)
    assert not np.any(np.isnan(V_barrier)), "Potentiel barrière contient NaN"
    assert np.all(np.isfinite(V_barrier)), "Potentiel barrière contient inf"


def test_infinite_well_invalid_n():
    """
    Test gestion erreur : n < 1 doit lever ValueError.
    """
    hbar = 1.054571817e-34
    mass = 9.1093837015e-31
    width = 1e-9
    
    well = InfiniteWell(width, mass, hbar)
    
    with pytest.raises(ValueError, match="n doit être ≥ 1"):
        well.energy_eigenvalue(0)
    
    with pytest.raises(ValueError, match="n doit être ≥ 1"):
        x = np.linspace(0, width, 100)
        well.eigenstate_wavefunction(-1, x)