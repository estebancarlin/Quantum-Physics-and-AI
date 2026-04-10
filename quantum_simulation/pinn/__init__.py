"""
quantum_simulation/pinn
========================
Module Physics-Informed Neural Networks (PINNs) pour l'équation de Schrödinger.

Ce module implémente deux solveurs PINN complémentaires aux méthodes
numériques classiques (Crank-Nicolson, opérateur splitté) du framework.

Solveurs disponibles
---------------------
TISESolver : Résout l'ETIS (Ĥψ = Eψ) — états propres et énergies propres.
TDSESolver : Résout l'ETDS (iℏ∂ψ/∂t = Ĥψ) — dynamique temporelle.

Réseaux
--------
TISENet           : MLP tanh pour TISE (énergie propre comme paramètre appris)
TDSENet           : MLP tanh pour TDSE (sorties ψ_real et ψ_imag)
FourierFeatureTISENet : Variante avec encodage positionnel Fourier

Fonctions de perte
-------------------
tise_total_loss : Perte TISE (résidu EDP + conditions aux limites + normalisation)
tdse_total_loss : Perte TDSE (résidus réel/imaginaire + condition initiale)

Utilitaires
-----------
gauss_legendre_quadrature : Intégration numérique précise pour normalisation
uniform_collocation       : Points de collocation uniformes
spacetime_collocation     : Points (x, t) pour TDSE
validation_report         : Comparaison prédictions vs solutions analytiques

Exemple rapide
--------------
>>> from quantum_simulation.pinn import TISESolver
>>> import torch
>>>
>>> def harmonic_potential(x):
...     return 0.5 * x**2
>>>
>>> solver = TISESolver(
...     potential_fn=harmonic_potential,
...     x_domain=(-5.0, 5.0),
...     use_trial_function=False,
... )
>>> result = solver.solve(n_epochs_adam=3000, n_steps_lbfgs=200, verbose=True)
>>> print(f"Énergie fondamentale : {result['E']:.4f} (exact : 0.5000)")

Références
----------
- Raissi et al. 2019 — Physics-informed neural networks (JCP 378:686–707)
- arxiv:2210.12522 — PINNs as Solvers for the Time-Dependent Schrödinger Eq.
- arxiv:2504.05367 — PINN solvers pour puits quantiques 1D
- arxiv:2405.13442 — Oscillateur anharmonique via PINNs
"""

from .network import SchrodingerNet, TISENet, TDSENet, FourierFeatureTISENet
from .losses import (
    tise_total_loss,
    tdse_total_loss,
    tise_pde_loss,
    tise_normalization_loss,
    tise_boundary_loss,
    tdse_pde_loss,
    tdse_initial_condition_loss,
)
from .utils import (
    gauss_legendre_quadrature,
    uniform_collocation,
    random_collocation,
    spacetime_collocation,
    boundary_points,
    l2_relative_error,
    wavefunction_overlap,
    check_normalization,
    energy_relative_error,
    validation_report,
    to_tensor,
    to_numpy,
    get_device,
)
from .tise_solver import TISESolver
from .tdse_solver import TDSESolver

__all__ = [
    # Solveurs
    "TISESolver",
    "TDSESolver",
    # Réseaux
    "SchrodingerNet",
    "TISENet",
    "TDSENet",
    "FourierFeatureTISENet",
    # Pertes TISE
    "tise_total_loss",
    "tise_pde_loss",
    "tise_normalization_loss",
    "tise_boundary_loss",
    # Pertes TDSE
    "tdse_total_loss",
    "tdse_pde_loss",
    "tdse_initial_condition_loss",
    # Utilitaires
    "gauss_legendre_quadrature",
    "uniform_collocation",
    "random_collocation",
    "spacetime_collocation",
    "boundary_points",
    "l2_relative_error",
    "wavefunction_overlap",
    "check_normalization",
    "energy_relative_error",
    "validation_report",
    "to_tensor",
    "to_numpy",
    "get_device",
]
