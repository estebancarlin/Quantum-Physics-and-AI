"""
Tests pour le solveur TDSE PINN.

Ces tests utilisent un réseau minimal et peu d'epochs pour s'exécuter
rapidement. On vérifie les propriétés structurelles (shapes, dérivées,
condition initiale) plutôt que la précision physique fine.

Cas testés
----------
- Particule libre : V(x) = 0, ψ₀ = Gaussienne
- Structure du réseau TDSENet
- Décomposition réel/imaginaire
- Conservation qualitative de la norme
"""

import numpy as np
import pytest
import torch

from quantum_simulation.pinn import TDSESolver
from quantum_simulation.pinn.network import TDSENet
from quantum_simulation.pinn.losses import tdse_pde_loss
from quantum_simulation.pinn.utils import (
    spacetime_collocation,
    check_normalization,
)


# ===========================================================================
# Conditions initiales de test
# ===========================================================================

def gaussian_psi0(x: np.ndarray, x0: float = 0.0, sigma: float = 1.0, k0: float = 2.0):
    """
    Paquet d'ondes gaussien : ψ₀(x) = N·exp(-(x-x0)²/4σ²)·exp(ik₀x).

    Retourne (psi_real, psi_imag) pour la condition initiale TDSE.
    """
    envelope = np.exp(-(x - x0)**2 / (4.0 * sigma**2))
    norm = (2.0 * np.pi * sigma**2) ** (-0.25)
    psi_r = norm * envelope * np.cos(k0 * x)
    psi_i = norm * envelope * np.sin(k0 * x)
    return psi_r, psi_i


def zero_potential_tdse(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def barrier_potential(x: torch.Tensor, V0: float = 1.0, width: float = 0.5):
    """Barrière rectangulaire centrée en 0."""
    return torch.where(torch.abs(x) < width, torch.tensor(V0), torch.zeros_like(x))


# ===========================================================================
# Tests architecture TDSENet
# ===========================================================================

class TestTDSENetArchitecture:
    """Tests unitaires de l'architecture TDSENet."""

    def test_output_shape(self):
        """TDSENet retourne deux outputs (psi_real, psi_imag) de shape (N, 1)."""
        net = TDSENet(
            x_domain=(-5.0, 5.0),
            t_domain=(0.0, 1.0),
            n_hidden=2,
            n_neurons=32,
        )
        x = torch.rand(100, 1)
        t = torch.rand(100, 1)
        psi_r, psi_i = net(x, t)
        assert psi_r.shape == (100, 1)
        assert psi_i.shape == (100, 1)

    def test_input_normalization(self):
        """normalize_inputs mappe correctement les bords vers ±1."""
        net = TDSENet(
            x_domain=(-5.0, 5.0),
            t_domain=(0.0, 2.0),
            n_hidden=2,
            n_neurons=32,
        )
        x_min = torch.tensor([[-5.0]])
        t_min = torch.tensor([[0.0]])
        x_max = torch.tensor([[5.0]])
        t_max = torch.tensor([[2.0]])

        xt_min = net.normalize_inputs(x_min, t_min)
        xt_max = net.normalize_inputs(x_max, t_max)

        assert torch.allclose(xt_min, torch.tensor([[-1.0, -1.0]]), atol=1e-6)
        assert torch.allclose(xt_max, torch.tensor([[1.0, 1.0]]), atol=1e-6)

    def test_autograd_works(self):
        """Les dérivées autograd de psi_r et psi_i par rapport à x et t."""
        net = TDSENet(
            x_domain=(-5.0, 5.0),
            t_domain=(0.0, 1.0),
            n_hidden=2,
            n_neurons=32,
        )
        x = torch.rand(10, 1, requires_grad=True)
        t = torch.rand(10, 1, requires_grad=True)
        psi_r, psi_i = net(x, t)

        # Doit pouvoir calculer ∂ψ_r/∂x et ∂ψ_i/∂t sans erreur
        dpsi_r_dx = torch.autograd.grad(
            psi_r.sum(), x, create_graph=True, retain_graph=True
        )[0]
        dpsi_i_dt = torch.autograd.grad(
            psi_i.sum(), t, create_graph=True, retain_graph=True
        )[0]

        assert dpsi_r_dx.shape == (10, 1)
        assert dpsi_i_dt.shape == (10, 1)


# ===========================================================================
# Tests des pertes TDSE
# ===========================================================================

class TestTDSELosses:
    """Tests des fonctions de perte TDSE."""

    def test_pde_loss_returns_scalar(self):
        """tdse_pde_loss retourne deux scalaires (loss_re, loss_im)."""
        x = torch.rand(50, 1, requires_grad=True)
        t = torch.rand(50, 1, requires_grad=True)

        net = TDSENet(
            x_domain=(-5.0, 5.0),
            t_domain=(0.0, 1.0),
            n_hidden=2,
            n_neurons=32,
        )
        psi_r, psi_i = net(x, t)
        l_re, l_im = tdse_pde_loss(psi_r, psi_i, x, t, zero_potential_tdse)

        assert l_re.shape == ()
        assert l_im.shape == ()
        assert l_re.item() >= 0
        assert l_im.item() >= 0

    def test_ic_loss_zero_for_correct_ic(self):
        """IC loss = 0 si prédiction = cible."""
        from quantum_simulation.pinn.losses import tdse_initial_condition_loss
        psi_r = torch.rand(100, 1)
        psi_i = torch.rand(100, 1)
        loss = tdse_initial_condition_loss(psi_r, psi_i, psi_r, psi_i)
        assert loss.item() < 1e-10


# ===========================================================================
# Tests solveur TDSE
# ===========================================================================

class TestTDSESolverStructure:
    """Tests structurels du TDSESolver (sans entraînement complet)."""

    def test_solver_initializes(self):
        """TDSESolver s'initialise sans erreur."""
        solver = TDSESolver(
            potential_fn=zero_potential_tdse,
            psi0_fn=gaussian_psi0,
            x_domain=(-8.0, 8.0),
            t_domain=(0.0, 1.0),
            n_colloc=100,
            n_ic=50,
            n_hidden=2,
            n_neurons=32,
        )
        assert solver.network is not None

    def test_predict_shape(self):
        """predict() retourne deux arrays de la bonne shape."""
        solver = TDSESolver(
            potential_fn=zero_potential_tdse,
            psi0_fn=gaussian_psi0,
            x_domain=(-8.0, 8.0),
            t_domain=(0.0, 1.0),
            n_colloc=100,
            n_ic=50,
            n_hidden=2,
            n_neurons=32,
        )
        x = np.linspace(-8.0, 8.0, 200)
        psi_r, psi_i = solver.predict(x, t=0.5)
        assert psi_r.shape == (200,)
        assert psi_i.shape == (200,)

    def test_predict_density_non_negative(self):
        """La densité |ψ|² est toujours positive."""
        solver = TDSESolver(
            potential_fn=zero_potential_tdse,
            psi0_fn=gaussian_psi0,
            x_domain=(-8.0, 8.0),
            t_domain=(0.0, 1.0),
            n_colloc=100,
            n_ic=50,
            n_hidden=2,
            n_neurons=32,
        )
        x = np.linspace(-8.0, 8.0, 200)
        density = solver.predict_density(x, t=0.5)
        assert np.all(density >= 0.0)

    def test_initial_condition_approximately_satisfied_after_short_training(self):
        """
        Après un court entraînement (500 epochs), ψ(x, t=0) ≈ ψ₀(x).

        Test qualitatif : overlap > 0.5.
        """
        solver = TDSESolver(
            potential_fn=zero_potential_tdse,
            psi0_fn=gaussian_psi0,
            x_domain=(-8.0, 8.0),
            t_domain=(0.0, 1.0),
            n_colloc=200,
            n_ic=100,
            n_hidden=2,
            n_neurons=32,
        )
        solver.solve(
            n_epochs_adam=500,
            n_steps_lbfgs=0,
            lr_adam=1e-3,
            lambda_pde=0.1,
            lambda_ic=10.0,
            lambda_norm=0.0,
            verbose=False,
        )

        x = np.linspace(-8.0, 8.0, 300)
        psi_r_pred, psi_i_pred = solver.predict(x, t=0.0)
        psi_r_target, psi_i_target = gaussian_psi0(x)

        psi_pred = psi_r_pred + 1j * psi_i_pred
        psi_target = psi_r_target + 1j * psi_i_target

        from quantum_simulation.pinn.utils import wavefunction_overlap
        overlap = wavefunction_overlap(psi_pred, psi_target, x)
        assert overlap > 0.5, (
            f"Condition initiale mal apprise : overlap = {overlap:.4f} < 0.5"
        )

    def test_norm_conservation_qualitative(self):
        """
        Après entraînement court, la norme reste dans [0.5, 2.0] pour tous t.

        Test lâche car peu d'epochs — vérifie seulement qu'on n'explose pas.
        """
        solver = TDSESolver(
            potential_fn=zero_potential_tdse,
            psi0_fn=gaussian_psi0,
            x_domain=(-8.0, 8.0),
            t_domain=(0.0, 1.0),
            n_colloc=200,
            n_ic=100,
            n_hidden=2,
            n_neurons=32,
        )
        solver.solve(
            n_epochs_adam=300,
            n_steps_lbfgs=0,
            lambda_pde=1.0,
            lambda_ic=10.0,
            lambda_norm=50.0,
            verbose=False,
        )

        x = np.linspace(-8.0, 8.0, 300)
        norms = solver.check_norm_over_time(x, t_values=[0.0, 0.5, 1.0])

        for t, norm in zip([0.0, 0.5, 1.0], norms):
            assert 0.5 <= norm <= 2.0, f"Norme explosive à t={t} : {norm:.4f}"

    def test_spacetime_collocation_shape(self):
        """spacetime_collocation retourne les bons shapes."""
        x_c, t_c = spacetime_collocation(-5.0, 5.0, 0.0, 2.0, 1000)
        assert x_c.shape == (1000, 1)
        assert t_c.shape == (1000, 1)
        assert x_c.requires_grad
        assert t_c.requires_grad
