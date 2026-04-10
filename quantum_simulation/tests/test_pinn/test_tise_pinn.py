"""
Tests pour le solveur TISE PINN.

Ces tests utilisent un nombre réduit d'epochs et un petit réseau pour
s'exécuter rapidement (<60s). Les tolérances sont plus larges que celles
du notebook NB09 qui utilise des paramètres ambitieux.

Systèmes testés
---------------
- Oscillateur harmonique : V(x) = ½x² (unités atomiques ℏ=m=ω=1)
  E_n = n + ½  →  E_0 = 0.5, E_1 = 1.5
- Puits infini sur [-π/2, π/2] : V=0
  E_n = (n+1)² = 1, 4, 9  (unités atomiques ℏ=m=1, L=π/2)

Note : on vérifie que le PINN converge qualitativement vers les bonnes
valeurs, pas la précision spectrale du Crank-Nicolson.
"""

import numpy as np
import pytest
import torch

from quantum_simulation.pinn import TISESolver
from quantum_simulation.pinn.utils import check_normalization, energy_relative_error
from quantum_simulation.pinn.network import TISENet, SchrodingerNet


# ===========================================================================
# Potentiels de test (Tensors PyTorch)
# ===========================================================================

def harmonic_potential(x: torch.Tensor) -> torch.Tensor:
    """V(x) = ½x² — oscillateur harmonique (unités atomiques)."""
    return 0.5 * x**2


def zero_potential(x: torch.Tensor) -> torch.Tensor:
    """V(x) = 0 — puits infini (le confinement vient des CL)."""
    return torch.zeros_like(x)


# Énergie exacte du puits infini sur [-L, L], niveau n (n=1 fondamental)
def infinite_well_energy(n: int, L: float, hbar: float = 1.0, m: float = 1.0) -> float:
    """E_n = n²π²ℏ²/(2mL²) — puits infini centré."""
    return (n**2 * np.pi**2 * hbar**2) / (2.0 * m * (2 * L)**2)


# ===========================================================================
# Paramètres d'entraînement rapide pour les tests
# ===========================================================================

FAST_SOLVE_KWARGS = dict(
    n_epochs_adam=2000,
    n_steps_lbfgs=100,
    lr_adam=1e-3,
    lambda_pde=1.0,
    lambda_bc=10.0,
    lambda_norm=100.0,
    verbose=False,
)


# ===========================================================================
# Tests architecture réseau
# ===========================================================================

class TestNetworkArchitecture:
    """Tests unitaires de l'architecture sans entraînement."""

    def test_schrodinger_net_output_shape(self):
        """SchrodingerNet produit le bon shape de sortie."""
        net = SchrodingerNet(n_input=1, n_output=1, n_hidden=2, n_neurons=32)
        x = torch.rand(50, 1)
        out = net(x)
        assert out.shape == (50, 1)

    def test_tise_net_has_energy_parameter(self):
        """TISENet possède un paramètre énergie appris."""
        net = TISENet(x_domain=(-5.0, 5.0), n_hidden=2, n_neurons=32)
        assert hasattr(net, "E")
        assert isinstance(net.E, torch.nn.Parameter)

    def test_tise_net_normalizes_input(self):
        """TISENet normalise les entrées vers [-1, 1]."""
        net = TISENet(x_domain=(-5.0, 5.0), n_hidden=2, n_neurons=32)
        # x=-5 → -1, x=5 → 1
        x_min = torch.tensor([[-5.0]])
        x_max = torch.tensor([[5.0]])
        assert torch.allclose(net.normalize_input(x_min), torch.tensor([[-1.0]]))
        assert torch.allclose(net.normalize_input(x_max), torch.tensor([[1.0]]))

    def test_tise_net_forward_shape(self):
        """TISENet forward retourne le bon shape."""
        net = TISENet(x_domain=(-5.0, 5.0), n_hidden=2, n_neurons=32)
        x = torch.rand(100, 1)
        psi = net(x)
        assert psi.shape == (100, 1)

    def test_trial_function_satisfies_bc(self):
        """La trial function vaut 0 aux bords du domaine."""
        solver = TISESolver(
            potential_fn=zero_potential,
            x_domain=(-np.pi / 2, np.pi / 2),
            n_colloc=50,
            use_trial_function=True,
            n_hidden=2,
            n_neurons=32,
        )
        x_bc = torch.tensor([[-np.pi / 2], [np.pi / 2]], dtype=torch.float32)
        psi_bc = solver._apply_trial_function(solver.network(x_bc), x_bc)
        assert torch.all(torch.abs(psi_bc) < 1e-6), (
            f"Trial function non nulle aux bords : {psi_bc.detach().numpy()}"
        )


# ===========================================================================
# Tests solveur TISE — Oscillateur harmonique
# ===========================================================================

class TestTISEHarmonicOscillator:
    """Tests du solveur TISE sur l'oscillateur harmonique."""

    @pytest.fixture(scope="class")
    def solver_result(self):
        """Entraîne le solveur une seule fois pour tous les tests de la classe."""
        solver = TISESolver(
            potential_fn=harmonic_potential,
            x_domain=(-5.0, 5.0),
            n_colloc=300,
            n_quad=100,
            use_trial_function=False,
            n_hidden=3,
            n_neurons=64,
            E_init=0.5,
        )
        result = solver.solve(**FAST_SOLVE_KWARGS)
        return solver, result

    def test_ground_state_energy_in_range(self, solver_result):
        """E_0 prédit doit être proche de 0.5 (±15%)."""
        solver, result = solver_result
        E_pred = result["E"]
        E_exact = 0.5
        error = energy_relative_error(E_pred, E_exact)
        assert error < 0.15, (
            f"Énergie fondamentale HO : {E_pred:.4f} (exact : {E_exact}, "
            f"erreur : {error*100:.1f}%)"
        )

    def test_normalization(self, solver_result):
        """||ψ_pred||² ∈ [0.90, 1.10]."""
        solver, result = solver_result
        x_np = np.linspace(-5.0, 5.0, 500)
        psi = solver.predict(x_np)
        norm = check_normalization(psi, x_np)
        assert 0.90 <= norm <= 1.10, f"Normalisation : {norm:.4f}"

    def test_history_loss_decreasing(self, solver_result):
        """La perte totale doit diminuer pendant l'entraînement."""
        _, result = solver_result
        history = result["history"]["loss"]
        # Comparer premier quart vs dernier quart
        n = len(history)
        first_quarter_avg = np.mean(history[:n // 4])
        last_quarter_avg = np.mean(history[3 * n // 4:])
        assert last_quarter_avg < first_quarter_avg, (
            f"La perte ne diminue pas : {first_quarter_avg:.4e} → {last_quarter_avg:.4e}"
        )

    def test_energy_in_history(self, solver_result):
        """L'historique d'énergie doit converger (variance finale faible)."""
        _, result = solver_result
        E_history = result["history"]["E"]
        n = len(E_history)
        E_final_std = np.std(E_history[3 * n // 4:])
        assert E_final_std < 0.5, (
            f"Énergie non convergée (std finale : {E_final_std:.4f})"
        )


# ===========================================================================
# Tests solveur TISE — Puits infini avec trial function
# ===========================================================================

class TestTISEInfiniteWell:
    """Tests du solveur TISE sur le puits infini."""

    @pytest.fixture(scope="class")
    def solver_result(self):
        L = np.pi / 2
        solver = TISESolver(
            potential_fn=zero_potential,
            x_domain=(-L, L),
            n_colloc=200,
            n_quad=100,
            use_trial_function=True,
            n_hidden=3,
            n_neurons=64,
            E_init=1.0,
        )
        result = solver.solve(**FAST_SOLVE_KWARGS)
        return solver, result, L

    def test_boundary_conditions_satisfied(self, solver_result):
        """ψ aux bords doit être ≈ 0 (garanti par trial function)."""
        solver, result, L = solver_result
        x_bc = np.array([-L, L])
        psi_bc = solver.predict(x_bc)
        assert np.all(np.abs(psi_bc) < 1e-5), (
            f"Conditions aux limites non satisfaites : {psi_bc}"
        )

    def test_energy_positive(self, solver_result):
        """L'énergie propre doit être positive."""
        solver, result, L = solver_result
        assert result["E"] > 0, f"Énergie négative : {result['E']:.4f}"

    def test_normalization_with_trial(self, solver_result):
        """Normalisation avec trial function."""
        solver, result, L = solver_result
        x_np = np.linspace(-L, L, 500)
        psi = solver.predict(x_np)
        norm = check_normalization(psi, x_np)
        assert 0.85 <= norm <= 1.15, f"Normalisation avec trial : {norm:.4f}"


# ===========================================================================
# Tests utilitaires
# ===========================================================================

class TestPINNUtils:
    """Tests des fonctions utilitaires du module pinn."""

    def test_gauss_legendre_quadrature_integrates_constant(self):
        """∫₋₁¹ 1 dx = 2."""
        from quantum_simulation.pinn.utils import gauss_legendre_quadrature
        x_q, w_q = gauss_legendre_quadrature(50, -1.0, 1.0)
        integral = torch.sum(w_q).item()
        assert abs(integral - 2.0) < 1e-10, f"∫₋₁¹ 1 dx = {integral:.10f} ≠ 2"

    def test_gauss_legendre_integrates_polynomial(self):
        """∫₀¹ x² dx = 1/3."""
        from quantum_simulation.pinn.utils import gauss_legendre_quadrature
        x_q, w_q = gauss_legendre_quadrature(20, 0.0, 1.0)
        integral = torch.sum(w_q * x_q**2).item()
        assert abs(integral - 1.0 / 3.0) < 1e-8, f"∫₀¹ x² dx = {integral:.10f} ≠ 1/3"

    def test_l2_error_zero_for_identical_functions(self):
        """L2 error = 0 pour ψ_pred = ψ_exact."""
        from quantum_simulation.pinn.utils import l2_relative_error
        x = np.linspace(-5, 5, 200)
        psi = np.exp(-x**2 / 2)
        assert l2_relative_error(psi, psi) < 1e-10

    def test_l2_error_sign_invariant(self):
        """L2 error = 0 pour ψ_pred = -ψ_exact (signe global)."""
        from quantum_simulation.pinn.utils import l2_relative_error
        x = np.linspace(-5, 5, 200)
        psi = np.exp(-x**2 / 2)
        assert l2_relative_error(-psi, psi) < 1e-10

    def test_overlap_perfect(self):
        """Overlap = 1 pour fonctions identiques normalisées."""
        from quantum_simulation.pinn.utils import wavefunction_overlap
        x = np.linspace(-5, 5, 500)
        dx = x[1] - x[0]
        psi = np.exp(-x**2 / 2) / (np.pi**0.25)  # normalisée
        overlap = wavefunction_overlap(psi, psi, x)
        assert abs(overlap - 1.0) < 1e-4, f"Overlap : {overlap:.6f}"

    def test_check_normalization(self):
        """Gaussienne normalisée : ∫|ψ|² dx ≈ 1."""
        from quantum_simulation.pinn.utils import check_normalization
        x = np.linspace(-10, 10, 2000)
        psi = np.exp(-x**2 / 2) / (np.pi**0.25)
        norm = check_normalization(psi, x)
        assert abs(norm - 1.0) < 1e-3, f"Normalisation : {norm:.6f}"

    def test_uniform_collocation_shape(self):
        """uniform_collocation retourne (N, 1) avec requires_grad."""
        from quantum_simulation.pinn.utils import uniform_collocation
        x = uniform_collocation(-5.0, 5.0, 100)
        assert x.shape == (100, 1)
        assert x.requires_grad

    def test_boundary_points(self):
        """boundary_points retourne exactement les deux bords."""
        from quantum_simulation.pinn.utils import boundary_points
        x_bc = boundary_points(-3.0, 3.0)
        assert x_bc.shape == (2, 1)
        assert float(x_bc[0, 0]) == pytest.approx(-3.0)
        assert float(x_bc[1, 0]) == pytest.approx(3.0)
