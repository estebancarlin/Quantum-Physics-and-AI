"""
quantum_simulation/pinn/utils.py
==================================
Utilitaires pour PINNs : quadrature, collocation, métriques de validation.
"""

import numpy as np
import torch
from typing import Callable


# ===========================================================================
# Quadrature numérique
# ===========================================================================

def gauss_legendre_quadrature(
    n_points: int,
    a: float,
    b: float,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Points et poids de Gauss-Legendre pour ∫ₐᵇ f(x) dx ≈ Σᵢ wᵢ f(xᵢ).

    Paramètres
    ----------
    n_points : nombre de points de quadrature
    a, b     : bornes d'intégration
    device   : device torch (cpu ou cuda)

    Retourne
    --------
    x_quad : Tensor (n_points, 1)
    w_quad : Tensor (n_points, 1)
    """
    xi, wi = np.polynomial.legendre.leggauss(n_points)
    # Changement de variable : [-1, 1] → [a, b]
    x_quad = 0.5 * (b - a) * xi + 0.5 * (a + b)
    w_quad = 0.5 * (b - a) * wi

    x_t = torch.tensor(x_quad, dtype=torch.float32).unsqueeze(1)
    w_t = torch.tensor(w_quad, dtype=torch.float32).unsqueeze(1)

    if device is not None:
        x_t = x_t.to(device)
        w_t = w_t.to(device)

    return x_t, w_t


def trapz_integrate(
    f: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Intégrale par trapèzes : ∫ f dx sur la grille x.

    Paramètres
    ----------
    f : Tensor (N,) ou (N, 1)
    x : Tensor (N,) ou (N, 1) — doit être trié

    Retourne
    --------
    Tensor scalaire
    """
    f_ = f.squeeze()
    x_ = x.squeeze()
    return torch.trapz(f_, x_)


# ===========================================================================
# Points de collocation
# ===========================================================================

def uniform_collocation(
    x_min: float,
    x_max: float,
    n_points: int,
    requires_grad: bool = True,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Grille uniforme de points de collocation en 1D.

    Retourne Tensor (n_points, 1) avec requires_grad=True pour autograd.
    """
    x = torch.linspace(x_min, x_max, n_points, dtype=torch.float32).unsqueeze(1)
    if device is not None:
        x = x.to(device)
    if requires_grad:
        x = x.requires_grad_(True)
    return x


def random_collocation(
    x_min: float,
    x_max: float,
    n_points: int,
    requires_grad: bool = True,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Points de collocation aléatoires uniformes en 1D.

    Retourne Tensor (n_points, 1) avec requires_grad=True pour autograd.
    """
    x = torch.rand(n_points, 1, dtype=torch.float32) * (x_max - x_min) + x_min
    if device is not None:
        x = x.to(device)
    if requires_grad:
        x = x.requires_grad_(True)
    return x


def spacetime_collocation(
    x_min: float,
    x_max: float,
    t_min: float,
    t_max: float,
    n_points: int,
    requires_grad: bool = True,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Points de collocation aléatoires dans l'espace-temps (x, t).

    Retourne
    --------
    x_c : Tensor (n_points, 1) avec requires_grad
    t_c : Tensor (n_points, 1) avec requires_grad
    """
    x = torch.rand(n_points, 1) * (x_max - x_min) + x_min
    t = torch.rand(n_points, 1) * (t_max - t_min) + t_min
    if device is not None:
        x = x.to(device)
        t = t.to(device)
    if requires_grad:
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
    return x, t


def boundary_points(
    x_min: float,
    x_max: float,
    device: torch.device = None,
) -> torch.Tensor:
    """Retourne les deux points aux bords du domaine [x_min, x_max]."""
    x_bc = torch.tensor([[x_min], [x_max]], dtype=torch.float32)
    if device is not None:
        x_bc = x_bc.to(device)
    return x_bc


# ===========================================================================
# Métriques de validation
# ===========================================================================

def l2_relative_error(
    psi_pred: np.ndarray,
    psi_exact: np.ndarray,
) -> float:
    """
    Erreur L2 relative : ||ψ_pred - ψ_exact||₂ / ||ψ_exact||₂.

    Gère le signe ambiguïtaire (ψ et -ψ sont équivalents quantiquement).
    """
    diff_pos = np.linalg.norm(psi_pred - psi_exact)
    diff_neg = np.linalg.norm(psi_pred + psi_exact)
    diff = min(diff_pos, diff_neg)
    return diff / (np.linalg.norm(psi_exact) + 1e-12)


def wavefunction_overlap(
    psi_a: np.ndarray,
    psi_b: np.ndarray,
    x: np.ndarray,
) -> float:
    """
    Fidélité (overlap) : |⟨ψ_a | ψ_b⟩| = |∫ ψ_a* ψ_b dx|.

    Paramètres
    ----------
    psi_a, psi_b : arrays 1D (réels ou complexes)
    x : grille spatiale

    Retourne
    --------
    float dans [0, 1]
    """
    integrand = np.conj(psi_a) * psi_b
    overlap = np.abs(np.trapz(integrand, x))
    return float(overlap)


def check_normalization(
    psi: np.ndarray,
    x: np.ndarray,
) -> float:
    """
    Vérifie ∫|ψ|² dx.

    Retourne
    --------
    float — doit être ≈ 1.0
    """
    return float(np.trapz(np.abs(psi) ** 2, x))


def energy_relative_error(
    E_pred: float,
    E_exact: float,
) -> float:
    """Erreur relative sur l'énergie propre : |E_pred - E_exact| / |E_exact|."""
    return abs(E_pred - E_exact) / (abs(E_exact) + 1e-12)


def validation_report(
    psi_pred: np.ndarray,
    psi_exact: np.ndarray,
    E_pred: float,
    E_exact: float,
    x: np.ndarray,
    label: str = "État",
) -> dict:
    """
    Rapport de validation complet pour un état propre TISE.

    Retourne
    --------
    dict avec : E_pred, E_exact, E_error, l2_error, overlap, norm
    """
    norm = check_normalization(psi_pred, x)
    l2_err = l2_relative_error(psi_pred, psi_exact)
    overlap = wavefunction_overlap(psi_pred, psi_exact, x)
    E_err = energy_relative_error(E_pred, E_exact)

    print(f"\n{'='*55}")
    print(f"  Validation — {label}")
    print(f"{'='*55}")
    print(f"  Énergie prédite  : {E_pred:.6f}")
    print(f"  Énergie exacte   : {E_exact:.6f}")
    print(f"  Erreur relative  : {E_err*100:.3f} %")
    print(f"  Erreur L2 (ψ)    : {l2_err*100:.3f} %")
    print(f"  Overlap |⟨ψ|ψ_ex⟩|: {overlap:.6f}")
    print(f"  Normalisation    : {norm:.6f}")

    status = "✓" if (E_err < 0.02 and l2_err < 0.05 and abs(norm - 1) < 0.01) else "✗"
    print(f"  Statut           : {status}")
    print(f"{'='*55}")

    return {
        "E_pred": E_pred,
        "E_exact": E_exact,
        "E_error": E_err,
        "l2_error": l2_err,
        "overlap": overlap,
        "norm": norm,
    }


# ===========================================================================
# Conversion numpy ↔ torch
# ===========================================================================

def to_tensor(
    arr: np.ndarray,
    requires_grad: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    """Convertit un array numpy en Tensor float32."""
    t = torch.tensor(arr, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    if requires_grad:
        t = t.requires_grad_(True)
    return t


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convertit un Tensor en array numpy (détaché)."""
    return t.detach().cpu().numpy()


def get_device() -> torch.device:
    """Retourne cuda si disponible, sinon cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
