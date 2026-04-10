"""
quantum_simulation/pinn/losses.py
===================================
Fonctions de perte pour PINNs Schrödinger.

Formulation des pertes
----------------------
TISE : L = λ_pde · L_pde + λ_bc · L_bc + λ_norm · L_norm
  - L_pde  = <(Ĥψ - E·ψ)²>  (résidu de l'équation aux valeurs propres)
  - L_bc   = ψ(bords)²       (conditions aux limites de Dirichlet)
  - L_norm = (∫|ψ|² dx - 1)² (normalisation de la fonction d'onde)

TDSE : L = λ_pde · (L_re + L_im) + λ_ic · L_ic + λ_norm · L_norm
  - L_re/L_im : résidus réel/imaginaire de iℏ∂ψ/∂t = Ĥψ
  - L_ic      : condition initiale ψ(x, 0) = ψ₀(x)

Références
----------
- Raissi et al. 2019 (JCP 378:686–707) — formulation originale PINNs
- arxiv:2210.12522 — TDSE avec décomposition réel/imaginaire
"""

import torch
from typing import Callable


# ===========================================================================
# Utilitaires pour la différentiation automatique
# ===========================================================================

def grad(output: torch.Tensor, input_: torch.Tensor) -> torch.Tensor:
    """Dérivée ∂output/∂input via autograd (retain_graph=True)."""
    return torch.autograd.grad(
        output,
        input_,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


def grad2(output: torch.Tensor, input_: torch.Tensor) -> torch.Tensor:
    """Dérivée seconde ∂²output/∂input² via autograd."""
    first = grad(output, input_)
    return grad(first, input_)


# ===========================================================================
# Pertes TISE
# ===========================================================================

def tise_pde_loss(
    psi: torch.Tensor,
    x: torch.Tensor,
    E: torch.Tensor,
    potential_fn: Callable,
    hbar: float = 1.0,
    m: float = 1.0,
) -> torch.Tensor:
    """
    Résidu de l'équation aux valeurs propres : Ĥψ - E·ψ = 0.

    Ĥψ = -(ℏ²/2m) ∂²ψ/∂x² + V(x)·ψ

    Paramètres
    ----------
    psi : Tensor (N, 1)  — ψ(x) prédit par le réseau
    x   : Tensor (N, 1)  — points de collocation (requires_grad=True)
    E   : Tensor scalaire — énergie propre apprise
    potential_fn : callable x→V(x), doit opérer sur Tensors
    hbar, m : constantes physiques (unités atomiques par défaut : 1)

    Retourne
    --------
    Tensor scalaire — perte MSE du résidu
    """
    psi_xx = grad2(psi, x)
    V = potential_fn(x)
    H_psi = -(hbar**2 / (2.0 * m)) * psi_xx + V * psi
    residual = H_psi - E * psi
    return torch.mean(residual**2)


def tise_boundary_loss(
    psi_boundary: torch.Tensor,
    target: float = 0.0,
) -> torch.Tensor:
    """
    Condition aux limites de Dirichlet : ψ(bords) = target (= 0).

    Paramètres
    ----------
    psi_boundary : Tensor (N_bc, 1) — ψ évalué aux bords du domaine
    target : valeur cible (0 pour puits infini / domaine borné)
    """
    return torch.mean((psi_boundary - target) ** 2)


def tise_normalization_loss(
    psi: torch.Tensor,
    x_quad: torch.Tensor,
    w_quad: torch.Tensor,
) -> torch.Tensor:
    """
    Contrainte de normalisation : ∫|ψ(x)|² dx = 1.

    Utilise la quadrature de Gauss-Legendre pour l'intégrale :
    ∫ f dx ≈ Σᵢ wᵢ f(xᵢ)

    Paramètres
    ----------
    psi    : Tensor (N_quad, 1) — ψ évalué aux points de quadrature
    x_quad : Tensor (N_quad, 1) — points de quadrature
    w_quad : Tensor (N_quad, 1) — poids de quadrature

    Retourne
    --------
    Tensor scalaire — (∫|ψ|² dx - 1)²
    """
    norm_integral = torch.sum(w_quad * psi**2)
    return (norm_integral - 1.0) ** 2


def tise_orthogonality_loss(
    psi_new: torch.Tensor,
    psi_prev_list: list,
    x_quad: torch.Tensor,
    w_quad: torch.Tensor,
) -> torch.Tensor:
    """
    Contrainte d'orthogonalité pour les états excités :
    ⟨ψ_n | ψ_m⟩ = 0  pour n ≠ m.

    Paramètres
    ----------
    psi_new       : Tensor (N_quad, 1) — état propre courant
    psi_prev_list : liste de Tensors — états propres précédents (détachés)
    x_quad, w_quad : points et poids de quadrature
    """
    loss = torch.tensor(0.0)
    for psi_prev in psi_prev_list:
        overlap = torch.sum(w_quad * psi_new * psi_prev.detach())
        loss = loss + overlap**2
    return loss


def tise_total_loss(
    psi: torch.Tensor,
    x_colloc: torch.Tensor,
    psi_bc: torch.Tensor,
    psi_quad: torch.Tensor,
    x_quad: torch.Tensor,
    w_quad: torch.Tensor,
    E: torch.Tensor,
    potential_fn: Callable,
    lambda_pde: float = 1.0,
    lambda_bc: float = 10.0,
    lambda_norm: float = 100.0,
    psi_prev_list: list = None,
    lambda_ortho: float = 10.0,
    hbar: float = 1.0,
    m: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Perte totale TISE avec décomposition des termes.

    Retourne
    --------
    loss_total : Tensor scalaire
    components : dict avec les valeurs individuelles (pour monitoring)
    """
    l_pde = tise_pde_loss(psi, x_colloc, E, potential_fn, hbar, m)
    l_bc = tise_boundary_loss(psi_bc)
    l_norm = tise_normalization_loss(psi_quad, x_quad, w_quad)

    total = lambda_pde * l_pde + lambda_bc * l_bc + lambda_norm * l_norm

    components = {
        "pde": l_pde.item(),
        "bc": l_bc.item(),
        "norm": l_norm.item(),
    }

    if psi_prev_list:
        l_ortho = tise_orthogonality_loss(
            psi_quad, psi_prev_list, x_quad, w_quad
        )
        total = total + lambda_ortho * l_ortho
        components["ortho"] = l_ortho.item()

    return total, components


# ===========================================================================
# Pertes TDSE
# ===========================================================================

def tdse_pde_loss(
    psi_r: torch.Tensor,
    psi_i: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    potential_fn: Callable,
    hbar: float = 1.0,
    m: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Résidus réel et imaginaire de l'ETDS : iℏ∂ψ/∂t = Ĥψ.

    Décomposé en ψ = ψ_r + i·ψ_i, l'ETDS donne deux équations réelles :
      Re : ∂ψ_r/∂t = -(ℏ/2m) ∂²ψ_i/∂x² + (V/ℏ)·ψ_i
      Im : ∂ψ_i/∂t = +(ℏ/2m) ∂²ψ_r/∂x² - (V/ℏ)·ψ_r

    Paramètres
    ----------
    psi_r, psi_i : Tensors (N, 1) — parties réelle et imaginaire
    x, t : Tensors (N, 1) avec requires_grad=True
    potential_fn : callable x→V(x)
    hbar, m : constantes physiques

    Retourne
    --------
    loss_re, loss_im : Tensors scalaires
    """
    V = potential_fn(x)
    alpha = hbar / (2.0 * m)
    beta = 1.0 / hbar

    psi_r_t = grad(psi_r, t)
    psi_i_t = grad(psi_i, t)
    psi_r_xx = grad2(psi_r, x)
    psi_i_xx = grad2(psi_i, x)

    # Re: ∂ψ_r/∂t + α·∂²ψ_i/∂x² - β·V·ψ_i = 0
    res_re = psi_r_t + alpha * psi_i_xx - beta * V * psi_i
    # Im: ∂ψ_i/∂t - α·∂²ψ_r/∂x² + β·V·ψ_r = 0
    res_im = psi_i_t - alpha * psi_r_xx + beta * V * psi_r

    return torch.mean(res_re**2), torch.mean(res_im**2)


def tdse_initial_condition_loss(
    psi_r_pred: torch.Tensor,
    psi_i_pred: torch.Tensor,
    psi_r_target: torch.Tensor,
    psi_i_target: torch.Tensor,
) -> torch.Tensor:
    """
    Condition initiale : ψ(x, t=0) = ψ₀(x).

    Paramètres
    ----------
    psi_r_pred, psi_i_pred : prédictions réseau à t=0
    psi_r_target, psi_i_target : valeurs de ψ₀(x)
    """
    loss_r = torch.mean((psi_r_pred - psi_r_target) ** 2)
    loss_i = torch.mean((psi_i_pred - psi_i_target) ** 2)
    return loss_r + loss_i


def tdse_normalization_loss(
    psi_r: torch.Tensor,
    psi_i: torch.Tensor,
    x_quad: torch.Tensor,
    w_quad: torch.Tensor,
) -> torch.Tensor:
    """
    Normalisation TDSE : ∫(|ψ_r|² + |ψ_i|²) dx = 1.

    Paramètres
    ----------
    psi_r, psi_i : Tensors (N_quad, 1) à t fixé
    x_quad, w_quad : points et poids de quadrature
    """
    density = psi_r**2 + psi_i**2
    norm_integral = torch.sum(w_quad * density)
    return (norm_integral - 1.0) ** 2


def tdse_total_loss(
    psi_r_colloc: torch.Tensor,
    psi_i_colloc: torch.Tensor,
    x_colloc: torch.Tensor,
    t_colloc: torch.Tensor,
    psi_r_ic: torch.Tensor,
    psi_i_ic: torch.Tensor,
    psi_r0_target: torch.Tensor,
    psi_i0_target: torch.Tensor,
    potential_fn: Callable,
    lambda_pde: float = 1.0,
    lambda_ic: float = 10.0,
    lambda_norm: float = 50.0,
    hbar: float = 1.0,
    m: float = 1.0,
    x_quad: torch.Tensor = None,
    w_quad: torch.Tensor = None,
) -> tuple[torch.Tensor, dict]:
    """
    Perte totale TDSE.

    Retourne
    --------
    loss_total : Tensor scalaire
    components : dict avec les valeurs individuelles
    """
    l_re, l_im = tdse_pde_loss(
        psi_r_colloc, psi_i_colloc, x_colloc, t_colloc, potential_fn, hbar, m
    )
    l_ic = tdse_initial_condition_loss(
        psi_r_ic, psi_i_ic, psi_r0_target, psi_i0_target
    )

    total = lambda_pde * (l_re + l_im) + lambda_ic * l_ic

    components = {
        "pde_re": l_re.item(),
        "pde_im": l_im.item(),
        "ic": l_ic.item(),
    }

    if x_quad is not None and w_quad is not None:
        # Normalisation évaluée sur les points de quadrature à t=0
        # (approximation : ||ψ(t)||≈||ψ(0)|| si bien entraîné)
        l_norm = tdse_normalization_loss(psi_r_ic, psi_i_ic, x_quad, w_quad)
        total = total + lambda_norm * l_norm
        components["norm"] = l_norm.item()

    return total, components
