"""
quantum_simulation/pinn/tdse_solver.py
=========================================
Solveur PINN pour l'Équation de Schrödinger Dépendante du Temps (TDSE).

Approche : décomposition réel/imaginaire ψ = ψ_r + i·ψ_i

L'ETDS complexe iℏ∂ψ/∂t = Ĥψ est décomposée en deux équations réelles :
  Re : ∂ψ_r/∂t = -(ℏ/2m)∂²ψ_i/∂x² + (V/ℏ)·ψ_i
  Im : ∂ψ_i/∂t = +(ℏ/2m)∂²ψ_r/∂x² - (V/ℏ)·ψ_r

Le réseau PINN prédit ψ_r(x,t) et ψ_i(x,t) conjointement sur tout
le domaine espace-temps [x_min, x_max] × [0, T].

Avantages vs Crank-Nicolson :
- Sans maillage (mesh-free)
- Surrogate : un réseau entraîné ≈ toutes les conditions initiales similaires
- Problèmes inverses : peut identifier V(x) à partir de données

Inconvénients :
- Précision moindre pour des évolutions longues (t >> 1)
- Coût d'entraînement supérieur à CN pour un problème donné

Références
----------
- arxiv:2210.12522 — PINNs as Solvers for the Time-Dependent Schrödinger Eq.
- Raissi et al. 2019 PINN original (JCP 378:686–707)
"""

import numpy as np
import torch
import torch.optim as optim
from typing import Callable, Optional

from .network import TDSENet
from .losses import tdse_total_loss
from .utils import (
    gauss_legendre_quadrature,
    spacetime_collocation,
    uniform_collocation,
    to_numpy,
    get_device,
    wavefunction_overlap,
    check_normalization,
)


class TDSESolver:
    """
    Résout iℏ∂ψ/∂t = Ĥψ via un PINN sur le domaine espace-temps.

    Paramètres
    ----------
    potential_fn : callable
        V(x) : Tensor (N, 1) → Tensor (N, 1).
    psi0_fn : callable
        ψ₀(x) : np.ndarray → (psi_r_np, psi_i_np) — condition initiale.
    x_domain : tuple (x_min, x_max)
    t_domain : tuple (t_min=0, t_max)
    n_colloc : int
        Nombre de points de collocation dans l'espace-temps.
    n_ic : int
        Nombre de points pour la condition initiale.
    n_quad : int
        Nombre de points de quadrature pour la normalisation.
    n_hidden, n_neurons : architecture du réseau.
    hbar, m : constantes physiques (unités atomiques par défaut).
    device : torch.device
    """

    def __init__(
        self,
        potential_fn: Callable,
        psi0_fn: Callable,
        x_domain: tuple,
        t_domain: tuple,
        n_colloc: int = 5000,
        n_ic: int = 500,
        n_quad: int = 200,
        n_hidden: int = 5,
        n_neurons: int = 128,
        hbar: float = 1.0,
        m: float = 1.0,
        device: torch.device = None,
    ):
        self.potential_fn = potential_fn
        self.psi0_fn = psi0_fn
        self.x_min, self.x_max = x_domain
        self.t_min, self.t_max = t_domain
        self.x_domain = x_domain
        self.t_domain = t_domain
        self.n_colloc = n_colloc
        self.n_ic = n_ic
        self.hbar = hbar
        self.m = m
        self.device = device or get_device()

        self.network = TDSENet(
            x_domain=x_domain,
            t_domain=t_domain,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
        ).to(self.device)

        # Points de quadrature (pour normalisation, sans grad)
        self.x_quad, self.w_quad = gauss_legendre_quadrature(
            n_quad, self.x_min, self.x_max, device=self.device
        )

        # Condition initiale (fixée une fois)
        self._x_ic, self._psi_r0, self._psi_i0 = self._build_ic(n_ic)

        self.history = {
            "loss": [],
            "loss_pde_re": [],
            "loss_pde_im": [],
            "loss_ic": [],
        }

    def _build_ic(
        self, n_ic: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construit les tenseurs de condition initiale."""
        x_np = np.linspace(self.x_min, self.x_max, n_ic)
        psi_r_np, psi_i_np = self.psi0_fn(x_np)

        x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1).to(self.device)
        psi_r_t = torch.tensor(psi_r_np, dtype=torch.float32).unsqueeze(1).to(self.device)
        psi_i_t = torch.tensor(psi_i_np, dtype=torch.float32).unsqueeze(1).to(self.device)
        return x_t, psi_r_t, psi_i_t

    def _eval_ic(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Évalue le réseau à t=0 sur les points de condition initiale."""
        t_zero = torch.zeros_like(self._x_ic).to(self.device)
        t_zero = t_zero.requires_grad_(True)
        x_ic = self._x_ic.requires_grad_(True)
        return self.network(x_ic, t_zero)

    def solve(
        self,
        n_epochs_adam: int = 10000,
        n_steps_lbfgs: int = 200,
        lr_adam: float = 1e-3,
        lambda_pde: float = 1.0,
        lambda_ic: float = 10.0,
        lambda_norm: float = 50.0,
        log_every: int = 1000,
        verbose: bool = True,
    ) -> dict:
        """
        Entraîne le réseau sur le domaine espace-temps complet.

        Paramètres
        ----------
        n_epochs_adam : epochs Adam
        n_steps_lbfgs : étapes L-BFGS (fine-tuning)
        lr_adam : taux d'apprentissage
        lambda_* : pondérations des termes de perte
        log_every : fréquence d'affichage
        verbose : affiche la progression

        Retourne
        --------
        dict : {history, network}
        """
        optimizer = optim.Adam(self.network.parameters(), lr=lr_adam)

        # Scheduler : réduit lr si plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=500, factor=0.5, min_lr=1e-5
        )

        for epoch in range(n_epochs_adam):
            optimizer.zero_grad()

            # Points de collocation espace-temps (rééchantillonnés chaque epoch)
            x_c, t_c = spacetime_collocation(
                self.x_min, self.x_max,
                self.t_min, self.t_max,
                self.n_colloc,
                requires_grad=True,
                device=self.device,
            )
            psi_r_c, psi_i_c = self.network(x_c, t_c)

            # Condition initiale
            psi_r_ic, psi_i_ic = self._eval_ic()

            # Quadrature pour normalisation (à t=0 comme approximation)
            x_q = self.x_quad.clone().requires_grad_(False)
            t_q_zero = torch.zeros_like(x_q).requires_grad_(False).to(self.device)
            with torch.no_grad():
                psi_r_q, psi_i_q = self.network(x_q, t_q_zero)

            loss, comps = tdse_total_loss(
                psi_r_colloc=psi_r_c,
                psi_i_colloc=psi_i_c,
                x_colloc=x_c,
                t_colloc=t_c,
                psi_r_ic=psi_r_ic,
                psi_i_ic=psi_i_ic,
                psi_r0_target=self._psi_r0,
                psi_i0_target=self._psi_i0,
                potential_fn=self.potential_fn,
                lambda_pde=lambda_pde,
                lambda_ic=lambda_ic,
                lambda_norm=lambda_norm,
                hbar=self.hbar,
                m=self.m,
                x_quad=self.x_quad,
                w_quad=self.w_quad,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            optimizer.step()
            scheduler.step(loss)

            self.history["loss"].append(loss.item())
            self.history["loss_pde_re"].append(comps["pde_re"])
            self.history["loss_pde_im"].append(comps["pde_im"])
            self.history["loss_ic"].append(comps["ic"])

            if verbose and epoch % log_every == 0:
                print(
                    f"  Epoch {epoch:6d} | Loss: {loss.item():.4e} | "
                    f"PDE_re: {comps['pde_re']:.4e} | "
                    f"PDE_im: {comps['pde_im']:.4e} | "
                    f"IC: {comps['ic']:.4e}"
                )

        # --- Fine-tuning L-BFGS ---
        if n_steps_lbfgs > 0 and verbose:
            print(f"\n  → Fine-tuning L-BFGS ({n_steps_lbfgs} étapes)...")

        if n_steps_lbfgs > 0:
            optimizer_lbfgs = optim.LBFGS(
                self.network.parameters(),
                lr=0.1,
                max_iter=n_steps_lbfgs,
                history_size=20,
                line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer_lbfgs.zero_grad()
                x_c, t_c = spacetime_collocation(
                    self.x_min, self.x_max,
                    self.t_min, self.t_max,
                    self.n_colloc // 2,
                    requires_grad=True,
                    device=self.device,
                )
                psi_r_c, psi_i_c = self.network(x_c, t_c)
                psi_r_ic, psi_i_ic = self._eval_ic()
                loss, _ = tdse_total_loss(
                    psi_r_colloc=psi_r_c,
                    psi_i_colloc=psi_i_c,
                    x_colloc=x_c,
                    t_colloc=t_c,
                    psi_r_ic=psi_r_ic,
                    psi_i_ic=psi_i_ic,
                    psi_r0_target=self._psi_r0,
                    psi_i0_target=self._psi_i0,
                    potential_fn=self.potential_fn,
                    lambda_pde=lambda_pde,
                    lambda_ic=lambda_ic,
                    lambda_norm=0.0,
                    hbar=self.hbar,
                    m=self.m,
                )
                loss.backward()
                return loss

            optimizer_lbfgs.step(closure)

        return {"history": self.history.copy(), "network": self.network}

    def predict(
        self,
        x: np.ndarray,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Évalue ψ_r(x, t) et ψ_i(x, t) sur une grille numpy.

        Paramètres
        ----------
        x : array 1D des positions
        t : instant temporel

        Retourne
        --------
        psi_r, psi_i : arrays 1D
        """
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(self.device)
        t_t = torch.full_like(x_t, t)

        with torch.no_grad():
            psi_r, psi_i = self.network(x_t, t_t)

        return to_numpy(psi_r).squeeze(), to_numpy(psi_i).squeeze()

    def predict_density(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Densité de probabilité |ψ(x, t)|².

        Paramètres
        ----------
        x : array 1D
        t : instant

        Retourne
        --------
        density : array 1D = ψ_r² + ψ_i²
        """
        psi_r, psi_i = self.predict(x, t)
        return psi_r**2 + psi_i**2

    def compare_with_reference(
        self,
        x: np.ndarray,
        t_values: list,
        ref_psi_fn: Callable,
    ) -> list:
        """
        Compare les densités PINN avec une référence (CN ou analytique).

        Paramètres
        ----------
        x : grille spatiale
        t_values : liste d'instants de comparaison
        ref_psi_fn : callable (x, t) → (psi_r, psi_i) — référence

        Retourne
        --------
        list de dicts : {t, overlap, norm_pinn, norm_ref}
        """
        results = []
        print("\n  Comparaison PINN vs Référence")
        print(f"  {'t':>8} | {'Overlap':>10} | {'Norm PINN':>10} | {'Norm Ref':>10}")
        print(f"  {'-'*46}")

        for t in t_values:
            psi_r_pinn, psi_i_pinn = self.predict(x, t)
            psi_r_ref, psi_i_ref = ref_psi_fn(x, t)

            psi_pinn = psi_r_pinn + 1j * psi_i_pinn
            psi_ref = psi_r_ref + 1j * psi_i_ref

            overlap = wavefunction_overlap(psi_pinn, psi_ref, x)
            norm_pinn = check_normalization(psi_pinn, x)
            norm_ref = check_normalization(psi_ref, x)

            print(
                f"  {t:8.3f} | {overlap:10.6f} | {norm_pinn:10.6f} | {norm_ref:10.6f}"
            )

            results.append({
                "t": t,
                "overlap": overlap,
                "norm_pinn": norm_pinn,
                "norm_ref": norm_ref,
            })

        return results

    def check_norm_over_time(
        self,
        x: np.ndarray,
        t_values: list,
    ) -> list:
        """
        Vérifie la conservation de la norme ∫|ψ(x,t)|² dx pour plusieurs t.

        Retourne
        --------
        list de floats — norme à chaque instant
        """
        norms = []
        for t in t_values:
            density = self.predict_density(x, t)
            norm = float(np.trapz(density, x))
            norms.append(norm)
        return norms
