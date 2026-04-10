"""
quantum_simulation/pinn/tise_solver.py
=========================================
Solveur PINN pour l'Équation de Schrödinger Indépendante du Temps (TISE).

Stratégie d'optimisation
-------------------------
1. Adam (lr=1e-3, ~5000 epochs) — convergence globale rapide
2. L-BFGS (~500 étapes) — fine-tuning de précision locale

Systèmes supportés (via potential_fn)
--------------------------------------
- Puits infini : V(x) = 0 avec trial function ψ_trial = (1 - (x/L)²)·ψ_NN
- Oscillateur harmonique : V(x) = ½ω²x²
- Puits fini, potentiel anharmonique, etc.

Références
----------
- arxiv:2504.05367 — PINN solvers pour puits quantiques 1D
- arxiv:2405.13442 — Oscillateur anharmonique via PINNs
"""

import numpy as np
import torch
import torch.optim as optim
from typing import Callable, Optional

from .network import TISENet, FourierFeatureTISENet
from .losses import tise_total_loss
from .utils import (
    gauss_legendre_quadrature,
    uniform_collocation,
    boundary_points,
    to_numpy,
    get_device,
    validation_report,
)


class TISESolver:
    """
    Résout Ĥψ = Eψ pour les états propres via un PINN.

    Paramètres
    ----------
    potential_fn : callable
        V(x) : Tensor (N, 1) → Tensor (N, 1). Doit opérer sur des Tensors.
    x_domain : tuple (x_min, x_max)
        Domaine spatial.
    n_colloc : int
        Nombre de points de collocation internes.
    n_quad : int
        Nombre de points de quadrature pour la normalisation.
    use_trial_function : bool
        Si True, applique ψ_trial(x) = (1 - (2x/L)²)·ψ_NN(x) pour
        encoder automatiquement les conditions aux limites (puits infini).
    use_fourier : bool
        Si True, utilise FourierFeatureTISENet (meilleur pour états excités).
    n_fourier : int
        Nombre de fréquences Fourier (si use_fourier=True).
    n_hidden : int
        Nombre de couches cachées.
    n_neurons : int
        Neurones par couche.
    hbar, m : float
        Constantes physiques (défaut : unités atomiques, hbar=m=1).
    device : torch.device
        Appareil de calcul (auto-détecté si None).
    """

    def __init__(
        self,
        potential_fn: Callable,
        x_domain: tuple,
        n_colloc: int = 500,
        n_quad: int = 200,
        use_trial_function: bool = False,
        use_fourier: bool = False,
        n_fourier: int = 16,
        n_hidden: int = 4,
        n_neurons: int = 128,
        hbar: float = 1.0,
        m: float = 1.0,
        device: torch.device = None,
    ):
        self.potential_fn = potential_fn
        self.x_min, self.x_max = x_domain
        self.x_domain = x_domain
        self.n_colloc = n_colloc
        self.n_quad = n_quad
        self.use_trial_function = use_trial_function
        self.hbar = hbar
        self.m = m
        self.device = device or get_device()

        # Réseau
        if use_fourier:
            self.network = FourierFeatureTISENet(
                x_domain=x_domain,
                n_fourier=n_fourier,
                n_hidden=n_hidden,
                n_neurons=n_neurons,
            ).to(self.device)
        else:
            self.network = TISENet(
                x_domain=x_domain,
                n_hidden=n_hidden,
                n_neurons=n_neurons,
            ).to(self.device)

        # Points de quadrature (fixes, pas de gradient nécessaire)
        self.x_quad, self.w_quad = gauss_legendre_quadrature(
            n_quad, self.x_min, self.x_max, device=self.device
        )

        # Points aux bords
        self.x_bc = boundary_points(self.x_min, self.x_max, device=self.device)

        # Historique d'entraînement
        self.history = {"loss": [], "E": [], "loss_pde": [], "loss_norm": []}

        # États propres précédents pour orthogonalité
        self._prev_states: list = []

    def _apply_trial_function(
        self, psi_nn: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """
        ψ_trial(x) = (1 - ((2x)/(x_max - x_min))²) · ψ_NN(x)

        Satisfait automatiquement ψ(x_min) = ψ(x_max) = 0.
        """
        L = (self.x_max - self.x_min) / 2.0
        x_centered = x - (self.x_max + self.x_min) / 2.0
        return (1.0 - (x_centered / L) ** 2) * psi_nn

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Évalue ψ(x) avec ou sans trial function."""
        psi_nn = self.network(x)
        if self.use_trial_function:
            return self._apply_trial_function(psi_nn, x)
        return psi_nn

    def _build_collocation(self) -> torch.Tensor:
        """Crée les points de collocation (uniforme)."""
        return uniform_collocation(
            self.x_min, self.x_max, self.n_colloc,
            requires_grad=True, device=self.device
        )

    def solve(
        self,
        n_epochs_adam: int = 5000,
        n_steps_lbfgs: int = 500,
        lr_adam: float = 1e-3,
        lambda_pde: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_norm: float = 100.0,
        lambda_ortho: float = 10.0,
        E_init: float = None,
        log_every: int = 500,
        verbose: bool = True,
    ) -> dict:
        """
        Résout pour l'état propre courant (fondamental si aucun état préalable).

        Paramètres
        ----------
        n_epochs_adam : epochs Adam pour convergence globale
        n_steps_lbfgs : étapes L-BFGS pour précision locale
        lr_adam : taux d'apprentissage Adam
        lambda_* : pondérations des termes de perte
        E_init : énergie initiale (remplace E_init du constructeur si fourni)
        log_every : fréquence d'affichage
        verbose : affiche la progression

        Retourne
        --------
        dict : résultats {E, psi_fn, history}
        """
        if E_init is not None:
            with torch.no_grad():
                self.network.E.copy_(torch.tensor([E_init]))

        optimizer_adam = optim.Adam(self.network.parameters(), lr=lr_adam)

        # --- Phase 1 : Adam ---
        for epoch in range(n_epochs_adam):
            optimizer_adam.zero_grad()

            x_c = self._build_collocation()
            psi_c = self._forward(x_c)

            # Points de quadrature (avec grad pour normalisation différentiable)
            x_q = self.x_quad.clone().requires_grad_(False)
            psi_q = self._forward(x_q)

            # Conditions aux limites (sans trial function)
            if self.use_trial_function:
                l_bc_val = torch.tensor(0.0, device=self.device)
                # Simuler une perte BC nulle compatible avec autograd
                psi_bc = torch.zeros(2, 1, device=self.device)
            else:
                psi_bc = self._forward(self.x_bc)

            loss, comps = tise_total_loss(
                psi=psi_c,
                x_colloc=x_c,
                psi_bc=psi_bc,
                psi_quad=psi_q,
                x_quad=x_q,
                w_quad=self.w_quad,
                E=self.network.E,
                potential_fn=self.potential_fn,
                lambda_pde=lambda_pde,
                lambda_bc=0.0 if self.use_trial_function else lambda_bc,
                lambda_norm=lambda_norm,
                psi_prev_list=self._prev_states if self._prev_states else None,
                lambda_ortho=lambda_ortho,
                hbar=self.hbar,
                m=self.m,
            )

            loss.backward()
            optimizer_adam.step()

            self.history["loss"].append(loss.item())
            self.history["E"].append(self.network.E.item())
            self.history["loss_pde"].append(comps["pde"])
            self.history["loss_norm"].append(comps["norm"])

            if verbose and epoch % log_every == 0:
                print(
                    f"  Epoch {epoch:5d} | Loss: {loss.item():.4e} | "
                    f"E: {self.network.E.item():.6f} | "
                    f"PDE: {comps['pde']:.4e} | Norm: {comps['norm']:.4e}"
                )

        # --- Phase 2 : L-BFGS ---
        if n_steps_lbfgs > 0:
            if verbose:
                print(f"\n  → Fine-tuning L-BFGS ({n_steps_lbfgs} étapes)...")

            optimizer_lbfgs = optim.LBFGS(
                self.network.parameters(),
                lr=1.0,
                max_iter=n_steps_lbfgs,
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer_lbfgs.zero_grad()
                x_c = self._build_collocation()
                psi_c = self._forward(x_c)
                x_q = self.x_quad.clone().requires_grad_(False)
                psi_q = self._forward(x_q)
                psi_bc = (
                    torch.zeros(2, 1, device=self.device)
                    if self.use_trial_function
                    else self._forward(self.x_bc)
                )
                loss, _ = tise_total_loss(
                    psi=psi_c,
                    x_colloc=x_c,
                    psi_bc=psi_bc,
                    psi_quad=psi_q,
                    x_quad=x_q,
                    w_quad=self.w_quad,
                    E=self.network.E,
                    potential_fn=self.potential_fn,
                    lambda_pde=lambda_pde,
                    lambda_bc=0.0 if self.use_trial_function else lambda_bc,
                    lambda_norm=lambda_norm,
                    psi_prev_list=self._prev_states if self._prev_states else None,
                    lambda_ortho=lambda_ortho,
                    hbar=self.hbar,
                    m=self.m,
                )
                loss.backward()
                return loss

            optimizer_lbfgs.step(closure)

            if verbose:
                E_final = self.network.E.item()
                print(f"  → L-BFGS terminé. E final = {E_final:.6f}")

        E_final = self.network.E.item()

        return {
            "E": E_final,
            "history": self.history.copy(),
            "network": self.network,
        }

    def solve_multiple(
        self,
        n_states: int = 3,
        E_inits: list = None,
        verbose: bool = True,
        **solve_kwargs,
    ) -> list:
        """
        Résout plusieurs états propres consécutivement.

        Chaque état est résolu avec une contrainte d'orthogonalité par
        rapport aux états précédents.

        Paramètres
        ----------
        n_states : nombre d'états propres à calculer
        E_inits  : liste de valeurs initiales pour E (optionnel)
        **solve_kwargs : arguments passés à solve()

        Retourne
        --------
        list de dicts {E, psi_values, history}
        """
        results = []
        x_eval = uniform_collocation(
            self.x_min, self.x_max, 500, requires_grad=False, device=self.device
        )

        for n in range(n_states):
            if verbose:
                print(f"\n{'='*55}")
                print(f"  Calcul état propre n={n}")
                print(f"{'='*55}")

            # Réinitialise le réseau pour chaque état
            from .network import TISENet, FourierFeatureTISENet
            E_init_n = E_inits[n] if E_inits else 0.5 + n
            self.network = self.network.__class__(
                x_domain=self.x_domain,
                n_hidden=self.network.net.net[0].in_features if hasattr(self.network.net, 'net') else 4,
                n_neurons=128,
                E_init=E_init_n,
            ).to(self.device)

            result = self.solve(E_init=E_init_n, verbose=verbose, **solve_kwargs)

            # Stocke ψ évalué sur grille pour orthogonalité future
            with torch.no_grad():
                psi_vals = self._forward(x_eval)
                # Normalise pour la contrainte d'orthogonalité
                norm = torch.sqrt(torch.sum(self.w_quad * self._forward(self.x_quad) ** 2))
                psi_normalized = psi_vals / (norm + 1e-8)

            result["psi_values"] = to_numpy(psi_vals)
            result["x_eval"] = to_numpy(x_eval)
            results.append(result)

            # Ajoute aux états précédents (détaché du graph)
            self._prev_states.append(psi_normalized.detach())

        return results

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Évalue ψ(x) sur une grille numpy.

        Paramètres
        ----------
        x : array 1D

        Retourne
        --------
        psi : array 1D (valeurs réelles normalisées)
        """
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            psi = self._forward(x_t)
        return to_numpy(psi).squeeze()

    def validate(
        self,
        analytical_energies: list,
        analytical_psi_fns: list,
        x_eval: np.ndarray = None,
    ) -> list:
        """
        Compare les prédictions PINN aux solutions analytiques.

        Paramètres
        ----------
        analytical_energies : liste des énergies exactes [E_0, E_1, ...]
        analytical_psi_fns : liste de callables x→ψ_exact(x)
        x_eval : grille d'évaluation numpy (créée auto si None)

        Retourne
        --------
        list de dicts de métriques
        """
        if x_eval is None:
            x_eval = np.linspace(self.x_min, self.x_max, 500)

        reports = []
        for n, (E_ex, psi_fn) in enumerate(
            zip(analytical_energies, analytical_psi_fns)
        ):
            psi_pred = self.predict(x_eval)
            psi_exact = psi_fn(x_eval)

            # Normalise les deux
            norm_pred = np.trapz(psi_pred**2, x_eval)
            norm_exact = np.trapz(psi_exact**2, x_eval)
            psi_pred_norm = psi_pred / (np.sqrt(norm_pred) + 1e-12)
            psi_exact_norm = psi_exact / (np.sqrt(norm_exact) + 1e-12)

            report = validation_report(
                psi_pred_norm,
                psi_exact_norm,
                self.network.E.item(),
                E_ex,
                x_eval,
                label=f"État n={n}",
            )
            reports.append(report)

        return reports
