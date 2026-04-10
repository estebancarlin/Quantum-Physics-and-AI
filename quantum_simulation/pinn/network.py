"""
quantum_simulation/pinn/network.py
===================================
Architectures de réseaux de neurones pour PINNs Schrödinger.

Références
----------
- Raissi et al. 2019 — Physics-informed neural networks (JCP 378:686–707)
- Tancik et al. 2020 — Fourier Features Let Networks Learn High Frequency Functions
- arxiv:2210.12522 — PINNs as Solvers for the Time-Dependent Schrödinger Equation
"""

import torch
import torch.nn as nn
import numpy as np


class SchrodingerNet(nn.Module):
    """
    MLP profond pour PINNs Schrödinger (TISE et TDSE).

    Architecture : couches linéaires avec activation tanh (infiniment
    dérivable, requis pour les résidus EDP d'ordre 2 via autograd).

    Modes d'utilisation
    -------------------
    TISE (1D) : Input x ∈ ℝ → Output [ψ(x)]  +  paramètre E séparé
    TDSE (1D) : Input (x, t) ∈ ℝ² → Output [ψ_real(x,t), ψ_imag(x,t)]

    Paramètres
    ----------
    n_input : int
        Dimension d'entrée (1 pour TISE, 2 pour TDSE).
    n_output : int
        Dimension de sortie (1 pour TISE, 2 pour TDSE).
    n_hidden : int
        Nombre de couches cachées (recommandé : 4–6).
    n_neurons : int
        Neurones par couche cachée (recommandé : 128–256).
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 4,
        n_neurons: int = 128,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output

        layers = []
        in_dim = n_input
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, n_neurons))
            layers.append(nn.Tanh())
            in_dim = n_neurons
        layers.append(nn.Linear(in_dim, n_output))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier uniform (recommandé pour tanh)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TISENet(nn.Module):
    """
    Réseau spécialisé pour l'Équation de Schrödinger Indépendante du Temps.

    Sorties : [ψ(x)]  +  énergie propre E comme paramètre appris.

    L'énergie E est un paramètre scalaire du réseau, optimisé
    conjointement avec les poids. Cela permet de traiter Ĥψ = Eψ
    comme un problème d'optimisation sans supervision sur E.

    Paramètres
    ----------
    x_domain : tuple (x_min, x_max)
        Domaine spatial pour la normalisation des entrées vers [-1, 1].
    n_hidden, n_neurons : voir SchrodingerNet.
    E_init : float
        Valeur initiale de l'énergie propre (défaut 0.5, ~état fondamental HO).
    """

    def __init__(
        self,
        x_domain: tuple,
        n_hidden: int = 4,
        n_neurons: int = 128,
        E_init: float = 0.5,
    ):
        super().__init__()
        self.x_min, self.x_max = x_domain

        self.net = SchrodingerNet(
            n_input=1, n_output=1, n_hidden=n_hidden, n_neurons=n_neurons
        )
        # Énergie propre comme paramètre appris (scalaire)
        self.E = nn.Parameter(torch.tensor([E_init]))

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise x ∈ [x_min, x_max] vers [-1, 1]."""
        return 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paramètres
        ----------
        x : Tensor shape (N, 1)

        Retourne
        --------
        psi : Tensor shape (N, 1)
        """
        x_norm = self.normalize_input(x)
        return self.net(x_norm)

    def get_energy(self) -> float:
        """Retourne la valeur courante de l'énergie propre apprise."""
        return self.E.item()


class TDSENet(nn.Module):
    """
    Réseau spécialisé pour l'Équation de Schrödinger Dépendante du Temps.

    Sorties : [ψ_real(x,t), ψ_imag(x,t)]

    La fonction d'onde complexe ψ = ψ_real + i·ψ_imag est décomposée
    en deux fonctions réelles pour rester compatible avec autograd standard.

    Paramètres
    ----------
    x_domain : tuple (x_min, x_max)
    t_domain : tuple (t_min, t_max)
    n_hidden, n_neurons : voir SchrodingerNet.
    """

    def __init__(
        self,
        x_domain: tuple,
        t_domain: tuple,
        n_hidden: int = 5,
        n_neurons: int = 128,
    ):
        super().__init__()
        self.x_min, self.x_max = x_domain
        self.t_min, self.t_max = t_domain

        self.net = SchrodingerNet(
            n_input=2, n_output=2, n_hidden=n_hidden, n_neurons=n_neurons
        )

    def normalize_inputs(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Normalise (x, t) vers [-1, 1] × [-1, 1]."""
        x_norm = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        t_norm = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        return torch.cat([x_norm, t_norm], dim=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Paramètres
        ----------
        x : Tensor shape (N, 1)
        t : Tensor shape (N, 1)

        Retourne
        --------
        psi_real : Tensor shape (N, 1)
        psi_imag : Tensor shape (N, 1)
        """
        xt = self.normalize_inputs(x, t)
        out = self.net(xt)
        return out[:, 0:1], out[:, 1:2]


class FourierFeatureTISENet(nn.Module):
    """
    TISE-Net avec encodage positionnel de Fourier.

    Ajoute des caractéristiques [sin(2πkx), cos(2πkx)] en entrée pour
    permettre au réseau d'apprendre les oscillations haute fréquence
    des états propres excités (utile pour n > 3).

    Référence : Tancik et al. 2020, "Fourier Features Let Networks Learn
    High Frequency Functions in Low Dimensional Domains".

    Paramètres
    ----------
    x_domain : tuple (x_min, x_max)
    n_fourier : int
        Nombre de fréquences de Fourier (k = 1, 2, ..., n_fourier).
        L'entrée devient de dimension 2·n_fourier + 1.
    n_hidden, n_neurons : voir SchrodingerNet.
    E_init : float
        Valeur initiale de l'énergie propre.
    """

    def __init__(
        self,
        x_domain: tuple,
        n_fourier: int = 16,
        n_hidden: int = 4,
        n_neurons: int = 128,
        E_init: float = 0.5,
    ):
        super().__init__()
        self.x_min, self.x_max = x_domain
        self.n_fourier = n_fourier

        # Fréquences enregistrées comme buffer (non-trainable)
        freqs = torch.arange(1, n_fourier + 1, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

        n_input_encoded = 1 + 2 * n_fourier
        self.net = SchrodingerNet(
            n_input=n_input_encoded,
            n_output=1,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
        )
        self.E = nn.Parameter(torch.tensor([E_init]))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Applique l'encodage Fourier : [x, sin(2πkx), cos(2πkx)]."""
        x_norm = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        # x_norm shape: (N, 1) → broadcasté avec freqs shape: (n_fourier,)
        angles = 2.0 * np.pi * self.freqs * x_norm  # (N, n_fourier)
        return torch.cat([x_norm, torch.sin(angles), torch.cos(angles)], dim=1)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.encode(x))

    def get_energy(self) -> float:
        return self.E.item()
