# quantum_simulation/experiments/gallery/__init__.py
"""
Galerie expériences 2D prédéfinies.

Catalogue expériences démonstratives :
- Double-slit (interférences)
- Quantum billiard (chaos)
- Vortex states (moment angulaire)
"""

from .double_slit_2d import DoubleSlitExperiment

__all__ = [
    'DoubleSlitExperiment',
    # 'QuantumBilliard2D',  # À implémenter
    # 'VortexStates2D',      # À implémenter
]