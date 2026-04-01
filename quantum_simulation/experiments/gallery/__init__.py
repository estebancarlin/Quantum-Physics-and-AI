# quantum_simulation/experiments/gallery/__init__.py
"""
Galerie expériences 2D prédéfinies.

Catalogue expériences démonstratives :
- Double-slit (interférences)
- Quantum billiard (chaos)
- Vortex states (moment angulaire)
"""

from .double_slit_2d import DoubleSlitExperiment
from .scattering_yukawa import ScatteringYukawa
from .rabi_oscillations import RabiOscillationsExperiment
from .hydrogen_fine_structure import HydrogenFineStructureExperiment

__all__ = [
    # Tome 1
    'DoubleSlitExperiment',
    # Tome 2
    'ScatteringYukawa',
    'RabiOscillationsExperiment',
    'HydrogenFineStructureExperiment',
]