"""
Module systems : systèmes physiques concrets (Tomes 1 & 2).
"""

from quantum_simulation.systems.free_particle import FreeParticle
from quantum_simulation.systems.harmonic_oscillator import HarmonicOscillator
from quantum_simulation.systems.potential_systems import InfiniteWell, FiniteWell, PotentialBarrier
from quantum_simulation.systems.hydrogen_structure import HydrogenFineStructure, HydrogenHyperfine
from quantum_simulation.systems.zeeman_stark import ZeemanEffect, StarkEffect
from quantum_simulation.systems.identical_particles import Symmetrizer, SlaterDeterminant, IdenticalParticlesScattering

__all__ = [
    # Tome 1
    'FreeParticle', 'HarmonicOscillator',
    'InfiniteWell', 'FiniteWell', 'PotentialBarrier',
    # Tome 2
    'HydrogenFineStructure', 'HydrogenHyperfine',
    'ZeemanEffect', 'StarkEffect',
    'Symmetrizer', 'SlaterDeterminant', 'IdenticalParticlesScattering',
]
