import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest
from scipy.signal import find_peaks

from quantum_simulation.utils.config_loader import load_config
from quantum_simulation.experiments.gallery.double_slit_2d import DoubleSlitExperiment


def test_double_slit_interference():
    """Motif interférence fentes Young."""
    config = load_config()
    hbar = config['physical_constants']['hbar']
    mass = config['physical_constants']['m_electron']

    exp_cfg = config['experiments']['double_slit_2d']
    k0 = exp_cfg['initial_state']['momentum_x'] / mass

    exp = DoubleSlitExperiment(config)
    results = exp.run()

    # Extraction densité écran
    screen_density = results['measurements']['screen_distribution']
    y_positions = results['measurements']['y_screen']

    # Détection pics (scipy.signal.find_peaks)
    peaks, _ = find_peaks(screen_density)

    # Interfrange mesuré
    interfrange_measured = np.mean(np.diff(y_positions[peaks]))

    # Théorique : Δy = λD/d
    wavelength = hbar / (mass * k0)
    distance_screen = exp_cfg['screen_distance']
    slit_separation = exp_cfg['slit_separation']
    interfrange_theory = wavelength * distance_screen / slit_separation

    assert abs(interfrange_measured - interfrange_theory) / interfrange_theory < 0.1  # 10%
