# quantum_simulation/tests/test_orchestration/test_comparisons.py
"""
Tests pour ComparisonEngine (comparaisons multi-expériences).
"""

import pytest
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum_simulation.orchestration.comparisons import ComparisonEngine, ComparisonReport
from quantum_simulation.core.state import WaveFunctionState


@pytest.fixture
def mock_results():
    """Résultats expériences simulés."""
    # Grille commune
    x = np.linspace(-1e-8, 1e-8, 1024)
    dx = x[1] - x[0]
    
    # ✅ FIX 1 : État final expérience 1 (gaussienne centrée)
    sigma = 2e-9
    psi1_raw = np.exp(-x**2 / (2 * sigma**2))
    
    # Normalisation correcte (discrète)
    norm1 = np.sqrt(np.sum(np.abs(psi1_raw)**2) * dx)
    psi1 = psi1_raw / norm1
    state1 = WaveFunctionState(x, psi1.astype(complex))
    
    # Vérification
    assert abs(state1.norm() - 1.0) < 1e-9, f"État 1 mal normé: {state1.norm()}"
    
    # ✅ FIX 2 : État final expérience 2 (gaussienne décalée)
    x0 = 1e-9
    psi2_raw = np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    # Normalisation correcte
    norm2 = np.sqrt(np.sum(np.abs(psi2_raw)**2) * dx)
    psi2 = psi2_raw / norm2
    state2 = WaveFunctionState(x, psi2.astype(complex))
    
    # Vérification
    assert abs(state2.norm() - 1.0) < 1e-9, f"État 2 mal normé: {state2.norm()}"
    
    results = [
        {
            'experiment_name': 'Exp1',
            'evolved_states': [state1],
            'initial_state': state1
        },
        {
            'experiment_name': 'Exp2',
            'evolved_states': [state2],
            'initial_state': state2
        }
    ]
    
    return results


def test_compare_observables_structure(mock_results):
    """Vérifie structure ComparisonReport."""
    engine = ComparisonEngine(hbar=1.054571817e-34)
    
    report = engine.compare_observables(mock_results)
    
    # Vérifier attributs
    assert isinstance(report, ComparisonReport)
    assert len(report.experiment_names) == 2
    assert 'mean_x' in report.metrics
    assert 'delta_x' in report.metrics
    assert 'heisenberg_product' in report.metrics
    
    # Vérifier dimensions
    assert report.metrics['mean_x'].shape == (2,)
    assert report.metrics['norm_final'].shape == (2,)


def test_compare_observables_values(mock_results):
    """Vérifie cohérence valeurs calculées."""
    engine = ComparisonEngine(hbar=1.054571817e-34)
    
    report = engine.compare_observables(mock_results)
    
    # ✅ Vérifications pré-test (debug)
    print(f"\n=== États de test ===")
    for i, res in enumerate(mock_results):
        state = res['evolved_states'][0]
        print(f"État {i+1}: norme = {state.norm():.10f}")
    
    # Expérience 1 : gaussienne centrée → ⟨X⟩ ≈ 0
    assert abs(report.metrics['mean_x'][0]) < 1e-10, \
        f"⟨X⟩ Exp1 attendu ~0, obtenu {report.metrics['mean_x'][0]:.3e}"
    
    # Expérience 2 : gaussienne décalée → ⟨X⟩ ≈ 1e-9
    # Tolérance ajustée pour discrétisation
    assert abs(report.metrics['mean_x'][1] - 1e-9) < 5e-10, \
        f"⟨X⟩ Exp2 attendu ~1e-9, obtenu {report.metrics['mean_x'][1]:.3e}"
    
    # ✅ Conservation norme (maintenant correcte)
    print(f"\n=== Normes finales ===")
    print(f"Exp1: {report.metrics['norm_final'][0]:.10f}")
    print(f"Exp2: {report.metrics['norm_final'][1]:.10f}")
    
    assert np.all(np.abs(report.metrics['norm_final'] - 1.0) < 1e-6), \
        f"Normes non conservées: {report.metrics['norm_final']}"


def test_compare_wavefunctions():
    """Test distances L² et fidélités."""
    engine = ComparisonEngine()
    
    x = np.linspace(-1e-8, 1e-8, 512)
    
    # Deux états identiques
    psi1 = np.exp(-x**2 / (2 * (2e-9)**2))
    psi1 /= np.linalg.norm(psi1) * np.sqrt(x[1] - x[0])
    state1 = WaveFunctionState(x, psi1.astype(complex))
    
    states_list1 = [state1, state1]
    states_list2 = [state1, state1]
    
    comparison = engine.compare_wavefunctions(states_list1, states_list2)
    
    # États identiques → distance = 0, fidélité = 1
    assert np.all(comparison['l2_distances'] < 1e-10)
    assert np.all(np.abs(comparison['fidelities'] - 1.0) < 1e-10)


def test_markdown_export(mock_results):
    """Test génération tableau Markdown."""
    engine = ComparisonEngine(hbar=1.054571817e-34)
    
    report = engine.compare_observables(mock_results)
    
    md_table = report.to_markdown_table()
    
    # Vérifier format
    assert '## Comparaison Expériences' in md_table
    assert '| Expérience |' in md_table
    assert 'Exp1' in md_table
    assert 'Exp2' in md_table


def test_statistical_tests(mock_results):
    """Validation tests statistiques."""
    engine = ComparisonEngine(hbar=1.054571817e-34)
    
    report = engine.compare_observables(mock_results)
    
    # Tests doivent être présents
    assert 'mean_x' in report.statistical_tests
    assert 'mean' in report.statistical_tests['mean_x']
    assert 'std' in report.statistical_tests['mean_x']
    assert 'relative_std' in report.statistical_tests['mean_x']


def test_recommendations_generation(mock_results):
    """Vérifier génération recommandations."""
    engine = ComparisonEngine(hbar=1.054571817e-34)
    
    report = engine.compare_observables(mock_results)
    
    # Recommandations doivent exister
    assert len(report.recommendations) > 0
    assert isinstance(report.recommendations[0], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])