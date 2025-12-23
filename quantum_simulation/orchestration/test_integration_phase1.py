# quantum_simulation/tests/test_orchestration/test_integration_phase1.py
"""
Tests intégration Phase 1 : Pipeline + Comparaisons + Rapports.
"""

import pytest
import sys
from pathlib import Path
import yaml

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum_simulation.orchestration.pipeline import ExperimentPipeline
from quantum_simulation.orchestration.comparisons import ComparisonEngine
from quantum_simulation.orchestration.reports import ReportGenerator
from quantum_simulation.experiments.wavepacket_evolution import WavePacketEvolution


@pytest.fixture
def config_full():
    """Configuration complète."""
    config_path = project_root / "quantum_simulation/config/parameters.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_full_pipeline_workflow(config_full, tmp_path):
    """
    Test workflow complet Phase 1 :
    1. Créer 2 expériences
    2. Exécuter pipeline
    3. Comparer résultats
    4. Générer rapports
    """
    # 1. Préparation expériences
    config1 = config_full.copy()
    config1['experiments']['wavepacket_evolution']['initial_state']['k0'] = 5e9
    
    config2 = config_full.copy()
    config2['experiments']['wavepacket_evolution']['initial_state']['k0'] = 8e9
    
    exp1 = WavePacketEvolution(config1)
    exp2 = WavePacketEvolution(config2)
    
    # 2. Exécution pipeline
    pipeline = ExperimentPipeline(
        [exp1, exp2],
        pipeline_config={
            'name': 'integration_test',
            'checkpoint_dir': str(tmp_path)
        }
    )
    
    results = pipeline.run(parallel=False)
    
    # Vérifications pipeline
    assert results.n_experiments == 2
    assert len(results.results) == 2
    
    # 3. Comparaison
    engine = ComparisonEngine(hbar=config_full['physical_constants']['hbar'])
    comparison = engine.compare_observables(results.results)
    
    # Vérifications comparaison
    assert len(comparison.experiment_names) == 2
    assert 'mean_p' in comparison.metrics
    
    # Impulsions moyennes différentes (k0 différents)
    p1 = comparison.metrics['mean_p'][0]
    p2 = comparison.metrics['mean_p'][1]
    assert abs(p2 - p1) > 1e-25  # Écart significatif
    
    # 4. Génération rapports
    reporter = ReportGenerator(output_dir=str(tmp_path))
    
    md_path = reporter.generate_markdown_report(results)
    json_path = reporter.generate_json_report(results)
    
    # Vérifications rapports
    assert Path(md_path).exists()
    assert Path(json_path).exists()
    assert Path(md_path).stat().st_size > 0
    assert Path(json_path).stat().st_size > 0


def test_checkpoint_recovery(config_full, tmp_path):
    """Test reprise pipeline depuis checkpoint."""
    # Préparation
    config1 = config_full.copy()
    exp1 = WavePacketEvolution(config1)
    
    # Pipeline initial
    pipeline = ExperimentPipeline(
        [exp1],
        pipeline_config={
            'name': 'checkpoint_test',
            'checkpoint_dir': str(tmp_path)
        }
    )
    
    # Checkpoint avant exécution
    checkpoint_file = tmp_path / "test_checkpoint.pkl"
    pipeline.checkpoint(str(checkpoint_file))
    
    # Nouveau pipeline charge checkpoint
    pipeline2 = ExperimentPipeline([], pipeline_config={})
    pipeline2.load_checkpoint(str(checkpoint_file))
    
    # Vérifications
    assert pipeline2.pipeline_name == 'checkpoint_test'
    assert checkpoint_file.exists()


def test_error_handling_continue(config_full, tmp_path, monkeypatch):
    """Test gestion erreurs avec continue_on_error=True."""
    
    # Expérience valide
    exp_valid = WavePacketEvolution(config_full)
    
    # Expérience qui échoue (config invalide)
    config_invalid = config_full.copy()
    config_invalid['experiments']['wavepacket_evolution']['initial_state']['sigma_x'] = -1e-9  # Invalide
    exp_invalid = WavePacketEvolution(config_invalid)
    
    pipeline = ExperimentPipeline(
        [exp_valid, exp_invalid],
        pipeline_config={
            'name': 'error_test',
            'continue_on_error': True
        }
    )
    
    # Doit terminer malgré erreur
    results = pipeline.run(parallel=False)
    
    # Vérifications
    assert results.n_experiments == 2
    assert len(results.errors) >= 1  # Au moins une erreur enregistrée


if __name__ == "__main__":
    pytest.main([__file__, "-v"])