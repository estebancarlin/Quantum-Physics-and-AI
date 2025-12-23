# quantum_simulation/tests/test_orchestration/test_pipeline.py

import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum_simulation.orchestration.pipeline import ExperimentPipeline, PipelineResults
from quantum_simulation.experiments.wavepacket_evolution import WavePacketEvolution
import yaml


@pytest.fixture
def test_config():
    """Configuration test minimale."""
    config_path = project_root / "quantum_simulation/config/parameters.yaml"
    
    # ✅ FIX : Spécifier encodage UTF-8 explicitement
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_pipeline_sequential(test_config):
    """Test pipeline séquentiel 2 expériences."""
    # 2 instances même expérience (variations config)
    exp1 = WavePacketEvolution(test_config)
    exp2 = WavePacketEvolution(test_config)
    
    pipeline = ExperimentPipeline(
        [exp1, exp2],
        pipeline_config={'name': 'test_sequential'}
    )
    
    results = pipeline.run(parallel=False)
    
    assert isinstance(results, PipelineResults)
    assert results.n_experiments == 2
    assert len(results.results) == 2


def test_pipeline_checkpoint(test_config, tmp_path):
    """Test sauvegarde/reprise checkpoint."""
    exp = WavePacketEvolution(test_config)
    
    pipeline = ExperimentPipeline([exp], pipeline_config={
        'name': 'test_checkpoint',
        'checkpoint_dir': str(tmp_path)
    })
    
    # Checkpoint avant exécution
    checkpoint_file = tmp_path / "test.pkl"
    pipeline.checkpoint(str(checkpoint_file))
    
    assert checkpoint_file.exists()
    
    # Charger checkpoint
    pipeline2 = ExperimentPipeline([], pipeline_config={})
    pipeline2.load_checkpoint(str(checkpoint_file))
    
    assert pipeline2.pipeline_name == 'test_checkpoint'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])