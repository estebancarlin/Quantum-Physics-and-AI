# quantum_simulation/tests/test_orchestration/test_reports.py
"""
Tests pour ReportGenerator (génération rapports).
"""

import pytest
import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum_simulation.orchestration.reports import ReportGenerator
from quantum_simulation.orchestration.pipeline import PipelineResults


@pytest.fixture
def mock_pipeline_results():
    """PipelineResults simulé."""
    results = PipelineResults(
        pipeline_name='test_pipeline',
        n_experiments=2,
        experiment_names=['Exp1', 'Exp2'],
        results=[
            {
                'experiment_name': 'Exp1',
                'validation': {'heisenberg': True, 'conservation': True},
                'all_validations_passed': True
            },
            {
                'experiment_name': 'Exp2',
                'validation': {'heisenberg': True, 'conservation': False},
                'all_validations_passed': False
            }
        ],
        execution_times=[1.2, 2.5],
        total_time=3.7,
        all_passed=False,
        errors=[]
    )
    return results


def test_markdown_generation(mock_pipeline_results, tmp_path):
    """Test génération rapport Markdown."""
    reporter = ReportGenerator(output_dir=str(tmp_path))
    
    md_path = reporter.generate_markdown_report(
        mock_pipeline_results,
        output_path=tmp_path / "test_report.md"
    )
    
    # Vérifier fichier créé
    assert Path(md_path).exists()
    
    # Vérifier contenu
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    assert 'test_pipeline' in content
    assert 'Exp1' in content
    assert 'Exp2' in content
    assert '✓ PASS' in content or '✗ FAIL' in content


def test_json_generation(mock_pipeline_results, tmp_path):
    """Test génération rapport JSON."""
    reporter = ReportGenerator(output_dir=str(tmp_path))
    
    json_path = reporter.generate_json_report(
        mock_pipeline_results,
        output_path=tmp_path / "test_data.json"
    )
    
    # Vérifier fichier créé
    assert Path(json_path).exists()
    
    # Vérifier structure JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    assert data['pipeline_name'] == 'test_pipeline'
    assert data['n_experiments'] == 2
    assert len(data['results_summary']) == 2


def test_html_generation_without_plotly(mock_pipeline_results, tmp_path, monkeypatch):
    """Test gestion absence plotly."""
    # Simuler absence plotly
    import quantum_simulation.orchestration.reports as reports_module
    monkeypatch.setattr(reports_module, 'PLOTLY_AVAILABLE', False)
    
    reporter = ReportGenerator(output_dir=str(tmp_path))
    
    with pytest.raises(ImportError, match="plotly requis"):
        reporter.generate_html_report(mock_pipeline_results)


def test_output_directory_creation():
    """Vérifier création automatique dossier sortie."""
    test_dir = Path("./test_reports_temp/")
    
    # Supprimer si existe
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
    
    reporter = ReportGenerator(output_dir=str(test_dir))
    
    # Vérifier création
    assert test_dir.exists()
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])