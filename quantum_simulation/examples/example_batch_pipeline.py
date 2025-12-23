# quantum_simulation/examples/example_batch_pipeline.py
"""
Démonstration Pipeline Orchestration (Phase 1).

Exécute 2 expériences en batch avec génération rapport.
"""

import sys
from pathlib import Path
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum_simulation.orchestration.pipeline import ExperimentPipeline
from quantum_simulation.orchestration.comparisons import ComparisonEngine
from quantum_simulation.orchestration.reports import ReportGenerator
from quantum_simulation.experiments.wavepacket_evolution import WavePacketEvolution


def load_config():
    """Charge configuration."""
    config_path = project_root / "quantum_simulation/config/parameters.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    print("="*70)
    print(" Démonstration Pipeline Orchestration")
    print("="*70)
    print()
    
    # 1. Configuration
    print("[1/4] Chargement configuration...")
    config = load_config()
    
    # 2. Création expériences
    print("[2/4] Préparation expériences...")
    
    # Variante 1 : Paquet gaussien avec k0=5e9
    config1 = config.copy()
    config1['experiments']['wavepacket_evolution']['initial_state']['k0'] = 5e9
    exp1 = WavePacketEvolution(config1)
    
    # Variante 2 : Paquet gaussien avec k0=8e9
    config2 = config.copy()
    config2['experiments']['wavepacket_evolution']['initial_state']['k0'] = 8e9
    exp2 = WavePacketEvolution(config2)
    
    print(f"  ✓ 2 expériences préparées")
    print()
    
    # 3. Exécution pipeline
    print("[3/4] Exécution pipeline...")
    pipeline = ExperimentPipeline(
        experiments=[exp1, exp2],
        pipeline_config={
            'name': 'batch_wavepacket',
            'continue_on_error': True
        }
    )
    
    results = pipeline.run(parallel=False)
    print()
    
    # 4. Comparaison résultats
    print("[4/4] Analyse comparative...")
    comparison_engine = ComparisonEngine(hbar=config['physical_constants']['hbar'])
    comparison_report = comparison_engine.compare_observables(results.results)
    
    print()
    print(comparison_report.to_markdown_table())
    print()
    
    # 5. Génération rapports
    print("Génération rapports...")
    reporter = ReportGenerator(output_dir="./quantum_simulation/results/reports/")
    
    # Rapport Markdown
    md_path = reporter.generate_markdown_report(results)
    print(f"  ✓ Rapport MD : {md_path}")
    
    # Rapport JSON
    json_path = reporter.generate_json_report(results)
    print(f"  ✓ Données JSON : {json_path}")
    
    # Rapport HTML (si plotly disponible)
    try:
        html_path = reporter.generate_html_report(results)
        print(f"  ✓ Rapport HTML : {html_path}")
    except ImportError:
        print("  ⚠️ Rapport HTML ignoré (plotly non installé)")
    
    print()
    print("="*70)
    print(" Pipeline complété avec succès !")
    print("="*70)


if __name__ == "__main__":
    main()