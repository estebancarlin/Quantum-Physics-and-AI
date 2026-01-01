# quantum_simulation/examples/example_gallery_2d.py
"""
Démonstration galerie expériences 2D.

Exécute séquentiellement :
1. Double-slit (interférences)
2. Puits infini 2D (dégénérescence)
3. Quantum dot (confinement)

Génère rapport comparatif.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum_simulation.utils.config_loader import load_config
from quantum_simulation.experiments.gallery.double_slit_2d import DoubleSlitExperiment
from quantum_simulation.orchestration.pipeline import ExperimentPipeline
from quantum_simulation.orchestration.reports import ReportGenerator


def main():
    print("="*70)
    print(" Galerie Expériences Quantiques 2D")
    print("="*70)
    print()
    
    # 1. Configuration
    print("[1/3] Chargement configuration...")
    config = load_config()
    print()
    
    # 2. Liste expériences
    print("[2/3] Préparation expériences...")
    experiments = [
        DoubleSlitExperiment(config),
        # QuantumBilliard2D(config, shape='stadium'),  # Phase 2C
        # VortexStates2D(config)                        # Phase 2C
    ]
    print(f"  ✓ {len(experiments)} expérience(s) préparée(s)")
    print()
    
    # 3. Pipeline
    print("[3/3] Exécution pipeline...")
    pipeline = ExperimentPipeline(
        experiments=experiments,
        pipeline_config={'name': 'gallery_2d'}
    )
    
    results = pipeline.run(parallel=False)
    print()
    
    # 4. Génération rapport
    print("Génération rapport...")
    reporter = ReportGenerator(output_dir='quantum_simulation/results/gallery_2d/')
    
    md_path = reporter.generate_markdown_report(results)
    json_path = reporter.generate_json_report(results)
    
    print(f"  ✓ Rapport Markdown : {md_path}")
    print(f"  ✓ Rapport JSON     : {json_path}")
    
    try:
        html_path = reporter.generate_html_report(results)
        print(f"  ✓ Rapport HTML     : {html_path}")
    except ImportError:
        print("  ⚠️ Rapport HTML skippé (plotly non installé)")
    
    print()
    print("="*70)
    print(" ✓ Galerie complète!")
    print("="*70)
    print(f" 📁 Résultats : quantum_simulation/results/gallery_2d/")
    print("="*70)


if __name__ == "__main__":
    main()