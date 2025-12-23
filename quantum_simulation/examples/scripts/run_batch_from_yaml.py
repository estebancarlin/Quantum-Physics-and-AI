# quantum_simulation/examples/scripts/run_batch_from_yaml.py
"""
Exécute pipeline depuis fichier batch YAML.

Usage:
    python run_batch_from_yaml.py batch_config.yaml
"""

import sys
from pathlib import Path
import argparse

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum_simulation.orchestration.batch_loader import BatchConfigLoader
from quantum_simulation.orchestration.pipeline import ExperimentPipeline
from quantum_simulation.orchestration.reports import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description='Run batch experiments from YAML')
    parser.add_argument('batch_config', type=str, help='Path to batch YAML config')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--output', type=str, default='./results/batch/', help='Output directory')
    
    args = parser.parse_args()
    
    print("="*70)
    print(" Batch Execution from YAML")
    print("="*70)
    print(f" Config file : {args.batch_config}")
    print(f" Mode        : {'Parallel' if args.parallel else 'Sequential'}")
    print("="*70)
    print()
    
    # 1. Chargement expériences
    print("[1/3] Loading experiments from YAML...")
    loader = BatchConfigLoader()
    experiments = loader.load_from_yaml(args.batch_config)
    print(f"  ✓ {len(experiments)} experiments loaded")
    print()
    
    # 2. Exécution pipeline
    print("[2/3] Running pipeline...")
    pipeline = ExperimentPipeline(
        experiments,
        pipeline_config={
            'name': Path(args.batch_config).stem,
            'continue_on_error': True
        }
    )
    
    results = pipeline.run(parallel=args.parallel, n_workers=args.workers)
    print()
    
    # 3. Génération rapports
    print("[3/3] Generating reports...")
    reporter = ReportGenerator(output_dir=args.output)
    
    md_path = reporter.generate_markdown_report(results)
    json_path = reporter.generate_json_report(results)
    
    print(f"  ✓ Markdown : {md_path}")
    print(f"  ✓ JSON     : {json_path}")
    
    try:
        html_path = reporter.generate_html_report(results)
        print(f"  ✓ HTML     : {html_path}")
    except ImportError:
        print("  ⚠️ HTML skipped (plotly not installed)")
    
    print()
    print("="*70)
    print(" Batch execution completed!")
    print("="*70)


if __name__ == "__main__":
    main()