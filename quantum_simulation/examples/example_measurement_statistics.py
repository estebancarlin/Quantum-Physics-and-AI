"""
Exemple : Validation statistiques mesure quantique.

Démontre :
- Distribution probabilités P(aₙ) = |⟨uₙ|ψ⟩|²
- Test χ² empirique vs théorique
- Réduction paquet d'ondes
"""

import sys
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum_simulation.experiments.measurement_statistics import MeasurementStatistics


def load_config():
    """Charge configuration depuis YAML."""
    config_file = project_root / "quantum_simulation/config/parameters.yaml"
    
    # CORRECTION : Spécifier encodage UTF-8
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def plot_distributions(results, save_dir="./quantum_simulation/results/"):
    """
    Affiche histogramme empirique vs distribution théorique.
    """
    analysis = results['analysis']
    dists = analysis['distributions']
    
    outcomes = dists['outcomes']
    eigenvalues = dists['eigenvalues']
    probs_theory = dists['theoretical_probabilities']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogramme empirique
    ax.hist(outcomes, bins=30, density=True, alpha=0.6, 
            label='Distribution empirique', color='steelblue')
    
    # Distribution théorique
    ax.stem(eigenvalues, probs_theory / (eigenvalues[1] - eigenvalues[0]), 
            linefmt='r-', markerfmt='ro', basefmt='none',
            label='Distribution théorique')
    
    ax.set_xlabel('Valeur mesurée')
    ax.set_ylabel('Densité probabilité')
    ax.set_title('Validation postulat mesure : Distribution empirique vs théorique')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{save_dir}/measurement_distributions_{results['system_type']}.png", dpi=150, bbox_inches='tight')
    plt.show()


def main():
    print("="*70)
    print(" Expérience : Statistiques Mesure Quantique")
    print("="*70)
    print()
    
    # 1. Configuration
    print("[1/3] Chargement configuration...")
    config = load_config()
    print()
    
    # 2. Exécution
    print("[2/3] Exécution expérience...")
    experiment = MeasurementStatistics(config)
    results = experiment.run()
    print()
    
    # 3. Affichage résultats
    print("[3/3] Résultats validation:")
    print("-" * 50)
    for test_name, passed in results['validation'].items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:30s} : {status}")
    print("-" * 50)
    
    # Add system_type from config to results
    results['system_type'] = config['experiments']['measurement_statistics']['system_type']
    print(f" Système : {results['system_type']}")
    
    # Analyse détaillée
    analysis = results['analysis']
    print("\n  Statistiques mesures:")
    print(f"    Valeur moyenne mesurée : {analysis['mean']['measured']:.6e}")
    print(f"    Valeur moyenne théorique: {analysis['mean']['theoretical']:.6e}")
    print(f"    Écart relatif          : {analysis['mean']['relative_error']:.2%}")
    
    # Visualisation
    print("\n  Génération visualisations...")
    plot_distributions(results)
    
    print("\n" + "="*70)
    print(f" Expérience terminée en {results['execution_time_seconds']:.2f}s")
    print("="*70)


if __name__ == "__main__":
    main()