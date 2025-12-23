# quantum_simulation/orchestration/comparisons.py
"""
Moteur comparaison quantitative entre expériences.

Métriques:
    - Écarts observables (⟨X⟩, ⟨P⟩, ΔX, ΔP)
    - Distances L² entre états
    - Fidélités quantiques
    - Tests statistiques (χ²)
"""

import numpy as np
import warnings
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from quantum_simulation.core.state import WaveFunctionState
from quantum_simulation.core.operators import PositionOperator, MomentumOperator


@dataclass
class ComparisonReport:
    """
    Rapport comparaison entre N expériences.
    
    Attributes:
        experiment_names: Noms expériences comparées
        metrics: Dict[metric_name, values] (N×M array)
        statistical_tests: Résultats tests significativité
        recommendations: Suggestions interprétation
    """
    experiment_names: List[str]
    metrics: Dict[str, np.ndarray]
    statistical_tests: Dict[str, Any]
    recommendations: List[str]
    
    def to_markdown_table(self) -> str:
        """Export tableau Markdown."""
        lines = ["## Comparaison Expériences\n"]
        
        # En-tête
        header = "| Expérience | " + " | ".join(self.metrics.keys()) + " |"
        separator = "|" + "---|" * (len(self.metrics) + 1)
        lines.extend([header, separator])
        
        # Lignes données
        for i, name in enumerate(self.experiment_names):
            row = f"| {name} |"
            for metric_values in self.metrics.values():
                value = metric_values[i]
                if isinstance(value, float):
                    row += f" {value:.6e} |"
                else:
                    row += f" {value} |"
            lines.append(row)
            
        return "\n".join(lines)


class ComparisonEngine:
    """
    Comparateur quantitatif multi-expériences.
    
    Usage:
        engine = ComparisonEngine()
        report = engine.compare_observables([result1, result2, result3])
        print(report.to_markdown_table())
    """
    
    def __init__(self, hbar: float = 1.054571817e-34):
        """
        Args:
            hbar: Constante Planck réduite (J·s)
        """
        self.hbar = hbar
        self.X = PositionOperator()
        self.P = MomentumOperator(hbar)
        
    def compare_observables(self, results_list: List[Dict[str, Any]]) -> ComparisonReport:
        """
        Compare valeurs moyennes observables entre N expériences.
        
        Args:
            results_list: Liste résultats expériences (format Experiment.run())
            
        Returns:
            ComparisonReport avec tableau (N × M observables)
        """
        n_exp = len(results_list)
        experiment_names = [r['experiment_name'] for r in results_list]
        
        # Extraction observables finales
        metrics = {
            'mean_x': np.zeros(n_exp),
            'mean_p': np.zeros(n_exp),
            'delta_x': np.zeros(n_exp),
            'delta_p': np.zeros(n_exp),
            'heisenberg_product': np.zeros(n_exp),
            'norm_final': np.zeros(n_exp)
        }
        
        for i, result in enumerate(results_list):
            # État final
            final_state = result['evolved_states'][-1] if result.get('evolved_states') else result.get('initial_state')
            
            if final_state is None:
                warnings.warn(f"Expérience {i} sans état final, skippée")
                continue
                
            # Observables
            metrics['mean_x'][i] = self.X.expectation_value(final_state)
            metrics['mean_p'][i] = self.P.expectation_value(final_state)
            metrics['delta_x'][i] = self.X.uncertainty(final_state)
            metrics['delta_p'][i] = self.P.uncertainty(final_state)
            metrics['heisenberg_product'][i] = metrics['delta_x'][i] * metrics['delta_p'][i]
            metrics['norm_final'][i] = final_state.norm()
        
        # Tests statistiques
        statistical_tests = self._perform_statistical_tests(metrics)
        
        # Recommandations
        recommendations = self._generate_recommendations(metrics, statistical_tests)
        
        return ComparisonReport(
            experiment_names=experiment_names,
            metrics=metrics,
            statistical_tests=statistical_tests,
            recommendations=recommendations
        )
    
    def compare_wavefunctions(self, 
                             states1: List[WaveFunctionState],
                             states2: List[WaveFunctionState],
                             label1: str = "Exp1",
                             label2: str = "Exp2") -> Dict[str, Any]:
        """
        Calcule distances entre séquences fonctions d'onde.
        
        Métriques:
            - Distance L² : ∫|ψ₁-ψ₂|² dx
            - Fidélité : |⟨ψ₁|ψ₂⟩|²
            
        Args:
            states1, states2: Séquences états (même longueur)
            label1, label2: Labels expériences
            
        Returns:
            Dict avec distances temporelles + moyennes
        """
        if len(states1) != len(states2):
            raise ValueError(f"Séquences longueurs différentes: {len(states1)} vs {len(states2)}")
            
        n_times = len(states1)
        l2_distances = np.zeros(n_times)
        fidelities = np.zeros(n_times)
        
        for i, (psi1, psi2) in enumerate(zip(states1, states2)):
            # Vérifier grilles identiques
            if not np.allclose(psi1.spatial_grid, psi2.spatial_grid):
                raise ValueError(f"Temps {i}: grilles spatiales différentes")
                
            # Distance L²
            diff = psi1.wavefunction - psi2.wavefunction
            l2_distances[i] = np.sqrt(np.sum(np.abs(diff)**2) * psi1.dx)
            
            # Fidélité F = |⟨ψ₁|ψ₂⟩|²
            overlap = psi1.inner_product(psi2)
            fidelities[i] = np.abs(overlap)**2
        
        return {
            'label1': label1,
            'label2': label2,
            'n_timesteps': n_times,
            'l2_distances': l2_distances,
            'fidelities': fidelities,
            'mean_l2_distance': np.mean(l2_distances),
            'mean_fidelity': np.mean(fidelities),
            'max_l2_distance': np.max(l2_distances),
            'min_fidelity': np.min(fidelities)
        }
    
    def _perform_statistical_tests(self, metrics: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Tests significativité écarts observables.
        
        Test: Variance relative ≪ 1 → valeurs cohérentes
        """
        tests = {}
        
        for metric_name, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if mean_val != 0:
                relative_std = std_val / np.abs(mean_val)
            else:
                relative_std = np.inf
                
            tests[metric_name] = {
                'mean': mean_val,
                'std': std_val,
                'relative_std': relative_std,
                'is_consistent': relative_std < 0.1  # 10% tolérance
            }
        
        return tests
    
    def _generate_recommendations(self, metrics: Dict[str, np.ndarray], 
                                 tests: Dict[str, Any]) -> List[str]:
        """Suggestions interprétation résultats."""
        recommendations = []
        
        # Conservation norme
        norms = metrics.get('norm_final', np.array([]))
        if len(norms) > 0 and np.any(np.abs(norms - 1.0) > 1e-6):
            recommendations.append(
                "⚠️ Conservation norme imparfaite détectée (erreur > 10⁻⁶). "
                "Vérifier paramètres évolution temporelle (dt, schéma)."
            )
        
        # Heisenberg
        heisenberg_products = metrics.get('heisenberg_product', np.array([]))
        heisenberg_limit = self.hbar / 2.0
        if len(heisenberg_products) > 0 and np.any(heisenberg_products < heisenberg_limit * 0.9):
            recommendations.append(
                "⚠️ Produit ΔX·ΔP < ℏ/2 détecté. Violation Heisenberg probable, "
                "revoir calcul incertitudes."
            )
        
        # Cohérence valeurs moyennes
        for metric_name, test_result in tests.items():
            if not test_result['is_consistent']:
                recommendations.append(
                    f"✓ Écart significatif détecté pour {metric_name} "
                    f"(std/mean = {test_result['relative_std']:.2%}). "
                    "Différences physiques probables entre expériences."
                )
        
        if not recommendations:
            recommendations.append("✓ Toutes métriques cohérentes. Résultats conformes.")
        
        return recommendations


if __name__ == "__main__":
    print("Module comparisons chargé avec succès")