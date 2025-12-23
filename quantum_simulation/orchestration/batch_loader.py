# quantum_simulation/orchestration/batch_loader.py
"""
Chargement configurations batch depuis YAML.

Permet définir multi-expériences avec variations paramétriques.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any
from quantum_simulation.experiments.base_experiment import Experiment


class BatchConfigLoader:
    """
    Chargeur configurations batch expériences.
    
    Usage:
        loader = BatchConfigLoader()
        experiments = loader.load_from_yaml('batch_config.yaml')
        pipeline = ExperimentPipeline(experiments)
    """
    
    def __init__(self, base_config_path: str = None):
        """
        Args:
            base_config_path: Chemin config globale (parameters.yaml)
        """
        if base_config_path is None:
            base_config_path = Path(__file__).parent.parent / "config/parameters.yaml"
        
        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)
    
    def load_from_yaml(self, batch_config_path: str) -> List[Experiment]:
        """
        Charge expériences depuis fichier batch YAML.
        
        Args:
            batch_config_path: Chemin fichier batch (voir examples/)
            
        Returns:
            Liste expériences instanciées
            
        Format YAML attendu:
            experiments:
              - name: "exp1"
                class: "WavePacketEvolution"
                config_override:
                  experiments:
                    wavepacket_evolution:
                      initial_state:
                        k0: 5e9
        """
        with open(batch_config_path, 'r', encoding='utf-8') as f:
            batch_config = yaml.safe_load(f)
        
        experiments = []
        
        for exp_config in batch_config.get('experiments', []):
            exp_class_name = exp_config['class']
            config_override = exp_config.get('config_override', {})
            
            # Fusion config base + override
            merged_config = self._merge_configs(self.base_config, config_override)
            
            # Instanciation dynamique
            exp_instance = self._instantiate_experiment(exp_class_name, merged_config)
            experiments.append(exp_instance)
        
        return experiments
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Fusion récursive configurations."""
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _instantiate_experiment(self, class_name: str, config: Dict) -> Experiment:
        """Instanciation dynamique expérience."""
        # Import dynamique
        if class_name == 'WavePacketEvolution':
            from quantum_simulation.experiments.wavepacket_evolution import WavePacketEvolution
            return WavePacketEvolution(config)
        elif class_name == 'MeasurementStatistics':
            from quantum_simulation.experiments.measurement_statistics import MeasurementStatistics
            return MeasurementStatistics(config)
        else:
            raise ValueError(f"Classe expérience inconnue: {class_name}")


if __name__ == "__main__":
    # Test chargement
    loader = BatchConfigLoader()
    print("✓ BatchConfigLoader chargé")