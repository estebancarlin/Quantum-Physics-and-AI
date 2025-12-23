"""
Pipeline orchestration pour exécutions batch d'expériences quantiques.

Fonctionnalités:
    - Exécution séquentielle ou parallèle
    - Gestion checkpoints (reprise après crash)
    - Collecte résultats structurés
    - Gestion erreurs robuste
"""

import time
import pickle
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

from quantum_simulation.experiments.base_experiment import Experiment


@dataclass
class PipelineResults:
    """
    Résultats pipeline multi-expériences.
    
    Attributes:
        pipeline_name: Nom identifiant pipeline
        n_experiments: Nombre expériences exécutées
        experiment_names: Liste noms expériences
        results: Liste résultats individuels
        execution_times: Temps calcul par expérience (s)
        total_time: Temps total pipeline (s)
        all_passed: True si toutes validations passées
        errors: Liste erreurs rencontrées
    """
    pipeline_name: str
    n_experiments: int
    experiment_names: List[str]
    results: List[Dict[str, Any]]
    execution_times: List[float]
    total_time: float
    all_passed: bool
    errors: List[Dict[str, str]] = field(default_factory=list)
    
    def summary(self) -> str:
        """Résumé textuel résultats."""
        lines = [
            f"{'='*70}",
            f" Pipeline: {self.pipeline_name}",
            f"{'='*70}",
            f" Expériences exécutées  : {self.n_experiments}",
            f" Temps total           : {self.total_time:.2f}s",
            f" Validations globales  : {'✓ PASS' if self.all_passed else '✗ FAIL'}",
            f"{'='*70}",
            "\n Détails par expérience:"
        ]
        
        for i, (name, time_s, result) in enumerate(zip(
            self.experiment_names, self.execution_times, self.results
        ), 1):
            status = "✓" if result.get('all_validations_passed', False) else "✗"
            lines.append(f"  {i}. {name:30s} [{time_s:6.2f}s] {status}")
            
        if self.errors:
            lines.append(f"\n ⚠️  {len(self.errors)} erreur(s) rencontrée(s)")
            for err in self.errors:
                lines.append(f"    - {err['experiment']}: {err['message']}")
        
        lines.append("="*70)
        return "\n".join(lines)


class ExperimentPipeline:
    """
    Orchestrateur séquentiel/parallèle d'expériences quantiques.
    
    Usage:
        experiments = [
            WavePacketEvolution(config1),
            MeasurementStatistics(config2)
        ]
        pipeline = ExperimentPipeline(experiments, pipeline_config={'name': 'test'})
        results = pipeline.run(parallel=True, n_workers=4)
        print(results.summary())
    """
    
    def __init__(self, experiments: List[Experiment], 
                 pipeline_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            experiments: Liste expériences à exécuter
            pipeline_config: Configuration pipeline
                - name: str (nom pipeline)
                - timeout: float (timeout par expérience, s)
                - continue_on_error: bool (si True, continue malgré erreurs)
                - checkpoint_dir: str (dossier checkpoints)
        """
        self.experiments = experiments
        self.config = pipeline_config or {}
        self.results_history: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, str]] = []
        
        # Configuration par défaut
        self.pipeline_name = self.config.get('name', 'unnamed_pipeline')
        self.timeout = self.config.get('timeout', None)
        self.continue_on_error = self.config.get('continue_on_error', True)
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints/'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self, parallel: bool = False, n_workers: int = 1) -> PipelineResults:
        """
        Exécute pipeline complet.
        
        Args:
            parallel: Si True, parallélise expériences indépendantes
            n_workers: Nombre processus parallèles (si parallel=True)
            
        Returns:
            PipelineResults avec métadonnées + résultats individuels
            
        Note:
            En mode parallèle, expériences doivent être indépendantes
            (pas de partage état entre expériences).
        """
        print(f"\n{'='*70}")
        print(f" Démarrage Pipeline: {self.pipeline_name}")
        print(f"{'='*70}")
        print(f" Mode           : {'Parallèle' if parallel else 'Séquentiel'}")
        print(f" Expériences    : {len(self.experiments)}")
        if parallel:
            print(f" Workers        : {n_workers}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        if parallel:
            results_data = self._run_parallel(n_workers)
        else:
            results_data = self._run_sequential()
            
        total_time = time.time() - start_time
        
        # Compilation résultats
        experiment_names = [exp.__class__.__name__ for exp in self.experiments]
        execution_times = [r.get('execution_time_seconds', 0.0) for r in results_data]
        all_passed = all(r.get('all_validations_passed', False) for r in results_data)
        
        pipeline_results = PipelineResults(
            pipeline_name=self.pipeline_name,
            n_experiments=len(self.experiments),
            experiment_names=experiment_names,
            results=results_data,
            execution_times=execution_times,
            total_time=total_time,
            all_passed=all_passed,
            errors=self.errors
        )
        
        print(f"\n{pipeline_results.summary()}")
        
        return pipeline_results
    
    def _run_sequential(self) -> List[Dict[str, Any]]:
        """Exécution séquentielle."""
        results = []
        
        for i, exp in enumerate(self.experiments, 1):
            exp_name = exp.__class__.__name__
            print(f"\n[{i}/{len(self.experiments)}] Exécution: {exp_name}")
            print("-" * 70)
            
            try:
                result = exp.run()
                results.append(result)
                self.results_history.append(result)
                
                # Checkpoint automatique
                if i % 5 == 0:  # Tous les 5 expériences
                    self._auto_checkpoint(i)
                    
            except Exception as e:
                error_msg = f"Erreur dans {exp_name}: {str(e)}"
                warnings.warn(error_msg, category=RuntimeWarning)
                self.errors.append({'experiment': exp_name, 'message': str(e)})
                
                if not self.continue_on_error:
                    raise RuntimeError(f"Pipeline arrêté: {error_msg}")
                else:
                    # Placeholder résultat vide
                    results.append({
                        'experiment_name': exp_name,
                        'error': str(e),
                        'all_validations_passed': False
                    })
        
        return results
    
    def _run_parallel(self, n_workers: int) -> List[Dict[str, Any]]:
        """
        Exécution parallèle.
        
        LIMITATION: Nécessite que Experiment.run() soit picklable.
        """
        results = [None] * len(self.experiments)
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Soumettre toutes tâches
            future_to_idx = {
                executor.submit(self._run_single_experiment, exp): i
                for i, exp in enumerate(self.experiments)
            }
            
            # Collecter résultats au fur et à mesure
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                exp_name = self.experiments[idx].__class__.__name__
                
                try:
                    result = future.result(timeout=self.timeout)
                    results[idx] = result
                    completed += 1
                    print(f"  [{completed}/{len(self.experiments)}] ✓ {exp_name}")
                    
                except Exception as e:
                    error_msg = f"Erreur dans {exp_name}: {str(e)}"
                    warnings.warn(error_msg, RuntimeWarning)
                    self.errors.append({'experiment': exp_name, 'message': str(e)})
                    
                    results[idx] = {
                        'experiment_name': exp_name,
                        'error': str(e),
                        'all_validations_passed': False
                    }
        
        return results
    
    @staticmethod
    def _run_single_experiment(experiment: Experiment) -> Dict[str, Any]:
        """Wrapper pour exécution parallèle."""
        return experiment.run()
    
    def checkpoint(self, filepath: Optional[str] = None):
        """
        Sauvegarde état pipeline (reprise calcul).
        
        Args:
            filepath: Chemin fichier checkpoint (.pkl)
        """
        if filepath is None:
            filepath = self.checkpoint_dir / f"{self.pipeline_name}_checkpoint.pkl"
        else:
            filepath = Path(filepath)
            
        checkpoint_data = {
            'pipeline_name': self.pipeline_name,
            'config': self.config,
            'results_history': self.results_history,
            'errors': self.errors,
            'n_completed': len(self.results_history)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        print(f"✓ Checkpoint sauvegardé: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Reprend pipeline depuis checkpoint.
        
        Args:
            filepath: Chemin fichier checkpoint
            
        Note:
            Expériences déjà complétées seront sautées.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {filepath}")
            
        with open(filepath, 'rb') as f:
            checkpoint_data = pickle.load(f)
            
        self.pipeline_name = checkpoint_data['pipeline_name']
        self.config = checkpoint_data['config']
        self.results_history = checkpoint_data['results_history']
        self.errors = checkpoint_data['errors']
        
        n_completed = checkpoint_data['n_completed']
        print(f"✓ Checkpoint chargé: {n_completed} expériences déjà complétées")
        
        # Retirer expériences déjà exécutées
        self.experiments = self.experiments[n_completed:]
        
    def _auto_checkpoint(self, iteration: int):
        """Checkpoint automatique tous les N pas."""
        auto_path = self.checkpoint_dir / f"{self.pipeline_name}_auto_{iteration}.pkl"
        self.checkpoint(str(auto_path))


if __name__ == "__main__":
    # Test minimal
    print("Module pipeline chargé avec succès")