# quantum_simulation/orchestration/__init__.py
"""
Module orchestration multi-expériences.

Composants:
    - ExperimentPipeline: Exécution batch séquentielle/parallèle
    - ComparisonEngine: Analyses comparatives quantitatives
    - ReportGenerator: Génération rapports multi-formats
"""

from .pipeline import ExperimentPipeline, PipelineResults
from .comparisons import ComparisonEngine, ComparisonReport
from .reports import ReportGenerator

__all__ = [
    'ExperimentPipeline',
    'PipelineResults',
    'ComparisonEngine',
    'ComparisonReport',
    'ReportGenerator'
]

__version__ = '1.0.0'