# quantum_simulation/orchestration/reports.py
"""
Génération rapports synthétiques multi-expériences.

Formats:
    - HTML interactif (plotly, navigation)
    - Markdown (export facile)
    - JSON structuré (archivage)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    import warnings
    warnings.warn("plotly non installé, rapports HTML désactivés", ImportWarning)


class ReportGenerator:
    """
    Générateur rapports multi-formats.
    
    Usage:
        reporter = ReportGenerator()
        reporter.generate_html_report(pipeline_results, "report.html")
        reporter.generate_json_report(pipeline_results, "data.json")
    """
    
    def __init__(self, output_dir: str = "./reports/"):
        """
        Args:
            output_dir: Dossier sortie rapports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_html_report(self, pipeline_results, 
                            output_path: Optional[str] = None) -> str:
        """
        Génère rapport HTML interactif.
        
        Args:
            pipeline_results: PipelineResults (orchestration.pipeline)
            output_path: Chemin fichier HTML (défaut: auto-généré)
            
        Returns:
            Chemin fichier généré
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly requis pour rapports HTML")
            
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"report_{pipeline_results.pipeline_name}_{timestamp}.html"
        else:
            output_path = Path(output_path)
            
        # Construction page HTML
        html_content = self._build_html_structure(pipeline_results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"✓ Rapport HTML généré: {output_path}")
        return str(output_path)
    
    def generate_markdown_report(self, pipeline_results,
                                 output_path: Optional[str] = None) -> str:
        """Génère rapport Markdown."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"report_{pipeline_results.pipeline_name}_{timestamp}.md"
        else:
            output_path = Path(output_path)
            
        md_lines = [
            f"# Rapport Pipeline: {pipeline_results.pipeline_name}",
            f"\n**Généré le**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "---\n",
            "## Résumé Exécution\n",
            f"- **Expériences**: {pipeline_results.n_experiments}",
            f"- **Temps total**: {pipeline_results.total_time:.2f}s",
            f"- **Statut global**: {'✓ PASS' if pipeline_results.all_passed else '✗ FAIL'}",
            "\n---\n",
            "## Détails Expériences\n",
            "| # | Nom | Temps (s) | Validations | Statut |",
            "|---|-----|-----------|-------------|--------|"
        ]
        
        for i, (name, time_s, result) in enumerate(zip(
            pipeline_results.experiment_names,
            pipeline_results.execution_times,
            pipeline_results.results
        ), 1):
            validations = result.get('validation', {})
            n_pass = sum(1 for v in validations.values() if v)
            n_total = len(validations)
            status = "✓" if result.get('all_validations_passed', False) else "✗"
            
            md_lines.append(f"| {i} | {name} | {time_s:.2f} | {n_pass}/{n_total} | {status} |")
        
        if pipeline_results.errors:
            md_lines.extend([
                "\n---\n",
                "## ⚠️ Erreurs Rencontrées\n"
            ])
            for err in pipeline_results.errors:
                md_lines.append(f"- **{err['experiment']}**: {err['message']}")
        
        md_content = "\n".join(md_lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        print(f"✓ Rapport Markdown généré: {output_path}")
        return str(output_path)
    
    def generate_json_report(self, pipeline_results,
                            output_path: Optional[str] = None) -> str:
        """Génère rapport JSON structuré (archivage)."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"data_{pipeline_results.pipeline_name}_{timestamp}.json"
        else:
            output_path = Path(output_path)
            
        # Sérialisation (exclure objets non-JSON)
        json_data = {
            'pipeline_name': pipeline_results.pipeline_name,
            'timestamp': datetime.now().isoformat(),
            'n_experiments': pipeline_results.n_experiments,
            'experiment_names': pipeline_results.experiment_names,
            'execution_times': pipeline_results.execution_times,
            'total_time': pipeline_results.total_time,
            'all_passed': pipeline_results.all_passed,
            'errors': pipeline_results.errors,
            'results_summary': [
                {
                    'experiment': name,
                    'validations_passed': result.get('all_validations_passed', False),
                    'validation_details': {k: bool(v) for k, v in result.get('validation', {}).items()}
                }
                for name, result in zip(pipeline_results.experiment_names, pipeline_results.results)
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
            
        print(f"✓ Rapport JSON généré: {output_path}")
        return str(output_path)
    
    def _build_html_structure(self, pipeline_results) -> str:
        """Construction HTML avec plotly."""
        # Graphique temps exécution
        fig_times = go.Figure(data=[
            go.Bar(
                x=pipeline_results.experiment_names,
                y=pipeline_results.execution_times,
                marker_color='steelblue'
            )
        ])
        fig_times.update_layout(
            title="Temps Exécution par Expérience",
            xaxis_title="Expérience",
            yaxis_title="Temps (s)"
        )
        
        # Graphique validations
        validations_data = []
        for result in pipeline_results.results:
            n_pass = sum(1 for v in result.get('validation', {}).values() if v)
            n_total = len(result.get('validation', {}))
            validations_data.append(n_pass / n_total if n_total > 0 else 0)
            
        fig_validations = go.Figure(data=[
            go.Bar(
                x=pipeline_results.experiment_names,
                y=[v * 100 for v in validations_data],
                marker_color=['green' if v == 1.0 else 'orange' for v in validations_data]
            )
        ])
        fig_validations.update_layout(
            title="Taux Réussite Validations (%)",
            xaxis_title="Expérience",
            yaxis_title="% Validations Passées"
        )
        
        # HTML complet
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Rapport Pipeline: {pipeline_results.pipeline_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .chart {{ margin: 30px 0; }}
    </style>
</head>
<body>
    <h1>Rapport Pipeline: {pipeline_results.pipeline_name}</h1>
    <p><strong>Généré le:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>Résumé</h2>
        <ul>
            <li><strong>Expériences:</strong> {pipeline_results.n_experiments}</li>
            <li><strong>Temps total:</strong> {pipeline_results.total_time:.2f}s</li>
            <li><strong>Statut:</strong> {'✓ PASS' if pipeline_results.all_passed else '✗ FAIL'}</li>
        </ul>
    </div>
    
    <div class="chart">
        {fig_times.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    
    <div class="chart">
        {fig_validations.to_html(full_html=False, include_plotlyjs=False)}
    </div>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    print("Module reports chargé avec succès")