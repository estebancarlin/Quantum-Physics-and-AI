# quantum_simulation/utils/config_loader.py
"""
Chargement configuration depuis YAML.

Fonction utilitaire partagée par tous exemples/expériences.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Charge configuration depuis parameters.yaml.
    
    Args:
        config_path: Chemin fichier YAML (optionnel)
                    Si None, utilise chemin par défaut
    
    Returns:
        Configuration complète (dict)
    
    Raises:
        FileNotFoundError: Si fichier introuvable
    
    Usage:
        >>> config = load_config()
        >>> hbar = config['physical_constants']['hbar']
    """
    if config_path is None:
        # Chemin par défaut (relatif à quantum_simulation/)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "quantum_simulation" / "config" / "parameters.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Fichier configuration introuvable : {config_path}\n"
            f"Vérifier que parameters.yaml existe dans quantum_simulation/config/"
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Validation cohérence h/ℏ
    h = config['physical_constants']['h']
    hbar = config['physical_constants']['hbar']
    expected_hbar = h / (2 * 3.141592653589793)
    
    if abs(hbar - expected_hbar) > 1e-40:
        import warnings
        warnings.warn(
            f"Incohérence h/ℏ : hbar={hbar:.15e} vs h/(2π)={expected_hbar:.15e}",
            RuntimeWarning
        )
    
    return config


if __name__ == "__main__":
    # Test chargement
    try:
        cfg = load_config()
        print("✓ Configuration chargée avec succès")
        print(f"  ℏ = {cfg['physical_constants']['hbar']:.15e} J·s")
        print(f"  m_e = {cfg['physical_constants']['m_electron']:.6e} kg")
    except Exception as e:
        print(f"✗ Erreur : {e}")