from pathlib import Path

STRUCTURE = {
    "config": ["parameters.yaml"],
    "core": [
        "constants.py",
        "state.py",
        "operators.py",
        "hilbert_space.py",
    ],
    "dynamics": [
        "evolution.py",
        "measurement.py",
    ],
    "systems": [
        "free_particle.py",
        "harmonic_oscillator.py",
        "potential_systems.py",
    ],
    "experiments": [
        "base_experiment.py",
        "wavepacket_evolution.py",
        "measurement_statistics.py",
    ],
    "validation": [
        "heisenberg_relations.py",
        "conservation_laws.py",
        "ehrenfest_theorem.py",
    ],
    "utils": [
        "numerical.py",
        "visualization.py",
    ],
}

PROJECT_NAME = "quantum_simulation"


def create_project_structure(base_path="."):
    root = Path(base_path) / PROJECT_NAME
    root.mkdir(parents=True, exist_ok=True)

    for folder, files in STRUCTURE.items():
        folder_path = root / folder
        folder_path.mkdir(exist_ok=True)

        # Rendre les dossiers importables
        (folder_path / "__init__.py").touch(exist_ok=True)

        for file in files:
            (folder_path / file).touch(exist_ok=True)

    print(f"✅ Project '{PROJECT_NAME}' created successfully.")


if __name__ == "__main__":
    create_project_structure()
