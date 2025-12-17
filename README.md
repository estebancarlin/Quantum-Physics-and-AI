# Quantum Simulation Framework

A comprehensive Python framework for simulating quantum mechanical systems based on Cohen-Tannoudji's "Mécanique Quantique" textbook. This project implements fundamental quantum mechanics postulates with rigorous validation of physical principles.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](quantum_simulation/tests/)

## 🎯 Project Overview

This framework provides:
- **Rigorous quantum mechanics simulations** following textbook postulates
- **Multiple quantum systems**: free particles, harmonic oscillators, infinite/finite potential wells
- **Complete measurement statistics** with wavefunction collapse
- **Physical validation**: Heisenberg uncertainty, Ehrenfest theorem, conservation laws
- **Configurable experiments** via YAML files
- **Comprehensive visualizations** of wavefunctions and observables

## 📚 Theoretical Foundation

All implementations are directly traceable to:
- **Cohen-Tannoudji, Diu, Laloë - Mécanique Quantique Tome I**
- Every equation references specific chapters and sections
- Complete documentation in Document de référence.md

### Core Physical Principles Implemented

| Principle | Rule ID | Implementation |
|-----------|---------|----------------|
| Schrödinger equation | R3.1, R3.2 | Time evolution with Crank-Nicolson |
| Born rule (measurement) | R2.2 | Probabilistic measurement outcomes |
| Wavefunction collapse | R2.3 | Post-measurement state reduction |
| Heisenberg uncertainty | R4.3 | ΔX·ΔP ≥ ℏ/2 validation |
| Ehrenfest theorem | R4.4 | Classical limit verification |
| Probability conservation | R5.1, R5.2 | ∂ρ/∂t + ∇·J = 0 |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/estebancarlin/Exploring_Quantum_Physics.git
cd Exploring_Quantum_Physics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Example Simulations

#### 1. Free Particle Wavepacket Evolution

```bash
python quantum_simulation/examples/example_wavepacket_free_particle.py
```

**Output**: Observes Gaussian wavepacket spreading over time with validation of:
- Heisenberg uncertainty relations at all times
- Probability conservation (norm = 1)
- Ehrenfest theorem (⟨P⟩/m = d⟨X⟩/dt)

#### 2. Measurement Statistics Validation

```bash
python quantum_simulation/examples/example_measurement_statistics.py
```

**Output**: Performs 1000+ measurements to validate:
- Born rule: empirical distribution matches |⟨ψ|uₙ⟩|²
- Wavefunction collapse: successive measurements give identical results
- Chi-squared test: p-value > 0.05 (statistical consistency)

**Recent Results**:
```
chi2_test               : ✓ PASS
wavefunction_collapse   : ✓ PASS
Mean energy measured    : 1.764e-19 J
Mean energy theoretical : 1.759e-19 J
Relative error          : 0.29%
```

## 🏗️ Project Architecture

```
quantum_simulation/
├── core/                  # Fundamental quantum objects
│   ├── state.py          # QuantumState, WaveFunctionState
│   ├── operators.py      # Observable, Hamiltonian, Position, Momentum
│   └── constants.py      # Physical constants (ℏ, m_e, etc.)
│
├── dynamics/              # Physical processes
│   ├── evolution.py      # Time evolution (Schrödinger equation)
│   └── measurement.py    # Quantum measurement & collapse
│
├── systems/               # Specific quantum systems
│   ├── free_particle.py          # V = 0 system
│   ├── harmonic_oscillator.py    # ℏω(n+½) energy levels
│   └── potential_systems.py      # Wells, barriers
│
├── experiments/           # Complete simulations
│   ├── base_experiment.py        # Abstract experiment class
│   ├── wavepacket_evolution.py   # Gaussian packet dynamics
│   └── measurement_statistics.py # Measurement postulate validation
│
├── validation/            # Physical principle validators
│   ├── heisenberg_relations.py   # ΔX·ΔP ≥ ℏ/2
│   ├── conservation_laws.py      # Continuity equation
│   └── ehrenfest_theorem.py      # d⟨X⟩/dt = ⟨P⟩/m
│
├── utils/                 # Auxiliary tools
│   ├── numerical.py      # FFT, gradients, integration
│   └── visualization.py  # Plotting functions
│
├── tests/                 # Unit tests (pytest)
├── examples/              # Runnable demonstrations
├── config/
│   └── parameters.yaml   # Centralized configuration
└── results/              # Generated figures and data
```

### Dependency Flow
```
experiments → systems → dynamics → core
     ↓          ↓         ↓         ↓
validation → accesses all layers
     ↓
   utils
```

**Key principle**: No reverse dependencies (e.g., `core` never imports `dynamics`)

## ⚙️ Configuration System

All physical and numerical parameters are centralized in parameters.yaml:

```yaml
physical_constants:
  hbar: 1.054571817e-34    # Reduced Planck constant (J·s)
  m_electron: 9.1093837015e-31

numerical_parameters:
  spatial_discretization:
    nx: 2048               # Grid points
    x_min: -5.0e-9         # meters
    x_max: 5.0e-9
  temporal_discretization:
    dt: 1.0e-17            # seconds
  tolerances:
    normalization_check: 1.0e-10
    heisenberg_inequality: 1.0e-10

experiments:
  wavepacket_evolution:
    initial_state:
      type: "gaussian"
      x0: 0.0
      sigma_x: 2.0e-9      # Width (meters)
      k0: 5.0e9            # Wavenumber (m⁻¹)
```

## 🧪 Running Tests

```bash
# All tests
pytest quantum_simulation/tests/ -v

# Specific test categories
pytest quantum_simulation/tests/test_core/ -v           # Core quantum objects
pytest quantum_simulation/tests/test_validation/ -v     # Physical principles
pytest quantum_simulation/tests/test_measurement_statistics.py -v

# With coverage report
pytest --cov=quantum_simulation quantum_simulation/tests/
```

**Test Status**: 40+ tests covering:
- State normalization and orthogonality
- Operator hermiticity
- Commutation relations [X, P] = iℏ
- Heisenberg uncertainty validation
- Probability conservation during evolution
- Measurement statistics (χ² tests)

## 📊 Example Outputs

### Wavepacket Evolution
![Wavepacket Spreading](quantum_simulation/results/state_initial.png)
*Gaussian wavepacket at t=0 and t=5fs showing quantum spreading*

### Observable Time Evolution
!Observables
*Position, momentum, and uncertainty evolution validating Heisenberg relations*

### Measurement Distribution
![Measurement Stats](quantum_simulation/results/measurement_distributions_infinite_well.png)
*1000 measurements vs theoretical Born rule predictions (χ² test: p=0.77)*

## 🎓 Educational Features

### 1. Complete Traceability
Every implemented equation includes:
```python
def expectation_value(self, state: QuantumState) -> float:
    """
    Compute ⟨A⟩ = ⟨ψ|A|ψ⟩
    
    Source: Cohen-Tannoudji, Chapter III, § C-4
    Implements Rule R4.1
    """
```

### 2. Physical Validation Built-In
All experiments automatically validate:
- **Heisenberg relations**: Ensures ΔX·ΔP ≥ ℏ/2 with configurable tolerance
- **Norm conservation**: Monitors ∫|ψ(t)|²dr = 1 throughout evolution
- **Ehrenfest theorem**: Verifies quantum-classical correspondence

### 3. Step-by-Step Experiment Workflow
```python
class Experiment(ABC):
    def run(self):
        self.prepare_initial_state()  # |ψ(t₀)⟩
        self.define_hamiltonian()     # H = P²/2m + V
        self.evolve_state()           # iℏ∂ψ/∂t = Hψ
        self.perform_measurements()   # Observables & statistics
        self.validate_physics()       # Check principles
        self.analyze_results()        # Generate reports
```

## 🔬 Implemented Quantum Systems

| System | Hamiltonian | Key Features |
|--------|-------------|--------------|
| **Free Particle** | H = P²/2m | Plane waves, Gaussian wavepackets, spreading dynamics |
| **Infinite Well** | V=0 (0<x<L), V=∞ elsewhere | Discrete energy levels En = n²π²ℏ²/2mL², standing waves |
| **Finite Well** | V=-V₀ (inside), V=0 (outside) | Bound + scattering states, numerical eigensolvers |
| **Harmonic Oscillator** | H = P²/2m + ½mω²X² | Ladder operators a/a†, Fock states \|n⟩, En = ℏω(n+½) |
| **Potential Barrier** | Step/rectangular barrier | Quantum tunneling, transmission coefficients |

## 📖 Documentation

- **[Document de référence](quantum_simulation/Document%20de%20référence.md)** (French): Complete theoretical foundation with 100+ references to textbook
- **Inline documentation**: All classes/methods include docstrings with equation sources
- **Configuration guide**: parameters.yaml with detailed comments

## 🛠️ Advanced Usage

### Custom Experiments

```python
from quantum_simulation.experiments.base_experiment import Experiment
from quantum_simulation.systems.free_particle import FreeParticle

class MyExperiment(Experiment):
    def prepare_initial_state(self):
        # Define custom initial state
        self.initial_state = FreeParticle(...).create_gaussian_wavepacket(...)
    
    def define_hamiltonian(self):
        # Define system Hamiltonian
        self.hamiltonian = ...
    
    def evolve_state(self):
        # Time evolution logic
        pass
    
    def perform_measurements(self):
        # Custom measurements
        pass
    
    def validate_physics(self) -> Dict[str, bool]:
        # Physical principle checks
        return {'heisenberg': True, ...}
```

### Numerical Methods

Current implementations:
- **Spatial discretization**: Uniform grid with finite differences (order 2)
- **Time integration**: Crank-Nicolson (implicit, unconditionally stable)
- **FFT support**: For momentum-space operations (planned)

Configurable via:
```yaml
numerical_parameters:
  integration_method: "crank_nicolson"  # or "runge_kutta", "split_operator"
  finite_difference_order: 2
```

## ⚠️ Current Limitations

1. **1D only**: 2D/3D support requires grid/Laplacian extensions
2. **No spin**: Pauli matrices not yet implemented
3. **Time-independent potentials**: V(r,t) requires algorithm modifications
4. **Spectral methods**: Full FFT-based evolution planned for future

See `Document de référence.md` § 8 for detailed roadmap.

## 🤝 Contributing

Contributions welcome! Please ensure:
1. All equations reference textbook sources
2. Physical validation tests included
3. Code follows existing architecture (layered dependencies)
4. Tests pass: `pytest quantum_simulation/tests/ -v`

## 📝 Citation

If using this framework for research/education:

```bibtex
@software{quantum_sim_2025,
  author = {Carlin, Esteban},
  title = {Quantum Simulation Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/estebancarlin/Exploring_Quantum_Physics}
}
```

Based on:
```bibtex
@book{cohen1977quantum,
  title={Mécanique Quantique},
  author={Cohen-Tannoudji, Claude and Diu, Bernard and Laloë, Franck},
  year={1977},
  publisher={Hermann}
}
```

## 📜 License

MIT License - see LICENSE for details

## 🙏 Acknowledgments

- **Theoretical foundation**: Cohen-Tannoudji, Diu & Laloë textbook
- **Numerical methods**: SciPy, NumPy communities
- **Testing framework**: pytest ecosystem

---

**Project Status**: Active development | Python 3.10+ | Educational/Research tool

**Contact**: [GitHub Issues](https://github.com/estebancarlin/Exploring_Quantum_Physics/issues) for questions/bugs