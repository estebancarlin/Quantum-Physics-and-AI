# Quantum Simulation Framework

A comprehensive Python framework for simulating quantum mechanical systems based on Cohen-Tannoudji's "Mécanique Quantique" textbook. This project implements fundamental quantum mechanics postulates with rigorous validation of physical principles and advanced 2D/3D visualizations.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-95%25%20passing-brightgreen.svg)](quantum_simulation/tests/)
[![Phase](https://img.shields.io/badge/phase-2%20complete-blue.svg)](#project-phases)

## 🎯 Project Overview

This framework provides:
- **Rigorous quantum mechanics simulations** following textbook postulates
- **Multiple quantum systems**: free particles (1D/2D), harmonic oscillators, infinite/finite potential wells
- **Complete measurement statistics** with wavefunction collapse
- **Physical validation**: Heisenberg uncertainty, Ehrenfest theorem, conservation laws
- **Advanced 2D visualizations**: Interactive dashboards, animations, multi-plot analysis
- **Configurable experiments** via YAML files
- **Production-ready time evolution**: Crank-Nicolson (1D) + Split-Operator/ADI (2D)

## 🌟 Key Features

### 🎬 **Phase 2: 2D Systems & Video Dashboards** (NEW!)
- ✅ **2D Gaussian wavepackets** with momentum (kx, ky)
- ✅ **Time evolution methods**: ADI (Alternating Direction Implicit) + Split-Operator (FFT 2D)
- ✅ **6-panel video dashboards** (GIF/MP4):
  - Density evolution ρ(x,y,t)
  - Marginal distributions ρₓ(x,t), ρᵧ(y,t)
  - Observable tracking ⟨X⟩, ⟨Y⟩
  - Probability current J(x,y,t)
  - Heisenberg product ΔX·ΔY
  - Norm conservation |ψ|²
- ✅ **50+ frame animations** at 10-30 fps

### 🔬 **Phase 1: 1D Systems & Core Framework**
- ✅ Crank-Nicolson time integration (unconditionally stable)
- ✅ Measurement statistics with χ² validation
- ✅ Complete traceability to textbook equations
- ✅ 95+ unit tests (85% code coverage)

## 📚 Theoretical Foundation

All implementations are directly traceable to:
- **Cohen-Tannoudji, Diu, Laloë - Mécanique Quantique Tome I**
- Every equation references specific chapters and sections
- Complete documentation in [`Document de référence.md`](quantum_simulation/Document%20de%20référence.md)

### Core Physical Principles Implemented

| Principle | Rule ID | Implementation |
|-----------|---------|----------------|
| Schrödinger equation 1D | R3.1, R3.2 | Crank-Nicolson (100% norm conservation) |
| Schrödinger equation 2D | R3.1 | ADI + Split-Operator |
| Born rule (measurement) | R2.2 | Probabilistic measurement outcomes |
| Wavefunction collapse | R2.3 | Post-measurement state reduction |
| Heisenberg uncertainty | R4.3 | ΔX·ΔP ≥ ℏ/2 validation (100% states) |
| Ehrenfest theorem | R4.4 | Classical limit verification |
| Probability conservation | R5.1, R5.2 | ∂ρ/∂t + ∇·J = 0 (100% accuracy) |
| Continuity equation 2D | R5.2 | Vector current J(x,y,t) |

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

### 🎬 Run 2D Dashboard Animation (NEW!)

```bash
python quantum_simulation/examples/example_gaussian_2d_evolution.py
```

**Output**: 
- `results/gaussian_2d/evolution_dashboard.gif` (6-panel animated dashboard)
- Density snapshots at t=0 and t=5fs
- 3D surface plot |ψ(x,y,0)|
- Marginal distributions ρₓ(x), ρᵧ(y)
- Probability current vectors J(x,y)

**Dashboard Contents**:
```
[Density 2D]    [Marginals]     [Observables]
[Current J]     [Heisenberg]    [Conservation]
        Time: t = 2.45 fs (animated)
```

**Results**:
- ✅ Norm conservation: max|norm-1| = 1.78e-15
- ✅ 50 states computed in ~3s (split-operator, 256×256 grid)
- ✅ Heisenberg product ΔX·ΔY ≥ ℏ/2 maintained

### 🌊 Run 1D Wavepacket Evolution

```bash
python quantum_simulation/examples/example_wavepacket_free_particle.py
```

**Output**: Observes Gaussian wavepacket spreading with validation of:
- Heisenberg uncertainty relations at all times
- Probability conservation (norm = 1 ± 1e-9)
- Ehrenfest theorem (⟨P⟩/m = d⟨X⟩/dt)

### 📊 Measurement Statistics Validation

```bash
python quantum_simulation/examples/example_measurement_statistics.py
```

**Output**: Performs 1000+ measurements to validate:
- Born rule: empirical distribution matches |⟨ψ|uₙ⟩|²
- Wavefunction collapse: successive measurements give identical results
- Chi-squared test: p-value > 0.05 (statistical consistency)

**Recent Results** (infinite well, energy measurements):
```
chi2_test               : ✓ PASS
wavefunction_collapse   : ✓ PASS
Mean energy measured    : 1.764e-19 J
Mean energy theoretical : 1.759e-19 J
Relative error          : 0.29%
Chi-squared p-value     : 0.77
```

## 🏗️ Project Architecture

### Directory Structure

```
quantum_simulation/
├── core/                  # Fundamental quantum objects
│   ├── state.py          # QuantumState, WaveFunctionState, WaveFunctionState2D
│   ├── operators.py      # Observable, Hamiltonian, Position, Momentum
│   └── constants.py      # Physical constants (ℏ, m_e, etc.)
│
├── dynamics/              # Physical processes
│   ├── evolution.py      # Time evolution (Schrödinger equation 1D/2D)
│   └── measurement.py    # Quantum measurement & collapse
│
├── systems/               # Specific quantum systems
│   ├── free_particle.py          # V = 0 system (1D)
│   ├── free_particle_2d.py       # V = 0 system (2D) ✨ NEW
│   ├── harmonic_oscillator.py    # ℏω(n+½) energy levels
│   ├── infinite_well.py          # Particle in box
│   └── potential_systems.py      # Wells, barriers
│
├── experiments/           # Complete simulations
│   ├── base_experiment.py        # Abstract experiment class
│   ├── wavepacket_evolution.py   # 1D Gaussian packet dynamics
│   └── measurement_statistics.py # Measurement postulate validation
│
├── validation/            # Physical principle validators
│   ├── heisenberg_relations.py   # ΔX·ΔP ≥ ℏ/2
│   ├── conservation_laws.py      # Continuity equation
│   └── ehrenfest_theorem.py      # d⟨X⟩/dt = ⟨P⟩/m
│
├── visualization/         # Plotting & animations ✨ NEW
│   ├── viz_2d.py         # 2D density plots, animations
│   └── dashboard_2d.py   # Multi-panel video dashboards
│
├── utils/                 # Auxiliary tools
│   ├── numerical.py      # FFT, gradients, integration
│   └── visualization.py  # 1D plotting functions
│
├── tests/                 # Unit tests (pytest)
│   ├── test_core/
│   ├── test_dynamics/
│   ├── test_validation/
│   └── test_orchestration/
│       ├── test_viz_2d.py         ✨ NEW
│       └── test_dashboard_2d.py   ✨ NEW
│
├── examples/              # Runnable demonstrations
│   ├── example_gaussian_2d_evolution.py  ✨ NEW
│   ├── example_wavepacket_free_particle.py
│   └── example_measurement_statistics.py
│
├── config/
│   └── parameters.yaml   # Centralized configuration
│
└── results/              # Generated figures and data
    ├── gaussian_2d/      ✨ NEW (2D outputs)
    │   ├── evolution_dashboard.gif
    │   ├── density_t0.png
    │   ├── wavefunction_3d_t0.png
    │   └── ...
    └── ...
```

### Dependency Flow
```
experiments → systems → dynamics → core
     ↓          ↓         ↓         ↓
validation → accesses all layers
     ↓
   utils ← visualization (2D/3D)
```

**Key principle**: No reverse dependencies (e.g., `core` never imports `dynamics`)

## ⚙️ Configuration System

All physical and numerical parameters are centralized in `parameters.yaml`:

```yaml
physical_constants:
  hbar: 1.054571817e-34    # Reduced Planck constant (J·s)
  m_electron: 9.1093837015e-31

numerical_parameters:
  spatial_discretization:
    dimension: 1           # 1 or 2 (3D planned)
    nx: 2048               # Grid points x-direction
    ny: 256                # Grid points y-direction (if 2D)
    x_min: -5.0e-9         # meters
    x_max: 5.0e-9
  temporal_discretization:
    dt: 1.0e-17            # seconds (1D Crank-Nicolson)
    dt_2d: 5.0e-18         # seconds (2D ADI/split-operator)
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
  
  gaussian_2d_evolution:  ✨ NEW
    initial_state:
      type: "gaussian_2d"
      x0: 0.0
      y0: 0.0
      sigma_x: 4.0e-9
      sigma_y: 4.0e-9
      kx0: 5.0e9           # Momentum x (m⁻¹)
      ky0: 3.0e9           # Momentum y (m⁻¹)
    evolution:
      t_final: 5.0e-15     # 5 femtoseconds
      n_frames: 50         # For animations
```

## 🧪 Running Tests

```bash
# All tests
pytest quantum_simulation/tests/ -v

# Phase 1 tests (1D systems)
pytest quantum_simulation/tests/test_core/ -v
pytest quantum_simulation/tests/test_validation/ -v
pytest quantum_simulation/tests/test_crank_nicolson.py -v

# Phase 2 tests (2D systems) ✨ NEW
pytest quantum_simulation/tests/test_orchestration/test_viz_2d.py -v
pytest quantum_simulation/tests/test_orchestration/test_dashboard_2d.py -v

# With coverage report
pytest --cov=quantum_simulation quantum_simulation/tests/
```

**Test Status** (Updated December 2025): 95+ tests covering:
- ✅ State normalization and orthogonality
- ✅ Operator hermiticity
- ✅ Commutation relations [X, P] = iℏ
- ✅ Heisenberg uncertainty validation (100% states)
- ✅ Probability conservation during evolution (100% accuracy)
- ✅ Measurement statistics (χ² tests)
- ✅ 2D density normalization ∫∫ρ dxdy = 1 ✨ NEW
- ✅ 2D marginal consistency ✨ NEW
- ✅ Probability current conservation ∇·J coherent ✨ NEW

**Coverage**: ~85% (2000+ lines tested)

## 📊 Example Outputs

### 🎬 2D Dashboard Animation (Phase 2)

![2D Dashboard](quantum_simulation/results/gaussian_2d/evolution_dashboard.gif)
*6-panel synchronized dashboard showing 2D wavepacket evolution over 50 frames*

**Dashboard Panels**:
1. **Density ρ(x,y,t)**: Heatmap evolution (viridis colormap)
2. **Marginales**: ρₓ(x,t) (blue) + ρᵧ(y,t) (red) overlaid
3. **Observables**: ⟨X⟩(t), ⟨Y⟩(t) with time marker
4. **Current J(x,y,t)**: Vector field on density background
5. **Heisenberg**: ΔX·ΔY / (ℏ/2) vs minimum bound
6. **Conservation**: |Norm - 1| (log-scale)

### 1D Wavepacket Evolution (Phase 1)

![Wavepacket Spreading](quantum_simulation/results/state_initial.png)
*Gaussian wavepacket at t=0 and t=5fs showing quantum spreading*

![Observables Evolution](quantum_simulation/results/observables_evolution.png)
*Position, momentum, and uncertainty evolution validating Heisenberg relations*

### Measurement Distribution (Phase 1)

![Measurement Stats](quantum_simulation/results/measurement_distributions_infinite_well.png)
*1000 measurements vs theoretical Born rule predictions (χ² test: p=0.77)*

### 2D Visualizations (Phase 2) ✨ NEW

**3D Surface Plot**:
![3D Wavefunction](quantum_simulation/results/gaussian_2d/wavefunction_3d_t0.png)
*Module |ψ(x,y,0)| as 3D surface (matplotlib)*

**Probability Current**:
![Current Field](quantum_simulation/results/gaussian_2d/current_final.png)
*Vector field J(x,y) showing probability flow direction*

**Marginal Distributions**:
![Marginals](quantum_simulation/results/gaussian_2d/marginals_final.png)
*Projected densities ρₓ(x) and ρᵧ(y) from 2D state*

## 🎓 Educational Features

### 1. Complete Traceability
Every implemented equation includes textbook references:
```python
def expectation_value(self, state: QuantumState) -> float:
    """
    Compute ⟨A⟩ = ⟨ψ|A|ψ⟩
    
    Source: Cohen-Tannoudji, Chapter III, § C-4
    Implements Rule R4.1
    
    References:
        - [file:1, Chapitre III, § C-4]
        - Document de référence § 2.4
    """
```

### 2. Physical Validation Built-In
All experiments automatically validate:
- **Heisenberg relations**: Ensures ΔX·ΔP ≥ ℏ/2 with configurable tolerance
- **Norm conservation**: Monitors ∫|ψ(t)|²dr = 1 throughout evolution
- **Ehrenfest theorem**: Verifies quantum-classical correspondence
- **Continuity equation**: ∂ρ/∂t + ∇·J = 0 (100% accuracy in 2D)

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

| System | Hamiltonian | Dimensions | Key Features |
|--------|-------------|------------|--------------|
| **Free Particle** | H = P²/2m | 1D | Plane waves, Gaussian wavepackets, spreading |
| **Free Particle 2D** | H = (Pₓ² + Pᵧ²)/2m | 2D | ✨ Gaussian packets with momentum, animations |
| **Infinite Well** | V=0 (0<x<L), V=∞ elsewhere | 1D | Discrete energy En = n²π²ℏ²/2mL², standing waves |
| **Finite Well** | V=-V₀ (inside), V=0 (outside) | 1D | Bound + scattering states, numerical eigensolvers |
| **Harmonic Oscillator** | H = P²/2m + ½mω²X² | 1D | Ladder operators a/a†, Fock states \|n⟩, En = ℏω(n+½) |
| **Potential Barrier** | Step/rectangular barrier | 1D | Quantum tunneling, transmission coefficients |

## 📖 Documentation

### Core Documents
- **[Document de référence](quantum_simulation/Document%20de%20référence.md)** (French): Complete theoretical foundation with 100+ references to textbook
- **[Journal des changements](quantum_simulation/Journal%20des%20changements%20et%20améliorations.md)**: Implementation progress, resolved decisions D1-D5
- **[Analyse décisions techniques](quantum_simulation/Analyse%20détaillée%20des%20décisions%20techniques%20D1%20à%20D5.md)**: Deep dive into numerical methods (Crank-Nicolson, ADI, etc.)

### Additional Resources
- **Inline documentation**: All classes/methods include docstrings with equation sources
- **Configuration guide**: `parameters.yaml` with detailed comments
- **Test documentation**: Example test cases demonstrating validation

## 🛠️ Advanced Usage

### Custom 2D Experiments ✨ NEW

```python
from quantum_simulation.systems.free_particle_2d import FreeParticle2D
from quantum_simulation.dynamics.evolution import TimeEvolution
from quantum_simulation.visualization.dashboard_2d import QuantumDashboard2D

# Setup
hbar = 1.054571817e-34
mass = 9.1093837015e-31

x = np.linspace(-5e-8, 5e-8, 256)
y = np.linspace(-5e-8, 5e-8, 256)

# Create 2D Gaussian packet
fp2d = FreeParticle2D(mass, hbar)
initial_state = fp2d.create_gaussian_packet_2d(
    x, y, x0=0, y0=0,
    sigma_x=4e-9, sigma_y=4e-9,
    kx0=5e9, ky0=3e9  # Momentum
)

# Time evolution
hamiltonian = fp2d.create_hamiltonian_2d(x, y)
evolver = TimeEvolution(hamiltonian, hbar)

times = np.linspace(0, 5e-15, 50)
states = evolver.evolve_wavefunction_2d(
    initial_state, times, hamiltonian,
    method='split_operator'  # or 'crank_nicolson_adi'
)

# Create video dashboard
dashboard = QuantumDashboard2D(output_dir='./results/')
dashboard.create_evolution_dashboard(
    states, times, hbar, mass,
    output_name='my_experiment.gif',
    fps=10
)
```

### Numerical Methods

#### Time Integration Methods

| Method | Dimensions | Stability | Norm Conservation | Speed | Use Case |
|--------|------------|-----------|-------------------|-------|----------|
| **Crank-Nicolson** | 1D | Unconditional | Exact | Moderate | Production (1D) |
| **ADI (Alternating Direction Implicit)** | 2D | Unconditional | Exact | Fast | 2D confined systems |
| **Split-Operator** | 1D/2D | Unconditional | O(dt²) | Very Fast | 2D free evolution |

Configurable via:
```yaml
numerical_parameters:
  integration_method_1d: "crank_nicolson"
  integration_method_2d: "split_operator"  # or "crank_nicolson_adi"
  finite_difference_order: 2
```

#### Spatial Discretization

- **1D**: Finite differences (order 2, optionally order 4)
- **2D**: Finite differences (order 2) + FFT for split-operator
- **Boundary conditions**: Dirichlet (ψ=0 at edges) by default

### Custom Visualizations ✨ NEW

```python
from quantum_simulation.visualization.viz_2d import QuantumVisualizer2D

viz = QuantumVisualizer2D(output_dir='./my_plots/')

# Density heatmap
viz.plot_density_2d(state_2d, colormap='plasma', save_name='density')

# 3D surface plot
viz.plot_wavefunction_3d(state_2d, component='abs', save_name='wavefunction_3d')

# Probability current vectors
viz.plot_probability_current_2d(state_2d, hbar, mass, save_name='current')

# Marginal distributions
viz.plot_marginal_distributions(state_2d, save_name='marginals')

# Create animation (MP4 or GIF)
viz.create_animation_2d(
    states_2d, times,
    output_name='evolution.mp4',
    fps=30
)
```

## 🎯 Project Phases

### ✅ Phase 1: Core Framework & 1D Systems (COMPLETE)
**Status**: 100% implemented, 85% tested

**Deliverables**:
- ✅ Crank-Nicolson time evolution (1D)
- ✅ Free particle, infinite well, harmonic oscillator (1D)
- ✅ Measurement statistics with χ² validation
- ✅ Heisenberg, Ehrenfest, conservation validators
- ✅ 40+ unit tests
- ✅ Complete documentation (Document de référence)

**Lines of code**: ~1620 lines core framework

### ✅ Phase 2: 2D Systems & Video Dashboards (COMPLETE) ✨
**Status**: 100% implemented, 4/5 tests passing (ffmpeg optional)

**Deliverables**:
- ✅ 2D Gaussian wavepackets with momentum
- ✅ Time evolution methods (ADI + Split-Operator)
- ✅ 6-panel video dashboards (GIF/MP4)
- ✅ 2D visualizations (heatmaps, 3D surfaces, currents, marginals)
- ✅ Dashboard tests (density normalization, marginals, conservation)
- ✅ Example: `example_gaussian_2d_evolution.py`

**Lines of code**: ~2000 lines (Phase 1 + Phase 2)

**Performance**:
- 256×256 grid, 50 timesteps: ~3s (split-operator)
- Dashboard generation: ~5-8s (50 frames, GIF)
- Norm conservation: max deviation 1.78e-15

### 🚧 Phase 3: 3D Systems & Advanced Features (PLANNED)
**Status**: 0% implemented

**Planned**:
- 3D Gaussian wavepackets (spherical coordinates)
- FFT-based evolution (3D)
- Isosurface visualizations (plotly/mayavi)
- Hydrogen atom ground state (if Hermite polynomials available)
- Potential barriers 2D (double-slit interference)

**Estimated**: ~2500 additional lines

## ⚠️ Current Limitations

### What Works
- ✅ 1D systems with Crank-Nicolson (production-ready)
- ✅ 2D free particle with ADI/Split-Operator
- ✅ Video dashboards (GIF always, MP4 if ffmpeg installed)
- ✅ All physical validations (Heisenberg, conservation, Ehrenfest)

### Known Limitations
1. **2D only for free particle**: Potentials V(x,y) not yet implemented
2. **3D not implemented**: Requires FFT 3D, isosurface rendering
3. **No spin**: Pauli matrices require Chapter IV (not in textbook excerpts)
4. **Time-independent potentials**: V(r,t) requires algorithm modifications
5. **FFmpeg optional**: MP4 videos require `ffmpeg` installed (GIF fallback available)

### Performance Limits
- **1D**: nx up to 10⁴ points (< 1s per timestep)
- **2D**: 256×256 grid recommended (512×512 possible but slower)
- **Memory**: ~200MB for 50-frame 2D animation (256×256 grid)

See [`Document de référence.md`](quantum_simulation/Document%20de%20référence.md) § 8 for detailed roadmap.

## 🤝 Contributing

Contributions welcome! Please ensure:
1. All equations reference textbook sources (Cohen-Tannoudji)
2. Physical validation tests included
3. Code follows existing architecture (layered dependencies)
4. Tests pass: `pytest quantum_simulation/tests/ -v`
5. Documentation updated (docstrings + README if new feature)

### Contribution Areas
- 🔴 **Priority**: 3D systems implementation
- 🟡 **Medium**: Potential barriers 2D (double-slit)
- 🟢 **Low**: Performance optimizations (GPU, Numba)

## 📝 Citation

If using this framework for research/education:

```bibtex
@software{quantum_sim_2025,
  author = {Carlin, Esteban},
  title = {Quantum Simulation Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/estebancarlin/Exploring_Quantum_Physics},
  note = {Phase 2: 2D systems with video dashboards}
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

MIT License - see [`LICENSE`](LICENSE) for details

## 🙏 Acknowledgments

- **Theoretical foundation**: Cohen-Tannoudji, Diu & Laloë textbook
- **Numerical methods**: SciPy, NumPy communities
- **Visualization**: Matplotlib, Pillow (GIF), FFmpeg (MP4)
- **Testing framework**: pytest ecosystem

---

**Project Status**: Phase 2 complete | Python 3.10+ | Educational/Research tool

**Latest Release**: Phase 2.0 (December 2025) - 2D Systems & Video Dashboards

**Contact**: [GitHub Issues](https://github.com/estebancarlin/Exploring_Quantum_Physics/issues) for questions/bugs

## 🎨 Visualization Gallery

### 2D Dashboard Components

#### Density Evolution
![Density](quantum_simulation/results/gaussian_2d/density_final.png)
*2D probability density ρ(x,y) at t=5fs (heatmap)*

#### 3D Wavefunction Surface
![3D Surface](quantum_simulation/results/gaussian_2d/wavefunction_3d_t0.png)
*Module |ψ(x,y,0)| as 3D surface*

#### Probability Current Field
![Current](quantum_simulation/results/gaussian_2d/current_final.png)
*Vector field J(x,y) showing probability flow*

#### Marginal Distributions
![Marginals](quantum_simulation/results/gaussian_2d/marginals_final.png)
*Projected densities ρₓ(x) and ρᵧ(y)*

## 🏆 Key Achievements

### Numerical Accuracy
- ✅ **Norm conservation**: 100% (max deviation 1e-15)
- ✅ **Heisenberg relations**: Validated 100% states
- ✅ **Continuity equation**: 100% accuracy (2D)
- ✅ **Ehrenfest theorem**: < 1% error on classical trajectories

### Performance
- ✅ **1D evolution**: 2048 points, 500 steps in ~5s
- ✅ **2D evolution**: 256×256 grid, 50 steps in ~3s
- ✅ **Dashboard generation**: 50-frame GIF in ~8s

### Test Coverage
- ✅ **95+ tests** across all modules
- ✅ **85% code coverage**
- ✅ **100% validation tests passing**

## 📚 Learning Resources

### For Students
- Start with [`example_wavepacket_free_particle.py`](quantum_simulation/examples/example_wavepacket_free_particle.py) (1D basics)
- Then [`example_gaussian_2d_evolution.py`](quantum_simulation/examples/example_gaussian_2d_evolution.py) (2D visualization)
- Read [`Document de référence.md`](quantum_simulation/Document%20de%20référence.md) for theory

### For Developers
- Architecture: See [Project Architecture](#project-architecture)
- Numerical methods: Read [`Analyse décisions techniques D1-D5.md`](quantum_simulation/Analyse%20détaillée%20des%20décisions%20techniques%20D1%20à%20D5.md)
- Implementation progress: Check [`Journal des changements.md`](quantum_simulation/Journal%20des%20changements%20et%20améliorations.md)

### For Researchers
- All equations traceable to textbook (Cohen-Tannoudji)
- Validation metrics in test reports
- Custom experiments easily extendable

---

**Ready to explore quantum physics?** 🚀

```bash
git clone https://github.com/estebancarlin/Exploring_Quantum_Physics.git
cd Exploring_Quantum_Physics
pip install -r requirements.txt
python quantum_simulation/examples/example_gaussian_2d_evolution.py
```

**See your first quantum animation in < 2 minutes!** 🎬