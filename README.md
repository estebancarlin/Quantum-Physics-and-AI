# Exploring Quantum Physics with AI

A rigorous Python framework for simulating quantum mechanical systems, grounded in Cohen-Tannoudji's *Mécanique Quantique* (Tomes I–III) and oriented toward AI-assisted quantum research. This project explores the intersection of numerical quantum mechanics and machine learning — from exact Crank-Nicolson integration to planned Neural Quantum States and PINNs.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-95%25%20passing-brightgreen.svg)](quantum_simulation/tests/)

---

## Overview

The framework implements quantum mechanics from the ground up with full traceability to textbook postulates:

- **Rigorous time evolution**: Crank-Nicolson 1D (unconditionally stable, exact norm conservation) + Split-Operator / ADI 2D
- **Quantum systems**: free particle (1D/2D), harmonic oscillator, infinite/finite wells, potential barriers, double slit
- **Measurement postulates**: Born rule with chi-squared validation, wavefunction collapse
- **Physical validators**: Heisenberg dX*dP >= hbar/2, Ehrenfest theorem, probability current d(rho)/dt + div(J) = 0
- **GPU acceleration**: CuPy-based sparse solvers, auto-detected
- **4 pedagogical Jupyter notebooks**: free particle, measurement postulates, harmonic oscillator, double slit 2D

The theoretical foundation is the three-volume *Mecanique Quantique* (Cohen-Tannoudji, Diu & Laloë), referenced throughout as rules R2.2-R6.3. PDFs are in [`references/`](references/).

---

## Repository Structure

```
.
├── quantum_simulation/          # Core framework package
│   ├── core/                    # Fundamental objects (state, operators, constants)
│   ├── dynamics/                # Time evolution (Crank-Nicolson, split-operator) + measurement
│   ├── systems/                 # Quantum systems (free particle 1D/2D, HO, wells, barriers)
│   ├── experiments/             # Full runnable experiments + gallery (double slit, tunneling)
│   ├── validation/              # Heisenberg, Ehrenfest, conservation law validators
│   ├── visualization/           # 2D/3D plots, dashboards, animations
│   ├── orchestration/           # Batch pipelines, comparisons, reports
│   ├── utils/                   # Numerical tools (FFT, gradients), GPU manager, config loader
│   ├── tests/                   # pytest suite (95+ tests, ~85% coverage)
│   ├── examples/
│   │   ├── notebooks/           # Pedagogical Jupyter notebooks (grounded in Cohen-Tannoudji)
│   │   └── *.py                 # Runnable example scripts
│   ├── benchmarks/              # CPU vs GPU performance benchmarks
│   ├── config/
│   │   └── parameters.yaml      # Centralized physical + numerical parameters
│   └── results/                 # Generated figures, animations, reports
│
├── references/                  # Cohen-Tannoudji — Mecanique Quantique Tomes I, II, III
├── requirements.txt
└── README.md
```

---

## Pedagogical Notebooks

Four notebooks grounded in Cohen-Tannoudji, runnable end-to-end:

| Notebook | Content | Key rules |
| --- | --- | --- |
| [01 — Particule libre & paquet d'ondes](quantum_simulation/examples/notebooks/01_particule_libre_wavepacket.ipynb) | CN time evolution, Ehrenfest, Heisenberg products | R3.1, R4.3, R4.4, R5.1 |
| [02 — Postulats de la mesure](quantum_simulation/examples/notebooks/02_postulats_mesure.ipynb) | Born rule, 1000 simulated measurements, chi-squared, collapse | R2.2, R2.3 |
| [03 — Oscillateur harmonique](quantum_simulation/examples/notebooks/03_oscillateur_harmonique.ipynb) | Spectrum, psi_n(x) via Hermite, algebra a/a+, coherent states | R6.1-R6.3 |
| [04 — Double fente 2D](quantum_simulation/examples/notebooks/04_double_fente_2d.ipynb) | ADI/split-operator 2D, Young fringes, norm conservation | R3.1, R5.1 |

```bash
jupyter notebook quantum_simulation/examples/notebooks/
```

---

## Quick Start

```bash
git clone https://github.com/estebancarlin/Exploring_Quantum_Physics.git
cd Exploring_Quantum_Physics
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Run examples

```bash
# 1D Gaussian wavepacket (Crank-Nicolson)
python quantum_simulation/examples/example_wavepacket_free_particle.py

# 2D evolution dashboard (split-operator, 50-frame GIF)
python quantum_simulation/examples/example_gaussian_2d_evolution.py

# Measurement statistics (Born rule + chi-squared validation)
python quantum_simulation/examples/example_measurement_statistics.py
```

### Run tests

```bash
# Full suite
pytest quantum_simulation/tests/ -v

# Crank-Nicolson validation (6 tests: norm, Ehrenfest, O(dt^2) convergence)
pytest quantum_simulation/tests/test_crank_nicolson.py -v

# With coverage
pytest --cov=quantum_simulation quantum_simulation/tests/
```

---

## Time Integration

| Method | Dim | Stability | Norm conservation | Use case |
| --- | --- | --- | --- | --- |
| **Crank-Nicolson** | 1D | Unconditional | Exact (machine eps) | All 1D systems |
| **ADI** | 2D | Unconditional | Exact | Confined 2D potentials |
| **Split-Operator (FFT)** | 2D | Unconditional | O(dt^2) | Free 2D evolution, large grids |

The Crank-Nicolson scheme solves `(I + iHdt/2hbar) psi^{n+1} = (I - iHdt/2hbar) psi^n` via `scipy.sparse.linalg.spsolve` on a CSR tridiagonal Hamiltonian.

```python
from quantum_simulation.dynamics.evolution import TimeEvolution
from quantum_simulation.systems.free_particle import FreeParticle

fp = FreeParticle(mass=9.109e-31, hbar=1.055e-34)
psi0 = fp.create_gaussian_packet(x_grid, x0=0, sigma=2e-9, k0=5e9)

evol = TimeEvolution(fp.hamiltonian)
psi_t = evol.evolve_wavefunction(psi0, t0=0, t=5e-15, dt=5e-18)
```

---

## GPU Acceleration

Automatic CuPy-based acceleration for grids above threshold (1D: nx > 1024, 2D: nx*ny > 256^2).

```bash
pip install cupy-cuda12x           # requires CUDA 12.x
export QUANTUM_USE_GPU=true        # or false to disable
python quantum_simulation/benchmarks/benchmark_gpu.py
```

| Operation | Grid | CPU | GPU | Speedup |
| --- | --- | --- | --- | --- |
| Laplacian 2D (FFT) | 512x512 | 42.7 ms | 3.8 ms | 11x |
| Split-operator 2D | 512x512, 50 steps | 6 min | 32 s | 11x |
| Crank-Nicolson 1D | 4096 pts, 100 steps | 12.4 s | 3.8 s | 3x |

---

## Implemented Systems

| System | Hamiltonian | Dim | Features |
| --- | --- | --- | --- |
| Free Particle | P^2/2m | 1D/2D | Gaussian wavepackets, spreading, analytical overlay |
| Infinite Well | V=0 (0<x<L), V=inf | 1D | Discrete levels E_n = n^2*pi^2*hbar^2/2mL^2 |
| Finite Well | -V_0 inside | 1D | Bound + scattering states |
| Harmonic Oscillator | P^2/2m + mw^2X^2/2 | 1D | Ladder operators a/a+, Fock states, coherent states |
| Potential Barrier | Step/rectangular | 1D | Tunneling, transmission coefficients |
| Double Slit | Barrier + slits | 2D | Young interference, fringe detection |

---

## Physical Validation

All experiments automatically verify:

| Principle | Rule | Tolerance |
| --- | --- | --- |
| Norm conservation | R5.1 | norm - 1 < 1e-9 |
| Heisenberg dX·dP >= hbar/2 | R4.3 | 100% states validated |
| Ehrenfest d[X]/dt = [P]/m | R4.4 | < 1% error |
| Born rule (empirical vs theoretical) | R2.2 | chi-squared p-value > 0.05 |
| Continuity equation (2D) | R5.2 | 100% accuracy |

---

## Visualization Gallery

6-panel synchronized dashboard from `example_gaussian_2d_evolution.py`:

```
[Density rho(x,y,t)]   [Marginals rho_x, rho_y]   [Observables <X>, <Y>]
[Current J(x,y,t)]     [Heisenberg dX*dY]          [Norm conservation]
```

Generated outputs are in [`quantum_simulation/results/`](quantum_simulation/results/):
- `gaussian_2d/evolution_dashboard.gif` — 50-frame 2D wavepacket dashboard
- `double_slit/` — double slit interference pattern and animation
- `measurement_distributions_infinite_well.png` — Born rule vs empirical (1000 measurements)

---

## Documentation

- [Document de reference](quantum_simulation/Document%20de%20référence.md) — full theoretical foundation, 100+ textbook references
- [Journal des changements](quantum_simulation/Journal%20des%20changements%20et%20améliorations.md) — implementation log, decisions D1-D5
- [Analyse des decisions techniques](quantum_simulation/Analyse%20détaillée%20des%20décisions%20techniques%20D1%20à%20D5.md) — numerical method choices (CN vs RK4, ADI, split-operator)

---

## Roadmap — AI Extensions

| Direction | Method | Status |
| --- | --- | --- |
| **Neural Quantum States** | NQS (NetKet / JAX) — variational ground states for N-body | Planned |
| **Physics-Informed Neural Networks** | PINNs for TDSE — NN solving Schrodinger equation | Planned |
| **Variational Quantum Eigensolver** | VQE in JAX — quantum-classical hybrid optimizer | Planned |
| **3D systems** | FFT-based 3D evolution, hydrogen atom, isosurfaces | Planned |

---

## Citation

```bibtex
@software{carlin2025quantum,
  author    = {Carlin, Esteban},
  title     = {Exploring Quantum Physics with AI},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/estebancarlin/Exploring_Quantum_Physics}
}
```

```bibtex
@book{cohen1977mecanique,
  title     = {Mecanique Quantique},
  author    = {Cohen-Tannoudji, Claude and Diu, Bernard and Laoloe, Franck},
  year      = {1977},
  publisher = {Hermann}
}
```

---

MIT License — see [LICENSE](LICENSE).
