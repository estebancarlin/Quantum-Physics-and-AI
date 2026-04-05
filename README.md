# Exploring Quantum Physics with AI

A rigorous Python framework for simulating quantum mechanical systems, grounded in Cohen-Tannoudji's *Mécanique Quantique* (Tomes I–III) and oriented toward AI-assisted quantum research. This project explores the intersection of numerical quantum mechanics and machine learning — from exact Crank-Nicolson integration to planned Neural Quantum States and PINNs.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-89%20passed%20%7C%200%20failed-brightgreen.svg)](quantum_simulation/tests/)
[![Notebooks](https://img.shields.io/badge/notebooks-8%2F8%20passing-brightgreen.svg)](quantum_simulation/examples/notebooks/)

---

## Overview

The framework implements quantum mechanics from the ground up with full traceability to textbook postulates:

- **Rigorous time evolution**: Crank-Nicolson 1D (unconditionally stable, exact norm conservation) + Split-Operator / ADI 2D
- **Quantum systems**: free particle (1D/2D), harmonic oscillator, infinite/finite wells, potential barriers, double slit
- **Measurement postulates**: Born rule with chi-squared validation, wavefunction collapse
- **Physical validators**: Heisenberg dX·dP ≥ ℏ/2, Ehrenfest theorem, probability current ∂ρ/∂t + ∇·J = 0
- **GPU acceleration**: CuPy-based sparse solvers, auto-detected
- **8 pedagogical Jupyter notebooks**: Tome 1 (free particle, measurement, harmonic oscillator, double slit 2D) + Tome 2 (spin/angular momentum, perturbation theory/Rabi, hydrogen fine structure, scattering/identical particles)

The theoretical foundation is the three-volume *Mecanique Quantique* (Cohen-Tannoudji, Diu & Laloë), referenced throughout as rules R2.2-R6.3. PDFs are in [`references/`](references/).

---

## Repository Structure

```text
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
│   ├── tests/                   # pytest suite (89 tests, 0 failures)
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

Eight notebooks grounded in Cohen-Tannoudji, all passing end-to-end execution with ✓ validators.

### Tome 1 — Quantum Mechanics Fundamentals

| Notebook | Content | Key rules |
| --- | --- | --- |
| [01 — Particule libre & paquet d'ondes](quantum_simulation/examples/notebooks/01_particule_libre_wavepacket.ipynb) | CN time evolution, Ehrenfest, Heisenberg products | R3.1, R4.3, R4.4, R5.1 |
| [02 — Postulats de la mesure](quantum_simulation/examples/notebooks/02_postulats_mesure.ipynb) | Born rule, 1000 simulated measurements, chi-squared, collapse | R2.2, R2.3 |
| [03 — Oscillateur harmonique](quantum_simulation/examples/notebooks/03_oscillateur_harmonique.ipynb) | Spectrum, ψₙ(x) via Hermite, algebra a/a†, coherent states | R6.1-R6.3 |
| [04 — Double fente 2D](quantum_simulation/examples/notebooks/04_double_fente_2d.ipynb) | ADI/split-operator 2D, Young fringes, norm conservation | R3.1, R5.1 |

### Tome 2 — Advanced Quantum Mechanics

| Notebook | Content | Key rules |
| --- | --- | --- |
| [05 — Spin et moment cinétique](quantum_simulation/examples/notebooks/05_spin_et_moment_cinetique.ipynb) | Spin-1/2 algebra, Bloch sphere, Clebsch-Gordan coefficients, coupled basis | R7.1, R8.1-R8.3 |
| [06 — Perturbations et Rabi](quantum_simulation/examples/notebooks/06_perturbations_et_rabi.ipynb) | Stationary perturbations (1st/2nd order), variational method, Rabi oscillations, Fermi's golden rule | R9.1-R9.5, R11.2, R11.3 |
| [07 — Hydrogène structure fine](quantum_simulation/examples/notebooks/07_hydrogene_structure_fine.ipynb) | Relativistic corrections, Darwin term, spin-orbit coupling, Zeeman effect, Stark effect, 21 cm hyperfine line | R10.1-R10.4 |
| [08 — Diffusion et particules identiques](quantum_simulation/examples/notebooks/08_diffusion_et_particules_identiques.ipynb) | Born approximation, phase shifts, partial wave expansion, optical theorem, symmetrization, Slater determinants, exchange scattering | R6.1-R6.4, R12.1-R12.3 |

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
# Full suite (89 passed, 0 failed)
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
| **Split-Operator (FFT)** | 2D | Unconditional | O(dt²) | Free 2D evolution, large grids |

The Crank-Nicolson scheme solves `(I + iHdt/2ℏ) ψ^{n+1} = (I - iHdt/2ℏ) ψ^n` via `scipy.sparse.linalg.spsolve` on a CSR tridiagonal Hamiltonian.

```python
from quantum_simulation.dynamics.evolution import TimeEvolution
from quantum_simulation.systems.free_particle import FreeParticle

fp = FreeParticle(mass=9.109e-31, hbar=1.055e-34)
psi0 = fp.create_gaussian_wavepacket(x_grid, x0=0, sigma_x=2e-9, k0=5e9)

evol = TimeEvolution(fp.hamiltonian)
psi_t = evol.evolve_wavefunction(psi0, t0=0, t=5e-15, dt=5e-18)
```

---

## GPU Acceleration

Automatic CuPy-based acceleration for grids above threshold (1D: nx > 1024, 2D: nx·ny > 256²).

```bash
pip install cupy-cuda12x           # requires CUDA 12.x
export QUANTUM_USE_GPU=true        # or false to disable
python quantum_simulation/benchmarks/benchmark_gpu.py
```

| Operation | Grid | CPU | GPU | Speedup |
| --- | --- | --- | --- | --- |
| Laplacian 2D (FFT) | 512×512 | 42.7 ms | 3.8 ms | 11× |
| Split-operator 2D | 512×512, 50 steps | 6 min | 32 s | 11× |
| Crank-Nicolson 1D | 4096 pts, 100 steps | 12.4 s | 3.8 s | 3× |

![GPU Benchmark](quantum_simulation/results/benchmarks/benchmark_gpu_summary.png)

---

## Implemented Systems

| System | Hamiltonian | Dim | Features |
| --- | --- | --- | --- |
| Free Particle | P²/2m | 1D/2D | Gaussian wavepackets, spreading, analytical overlay |
| Infinite Well | V=0 (0<x<L), V=∞ | 1D | Discrete levels Eₙ = n²π²ℏ²/2mL² |
| Finite Well | −V₀ inside | 1D | Bound + scattering states |
| Harmonic Oscillator | P²/2m + mω²X²/2 | 1D | Ladder operators a/a†, Fock states, coherent states |
| Potential Barrier | Step/rectangular | 1D | Tunneling, transmission coefficients |
| Double Slit | Barrier + slits | 2D | Young interference, fringe detection |

---

## Physical Validation

All experiments automatically verify:

| Principle | Rule | Tolerance |
| --- | --- | --- |
| Norm conservation | R5.1 | \|norm − 1\| < 1e-8 |
| Heisenberg dX·dP ≥ ℏ/2 | R4.3 | 100% states validated |
| Ehrenfest d⟨X⟩/dt = ⟨P⟩/m | R4.4 | < 1% error |
| Born rule (empirical vs theoretical) | R2.2 | chi-squared p-value > 0.05 |
| Continuity equation (2D) | R5.2 | 100% accuracy |

---

## Visualization Gallery

### Notebook 01 — Free Particle & Wavepacket

#### Wavepacket Evolution (Crank-Nicolson, 1D)

![Wavepacket Evolution](quantum_simulation/results/01_wavepacket_evolution.png)

*Gaussian wavepacket propagating under Crank-Nicolson integration. Norm conserved to machine precision throughout.*

#### Ehrenfest Theorem Validation

![Ehrenfest](quantum_simulation/results/01_ehrenfest.png)

*⟨X⟩(t) follows classical trajectory ⟨X⟩ = x₀ + (ℏk₀/m)t. Ehrenfest theorem (R4.4) satisfied to < 1% error.*

#### Heisenberg Uncertainty Products

![Heisenberg](quantum_simulation/results/01_heisenberg.png)

*ΔX·ΔP ≥ ℏ/2 verified at each timestep. Gaussian wavepacket saturates the bound at t=0.*

---

### Notebook 02 — Measurement Postulates (Born Rule)

#### Born Rule — Infinite Well

![Measurement Distributions (Infinite Well)](quantum_simulation/results/measurement_distributions_infinite_well.png)

*1000 energy measurements on a superposition state in the infinite well. Empirical histogram vs theoretical |cₙ|² predictions. Chi-squared test: p-value = 0.77.*

#### Born Rule — Free Particle (momentum)

![Measurement Distributions (Free Particle)](quantum_simulation/results/measurement_distributions_free_particle.png)

*Momentum measurement distribution for a Gaussian wavepacket. Empirical outcome vs |φ(p)|² (Fourier transform of ψ).*

#### Born Rule Validation

![Born Rule](quantum_simulation/results/02_born_rule.png)

*Systematic Born rule validation (R2.2): empirical vs theoretical probabilities across energy eigenstates.*

#### Wavefunction Collapse

![Wavefunction Collapse](quantum_simulation/results/02_wavefunction_collapse.png)

*Wavefunction collapse (R2.3): state before and after measurement. Successive measurements return the same eigenvalue with certainty.*

---

### Notebook 03 — Harmonic Oscillator

#### Energy Spectrum Eₙ = ℏω(n + ½)

![HO Spectrum](quantum_simulation/results/03_ho_spectrum.png)

*Discrete energy levels superimposed on the parabolic potential V(x) = ½mω²x². Zero-point energy E₀ = ℏω/2 ≠ 0 (R6.1).*

#### Wavefunctions ψₙ(x) — Hermite Polynomials

![HO Wavefunctions](quantum_simulation/results/03_ho_wavefunctions.png)

*First 6 eigenfunctions ψₙ(x) computed via `scipy.special.eval_hermite`. Orthonormality verified: max|⟨ψᵢ|ψⱼ⟩ − δᵢⱼ| < 1e-6 (R6.3).*

#### Orthonormality Matrix

![HO Orthonormality](quantum_simulation/results/03_ho_orthonorm.png)

*Overlap matrix |⟨ψᵢ|ψⱼ⟩| for n=0..5. Off-diagonal elements < 1e-6, confirming orthonormality of the Fock basis.*

#### Coherent States |α⟩ — Classical Limit

![Coherent States](quantum_simulation/results/03_coherent_states.png)

*Coherent states |α⟩ for α = 0.5, 1, 2, 3. Probability density is Gaussian, centered at the classical turning point ⟨X⟩ = √(2ℏ/mω) Re(α). Saturates Heisenberg bound: ΔX·ΔP = ℏ/2.*

---

### Notebook 04 — Double Slit 2D

#### Initial State ψ(x,y,0)

![Initial State](quantum_simulation/results/04_initial_state.png)

*2D Gaussian wavepacket before the double-slit barrier. σₓ = 3 nm, σᵧ = 1.5 nm.*

#### Double-Slit Barrier Potential

![Double Slit Potential](quantum_simulation/results/04_double_slit_potential.png)

*Potential landscape: two slits of width 2 nm separated by d = 10 nm in a hard barrier.*

#### Wavepacket Diffraction Through Double Slit

![Double Slit Evolution](quantum_simulation/results/04_double_slit_evolution.png)

*Density ρ(x,y,t) after passing through the barrier. Interference fringes emerge in the far field.*

#### Interference Pattern at the Screen

![Interference Pattern](quantum_simulation/results/04_interference_pattern.png)

*Intensity distribution I(y) at the detection screen (x = barrier + 100 nm). Fringe spacing Δy = λD/d consistent with Young's formula.*

#### Norm Conservation

![Norm Conservation](quantum_simulation/results/04_norm_conservation.png)

*||ψ(t)||² as a function of time during the 2D evolution. Deviation < 1e-8 throughout (R5.1).*

---

### 2D Wavepacket Evolution — Animated Dashboard

![2D Evolution Dashboard](quantum_simulation/results/gaussian_2d/evolution_dashboard.gif)

*6-panel synchronized dashboard: density ρ(x,y,t), marginals ρₓ/ρᵧ, observables ⟨X⟩⟨Y⟩, probability current J, Heisenberg product ΔX·ΔY, norm conservation over 50 frames.*

---

### 2D Wavepacket — Density & 3D Surface

| Initial density ρ(x,y,0) | Wavefunction \|ψ(x,y,0)\| |
| --- | --- |
| ![Density t=0](quantum_simulation/results/gaussian_2d/density_t0.png) | ![3D Surface](quantum_simulation/results/gaussian_2d/wavefunction_3d_t0.png) |

| Final density ρ(x,y,t=5fs) | Probability current J(x,y) |
| --- | --- |
| ![Density final](quantum_simulation/results/gaussian_2d/density_final.png) | ![Current field](quantum_simulation/results/gaussian_2d/current_final.png) |

---

### 2D Wavepacket — Marginal Distributions

![Marginals](quantum_simulation/results/gaussian_2d/marginals_final.png)

*Projected densities ρₓ(x) and ρᵧ(y) extracted from the 2D state at t = 5 fs.*

---

### Double Slit Experiment — Animated Diffraction

![Double Slit Evolution](quantum_simulation/results/double_slit/double_slit_evolution.gif)

*2D wavepacket diffracting through a double-slit barrier — density ρ(x,y,t) animated.*

![Screen Distribution](quantum_simulation/results/double_slit/screen_distribution.png)

*Intensity distribution at the detection screen. Fringe spacing Δy = λD/d consistent with Young's formula.*

---

### Notebook 05 — Spin-1/2 et Couplage de Moment Cinétique

#### Sphère de Bloch

![Bloch Sphere](quantum_simulation/results/05_bloch_sphere.png)

*Représentation de Bloch d'un état spin-1/2 arbitraire |χ⟩ = cos(θ/2)|+⟩ + e^(iφ)sin(θ/2)|−⟩. Vecteur de Bloch |n| = 1 (état pur). Règle R7.1.*

---

### Notebook 06 — Perturbations et Oscillations de Rabi

#### Niveaux perturbés de l'oscillateur harmonique

![Perturbation Levels](quantum_simulation/results/06_perturbation_levels.png)

*Corrections E⁽¹⁾ et E⁽²⁾ pour l'OHQ perturbé par W = λx². Comparaison perturbatif/exact et méthode variationnelle (borne supérieure). Règles R9.1-R9.4.*

#### Méthode variationnelle

![Variational](quantum_simulation/results/06_variational.png)

*Minimisation de ⟨H⟩ sur les états gaussiens d'écart-type σ. L'énergie variationnelle est strictement supérieure à E₀ (borne variationnelle, Règle R9.5).*

#### Oscillations de Rabi

![Rabi Oscillations](quantum_simulation/results/06_rabi_oscillations.png)

*P₂(t) pour un système à deux niveaux à résonance (δ=0) et hors résonance (δ=2Ω_R). Inversion complète P₂(T_π)=1 à résonance. Règle R11.3.*

#### Précession sur la sphère de Bloch

![Bloch Rabi](quantum_simulation/results/06_bloch_rabi.png)

*Trajectoire du vecteur de Bloch pendant l'oscillation de Rabi. Précession autour de l'axe x à résonance — inversion complète en T_π.*

---

### Notebook 07 — Structure Fine et Hyperfine de l'Hydrogène

#### Diagramme de structure fine (n=2)

![Fine Structure](quantum_simulation/results/07_fine_structure.png)

*Niveaux 2s₁/₂, 2p₁/₂, 2p₃/₂ avec corrections relativiste, Darwin et spin-orbite. Facteurs de Landé g_J = 2/3 (2p₁/₂) et 4/3 (2p₃/₂) exacts. Règle R10.1.*

#### Effet Zeeman

![Zeeman](quantum_simulation/results/07_zeeman.png)

*Levée de dégénérescence des niveaux 2p en champ magnétique externe. Splitting linéaire en B proportionnel à g_J m_J μ_B. Règle R10.3.*

#### Effet Stark linéaire (n=2)

![Stark](quantum_simulation/results/07_stark.png)

*Splitting Stark des quatre sous-niveaux n=2 en champ électrique. Splitting linéaire ΔE = 3eεa₀ vérifié à < 0.01 % (Règle R10.4). Raie 21 cm à ν = 1420.4 MHz ± 0.1 % (NIST).*

---

### Notebook 08 — Diffusion et Particules Identiques

#### Déphasages partiels et sections efficaces (Born)

![Phase Shifts](quantum_simulation/results/08_phase_shifts.png)

*Déphasages δ_l et sections efficaces partielles σ_l pour un potentiel de Yukawa. Optique theorem vérifié à < 1 %. Règle R6.4.*

#### Section efficace différentielle — approximation de Born

![Born Cross Section](quantum_simulation/results/08_born_cross_section.png)

*dσ/dΩ(θ) numérique vs analytique pour le potentiel de Yukawa. Accord exact. Régime Born validé : V₀/(ℏ²k²/2m) ≪ 1. Règles R6.1-R6.3.*

#### Densité à deux particules — bosons vs fermions

![Identical Particles](quantum_simulation/results/08_identical_particles.png)

*|Ψ(x₁,x₂)|² pour bosons (symétrique), fermions (antisymétrique) et classique. Effet d'échange : bosons antibunching→ bunching, fermions exclusion de Pauli visible. Règle R12.1.*

#### Diffusion de particules identiques

![Identical Scattering](quantum_simulation/results/08_identical_scattering.png)

*Section efficace de diffusion pour bosons et fermions comparée à des particules distinguables. Bosons : σ(π/2) = 2σ_classique, fermions : σ(π/2) = 0 (Règle R12.3).*

---

## Test Suite

```bash
pytest quantum_simulation/tests/ -v
```

```text
89 passed, 0 failed, 4 skipped  (415s)
```

| Test file | Tests | Coverage |
| --- | --- | --- |
| test_crank_nicolson.py | 6 | Norm, Ehrenfest, O(dt²) convergence |
| test_harmonic_oscillators.py | 9 | Spectrum, algebra [a,a†]=1, ladder actions |
| test_measurement_statistics.py | 6 | Born rule, collapse, chi-squared |
| test_operators.py | 8 | Hermiticity, commutators, expectation values |
| test_potential_systems.py | 9 | Well levels, tunneling, continuity |
| test_state.py | 7 | Normalization, inner product, probability |
| test_gpu/ | 16 | CPU/GPU equivalence, speedup, memory |
| test_validation/ | 9 | Heisenberg, Ehrenfest, conservation, orthonormality |
| test_orchestration/ | 16 | Pipeline, reports, comparisons, viz |

---

## Documentation

- [Document de référence](quantum_simulation/Document%20de%20référence.md) — full theoretical foundation, 100+ textbook references
- [Journal des changements](quantum_simulation/Journal%20des%20changements%20et%20améliorations.md) — implementation log, all decisions D1-D5, bug fixes 2026-03-30
- [Analyse des décisions techniques](quantum_simulation/Analyse%20détaillée%20des%20décisions%20techniques%20D1%20à%20D5.md) — numerical method choices (CN vs RK4, ADI, split-operator)

---

## Roadmap — AI Extensions

| Direction | Method | Status |
| --- | --- | --- |
| **Neural Quantum States** | NQS (NetKet / JAX) — variational ground states for N-body | Planned |
| **Physics-Informed Neural Networks** | PINNs for TDSE — NN solving Schrödinger equation | Planned |
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
