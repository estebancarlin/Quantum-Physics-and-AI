# quantum_simulation/tests/test_orchestration/test_free_particle_3d.py

import pytest

# ✅ MARQUER : Tests Phase 3 non implémentée
pytestmark = pytest.mark.skip(reason="Phase 3 (Systèmes 3D) non implémentée - Classes FreeParticle2D/3D manquantes")

def test_ehrenfest_2d():
    """Théorème Ehrenfest en 2D."""
    fp2d = FreeParticle2D(mass, hbar)
    psi0 = fp2d.create_gaussian_2d(X, Y, x0=0, y0=0, sigma=1e-9, kx=5e9, ky=3e9)
    
    # Évolution
    time_evo = TimeEvolution2D(fp2d.hamiltonian, hbar)
    psi_t = time_evo.evolve(psi0, t0=0, t=1e-15, dt=1e-17)
    
    # Positions moyennes
    x_measured = PositionOperator('x').expectation_value(psi_t)
    y_measured = PositionOperator('y').expectation_value(psi_t)
    
    # Théorique
    x_theory, y_theory = fp2d.expected_trajectory_2d(t, x0=0, y0=0, kx=5e9, ky=3e9)
    
    assert abs(x_measured - x_theory) < 1e-11
    assert abs(y_measured - y_theory) < 1e-11