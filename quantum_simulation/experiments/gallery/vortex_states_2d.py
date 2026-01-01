# quantum_simulation/experiments/gallery/vortex_states_2d.py
"""
États vortex quantiques 2D.

États avec moment angulaire L ≠ 0 :
    ψ(r,θ) = R(r) exp(imθ)

Applications :
- Superfluides (quantified vortex)
- Condensats Bose-Einstein
- Optique quantique (modes Laguerre-Gauss)

Observables :
- Courant circulaire J(r,θ)
- Moment angulaire ⟨Lz⟩
- Phase topologique
"""

class VortexStates2D(Experiment):
    """États tourbillons dans piège harmonique."""
    
    def prepare_initial_state(self):
        """État avec phase exp(imθ)."""
        # ψ(x,y) = r^|m| exp(imθ) exp(-r²/2σ²)
        pass
    
    def analyze_results(self):
        """
        Visualisations :
        - Densité ρ(x,y) avec trou central
        - Phase arg[ψ(x,y)] (discontinuité 2πm)
        - Courant circulaire J_θ(r)
        """
        pass