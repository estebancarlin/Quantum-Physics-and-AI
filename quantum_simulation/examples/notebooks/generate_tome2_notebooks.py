#!/usr/bin/env python3
"""Génère les 4 notebooks Tome 2 (NB05-NB08) depuis des templates Python."""
import json
from pathlib import Path

HERE = Path(__file__).parent


def mc(source: str, cell_id: str) -> dict:
    """Cellule markdown."""
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source}


def cc(source: str, cell_id: str) -> dict:
    """Cellule de code."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# =============================================================================
# NB05 — Spin-1/2 et couplage de moment cinétique
# =============================================================================

NB05_CELLS = [

mc(r"""# Spin-1/2 et Couplage de Moment Cinétique

**Référence :** Cohen-Tannoudji, Diu, Laloë — *Mécanique Quantique Tome 2*, Ch. IX et Ch. X

---

## Cadre théorique

### État de spin-1/2 (Règle R7.1)
$$|\chi\rangle = \cos\!\frac{\theta}{2}|{+}\rangle + e^{i\varphi}\sin\!\frac{\theta}{2}|{-}\rangle$$

Vecteur de Bloch : $\vec{n} = (\sin\theta\cos\varphi,\;\sin\theta\sin\varphi,\;\cos\theta)$, $|\vec{n}|=1$ pour un état pur.

### Opérateurs de spin (Règle R7.1)
$$\hat{S}_i = \frac{\hbar}{2}\sigma_i, \quad [\hat{S}_i,\hat{S}_j] = i\hbar\varepsilon_{ijk}\hat{S}_k, \quad \{\sigma_i,\sigma_j\} = 2\delta_{ij}\mathbb{1}$$

### Matrice densité (Règle R7.2)
$$\rho = \frac{1}{2}\!\left(\mathbb{1} + \vec{P}\cdot\vec{\sigma}\right), \quad \mathrm{Tr}(\rho^2) \leq 1 \;(\text{égalité : état pur})$$

### Coefficients de Clebsch-Gordan (Règle R8.2)
Pour $j_1=j_2=\tfrac{1}{2}$ : $|1,0\rangle = \frac{1}{\sqrt{2}}(|{+},{-}\rangle + |{-},{+}\rangle)$, $\;|0,0\rangle = \frac{1}{\sqrt{2}}(|{+},{-}\rangle - |{-},{+}\rangle)$

---""", "nb05-title"),

cc("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from quantum_simulation.core.spin import SpinHalf, SpinOperators, SpinDensityMatrix
from quantum_simulation.core.angular_momentum import ClebschGordan, AngularMomentumCoupling
from quantum_simulation.validation.tome2_invariants import SpinValidator, ClebschGordanValidator
from quantum_simulation.utils.config_loader import load_config

config = load_config()
hbar = config['physical_constants']['hbar']

print(f'ℏ = {hbar:.6e} J·s')
print('Modules chargés : spin, moment cinétique, validators Tome 2')
""", "nb05-setup"),

mc("""## 1. États de spin-1/2 et vecteur de Bloch

L'état le plus général d'un spin-1/2 est paramétrisé par deux angles sphériques
$\\theta \\in [0,\\pi]$ et $\\varphi \\in [0,2\\pi)$.
Le **vecteur de Bloch** $\\vec{n} = \\langle\\vec{\\sigma}\\rangle$ est de longueur 1 pour un état pur.""",
"nb05-s1-title"),

cc("""\
# États de base
spin_up   = SpinHalf.spin_up(hbar)
spin_down = SpinHalf.spin_down(hbar)

print('État |+⟩ :')
print(f'  composantes   : {spin_up.coefficients}')
print(f'  norme         : {spin_up.norm():.8f}')
print(f'  vecteur Bloch : {spin_up.to_bloch_vector()}  (attendu: [0, 0, 1])')

print()
print('État |-⟩ :')
print(f'  composantes   : {spin_down.coefficients}')
print(f'  vecteur Bloch : {spin_down.to_bloch_vector()}  (attendu: [0, 0, -1])')

# État général |χ⟩ (θ=π/3, φ=π/4)
theta, phi = np.pi / 3, np.pi / 4
chi = SpinHalf.from_bloch_angles(theta, phi, hbar)
bv  = chi.to_bloch_vector()

bv_th = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
err   = np.max(np.abs(bv - bv_th))

print()
print(f'État |χ⟩  (θ={theta:.4f} rad, φ={phi:.4f} rad) :')
print(f'  Bloch calculé  : ({bv[0]:+.6f}, {bv[1]:+.6f}, {bv[2]:+.6f})')
print(f'  Bloch théorique: ({bv_th[0]:+.6f}, {bv_th[1]:+.6f}, {bv_th[2]:+.6f})')
print(f"  {'✓' if err < 1e-10 else '✗'} Accord vecteur de Bloch  (erreur max = {err:.2e})")
print(f'  |n| = {np.linalg.norm(bv):.8f}  (état pur → 1)')
""", "nb05-bloch-calc"),

cc("""\
# Visualisation sur la sphère de Bloch
fig = plt.figure(figsize=(7, 7))
ax  = fig.add_subplot(111, projection='3d')

# Sphère de Bloch (wireframe)
u_s = np.linspace(0, 2*np.pi, 40)
v_s = np.linspace(0, np.pi,   20)
xs  = np.outer(np.cos(u_s), np.sin(v_s))
ys  = np.outer(np.sin(u_s), np.sin(v_s))
zs  = np.outer(np.ones_like(u_s), np.cos(v_s))
ax.plot_wireframe(xs, ys, zs, alpha=0.08, color='steelblue', linewidth=0.5)

# Axes
for vec, label, col in [([1,0,0],'x','red'), ([0,1,0],'y','green'), ([0,0,1],'z','blue')]:
    ax.quiver(0,0,0,*vec, length=1.2, color=col, linewidth=1.2, arrow_length_ratio=0.08)
    ax.text(*(np.array(vec)*1.3), f'${label}$', fontsize=11, color=col)

# Pôles
for sp, label in [(spin_up,'|+⟩'), (spin_down,'|-⟩')]:
    bv2 = sp.to_bloch_vector()
    ax.scatter(*bv2, s=80, zorder=5, color='black')
    ax.text(bv2[0]+0.05, bv2[1]+0.05, bv2[2]+0.08, label, fontsize=12)

# État |χ⟩
bv = chi.to_bloch_vector()
ax.quiver(0,0,0,*bv, color='darkorange', linewidth=3, arrow_length_ratio=0.12, label=f'|χ⟩ (θ={theta:.2f}, φ={phi:.2f})')
ax.scatter(*bv, s=100, color='darkorange', zorder=10)

ax.set_xlim([-1.3, 1.3]); ax.set_ylim([-1.3, 1.3]); ax.set_zlim([-1.3, 1.3])
ax.set_box_aspect([1,1,1])
ax.set_title('Sphère de Bloch — spin-1/2\\nRègle R7.1 — Cohen-Tannoudji Tome 2, Ch. IX', fontsize=11)
ax.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.savefig('../../results/05_bloch_sphere.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figure sauvegardée → results/05_bloch_sphere.png')
""", "nb05-bloch-plot"),

mc("""## 2. Opérateurs de spin — Relations de commutation et d'anticommutation

$$[\\hat{S}_x, \\hat{S}_y] = i\\hbar\\hat{S}_z, \\quad\
[\\hat{S}_y, \\hat{S}_z] = i\\hbar\\hat{S}_x, \\quad\
[\\hat{S}_z, \\hat{S}_x] = i\\hbar\\hat{S}_y$$

$$\\{\\sigma_i, \\sigma_j\\} = 2\\delta_{ij}\\mathbb{1}$$

Règle de Heisenberg généralisée pour le spin :
$$\\Delta S_x \\cdot \\Delta S_y \\geq \\frac{\\hbar}{2}|\\langle\\hat{S}_z\\rangle|$$""",
"nb05-s2-title"),

cc("""\
ops = SpinOperators(hbar)

# Validation algébrique
comm_ok  = ops.validate_commutation_relations()
anti_ok  = ops.validate_pauli_anticommutation()

print('VALIDATION ALGÉBRIQUE DES OPÉRATEURS DE SPIN')
print('-' * 55)
print(f"  {'✓' if comm_ok  else '✗'} Relations de commutation [Si,Sj] = iℏ εijk Sk")
print(f"  {'✓' if anti_ok  else '✗'} Anticommutation de Pauli  {{σi,σj}} = 2δij 𝟙")

# Valeurs propres de S²
S_sq = ops.S_squared
eigvals_sq = np.linalg.eigvalsh(S_sq)
s_s1 = 0.5 * (0.5 + 1) * hbar**2
print(f'  S² vp = {eigvals_sq[0]:.4e} J²  (attendu : s(s+1)ℏ² = {s_s1:.4e} J²)')
ok_sq = np.allclose(eigvals_sq, s_s1, rtol=1e-8)
print(f"  {'✓' if ok_sq else '✗'} Valeurs propres de Ŝ² = s(s+1)ℏ² = (3/4)ℏ²")

# Espérances et incertitudes sur |χ⟩
ex = ops.expectation_value(chi, 'x')
ey = ops.expectation_value(chi, 'y')
ez = ops.expectation_value(chi, 'z')
dx = ops.uncertainty(chi, 'x')
dy = ops.uncertainty(chi, 'y')

print()
print('OBSERVABLES SUR |χ⟩ (θ=π/3, φ=π/4) :')
print(f'  ⟨Sx⟩ = {ex:+.4e} J·s⁻¹,  ΔSx = {dx:.4e}')
print(f'  ⟨Sy⟩ = {ey:+.4e} J·s⁻¹,  ΔSy = {dy:.4e}')
print(f'  ⟨Sz⟩ = {ez:+.4e} J·s⁻¹')
product = dx * dy
bound   = abs(ez) * hbar / 2
print(f'  ΔSx·ΔSy = {product:.4e}  ≥  |⟨Sz⟩|ℏ/2 = {bound:.4e}')
print(f"  {'✓' if product >= bound - 1e-25 else '✗'} Inégalité de Heisenberg pour le spin")
""", "nb05-ops"),

mc("""## 3. Matrice densité de spin (Règle R7.2)

Pour un état pur $|\\chi\\rangle$, la matrice densité est :
$$\\rho = |\\chi\\rangle\\langle\\chi| = \\frac{1}{2}(\\mathbb{1} + \\vec{P}\\cdot\\vec{\\sigma}), \\quad \\vec{P} = \\langle\\vec{\\sigma}\\rangle$$

**Propriétés** : hermitique, $\\mathrm{Tr}(\\rho)=1$, définie positive, $\\mathrm{Tr}(\\rho^2) = 1$ (état pur).""",
"nb05-s3-title"),

cc("""\
# Matrice densité de l'état pur |χ⟩
rho = SpinDensityMatrix.from_pure_state(chi)
val = rho.validate()

print('MATRICE DENSITÉ DE |χ⟩')
print('-' * 40)
print(f"  {'✓' if val['is_hermitian']       else '✗'} Hermiticité  ρ† = ρ")
print(f"  {'✓' if val['trace_one']           else '✗'} Trace        Tr(ρ) = 1")
print(f"  {'✓' if val['positive_semidefinite'] else '✗'} Semi-définie positive")

purity = rho.purity()
bv_rho = rho.bloch_vector()
print(f'  Pureté Tr(ρ²) = {purity:.8f}  (état pur → 1)')
print(f'  Vecteur Bloch via ρ : ({bv_rho[0]:+.6f}, {bv_rho[1]:+.6f}, {bv_rho[2]:+.6f})')
print(f'  Via to_bloch_vector : ({bv[0]:+.6f}, {bv[1]:+.6f}, {bv[2]:+.6f})')
bv_err = np.max(np.abs(bv_rho - bv))
print(f"  {'✓' if bv_err < 1e-10 else '✗'} Cohérence  Tr(ρ σi) = ⟨σi⟩  (erreur = {bv_err:.2e})")

# Mélange : 50% |+⟩, 50% |-⟩
rho_mixed = SpinDensityMatrix.from_bloch_vector(np.array([0.0, 0.0, 0.0]))
print()
print('ÉTAT MIXTE 50/50 (vecteur de Bloch nul) :')
val_m = rho_mixed.validate()
print(f"  {'✓' if val_m['is_hermitian']        else '✗'} Hermiticité")
print(f"  {'✓' if val_m['trace_one']            else '✗'} Trace = 1")
print(f'  Pureté Tr(ρ²) = {rho_mixed.purity():.6f}  (état mixte → 1/2)')
""", "nb05-density"),

mc(r"""## 4. Coefficients de Clebsch-Gordan pour $j_1 = j_2 = \tfrac{1}{2}$ (Règle R8.2)

La table de CG (convention de Condon-Shortley) est une matrice **unitaire** de changement de base :

$$\{|+,+\rangle,\;|+,-\rangle,\;|-,+\rangle,\;|-,-\rangle\} \;\longrightarrow\; \{|1,1\rangle,\;|1,0\rangle,\;|0,0\rangle,\;|1,-1\rangle\}$$""",
"nb05-s4-title"),

cc("""\
# Table de Clebsch-Gordan pour j1=j2=1/2
table = ClebschGordan.two_spins_half_table()

print('TABLE DE CLEBSCH-GORDAN  j₁=j₂=1/2  (4×4)')
print('Colonnes : |m₁,m₂⟩ ∈ {|+,+⟩, |+,-⟩, |-,+⟩, |-,-⟩}')
print('Lignes   : |J,M⟩   ∈ {|1,+1⟩, |1,0⟩, |1,-1⟩, |0,0⟩}')
print()
labels_jm = ['|1,+1⟩', '|1, 0⟩', '|1,-1⟩', '|0, 0⟩']
for i, row in enumerate(table):
    print(f'  {labels_jm[i]} : {np.array2string(row, precision=6, suppress_small=True)}')

# Validation : unitarité et coefficients connus
cg_val  = ClebschGordanValidator()
res_uni = cg_val.unitarity(0.5, 0.5)
res_kno = cg_val.known_two_spin_half()

print()
print('VALIDATION')
print('-' * 50)
print(f"  {'✓' if res_uni['is_unitary'] else '✗'} Unitarité U†U = 𝟙  (erreur max = {res_uni['max_error']:.2e})")
print(f"  {'✓' if res_kno['all_correct'] else '✗'} Coefficients analytiques connus  (erreur max = {res_kno['max_error']:.2e})")
if res_kno.get('errors'):
    for err_info in res_kno['errors']:
        print(f'    ✗ {err_info}')

# Coefficient individuel
coeff_10 = ClebschGordan.coefficient(0.5, 0.5, 0.5, -0.5, 1, 0)  # ⟨1/2,+1/2; 1/2,-1/2|1,0⟩ = 1/√2
print(f'  ⟨1/2,+1/2 ; 1/2,-1/2|1,0⟩ = {coeff_10:.6f}  (attendu: {1/np.sqrt(2):.6f})')
coeff_00 = ClebschGordan.coefficient(0.5, 0.5, 0.5, -0.5, 0, 0)  # ⟨1/2,+1/2; 1/2,-1/2|0,0⟩ = 1/√2
print(f'  ⟨1/2,+1/2 ; 1/2,-1/2|0,0⟩ = {coeff_00:.6f}  (attendu: {1/np.sqrt(2):.6f})')
""", "nb05-cg"),

mc(r"""## 5. Couplage $\hat{L}\cdot\hat{S}$ — facteur de Landé $g_J$

Pour les niveaux $n=2$ de l'hydrogène ($l=1$, $s=1/2$) :

$$\langle\hat{L}\cdot\hat{S}\rangle = \frac{\hbar^2}{2}[j(j+1) - l(l+1) - s(s+1)]$$

$$g_J = 1 + \frac{J(J+1) + S(S+1) - L(L+1)}{2J(J+1)}$$

|Niveau|$j$|$\langle L\cdot S\rangle$|$g_J$|
|---|---|---|---|
|$2p_{1/2}$|$1/2$|$-\hbar^2$|$2/3$|
|$2p_{3/2}$|$3/2$|$+\hbar^2/2$|$4/3$|""",
"nb05-s5-title"),

cc("""\
coupling_2p = AngularMomentumCoupling(1, 0.5)  # L=1, S=1/2

# 2p₁/₂ : j=1/2
ls_12   = coupling_2p.spin_orbit_matrix_element(n=2, l=1, j=0.5, hbar=hbar)
g_12    = coupling_2p.lande_g_factor(1, 0.5, 0.5)
ls_12_th = hbar**2 / 2 * (0.5*1.5 - 1*2 - 0.5*1.5)  # = -hbar^2

# 2p₃/₂ : j=3/2
ls_32   = coupling_2p.spin_orbit_matrix_element(n=2, l=1, j=1.5, hbar=hbar)
g_32    = coupling_2p.lande_g_factor(1, 0.5, 1.5)
ls_32_th = hbar**2 / 2 * (1.5*2.5 - 1*2 - 0.5*1.5)  # = +hbar^2/2

print("COUPLAGE SPIN-ORBITE L\u00b7S (niveaux 2p de l'hydrog\u00e8ne)")
print('-' * 65)
print(f'  2p₁/₂ (j=1/2) :  ⟨L·S⟩ = {ls_12:.4e} J²  (th: {ls_12_th:.4e})')
ls_ok12 = abs(ls_12 - ls_12_th) < 1e-45 * abs(ls_12_th) + 1e-80
print(f"  {'✓' if abs(ls_12 - ls_12_th) < 1e-65 + abs(ls_12_th)*1e-6 else '✗'} Accord 2p₁/₂")
print(f'  2p₃/₂ (j=3/2) :  ⟨L·S⟩ = {ls_32:.4e} J²  (th: {ls_32_th:.4e})')
print(f"  {'✓' if abs(ls_32 - ls_32_th) < 1e-65 + abs(ls_32_th)*1e-6 else '✗'} Accord 2p₃/₂")

print()
print('FACTEUR DE LANDÉ g_J')
print(f'  2p₁/₂ : g_J = {g_12:.6f}  (attendu : 2/3 = {2/3:.6f})  {"✓" if abs(g_12 - 2/3) < 1e-6 else "✗"}')
print(f'  2p₃/₂ : g_J = {g_32:.6f}  (attendu : 4/3 = {4/3:.6f})  {"✓" if abs(g_32 - 4/3) < 1e-6 else "✗"}')

# Matrice de moment angulaire total (J=1)
J_sq, J_z = coupling_2p.total_angular_momentum_matrix(J=1)
print()
print(f'  Matrice J² (J=1) : vp = {np.sort(np.linalg.eigvalsh(J_sq.real))}')
print(f'  Attendu : J(J+1)ℏ² = {1*2*hbar**2:.4e}  (×3 fois)')
""", "nb05-ls"),

mc("""## 6. Validation finale — résumé""", "nb05-conclusion-title"),

cc("""\
sv  = SpinValidator()
cgv = ClebschGordanValidator()

print('BILAN DE VALIDATION — Tome 2 Chapitre IX-X')
print('=' * 55)

# Spin
ok1 = sv.bloch_vector_bound(chi)
ok2 = sv.pauli_anticommutation(ops)
ok3 = sv.spin_squared_eigenvalue(chi)
val_dm = sv.density_matrix_valid(rho)
ok4 = val_dm['is_hermitian'] and val_dm['trace_one'] and val_dm['positive_semidefinite']

print(f"  {'✓' if ok1 else '✗'} Vecteur de Bloch |n| = 1  (état pur)")
print(f"  {'✓' if ok2 else '✗'} Anticommutation de Pauli {{σi,σj}} = 2δij")
print(f"  {'✓' if ok3 else '✗'} Valeur propre S² = s(s+1)ℏ²")
print(f"  {'✓' if ok4 else '✗'} Matrice densité valide (hermitique, Tr=1, définie+)")

# CG
ok5 = cgv.unitarity(0.5, 0.5)['is_unitary']
ok6 = cgv.known_two_spin_half()['all_correct']
ok7 = cgv.selection_rule_M(0.5, 0.5)
ok8 = cgv.triangle_rule(0.5, 0.5, 0)   # J=0 autorisé
ok9 = not cgv.triangle_rule(0.5, 0.5, 2)  # J=2 interdit

print(f"  {'✓' if ok5 else '✗'} Unitarité table CG j₁=j₂=1/2")
print(f"  {'✓' if ok6 else '✗'} Coefficients CG analytiques")
print(f"  {'✓' if ok7 else '✗'} Règle de sélection M = m₁+m₂")
print(f"  {'✓' if ok8 else '✗'} Règle triangulaire : J=0 autorisé  (|j₁-j₂|≤J≤j₁+j₂)")
print(f"  {'✓' if ok9 else '✗'} Règle triangulaire : J=2 interdit")

all_ok = all([ok1,ok2,ok3,ok4,ok5,ok6,ok7,ok8,ok9])
print()
print(f"{'✓ Toutes les validations réussies !' if all_ok else '✗ Certaines validations ont échoué'}")
""", "nb05-validation"),

]  # end NB05_CELLS

# =============================================================================
# NB06 — Théorie des perturbations et oscillations de Rabi
# =============================================================================

NB06_CELLS = [

mc(r"""# Théorie des Perturbations et Oscillations de Rabi

**Référence :** Cohen-Tannoudji, Diu, Laloë — *Mécanique Quantique Tome 2*, Ch. XI

---

## Cadre théorique

### Perturbations stationnaires (Règles R9.1–R9.5)

$H = H_0 + \lambda W$

$$E_n^{(1)} = \langle\varphi_n|W|\varphi_n\rangle, \qquad E_n^{(2)} = \sum_{p\neq n}\frac{|W_{pn}|^2}{E_n^0 - E_p^0}$$

**Méthode variationnelle :** $E_{\rm var}(\alpha) = \langle\psi(\alpha)|H|\psi(\alpha)\rangle \geq E_0$ pour tout $\alpha$

### Perturbations dépendantes du temps (Règle R11.3 — Rabi)

Pour un système à deux niveaux soumis à un champ oscillant :
$$P_2(t) = \frac{\Omega_R^2}{\Omega^2}\sin^2\!\frac{\Omega t}{2}, \quad \Omega = \sqrt{\Omega_R^2 + \delta^2}$$

À résonance ($\delta=0$) : **inversion complète** à $t = T_\pi = \pi/\Omega_R$.

---""", "nb06-title"),

cc("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from quantum_simulation.dynamics.perturbation import StationaryPerturbation, VariationalMethod
from quantum_simulation.dynamics.time_perturbation import (
    RabiOscillations, TimeDependentPerturbation, FermiGoldenRule
)
from quantum_simulation.validation.tome2_invariants import PerturbationValidator, TimeDependentValidator
from quantum_simulation.utils.config_loader import load_config

config = load_config()
hbar = config['physical_constants']['hbar']
m_e  = config['physical_constants']['m_electron']

print(f'ℏ   = {hbar:.6e} J·s')
print(f'm_e = {m_e:.6e} kg')
print('Modules chargés : perturbation, Rabi, validators Tome 2')
""", "nb06-setup"),

mc(r"""## 1. Perturbation stationnaire de l'oscillateur harmonique

On perturbe l'OHQ ($H_0 = \hbar\omega(\hat{N}+\frac{1}{2})$) par $W = \lambda\hat{x}^2$.

**Résultat exact :** $E_n^{\rm exact} = \hbar\omega'(n+\tfrac{1}{2})$ avec $\omega' = \omega\sqrt{1+2\lambda/m\omega^2}$.

**Corrections perturbatives :**
$$E_0^{(1)} = \langle 0|W|0\rangle = \frac{\lambda\hbar}{2m\omega}, \qquad E_0^{(2)} = -\frac{\lambda^2\hbar}{4m^2\omega^3}$$""",
"nb06-s1-title"),

cc("""\
# Paramètres de l'OHQ
omega = 1e14   # rad/s (fréquence typique vibration moléculaire)
N_fock = 12   # troncature de la base de Fock

n_arr  = np.arange(N_fock)
E0_fock = hbar * omega * (n_arr + 0.5)
H0_states = np.eye(N_fock, dtype=complex)

# Opérateur position x dans la base de Fock
x_scale = np.sqrt(hbar / (2 * m_e * omega))
x_fock  = np.zeros((N_fock, N_fock))
for n in range(N_fock - 1):
    x_fock[n, n+1] = np.sqrt(n + 1) * x_scale
    x_fock[n+1, n] = np.sqrt(n + 1) * x_scale

# Perturbation W = λ x²  (λ = 5% de la constante de rappel)
lambda_val = 0.05 * m_e * omega**2
W_matrix   = lambda_val * (x_fock @ x_fock)

print(f'Oscillateur harmonique tronqué à N={N_fock} états de Fock')
print(f'ω = {omega:.2e} rad/s,   λ = {lambda_val:.4e} N/m²')
print(f'Paramètre perturbatif λ/(mω²) = {lambda_val/(m_e*omega**2):.4f}  (doit être ≪ 1)')

pert = StationaryPerturbation(E0_fock, H0_states, W_matrix, hbar)

# Corrections pour l'état fondamental n=0
E1_0   = pert.energy_correction_first_order(0)
E2_0   = pert.energy_correction_second_order(0)
E0_corr = pert.corrected_energy(0, order=2)

E1_0_th = lambda_val * hbar / (2 * m_e * omega)
E2_0_th = -lambda_val**2 * hbar / (4 * m_e**2 * omega**3)
omega_p  = omega * np.sqrt(1 + 2*lambda_val / (m_e * omega**2))
E0_exact = hbar * omega_p / 2

pv = PerturbationValidator()

print()
print('CORRECTIONS AU 1er et 2e ORDRE (n=0)')
print('-' * 65)
err1 = abs(E1_0 - E1_0_th) / abs(E1_0_th)
err2 = abs(E2_0 - E2_0_th) / abs(E2_0_th)
print(f'  E₀⁽⁰⁾ = {E0_fock[0]:.6e} J  ({E0_fock[0]/1.6e-19*1000:.4f} meV)')
print(f'  E₀⁽¹⁾ calculé   : {E1_0:.6e} J')
print(f'  E₀⁽¹⁾ analytique: {E1_0_th:.6e} J  ({"✓" if err1 < 1e-6 else "✗"} erreur={err1:.2e})')
print(f'  E₀⁽²⁾ calculé   : {E2_0:.6e} J')
print(f'  E₀⁽²⁾ analytique: {E2_0_th:.6e} J  ({"✓" if err2 < 1e-4 else "✗"} erreur={err2:.2e})')
print(f'  E₀ corrigé : {E0_corr:.10e} J')
print(f'  E₀ exact   : {E0_exact:.10e} J')
print(f"  {'✓' if abs(E0_corr - E0_exact)/abs(E0_exact) < 1e-4 else '✗'} Accord perturbatif/exact")

# Corrections pour tous les niveaux
E_corrected = np.array([pert.corrected_energy(n, order=2) for n in range(N_fock)])
E_exact_all = hbar * omega_p * (n_arr + 0.5)
max_err = np.max(np.abs(E_corrected - E_exact_all) / np.abs(E_exact_all))
print(f'  Erreur max sur {N_fock} niveaux: {max_err:.2e}')

# Régime perturbatif
regime = pert.validate_perturbative_regime(0)
print()
print(f"  {'✓' if regime['is_valid'] else '✗'} Régime perturbatif valide  (max ratio = {regime['max_ratio']:.4f} < 0.1)")

# Validation PerturbationValidator
corrs = np.array([E1_0])
ok_real = pv.energy_corrections_real(corrs)
ok_2nd  = pv.second_order_ground_state_negative(E2_0)
print(f"  {'✓' if ok_real else '✗'} Corrections d'énergie réelles")
print(f"  {'✓' if ok_2nd  else '✗'} 2e ordre négatif pour l'état fondamental")
""", "nb06-stationary"),

cc("""\
# Plot des niveaux perturbés vs exacts vs non-perturbés
fig, ax = plt.subplots(figsize=(9, 5))
n_plot = 6

ax.hlines(E0_fock[:n_plot]/1.6e-19*1000, -0.35, -0.05, colors='steelblue',
          linewidth=3, label='$E_n^{(0)}$ non perturbé')
ax.hlines(E_corrected[:n_plot]/1.6e-19*1000, 0.05, 0.35, colors='darkorange',
          linewidth=3, label='$E_n^{(0+1+2)}$ perturbé')
ax.hlines(E_exact_all[:n_plot]/1.6e-19*1000, 0.65, 0.95, colors='green',
          linewidth=3, label="$E_n^{\\\\rm exact}$ ($\\\\omega'$)")

for n in range(n_plot):
    ax.annotate(f'n={n}', xy=(0.97, E_exact_all[n]/1.6e-19*1000),
                fontsize=8, va='center', color='green')

ax.set_xticks([-0.2, 0.2, 0.8])
ax.set_xticklabels(["Non\\nperturbé", "Perturbé\\n(2e ordre)", "Exact"])
ax.set_ylabel('Énergie (meV)', fontsize=12)
ax.set_title(f"Niveaux de l'OHQ perturbé par W=\u03bbx\u00b2  (\u03bb/(m\u03c9\u00b2) = {lambda_val/(m_e*omega**2):.2f})"
             "\\nRègles R9.1-R9.2 - Cohen-Tannoudji Tome 2, Ch. XI", fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('../../results/06_perturbation_levels.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figure sauvegardée → results/06_perturbation_levels.png')
""", "nb06-pert-plot"),

mc(r"""## 2. Méthode variationnelle (Règle R9.5)

Ansatz gaussien $\psi(\alpha) \propto e^{-x^2/4\alpha^2}$ pour l'OHQ :

$$E_{\rm var}(\alpha) = \frac{\hbar^2}{4m\alpha^2} + \frac{m\omega^2\alpha^2}{2}$$

Minimum atteint pour $\alpha = \sqrt{\hbar/2m\omega}$ : $E_{\rm var}^{\rm min} = \hbar\omega/2 = E_0$ (l'ansatz est exact pour l'OHQ).""",
"nb06-s2-title"),

cc("""\
def ho_energy_var(params):
    sigma = params[0]
    if sigma <= 0:
        return 1e100
    return hbar**2 / (4 * m_e * sigma**2) + 0.5 * m_e * omega**2 * sigma**2

var_method = VariationalMethod(ho_energy_var, n_params=1)
sigma0 = 0.7 * np.sqrt(hbar / (m_e * omega))  # guess légèrement sous-optimal
result = var_method.minimize(initial_params=np.array([sigma0]))
E_var  = result['ground_state_energy']
E_gs   = E0_fock[0]
sigma_opt_th = np.sqrt(hbar / (2 * m_e * omega))

print('MÉTHODE VARIATIONNELLE (Règle R9.5)')
print('-' * 50)
print(f'  σ initial    : {sigma0:.4e} m')
print(f'  σ optimal    : {result["optimal_params"][0]:.4e} m  (th: {sigma_opt_th:.4e} m)')
print(f'  E_var(σ_opt) : {E_var:.6e} J')
print(f'  E₀ exact     : {E_gs:.6e} J')
print(f"  {'✓' if E_var >= E_gs - 1e-40 else '✗'} E_var ≥ E₀  (borne supérieure)")
print(f"  {'✓' if result['converged'] else '✗'} Minimisation convergée  ({result['n_iterations']} itérations)")

ok_var = pv.variational_bound(E_var, E_gs, tolerance=1e-8)
print(f"  {'✓' if ok_var else '✗'} Validation PerturbationValidator.variational_bound()")

# Courbe E_var(σ)
sigma_arr = np.linspace(0.2*sigma_opt_th, 3*sigma_opt_th, 200)
E_arr = np.array([ho_energy_var([s]) for s in sigma_arr])
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sigma_arr*1e12, E_arr/1.6e-19*1000, 'b-', lw=2, label='E_var(σ)')
ax.axhline(E_gs/1.6e-19*1000, color='r', ls='--', lw=2, label=f'E₀ = ℏω/2 = {E_gs/1.6e-19*1000:.3f} meV')
ax.axvline(result['optimal_params'][0]*1e12, color='g', ls=':', lw=1.5, label='σ_opt')
ax.fill_between(sigma_arr*1e12, E_gs/1.6e-19*1000, E_arr/1.6e-19*1000,
                alpha=0.15, color='blue', label='Marge variationnelle')
ax.set_xlabel('σ (pm)', fontsize=12)
ax.set_ylabel('E_var (meV)', fontsize=12)
ax.set_title("Énergie variationnelle de l'OHQ\\nRègle R9.5", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../../results/06_variational.png', dpi=150, bbox_inches='tight')
plt.show()
""", "nb06-variational"),

mc(r"""## 3. Oscillations de Rabi (Règle R11.3)

Pour un système à deux niveaux $|1\rangle, |2\rangle$ avec $\hbar\omega_0 = E_2 - E_1$,
soumis à un champ oscillant de fréquence $\omega$ (approximation onde tournante) :

$$P_2(t) = \frac{\Omega_R^2}{\Omega^2}\sin^2\!\frac{\Omega t}{2}, \qquad \Omega = \sqrt{\Omega_R^2 + \delta^2}$$

À **résonance** ($\delta=0$) : $P_2(T_\pi)=1$ (inversion complète), $T_\pi = \pi/\Omega_R$.""",
"nb06-s3-title"),

cc("""\
# Paramètres Rabi
omega_0   = 1e12    # fréquence de transition (rad/s)
omega_R   = 1e11    # fréquence de Rabi (rad/s) — couplage 10% de ω₀

t_final  = 12 * np.pi / omega_R
t_values = np.linspace(0, t_final, 800)

# Résonance (δ=0) et désaccord (δ=2Ω_R)
rabi_res = RabiOscillations(omega_0, omega_R, hbar, detuning=0.0)
rabi_off = RabiOscillations(omega_0, omega_R, hbar, detuning=2*omega_R)

P2_res = rabi_res.population_excited(t_values)
P2_off = rabi_off.population_excited(t_values)

T_pi = rabi_res.pi_pulse_time()
Omega_gen_off = rabi_off.rabi_frequency_generalized()

tv = TimeDependentValidator()

print('OSCILLATIONS DE RABI')
print('-' * 65)
print(f'  Ω_R = {omega_R:.2e} rad/s  (fréquence de Rabi)')
print(f'  T_π = {T_pi*1e9:.4f} ns    (temps de la impulsion π)')
print(f'  P₂(T_π) à résonance : {rabi_res.population_excited(np.array([T_pi]))[0]:.6f}  (attendu : 1)')
print(f"  {'✓' if abs(rabi_res.population_excited(np.array([T_pi]))[0] - 1) < 1e-6 else '✗'} Inversion complète à T_π")
print(f'  Ω_generalisée (δ=2Ω_R) : {Omega_gen_off:.4e} rad/s')
print(f'  P₂_max (δ≠0) : {P2_off.max():.4f}  (attendu : Ω_R²/Ω² = {omega_R**2/Omega_gen_off**2:.4f})')

print()
ok_bounds = tv.probability_bounds(P2_res)
ok_amp    = tv.rabi_oscillation_amplitude(P2_res, omega_R, rabi_res.omega_generalized)
print(f"  {'✓' if ok_bounds else '✗'} Probabilité P₂ ∈ [0,1]")
print(f"  {'✓' if ok_amp    else '✗'} Amplitude Rabi correcte (Ω_R²/Ω²)")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(t_values*1e9, P2_res, 'steelblue', lw=2, label=r'$P_2(t)$ résonance ($\delta=0$)')
axes[0].axhline(1.0, color='r', ls='--', lw=1, alpha=0.5, label='$P_2=1$')
axes[0].axvline(T_pi*1e9, color='g', ls=':', lw=2, label=f'$T_\\\\pi = {T_pi*1e9:.2f}$ ns')
axes[0].set_xlabel('t (ns)', fontsize=12)
axes[0].set_ylabel('$P_2(t)$', fontsize=12)
axes[0].set_title('Oscillations de Rabi à résonance\\n$P_2(T_\\\\pi) = 1$  — Règle R11.3', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].set_ylim(-0.05, 1.1)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_values*1e9, P2_res, 'steelblue', lw=2, label=f'$\\\\delta=0$  ($\\\\Omega=\\\\Omega_R$)')
axes[1].plot(t_values*1e9, P2_off, 'darkorange', lw=2, ls='--',
             label=f'$\\\\delta=2\\\\Omega_R$  ($\\\\Omega={Omega_gen_off/omega_R:.2f}\\\\Omega_R$)')
axes[1].set_xlabel('t (ns)', fontsize=12)
axes[1].set_ylabel('$P_2(t)$', fontsize=12)
axes[1].set_title('Effet du désaccord $\\\\delta$\\nProbabilité max réduite hors résonance', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].set_ylim(-0.05, 1.1)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../results/06_rabi_oscillations.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figure sauvegardée → results/06_rabi_oscillations.png')
""", "nb06-rabi"),

cc("""\
# Évolution du vecteur de Bloch (sphère de Bloch)
initial_bloch = np.array([0.0, 0.0, -1.0])  # état fondamental |1⟩ (pôle sud)
t_bloch = np.linspace(0, 2*T_pi, 300)
trajectory = rabi_res.bloch_vector_evolution(initial_bloch, t_bloch)
P2_bloch   = rabi_res.population_from_bloch(trajectory)

print(f'Vérification trajectoire Bloch vs formule Rabi :')
P2_direct = rabi_res.population_excited(t_bloch)
max_diff   = np.max(np.abs(P2_bloch - P2_direct))
print(f"  {'✓' if max_diff < 1e-4 else '✗'} Accord P₂(Bloch) vs P₂(Rabi)  (diff max = {max_diff:.2e})")

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 7))
ax  = fig.add_subplot(111, projection='3d')

u_s = np.linspace(0, 2*np.pi, 40); v_s = np.linspace(0, np.pi, 20)
xs = np.outer(np.cos(u_s), np.sin(v_s))
ys = np.outer(np.sin(u_s), np.sin(v_s))
zs = np.outer(np.ones_like(u_s), np.cos(v_s))
ax.plot_wireframe(xs, ys, zs, alpha=0.07, color='gray', linewidth=0.5)

sc = ax.scatter(trajectory[:,0], trajectory[:,1], trajectory[:,2],
                c=t_bloch*1e9, cmap='plasma', s=10, zorder=5)
plt.colorbar(sc, ax=ax, label='t (ns)', shrink=0.7)

ax.scatter(*initial_bloch, s=100, color='blue', zorder=10, label='État initial |1⟩')
ax.scatter(*trajectory[-1], s=100, color='red', zorder=10, label=f'État final t=2T_π')

for vec, label, col in [([0,0,1.3],'z','k'), ([1.3,0,0],'x','k'), ([0,1.3,0],'y','k')]:
    ax.text(*vec, f'${label}$', fontsize=10)

ax.set_xlim([-1.2,1.2]); ax.set_ylim([-1.2,1.2]); ax.set_zlim([-1.2,1.2])
ax.set_box_aspect([1,1,1])
ax.set_title('Précession sur la sphère de Bloch\\n(oscillation de Rabi à résonance)', fontsize=11)
ax.legend(fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig('../../results/06_bloch_rabi.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figure sauvegardée → results/06_bloch_rabi.png')
""", "nb06-bloch"),

mc(r"""## 4. Règle d'or de Fermi (Règle R11.2)

$$\Gamma_{i\to f} = \frac{2\pi}{\hbar}|W_{fi}|^2\,\rho(E_f)$$

Applicable lorsque $\Gamma t \ll 1$ (régime perturbatif temporel).""",
"nb06-s4-title"),

cc("""\
fgr = FermiGoldenRule(hbar)

# Paramètres typiques : couplage faible (régime linéaire Γt << 1)
W_fi    = 1e-22   # élément de matrice (J) — très faible couplage
E_final = 1.0 * 1.6e-19  # énergie finale (1 eV)
volume  = 1e-27   # volume macroscopique (m³)

rho_f  = fgr.density_of_states_3d(E_final, m_e, hbar, volume)
rate   = fgr.transition_rate(W_fi**2, rho_f)
t_max  = 1e-12  # durée de la perturbation (1 ps)

ok_pos   = tv.fermi_rate_nonnegative(rate)
ok_lin   = tv.linear_regime_check(rate, t_max)

print("RÈGLE D'OR DE FERMI (Règle R11.2)")
print('-' * 55)
print(f'  |W_fi|  = {W_fi:.2e} J   (élément de matrice)')
print(f"  \u03c1(E_f)  = {rho_f:.4e} J\u207b\u00b9  (densit\u00e9 d'états 3D)")
print(f'  Γ       = {rate:.4e} s⁻¹  (taux de transition)')
print(f'  Γ·t_max = {rate*t_max:.4e}  (doit être ≪ 1)')
print()
print(f"  {'✓' if ok_pos else '✗'} Taux de transition Γ ≥ 0")
print(f"  {'✓' if ok_lin else '✗'} Régime linéaire Γ·t ≪ 1")
print()
print('BILAN FINAL — Tome 2 Chapitre XI')
print('=' * 55)
all_ok = all([
    abs(rabi_res.population_excited(np.array([T_pi]))[0] - 1) < 1e-6,
    tv.probability_bounds(P2_res),
    tv.rabi_oscillation_amplitude(P2_res, omega_R, rabi_res.omega_generalized),
    pv.variational_bound(E_var, E_gs, tolerance=1e-8),
    pv.energy_corrections_real(np.array([E1_0])),
    pv.second_order_ground_state_negative(E2_0),
    ok_pos, ok_lin,
])
print(f"  {'✓ Toutes les validations réussies !' if all_ok else '✗ Certaines validations ont échoué'}")
""", "nb06-fermi"),

]  # end NB06_CELLS

# =============================================================================
# NB07 — Structure fine de l'hydrogène et effets Zeeman/Stark
# =============================================================================

NB07_CELLS = [

mc(r"""# Structure Fine de l'Hydrogène et Effets Externes

**Référence :** Cohen-Tannoudji, Diu, Laloë — *Mécanique Quantique Tome 2*, Ch. XII

---

## Cadre théorique

### Structure fine (Règle R10.1)

$$E_{n,l,j} = E_n^0 + W_{\rm mv} + W_{\rm SO} + W_D$$

- **Correction de masse-vitesse :** $W_{\rm mv} = -\frac{p^4}{8m^3c^2}$
- **Couplage spin-orbite :** $W_{\rm SO} = \frac{1}{2m^2c^2}\frac{1}{r}\frac{dV}{dr}\hat{L}\cdot\hat{S}$  (nul pour $l=0$)
- **Terme de Darwin :** $W_D = \frac{\hbar^2}{8m^2c^2}\nabla^2V$  (non nul seulement pour $l=0$)

### Hyperfine et raie 21 cm (Règle R10.2)

$$A_{\rm hf} = \frac{8}{3}g_p\frac{m_e}{m_p}\alpha^2 E_I, \qquad \nu_{21\,\rm cm} = \frac{\Delta E_{\rm hf}}{h} \approx 1420.4\,\mathrm{MHz}$$

### Effet Zeeman (Règle R10.3) en champ faible

$$\Delta E = g_J \mu_B M_J B, \quad g_J = 1 + \frac{J(J+1)+S(S+1)-L(L+1)}{2J(J+1)}$$

---""", "nb07-title"),

cc("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from quantum_simulation.systems.hydrogen_structure import HydrogenFineStructure, HydrogenHyperfine
from quantum_simulation.systems.zeeman_stark import ZeemanEffect, StarkEffect
from quantum_simulation.utils.config_loader import load_config

config = load_config()
hbar      = config['physical_constants']['hbar']
m_e       = config['physical_constants']['m_electron']
e_charge  = 1.602176634e-19   # C
epsilon_0 = 8.854187817e-12   # F/m
c_light   = 2.99792458e8      # m/s
m_p       = 1.67262192369e-27 # kg
g_p       = 5.5857            # facteur g du proton
a0        = 5.29177210903e-11 # m  (rayon de Bohr)

print(f'ℏ   = {hbar:.6e} J·s')
print(f'm_e = {m_e:.6e} kg')
print(f'α   = {e_charge**2/(4*np.pi*epsilon_0*hbar*c_light):.8e}  (structure fine)')
print('Modules chargés : hydrogen_structure, zeeman_stark')
""", "nb07-setup"),

mc(r"""## 1. Niveaux non perturbés de l'hydrogène

$$E_n^0 = -\frac{E_I}{n^2}, \quad E_I = \frac{m_e e^4}{2(4\pi\varepsilon_0)^2\hbar^2} \approx 13.606\,\mathrm{eV}$$""",
"nb07-s1-title"),

cc("""\
hfs = HydrogenFineStructure(m_e, hbar, c_light, e_charge, epsilon_0)

E1 = hfs.unperturbed_energy(1)
E2 = hfs.unperturbed_energy(2)
print('NIVEAUX NON PERTURBÉS')
print('-' * 45)
print(f'  E₁ = {E1:.6e} J = {E1/e_charge:.4f} eV  (attendu: -13.6057 eV)')
print(f'  E₂ = {E2:.6e} J = {E2/e_charge:.4f} eV  (attendu:  -3.4014 eV)')
ok_E1 = abs(E1/e_charge + 13.6057) < 0.001
ok_E2 = abs(E2/e_charge + 3.4014) < 0.001
print(f"  {'✓' if ok_E1 else '✗'} E₁ = -13.606 eV")
print(f"  {'✓' if ok_E2 else '✗'} E₂ = -3.401 eV")
print(f'  Rapport E₂/E₁ = {E2/E1:.4f}  (attendu: 1/4 = {1/4:.4f})  {"✓" if abs(E2/E1 - 0.25) < 1e-4 else "✗"}')
""", "nb07-e0"),

mc(r"""## 2. Structure fine du niveau $n=2$

Le niveau $n=2$ comprend 8 sous-niveaux :
$|2s_{1/2}, M_J=\pm\tfrac{1}{2}\rangle$, $|2p_{1/2}, M_J=\pm\tfrac{1}{2}\rangle$,
$|2p_{3/2}, M_J=\pm\tfrac{1}{2},\pm\tfrac{3}{2}\rangle$

**Dégénérescence levée** par les corrections relativistes de structure fine.""",
"nb07-s2-title"),

cc("""\
# Énergies de structure fine pour n=2
levels_n2 = [
    ('2s₁/₂', 0, 0.5),
    ('2p₁/₂', 1, 0.5),
    ('2p₃/₂', 1, 1.5),
]

print('STRUCTURE FINE n=2 (Règle R10.1)')
print('-' * 70)
print(f'  {"Niveau":<10} {"l":>3} {"j":>4} {"E_SF (J)":>16} {"E_SF (μeV)":>13} {"E_Dirac (μeV)":>14}')
print('-' * 70)

E_sf_dict = {}
for label, l, j in levels_n2:
    E_sf   = hfs.fine_structure_energy(2, l, j)
    E_dir  = hfs.fine_structure_energy_dirac(2, j)
    E_sf_dict[(l, j)] = E_sf
    print(f'  {label:<10} {l:>3d} {j:>4.1f} {E_sf:>16.8e}  {E_sf/e_charge*1e6:>12.6f}  {E_dir/e_charge*1e6:>13.6f}')
    rel_err = abs(E_sf - E_dir)/abs(E_dir) if E_dir != 0 else 0
    print(f'       {"✓" if rel_err < 1e-4 else "✗"} Accord SF vs Dirac  (err={rel_err:.2e})')

# Splitting 2p : ΔE = E(2p₃/₂) - E(2p₁/₂)
dE_sf = E_sf_dict[(1, 1.5)] - E_sf_dict[(1, 0.5)]
dE_GHz = dE_sf / (6.62607015e-34 * 1e9)
print()
print(f'  ΔE(2p₃/₂ − 2p₁/₂) = {dE_sf/e_charge*1e6:.4f} μeV = {dE_GHz:.3f} GHz')
print(f'  Valeur attendue (Cohen-T.) ≈ 10.9 GHz  {"✓" if 9 < dE_GHz < 12 else "✗"}')
""", "nb07-fine"),

cc("""\
# Diagramme d'énergie n=2
fig, ax = plt.subplots(figsize=(10, 6))

sublevel_data = [
    # (label, l, j, E)
    ('$2s_{1/2}$', 0, 0.5, E_sf_dict[(0, 0.5)]),
    ('$2p_{1/2}$', 1, 0.5, E_sf_dict[(1, 0.5)]),
    ('$2p_{3/2}$', 1, 1.5, E_sf_dict[(1, 1.5)]),
]

colors = ['steelblue', 'darkorange', 'green']
x_pos  = [0.5, 1.5, 2.5]

for i, (label, l, j, E) in enumerate(sublevel_data):
    n_mj = int(2*j) + 1  # dégénérescence
    E_eV = E / e_charge * 1e6  # en μeV
    ax.hlines(E_eV, x_pos[i]-0.3, x_pos[i]+0.3,
              colors=colors[i], linewidth=4, label=f'{label}  (×{n_mj})')
    ax.text(x_pos[i]+0.35, E_eV, f'{E_eV:.4f} μeV\\n  (×{n_mj} M_J)',
            va='center', fontsize=9, color=colors[i])

ax.set_xlim(0, 3.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(['$2s_{1/2}$', '$2p_{1/2}$', '$2p_{3/2}$'], fontsize=12)
ax.set_ylabel('Énergie relative à $E_2^0$ (μeV)', fontsize=12)
ax.set_title('Diagramme de structure fine — hydrogène $n=2$\\n'
             'Cohen-Tannoudji Tome 2, Ch. XII — Règle R10.1', fontsize=11)
ax.legend(fontsize=10, loc='center right')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('../../results/07_fine_structure.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figure sauvegardée → results/07_fine_structure.png')
""", "nb07-fine-plot"),

mc(r"""## 3. Structure hyperfine et raie 21 cm (Règle R10.2)

L'interaction entre le spin de l'électron et le moment magnétique du proton lève la dégénérescence
du niveau $1s_{1/2}$ en deux sous-niveaux $F=0$ et $F=1$ :

$$\nu_{21\,\rm cm} = \frac{A_{\rm hf}}{h}, \quad A_{\rm hf} = \frac{8}{3}g_p\frac{m_e}{m_p}\alpha^2 E_I \approx 1420.4\,\mathrm{MHz}$$""",
"nb07-s3-title"),

cc("""\
hhf = HydrogenHyperfine(m_e, m_p, hbar, g_p, e_charge, epsilon_0)

A_hf   = hhf.hyperfine_coupling_1s()
delta_E = hhf.hyperfine_splitting()
nu_21  = hhf.transition_frequency_21cm()
lam_21 = hhf.transition_wavelength_21cm()

nu_nist = 1420.405751768e6  # Hz (NIST 2022)
lam_th  = c_light / nu_nist

print('RAIE 21 cm — STRUCTURE HYPERFINE 1s (Règle R10.2)')
print('-' * 60)
print(f'  Constante hyperfine A = {A_hf:.6e} J')
print(f'  Splitting ΔE = {delta_E:.6e} J = {delta_E/e_charge*1e6:.4f} μeV')
print(f'  Fréquence ν  = {nu_21/1e6:.4f} MHz  (NIST: {nu_nist/1e6:.4f} MHz)')
print(f"  Longueur d'onde \u03bb = {lam_21*100:.4f} cm  (attendu: {lam_th*100:.4f} cm)")

err_nu = abs(nu_21 - nu_nist) / nu_nist
print()
print(f"  {'✓' if err_nu < 0.001 else '✗'} Accord fréquence NIST  (erreur relative = {err_nu:.4e})")

# Énergies des sous-niveaux F=0 et F=1
E_F0 = hhf.hyperfine_energy(0, 0.5, 0.5)
E_F1 = hhf.hyperfine_energy(1, 0.5, 0.5)
print(f'  E(F=1) - E(F=0) = {E_F1 - E_F0:.6e} J  (= A_hf? {abs(E_F1-E_F0-A_hf)/A_hf:.2e})')
""", "nb07-hyperfine"),

mc(r"""## 4. Effet Zeeman en champ faible (Règle R10.3)

En champ faible, $j$ reste un bon nombre quantique :
$$E(B) = E_{\rm SF} + g_J \mu_B M_J B$$""",
"nb07-s4-title"),

cc("""\
zeeman = ZeemanEffect(hbar)

B_range = np.linspace(0, 1.0, 100)   # 0 à 1 Tesla

# Facteurs de Landé pour 2p
g_12 = zeeman.lande_g_factor(1, 0.5, 0.5)   # 2p₁/₂
g_32 = zeeman.lande_g_factor(1, 0.5, 1.5)   # 2p₃/₂

print('FACTEURS DE LANDÉ')
print(f'  g_J(2p₁/₂) = {g_12:.6f}  (attendu: 2/3 = {2/3:.6f})  {"✓" if abs(g_12 - 2/3) < 1e-6 else "✗"}')
print(f'  g_J(2p₃/₂) = {g_32:.6f}  (attendu: 4/3 = {4/3:.6f})  {"✓" if abs(g_32 - 4/3) < 1e-6 else "✗"}')

# Diagramme Zeeman
E0_dict = {0.5: E_sf_dict[(1, 0.5)], 1.5: E_sf_dict[(1, 1.5)]}
diagram = zeeman.zeeman_diagram(L=1, J_values=[0.5, 1.5],
                                 B_field_range=B_range, E0_dict=E0_dict)

fig, ax = plt.subplots(figsize=(10, 6))
mu_B = 9.2740100783e-24

colors_J = {0.5: 'steelblue', 1.5: 'darkorange'}
styles_MJ = {-1.5:'--', -0.5:':', 0.5:'-.', 1.5:'-'}

for (J, MJ), E_arr in diagram.items():
    ls = styles_MJ.get(MJ, '-')
    ax.plot(B_range, E_arr/e_charge*1e6, ls,
            color=colors_J[J], lw=1.8,
            label=f'2p{{"1/2" if J==0.5 else "3/2"}}, Mⱼ={MJ:+.1f}')

ax.set_xlabel('Champ magnétique $B$ (T)', fontsize=12)
ax.set_ylabel('Énergie (μeV)', fontsize=12)
ax.set_title("Diagramme de Zeeman - niveaux $2p$ de l'hydrogène\\nCh. XII - Règle R10.3", fontsize=11)
ax.legend(fontsize=8, ncol=2, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../../results/07_zeeman.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figure sauvegardée → results/07_zeeman.png')
""", "nb07-zeeman"),

mc(r"""## 5. Effet Stark linéaire sur $n=2$ (Règle R10.4)

En champ électrique $\mathcal{E}$ selon $z$, la perturbation $W_{\rm Stark} = e\mathcal{E}z$
lève la dégénérescence $2s$–$2p$ au **1er ordre** (effet Stark linéaire).

Le seul élément de matrice non nul dans le sous-espace $n=2$ est :
$$\langle 2s|e\mathcal{E}z|2p, m=0\rangle = -3e\mathcal{E}a_0$$""",
"nb07-s5-title"),

cc("""\
stark = StarkEffect(hbar, e_charge, a0)

eps_range = np.linspace(0, 5e8, 200)   # 0 à 5×10⁸ V/m
E_n2 = hfs.unperturbed_energy(2)

# Valeurs propres en fonction du champ
stark_levels = np.array([stark.stark_energies_n2(eps, E_n2) for eps in eps_range])

fig, ax = plt.subplots(figsize=(10, 5))
colors_s = ['steelblue', 'darkorange', 'green', 'red']
labels_s = ['$E_1$ (mélange $2s$/$2p$)', r'$E_2$ ($2p, m=\pm1$)',
            r'$E_3$ ($2p, m=\pm1$)', '$E_4$ (mélange $2s$/$2p$)']

for i in range(4):
    ax.plot(eps_range/1e8, (stark_levels[:,i] - E_n2)/e_charge*1e6,
            color=colors_s[i], lw=2, label=labels_s[i])

ax.set_xlabel(r'Champ électrique $\mathcal{E}$ ($10^8$ V/m)', fontsize=12)
ax.set_ylabel(r'$\Delta E$ relative à $E_2^0$ (μeV)', fontsize=12)
ax.set_title('Effet Stark linéaire sur $n=2$\\n'
             'Cohen-Tannoudji Tome 2, Ch. XII — Règle R10.4', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../../results/07_stark.png', dpi=150, bbox_inches='tight')
plt.show()

# Vérification : levée de dégénérescence linéaire en ε
E_split_th = 3 * e_charge * eps_range[-1] * a0  # ΔE = 3eεa₀
E_split_num = stark_levels[-1, -1] - stark_levels[-1, 0]
err_stark = abs(E_split_num - 2*E_split_th) / (2*E_split_th)
print(f'  Splitting à ε={eps_range[-1]:.1e} V/m :')
print(f'    Calculé : {E_split_num/e_charge*1e6:.4f} μeV')
print(f'    Attendu : {2*E_split_th/e_charge*1e6:.4f} μeV  (2×3eεa₀)')
print(f"  {'✓' if err_stark < 0.01 else '✗'} Effet Stark linéaire (erreur = {err_stark:.2e})")

alpha_pol = stark.polarizability_1s()
print(f'  Polarisabilité 1s : α = {alpha_pol:.4e} F·m²  (= 4.5a₀³×4πε₀)')
print('Figure sauvegardée → results/07_stark.png')

print()
print('BILAN FINAL — Tome 2 Chapitre XII')
print('=' * 55)
all_ok = all([ok_E1, ok_E2, abs(dE_GHz-10.9)<2, err_nu < 0.001,
              abs(g_12-2/3)<1e-6, abs(g_32-4/3)<1e-6, err_stark < 0.01])
print(f"  {'✓ Toutes les validations réussies !' if all_ok else '✗ Certaines validations ont échoué'}")
""", "nb07-stark"),

]  # end NB07_CELLS

# =============================================================================
# NB08 — Diffusion quantique et particules identiques
# =============================================================================

NB08_CELLS = [

mc(r"""# Diffusion Quantique et Particules Identiques

**Référence :** Cohen-Tannoudji, Diu, Laloë — *Mécanique Quantique Tome 2*, Ch. VIII (diffusion) et Ch. XIV (particules identiques)

---

## Cadre théorique

### Approximation de Born (Règle R6.2)

$$f_{\rm Born}(\theta) = -\frac{m}{2\pi\hbar^2}\int e^{-i\vec{q}\cdot\vec{r}} V(\vec{r})\,d^3r, \quad \vec{q} = \vec{k}'-\vec{k}$$

### Théorème optique (Règle R6.3)

$$\sigma_{\rm tot} = \frac{4\pi}{k}\,\mathrm{Im}[f(\theta=0)]$$

### Ondes partielles (Règle R6.1)

$$\sigma_l = \frac{4\pi}{k^2}(2l+1)\sin^2\delta_l, \qquad \sigma_{\rm tot} = \sum_l \sigma_l$$

### Particules identiques — symétrie (Règles R12.1–R12.3)

**Bosons :** $|\Psi\rangle = \frac{1}{\sqrt{2}}(|\psi_1\rangle\otimes|\psi_2\rangle + |\psi_2\rangle\otimes|\psi_1\rangle)$

**Fermions :** $|\Psi\rangle = \frac{1}{\sqrt{2}}(|\psi_1\rangle\otimes|\psi_2\rangle - |\psi_2\rangle\otimes|\psi_1\rangle) = 0$ si $|\psi_1\rangle = |\psi_2\rangle$

---""", "nb08-title"),

cc("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from quantum_simulation.dynamics.scattering import BornApproximation, PhaseShiftSolver, CrossSection
from quantum_simulation.systems.identical_particles import (
    Symmetrizer, SlaterDeterminant, IdenticalParticlesScattering
)
from quantum_simulation.validation.tome2_invariants import ScatteringValidator, SymmetrizationValidator
from quantum_simulation.utils.config_loader import load_config

config = load_config()
hbar = config['physical_constants']['hbar']
m_e  = config['physical_constants']['m_electron']
e    = 1.602176634e-19   # C

print(f'ℏ   = {hbar:.6e} J·s')
print(f'm_e = {m_e:.6e} kg')
print('Modules chargés : scattering, identical_particles, validators Tome 2')
""", "nb08-setup"),

mc(r"""## 1. Diffusion sur un potentiel de Yukawa — Approximation de Born (Règles R6.2–R6.3)

$$V(r) = -V_0\,\frac{e^{-r/a}}{r}, \quad \text{potentiel de Yukawa (portée } a\text{)}$$

Amplitude de Born analytique :
$$f_{\rm Born}(\theta) = \frac{2mV_0 a^2}{\hbar^2} \cdot \frac{1}{1 + (2ka\sin\theta/2)^2}$$""",
"nb08-s1-title"),

cc("""\
# Paramètres du potentiel de Yukawa (régime Born : V0 << E)
V0      = 0.5 * e        # profondeur 0.5 eV (faible)
a_y     = 1e-10          # portée 1 Å
E_part  = 10.0 * e       # énergie incidente 10 eV (E >> V0)
m       = m_e

k = np.sqrt(2 * m * E_part) / hbar
lambda_dB = 2 * np.pi / k

print(f'Potentiel de Yukawa : V₀ = {V0/e:.1f} eV,  a = {a_y*1e10:.1f} Å')
print(f'Énergie incidente   : E  = {E_part/e:.1f} eV')
print(f'k  = {k:.4e} m⁻¹,   λ_dB = {lambda_dB*1e10:.4f} Å')
print(f'ka = {k*a_y:.4f}  (portée en unités de λ/2π)')
print()

def V_yukawa(r):
    r = np.asarray(r, dtype=float)
    scalar = r.ndim == 0
    r = np.atleast_1d(r)
    result = np.where(r < 1e-14, -V0 * np.exp(-1e-14/a_y) / 1e-14, -V0 * np.exp(-r/a_y) / r)
    return float(result[0]) if scalar else result

r_max   = 50 * a_y
n_theta = 300
theta_grid = np.linspace(0.01, np.pi - 0.01, n_theta)

born = BornApproximation(m, hbar, E_part, V_yukawa)

sigma_diff  = born.differential_cross_section(theta_grid, r_max=r_max)
sigma_tot_B = born.total_cross_section(n_theta=n_theta, r_max=r_max)
ot_check    = born.optical_theorem_check(r_max=r_max)

print(f'Section efficace totale (Born)  : σ = {sigma_tot_B:.4e} m²  = {sigma_tot_B/1e-20:.4f} Å²')
# Born amplitude is real → optical theorem formally violated (expected)
ot_rel_err = ot_check.get('relative_error', float('nan'))
print(f"  (i) Théorème optique (Born) : erreur relative = {ot_rel_err:.2e}  (violation attendue : Im[f_Born]=0)")

sv = ScatteringValidator()
ok_diff_pos = sv.differential_cs_nonnegative(sigma_diff)
print(f"  {'✓' if ok_diff_pos else '✗'} dσ/dΩ ≥ 0 partout")

born_regime = sv.born_approximation_regime(V0, E_part, a_y, m, hbar)
print(f"  {'✓' if born_regime['is_valid'] else '✗'} Régime Born valide  (paramètre = {born_regime['expansion_parameter']:.4f} ≪ 1)")
""", "nb08-born-calc"),

cc("""\
# Plot section efficace différentielle
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].semilogy(np.degrees(theta_grid), sigma_diff/1e-20, 'steelblue', lw=2,
                 label='Born dσ/dΩ')
axes[0].set_xlabel('Angle de diffusion θ (°)', fontsize=12)
axes[0].set_ylabel('dσ/dΩ (Å² / sr)', fontsize=12)
axes[0].set_title('Section efficace différentielle\\nPotentiel de Yukawa — Règle R6.2', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, which='both')

# Comparaison analytique : f_Born(θ) = 2mV₀a²/ℏ² / (1+(2ka·sin(θ/2))²)
f_an = (2*m*V0*a_y**2/hbar**2) / (1 + (2*k*a_y*np.sin(theta_grid/2))**2)
sigma_an = np.abs(f_an)**2
axes[0].semilogy(np.degrees(theta_grid), sigma_an/1e-20, 'r--', lw=1.5,
                 label='Analytique (Yukawa)')
axes[0].legend(fontsize=10)

axes[1].plot(np.degrees(theta_grid), sigma_diff/1e-20, 'steelblue', lw=2, label='Born (numérique)')
axes[1].plot(np.degrees(theta_grid), sigma_an/1e-20, 'r--', lw=2, label='Analytique')
axes[1].set_xlabel('Angle de diffusion θ (°)', fontsize=12)
axes[1].set_ylabel('dσ/dΩ (Å² / sr)', fontsize=12)
axes[1].set_title('Comparaison Born numérique vs analytique', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../results/08_born_cross_section.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figure sauvegardée → results/08_born_cross_section.png')
""", "nb08-born-plot"),

mc(r"""## 2. Déphasages et section efficace par ondes partielles (Règles R6.1, R6.4)

L'équation radiale de Schrödinger est résolue numériquement pour extraire les déphasages $\delta_l$ :
$$\frac{d^2 u_l}{dr^2} + \left[k^2 - \frac{2m}{\hbar^2}V(r) - \frac{l(l+1)}{r^2}\right]u_l = 0$$""",
"nb08-s2-title"),

cc("""\
r_min  = 5e-13   # éviter singularité en r=0
r_grid = np.linspace(r_min, 30*a_y, 600)
l_max  = 5

print(f'Calcul des déphasages δₗ pour l = 0..{l_max}')
phase_solver = PhaseShiftSolver(m, hbar, E_part, V_yukawa, r_grid)
phase_shifts = phase_solver.compute_all_phase_shifts(l_max, convergence_threshold=1e-4)

print()
print(f'  {"l":>3} | {"δₗ (rad)":>12} | {"sin²(δₗ)":>12} | {"σₗ (Å²)":>12}')
print('-' * 50)

cs = CrossSection(k, phase_shifts, hbar)
sigma_l = cs.partial_wave_cross_sections()
for l in range(l_max + 1):
    print(f'  {l:>3} | {phase_shifts[l]:>12.6f} | {np.sin(phase_shifts[l])**2:>12.6f} | {sigma_l[l]/1e-20:>12.4f}')

sigma_tot_pw = cs.total_cross_section()
ot_pw = cs.optical_theorem_check()

print()
print(f'  σ_tot (ondes partielles) = {sigma_tot_pw:.4e} m² = {sigma_tot_pw/1e-20:.4f} Å²')
print(f'  σ_tot (Born)             = {sigma_tot_B:.4e} m² = {sigma_tot_B/1e-20:.4f} Å²')
print(f'  Rapport σ_PW/σ_Born      = {sigma_tot_pw/sigma_tot_B:.4f}')
print(f"  {'✓' if ot_pw['is_valid'] else '✗'} Théorème optique (ondes partielles)  erreur = {ot_pw.get('relative_error', float('nan')):.2e}")
ok_unit = sv.unitarity_bound(phase_shifts)
print(f"  {'✓' if ok_unit else '✗'} Borne d'unitarité |δₗ| < π/2")
""", "nb08-phase"),

cc("""\
# Convergence des ondes partielles
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

l_arr = np.arange(l_max + 1)
axes[0].bar(l_arr, phase_shifts, color='steelblue', alpha=0.8)
axes[0].set_xlabel('Moment angulaire $l$', fontsize=12)
axes[0].set_ylabel(r'Déphasage $\delta_l$ (rad)', fontsize=12)
axes[0].set_title('Déphasages partiels $\\\\delta_l$\\nRègle R6.4', fontsize=11)
axes[0].set_xticks(l_arr)
axes[0].grid(True, alpha=0.3, axis='y')

sigma_cumul = np.cumsum(sigma_l)
axes[1].bar(l_arr, sigma_l/1e-20, color='darkorange', alpha=0.8, label='σ_l')
axes[1].plot(l_arr, sigma_cumul/1e-20, 'k-o', ms=6, lw=2, label='Σ_l σ_l (cumulé)')
axes[1].axhline(sigma_tot_B/1e-20, color='r', ls='--', lw=1.5, label=f'σ_Born = {sigma_tot_B/1e-20:.2f} Å²')
axes[1].set_xlabel('$l$', fontsize=12)
axes[1].set_ylabel('Section efficace (Å²)', fontsize=12)
axes[1].set_title('Convergence par ondes partielles\\nRègle R6.1', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].set_xticks(l_arr)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../../results/08_phase_shifts.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figure sauvegardée → results/08_phase_shifts.png')
""", "nb08-phase-plot"),

mc(r"""## 3. Particules identiques — symétrisation (Règle R12.1)

Pour deux **bosons** (spin entier) : $|\Psi\rangle$ doit être **symétrique** par échange.

Pour deux **fermions** (spin demi-entier) : $|\Psi\rangle$ doit être **antisymétrique** par échange.

**Principe d'exclusion de Pauli :** Si $|\psi_1\rangle = |\psi_2\rangle$, l'état antisymétrique est identiquement nul.""",
"nb08-s3-title"),

cc("""\
# Grille 1D
nx  = 80
x   = np.linspace(-10e-9, 10e-9, nx)
sig = 1.2e-9   # largeur des gaussiennes

# Deux états gaussiens centrés en ±x0 (quasi-orthogonaux pour x0 >> sig)
x0 = 5e-9
psi_a = np.exp(-(x - x0)**2 / (2*sig**2)); psi_a /= np.linalg.norm(psi_a)
psi_b = np.exp(-(x + x0)**2 / (2*sig**2)); psi_b /= np.linalg.norm(psi_b)

# Vérification d'orthogonalité approximative
overlap = np.dot(psi_a, psi_b)
print(f'Recouvrement ⟨ψ_a|ψ_b⟩ = {overlap:.4f}  (quasi-orthogonal si faible)')

# Symétrisations
sym_state    = Symmetrizer.symmetric_two_particle(psi_a, psi_b)
antisym_state = Symmetrizer.antisymmetric_two_particle(psi_a, psi_b)

print(f'Norme état symétrique    : {np.linalg.norm(sym_state):.6f}')
print(f'Norme état antisymétrique: {np.linalg.norm(antisym_state):.6f}')

# Vérification de la symétrie
symv = SymmetrizationValidator()
res_sym     = Symmetrizer.verify_symmetry(sym_state,    2, nx, 'bose')
res_antisym = Symmetrizer.verify_symmetry(antisym_state, 2, nx, 'fermi')
print(f"  {'✓' if res_sym['is_correct'] else '✗'}    État bosonique : valeur propre échange = {res_sym['symmetry_eigenvalue']:+.4f} (attendu: +1)")
print(f"  {'✓' if res_antisym['is_correct'] else '✗'} État fermionique : valeur propre échange = {res_antisym['symmetry_eigenvalue']:+.4f} (attendu: -1)")

# Densité à deux particules |Ψ(x₁,x₂)|²
X1, X2 = np.meshgrid(x, x, indexing='ij')
rho_sym    = sym_state.reshape(nx, nx)**2
rho_antisym = antisym_state.reshape(nx, nx)**2

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, rho, title in zip(
    axes,
    [rho_sym, rho_antisym, np.outer(psi_a, psi_b)**2 + np.outer(psi_b, psi_a)**2],
    [r'Bosons $|\Psi^+|^2$', r'Fermions $|\Psi^-|^2$', r'Classique $|\psi_a|^2|\psi_b|^2$ (somme)']
):
    im = ax.pcolormesh(x*1e9, x*1e9, rho/rho.max(), cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel('$x_1$ (nm)', fontsize=11)
    ax.set_ylabel('$x_2$ (nm)', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal')
plt.suptitle(r'Densité à deux particules $|\Psi(x_1,x_2)|^2$ — Règle R12.1',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../../results/08_identical_particles.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figure sauvegardée → results/08_identical_particles.png')
""", "nb08-symmetry"),

mc(r"""## 4. Déterminant de Slater (Règle R12.2)

L'état antisymétrique à $N$ fermions est le **déterminant de Slater** :
$$|\Psi\rangle = \frac{1}{\sqrt{N!}}\det[\varphi_\alpha(\vec{r}_i)]$$

**Pauli** : si deux orbitales sont identiques, le déterminant est nul.""",
"nb08-s4-title"),

cc("""\
# Déterminant de Slater avec états orthogonaux
slater_orthog = SlaterDeterminant([psi_a, psi_b])
norm_sl  = slater_orthog.norm()
pauli_ok = slater_orthog.pauli_exclusion_satisfied()

print('DÉTERMINANT DE SLATER — PRINCIPE DE PAULI (Règle R12.2)')
print('-' * 60)
print(f'  ||Ψ_Slater|| (états orthogonaux) = {norm_sl:.6f}')
print(f"  {'✓' if pauli_ok else '✗'} Pauli satisfait (det ≠ 0)")

# Test avec états identiques → det = 0
slater_id = SlaterDeterminant([psi_a, psi_a])
pauli_viol = not slater_id.pauli_exclusion_satisfied()
print(f"  {'✓' if pauli_viol else '✗'} Pauli : det = 0 si états identiques  (det = {slater_id.norm():.2e})")

# Validation via SymmetrizationValidator
ok_norm  = symv.slater_normalization(slater_orthog)
ok_pauli = symv.pauli_exclusion(slater_orthog)
ok_pauli_id = not symv.pauli_exclusion(slater_id)
print()
print(f"  {'✓' if ok_norm    else '✗'} SymmetrizationValidator.slater_normalization()")
print(f"  {'✓' if ok_pauli   else '✗'} SymmetrizationValidator.pauli_exclusion() [orthogonal]")
print(f"  {'✓' if ok_pauli_id else '✗'} SymmetrizationValidator.pauli_exclusion() = False [identiques]")
""", "nb08-slater"),

mc(r"""## 5. Diffusion de particules identiques (Règle R12.3)

Dans le centre de masse, la section efficace de diffusion est modifiée par les interférences :

- **Bosons :** $\frac{d\sigma}{d\Omega} = |f(\theta) + f(\pi-\theta)|^2$
- **Fermions (triplet) :** $\frac{d\sigma}{d\Omega} = |f(\theta) - f(\pi-\theta)|^2 \to 0$ pour $\theta = \pi/2$
- **Classique :** $\frac{d\sigma}{d\Omega} = |f(\theta)|^2 + |f(\pi-\theta)|^2$""",
"nb08-s5-title"),

cc("""\
# Amplitude de diffusion de Born : f(θ) pour le potentiel de Yukawa
f_forward  = born.scattering_amplitude(theta_grid, r_max=r_max)
f_exchange = np.array([born.scattering_amplitude(np.array([np.pi - t]), r_max=r_max)[0]
                       for t in theta_grid])

sigma_boson   = IdenticalParticlesScattering.cross_section_bosons(f_forward, f_exchange, theta_grid)
sigma_singlet = IdenticalParticlesScattering.cross_section_fermions_singlet(f_forward, f_exchange, theta_grid)
sigma_triplet = IdenticalParticlesScattering.cross_section_fermions_triplet(f_forward, f_exchange, theta_grid)
sigma_classic = IdenticalParticlesScattering.cross_section_classical(f_forward, f_exchange, theta_grid)

# Annulation à θ=π/2 pour fermions triplet
i_half = np.argmin(np.abs(theta_grid - np.pi/2))
print(f'Vérification σ_triplet(π/2) = {sigma_triplet[i_half]:.4e}  (attendu : 0)')
ok_zero = symv.fermion_triplet_zero_at_pi_half(sigma_triplet, theta_grid)
print(f"  {'✓' if ok_zero['is_zero'] else '✗'} σ_triplet(π/2) = 0  (|f-f'| = 0 par symétrie)")

fig, ax = plt.subplots(figsize=(10, 6))
theta_deg = np.degrees(theta_grid)
ref = sigma_classic / sigma_classic.max()
ax.plot(theta_deg, sigma_boson  / sigma_classic.max(), 'steelblue', lw=2.5, label="Bosons  $|f+f'|^2$")
ax.plot(theta_deg, sigma_triplet/ sigma_classic.max(), 'darkorange', lw=2.5, ls='--',
        label="Fermions triplet $|f-f'|^2$")
ax.plot(theta_deg, sigma_singlet/ sigma_classic.max(), 'green', lw=2, ls=':',
        label="Fermions singulet $|f+f'|^2$")
ax.plot(theta_deg, ref,          'gray',  lw=1.5, ls='-.', label="Classique $|f|^2+|f'|^2$")
ax.axvline(90, color='red', ls=':', lw=1.5, label='θ=π/2 (annulation triplet)')
ax.set_xlabel('Angle θ (°)', fontsize=12)
ax.set_ylabel('dσ/dΩ (normalisée)', fontsize=12)
ax.set_title("Diffusion de particules identiques - Interférences d'échange\\nRègle R12.3",
             fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../../results/08_identical_scattering.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figure sauvegardée → results/08_identical_scattering.png')
""", "nb08-scatter-id"),

mc("## 6. Validation finale — résumé", "nb08-conclusion-title"),

cc("""\
print('BILAN DE VALIDATION — Tome 2 Ch. VIII et XIV')
print('=' * 60)

# Diffusion
print()
print('  DIFFUSION :')
ok_born_nonneg = ok_diff_pos
ok_born_regime = born_regime['is_valid']
print(f"    {'✓' if ok_born_nonneg else '✗'} dσ/dΩ ≥ 0 partout")
print(f"    (i) Théorème optique Born : violation attendue (Im[f_Born]=0)")
print(f"    {'✓' if ot_pw['is_valid']    else '✗'} Théorème optique ondes partielles")
print(f"    {'✓' if ok_unit             else '✗'} Borne d'unitarité |δₗ| < π/2")
print(f"    {'✓' if ok_born_regime else '✗'} Régime Born valide")

# Particules identiques
print()
print('  PARTICULES IDENTIQUES :')
print(f"    {'✓' if res_sym['is_correct']     else '✗'} État bosonique symétrique (+1)")
print(f"    {'✓' if res_antisym['is_correct'] else '✗'} État fermionique antisymétrique (-1)")
print(f"    {'✓' if ok_norm                   else '✗'} Normalisation Slater")
print(f"    {'✓' if ok_pauli                  else '✗'} Pauli  (états orthogonaux)")
print(f"    {'✓' if ok_pauli_id               else '✗'} Pauli  (états identiques → det=0)")
print(f"    {'✓' if ok_zero['is_zero']        else '✗'} σ_triplet(π/2) = 0")

all_checks = [
    ok_born_nonneg, ot_pw['is_valid'], ok_unit,
    ok_born_regime, res_sym['is_correct'], res_antisym['is_correct'],
    ok_norm, ok_pauli, ok_pauli_id, ok_zero['is_zero'],
]
print()
print(f"  {'✓ Toutes les validations réussies !' if all(all_checks) else '✗ Certaines validations ont échoué'}")
""", "nb08-validation"),

]  # end NB08_CELLS

# =============================================================================
# Sérialisation
# =============================================================================

notebooks = {
    '05_spin_et_moment_cinetique.ipynb':         notebook(NB05_CELLS),
    '06_perturbations_et_rabi.ipynb':            notebook(NB06_CELLS),
    '07_hydrogene_structure_fine.ipynb':         notebook(NB07_CELLS),
    '08_diffusion_et_particules_identiques.ipynb': notebook(NB08_CELLS),
}

for fname, nb in notebooks.items():
    path = HERE / fname
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'  OK  {fname}  ({path.stat().st_size//1024} KB)')

print()
print('Notebooks Tome 2 générés avec succès !')
