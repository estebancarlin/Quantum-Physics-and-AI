# Document de Référence — Mécanique Quantique, Tome II
## Cohen-Tannoudji, Diu, Laloë — Spécification pour Implémentation Python

**Chapitres couverts** : VIII (Collisions), IX (Spin), X (Composition des moments cinétiques), XI (Perturbations stationnaires), XII (Structure fine et hyperfine de H), XIII (Perturbations dépendant du temps), XIV (Particules identiques)

**Convention de source** : `[file:1, Chap. N, § X-Y]` réfère au PDF fourni (*Mécanique quantique — Tome II*).

**Notebooks pédagogiques** (tous ✓ fin-à-fin) :

| Notebook | Chapitres | Modules Python |
| --- | --- | --- |
| [NB05 — Spin et moment cinétique](examples/notebooks/05_spin_et_moment_cinetique.ipynb) | IX, X | `core/spin.py`, `core/angular_momentum.py` |
| [NB06 — Perturbations et Rabi](examples/notebooks/06_perturbations_et_rabi.ipynb) | XI, XIII | `dynamics/perturbation.py`, `dynamics/time_perturbation.py` |
| [NB07 — Hydrogène structure fine](examples/notebooks/07_hydrogene_structure_fine.ipynb) | XII | `systems/hydrogen_structure.py`, `systems/zeeman_stark.py` |
| [NB08 — Diffusion et particules identiques](examples/notebooks/08_diffusion_et_particules_identiques.ipynb) | VIII, XIV | `dynamics/scattering.py`, `systems/identical_particles.py` |

---

## 1. Cadre théorique issu du cours

### 1.1 Théorie élémentaire des collisions (Chapitre VIII)

#### 1.1.1 Section efficace de diffusion

**Énoncé** : Le nombre de particules diffusées par unité de temps dans l'angle solide dΩ autour de la direction (θ, ϕ) est proportionnel au flux incident F_i et à dΩ.

```
dn = F_i * σ(θ, ϕ) * dΩ                                    (A-3)
σ_tot = ∫ σ(θ, ϕ) dΩ                                        (A-5)
```
**Source** : [file:1, Chap. VIII, § A-3]

**Propriétés** :
- σ(θ, ϕ) est homogène à une surface (unité : barn = 10⁻²⁴ cm²)
- Le potentiel V(r) doit décroître plus vite que 1/r à l'infini (exclut le potentiel coulombien)

#### 1.1.2 États stationnaires de diffusion et amplitude de diffusion

**Énoncé** : L'état stationnaire de diffusion associé à l'énergie E = ℏ²k²/(2μ) a le comportement asymptotique :

```
v_k^(diff)(r) ∼ e^(ikz) + f_k(θ, ϕ) * e^(ikr) / r    pour r → ∞    (B-9)
```
**Source** : [file:1, Chap. VIII, § B-1-b]

**Propriétés** :
- f_k(θ, ϕ) est l'amplitude de diffusion, seule quantité dépendant du potentiel V(r)
- Le premier terme est l'onde plane incidente, le second est l'onde sphérique diffusée sortante
- La section efficace est reliée à l'amplitude de diffusion par : σ(θ, ϕ) = |f_k(θ, ϕ)|²    (B-24)

#### 1.1.3 Équation intégrale de la diffusion

**Énoncé** : L'état stationnaire de diffusion vérifie l'équation intégrale :

```
v_k^(diff)(r) = e^(ik_i·r) - (1/4π) ∫ d³r' * G₊(r - r') * U(r') * v_k^(diff)(r')
```
où :
```
G₊(r - r') = e^(ik|r - r'|) / |r - r'|                     (B-31)
U(r) = (2μ/ℏ²) * V(r)                                       (B-6)
```
**Source** : [file:1, Chap. VIII, § B-3]

#### 1.1.4 Approximation de Born

**Énoncé** : Au premier ordre en U, l'amplitude de diffusion est la transformée de Fourier du potentiel :

```
f_k^(B)(θ, ϕ) = -(1/4π) ∫ d³r' * e^(-iK·r') * U(r')        (B-47)
```
où K = k_d - k_i est le vecteur d'onde transféré.

La section efficace de Born s'écrit :
```
σ_k^(B)(θ, ϕ) = (μ²/(4π²ℏ⁴)) * |∫ d³r * e^(-iK·r) * V(r)|²   (B-48)
```
**Source** : [file:1, Chap. VIII, § B-4]

**Propriétés** :
- Valide si le potentiel est faible (λ ≪ 1)
- La section efficace est le carré du module de la transformée de Fourier du potentiel
- Le vecteur K a pour module : |K| = 2k sin(θ/2)

#### 1.1.5 Méthode des déphasages (potentiel central)

**Énoncé** : Pour un potentiel central V(r), l'amplitude de diffusion s'exprime en fonction des déphasages δ_l :

```
f_k(θ) = (1/k) Σ_{l=0}^∞ √(4π(2l+1)) * e^(iδ_l) * sin(δ_l) * Y_l^0(θ)   (C-55)
```

La section efficace différentielle :
```
σ(θ) = |f_k(θ)|² = (1/k²) |Σ_{l=0}^∞ √(4π(2l+1)) * e^(iδ_l) * sin(δ_l) * Y_l^0(θ)|²   (C-56)
```

La section efficace totale :
```
σ_tot = (4π/k²) Σ_{l=0}^∞ (2l+1) * sin²(δ_l)                (C-58)
```
**Source** : [file:1, Chap. VIII, § C-4]

**Propriétés** :
- Pour un potentiel de portée r₀, seuls les déphasages avec l ≲ k*r₀ sont significatifs
- À basse énergie (k*r₀ ≪ 1), seul δ₀ (onde s) contribue : σ_tot ≈ 4π sin²(δ₀)/k²
- Théorème optique : σ_tot = (4π/k) Im[f_k(0)]

**LIMITE** : La diffusion par un potentiel coulombien (1/r) est exclue du formalisme présenté.

---

### 1.2 Le spin de l'électron (Chapitre IX)

#### 1.2.1 Postulats de Pauli pour le spin

**Énoncé** : L'électron possède un moment cinétique intrinsèque S (le spin) dont les composantes vérifient les relations de commutation des moments cinétiques.

```
[S_x, S_y] = iℏ S_z    (et permutations circulaires)          (A-4)
S² |s,m⟩ = s(s+1)ℏ² |s,m⟩                                     (A-5a)
S_z |s,m⟩ = mℏ |s,m⟩                                           (A-5b)
```

L'espace des états totaux est le produit tensoriel :
```
E = E_r ⊗ E_s                                                  (A-6)
```
**Source** : [file:1, Chap. IX, § A-2]

**Propriétés** :
- L'électron a un spin s = 1/2, dim(E_s) = 2
- Le moment magnétique de spin : M_S = 2(μ_B/ℏ)S (rapport gyromagnétique anormal)
- μ_B = qℏ/(2m_e) est le magnéton de Bohr
- Toute observable de spin commute avec toute observable orbitale

#### 1.2.2 Propriétés du spin 1/2

**Énoncé** : Base {|+⟩, |−⟩} de E_s, états propres communs à S² et S_z :

```
S² |±⟩ = (3/4)ℏ² |±⟩                                          (B-1a)
S_z |±⟩ = ±(1/2)ℏ |±⟩                                         (B-1b)
```

Matrices de Pauli (représentation matricielle de S dans la base {|+⟩, |−⟩}) :
```
S_x = (ℏ/2)σ_x,  S_y = (ℏ/2)σ_y,  S_z = (ℏ/2)σ_z

σ_x = [[0, 1], [1, 0]]
σ_y = [[0, -i], [i, 0]]
σ_z = [[1, 0], [0, -1]]
```

Propriétés des matrices de Pauli :
```
σ_i² = I                                                        
σ_x σ_y = iσ_z    (et permutations circulaires)
{σ_i, σ_j} = 2δ_{ij}I                                          
Tr(σ_i) = 0
```
**Source** : [file:1, Chap. IX, § B]

#### 1.2.3 Spineur à deux composantes

**Énoncé** : Un état quelconque d'une particule de spin 1/2 s'écrit dans la base {|r⟩} ⊗ {|±⟩} comme un spineur :

```
|ψ⟩ → ψ(r) = [ψ₊(r), ψ₋(r)]^T

avec ψ_±(r) = ⟨r, ±|ψ⟩
```

Normalisation :
```
∫ d³r (|ψ₊(r)|² + |ψ₋(r)|²) = 1
```
**Source** : [file:1, Chap. IX, § C]

---

### 1.3 Composition des moments cinétiques (Chapitre X)

#### 1.3.1 Moment cinétique total

**Énoncé** : Pour deux sous-systèmes de moments cinétiques J₁ et J₂ commutant entre eux, le moment cinétique total J = J₁ + J₂ est un moment cinétique.

```
J = J₁ + J₂                                                    (A-21)
[J_i, J_j] = iℏ ε_{ijk} J_k
```
**Source** : [file:1, Chap. X, § A]

**Propriétés** :
- Si [J₁, J₂] = 0, alors J est un moment cinétique
- J² et J_z commutent avec l'hamiltonien quand seul J_total est conservé
- Deux bases de l'espace des états :
  - Base « non couplée » : |j₁, m₁; j₂, m₂⟩ (vecteurs propres de J₁², J₁z, J₂², J₂z)
  - Base « couplée » : |j₁, j₂; J, M⟩ (vecteurs propres de J₁², J₂², J², J_z)

#### 1.3.2 Composition de deux spins 1/2

**Énoncé** : Deux spins 1/2 donnent un triplet (S=1) et un singulet (S=0) :

```
Spin total S = S₁ + S₂
S² = S₁² + S₂² + 2S₁·S₂                                      (B-5)
S₁·S₂ = (1/2)(S₁₊S₂₋ + S₁₋S₂₊) + S₁zS₂z                    (B-6)
```

États propres (base couplée) :
```
|S=1, M=1⟩  = |+,+⟩                                    (triplet)
|S=1, M=0⟩  = (1/√2)(|+,−⟩ + |−,+⟩)                   (triplet)
|S=1, M=−1⟩ = |−,−⟩                                    (triplet)
|S=0, M=0⟩  = (1/√2)(|+,−⟩ − |−,+⟩)                   (singulet)
```
**Source** : [file:1, Chap. X, § B]

#### 1.3.3 Composition générale : coefficients de Clebsch-Gordan

**Énoncé** : Le changement de base entre bases couplée et non couplée fait intervenir les coefficients de Clebsch-Gordan :

```
|j₁, j₂; J, M⟩ = Σ_{m₁, m₂} ⟨j₁, m₁; j₂, m₂|J, M⟩ |j₁, m₁; j₂, m₂⟩    (C-28)
```

Conditions :
```
M = m₁ + m₂
|j₁ − j₂| ≤ J ≤ j₁ + j₂    (règle du triangle)
```
**Source** : [file:1, Chap. X, § C]

**Propriétés des coefficients de Clebsch-Gordan** :
- Réels (avec la convention de Condon-Shortley)
- Relations d'orthogonalité
- Nuls si M ≠ m₁ + m₂
- Tables disponibles dans le complément B_X

#### 1.3.4 Théorème de Wigner-Eckart

**Énoncé** : Les éléments de matrice d'un opérateur vectoriel V dans la base |j₁, j₂; J, M⟩ sont proportionnels aux coefficients de Clebsch-Gordan.

```
⟨j₁, j₂; J', M'|V_q|j₁, j₂; J, M⟩ = ⟨J', M'|J, M; 1, q⟩ * ⟨J'||V||J⟩ / √(2J'+1)
```
**Source** : [file:1, Chap. X, Complément D_X]

Application : facteur de Landé g_J :
```
g_J = 1 + [J(J+1) + S(S+1) − L(L+1)] / [2J(J+1)]
```

---

### 1.4 Théorie des perturbations stationnaires (Chapitre XI)

#### 1.4.1 Position du problème

**Énoncé** : L'hamiltonien H = H₀ + W, où W = λŴ est petit devant H₀ (λ ≪ 1). On développe les valeurs propres et états propres en puissances de λ.

```
H(λ) = H₀ + λŴ                                                 (A-5)
E(λ) = ε₀ + λε₁ + λ²ε₂ + ...                                   (A-7a)
|ψ(λ)⟩ = |0⟩ + λ|1⟩ + λ²|2⟩ + ...                              (A-7b)
```
**Source** : [file:1, Chap. XI, § A]

#### 1.4.2 Perturbation d'un niveau non dégénéré

**Correction à l'énergie (1er ordre)** :
```
E_n^(1) = ⟨ϕ_n|W|ϕ_n⟩                                          (B-5)
```

**Correction à l'état (1er ordre)** :
```
|ψ_n^(1)⟩ = |ϕ_n⟩ + Σ_{p≠n} Σ_i [⟨ϕ_p^i|W|ϕ_n⟩ / (E_n⁰ − E_p⁰)] |ϕ_p^i⟩   (B-11)
```

**Correction à l'énergie (2e ordre)** :
```
E_n^(2) = Σ_{p≠n} Σ_i |⟨ϕ_p^i|W|ϕ_n⟩|² / (E_n⁰ − E_p⁰)       (B-15)
```
**Source** : [file:1, Chap. XI, § B]

**Propriétés** :
- La correction du 1er ordre est la valeur moyenne de W dans l'état non perturbé
- Au 2e ordre, les niveaux se « repoussent » mutuellement
- Majoration : |ε₂| ≤ (1/ΔE) Σ_{p≠n,i} |⟨ϕ_p^i|Ŵ|ϕ_n⟩|²
- Validité : éléments de matrice de W petits devant les écarts d'énergie E_n⁰ − E_p⁰

#### 1.4.3 Perturbation d'un niveau dégénéré

**Énoncé** : Pour un niveau E_n⁰ de dégénérescence g_n, la correction au 1er ordre s'obtient en diagonalisant la matrice g_n × g_n de la perturbation restreinte au sous-espace propre.

```
Matrice W_{ij} = ⟨ϕ_n^i|W|ϕ_n^j⟩,    i,j = 1,...,g_n
```

Les corrections à l'énergie au 1er ordre sont les valeurs propres de cette matrice.
**Source** : [file:1, Chap. XI, § C]

#### 1.4.4 Méthode des variations (Complément E_XI)

**Énoncé** : Pour tout état normé |ψ⟩, ⟨ψ|H|ψ⟩ ≥ E₀ (énergie fondamentale).

```
⟨ψ(α)|H|ψ(α)⟩ ≥ E₀     pour tout paramètre variationnel α
```
On minimise l'énergie moyenne par rapport aux paramètres.
**Source** : [file:1, Chap. XI, Complément E_XI]

---

### 1.5 Structure fine et hyperfine de l'atome d'hydrogène (Chapitre XII)

#### 1.5.1 Hamiltonien de structure fine

**Énoncé** : Les corrections relativistes à l'hamiltonien de l'atome d'hydrogène :

```
H = m_e c² + H₀ + W_mv + W_SO + W_D + ...                      (B-1)

H₀ = P²/(2m_e) + V(R)        (hamiltonien non relativiste)
W_mv = −P⁴/(8m_e³c²)          (variation de masse avec la vitesse)
W_SO = (1/(2m_e²c²)) * (1/R)(dV/dR) * L·S   (couplage spin-orbite)
W_D = (ℏ²/(8m_e²c²)) * ΔV(R)  (terme de Darwin)
```
**Source** : [file:1, Chap. XII, § B-1]

**Ordres de grandeur** :
```
W_mv/H₀ ≈ W_SO/H₀ ≈ W_D/H₀ ≈ α² ≈ (1/137)² ≈ 5 × 10⁻⁵
```

#### 1.5.2 Résultats pour la structure fine de n=2

**Énoncé** : Le niveau n=2 de l'atome d'hydrogène se décompose en :

```
Sous-niveaux de structure fine (dépendent de j, pas de l séparément) :
  j = 1/2 : E_SF(2, j=1/2) = E₂⁰ − (5/128) m_e c² α⁴
  j = 3/2 : E_SF(2, j=3/2) = E₂⁰ − (1/128) m_e c² α⁴
```
**Source** : [file:1, Chap. XII, § C]

#### 1.5.3 Hamiltonien hyperfin

**Énoncé** : L'interaction entre le moment magnétique de l'électron et celui du proton (spin I_p = 1/2) :

```
W_hf = A * S·I     (pour l'état 1s)

avec A = (16/3) E_F = (16/3) g_p (m_e/M_p) α² E_I
```
**Source** : [file:1, Chap. XII, § D]

**Propriétés** :
- W_hf/W_SF ≈ m_e/M_p ≈ 5 × 10⁻⁴
- Fréquence de la transition hyperfine du fondamental 1s : ν ≈ 1420 MHz (raie à 21 cm)

#### 1.5.4 Effet Zeeman

**Énoncé** : En champ magnétique extérieur B₀, l'hamiltonien supplémentaire :

```
W_Z = −(μ_B/ℏ)(L + 2S)·B₀ = −(μ_B/ℏ)(J + S)·B₀
```

Trois régimes selon la force relative de B₀ :
- Champ faible : W_Z ≪ W_SF (effet Zeeman anomal, levée en m_J)
- Champ fort : W_Z ≫ W_SF (effet Paschen-Back)
- Champ intermédiaire : diagonalisation numérique nécessaire
**Source** : [file:1, Chap. XII, § E]

#### 1.5.5 Effet Stark (Complément E_XII)

**Énoncé** : Atome d'hydrogène dans un champ électrique statique ε₀ :

```
W_Stark = qε₀ Z = qε₀ R cos θ
```

- Niveau n=1 : correction au 2e ordre (pas de dégénérescence orbitale exploitable)
- Niveau n=2 : dégénérescence 2s-2p levée au 1er ordre (effet Stark linéaire)
**Source** : [file:1, Chap. XII, Complément E_XII]

---

### 1.6 Perturbations dépendant du temps (Chapitre XIII)

#### 1.6.1 Probabilité de transition au 1er ordre

**Énoncé** : Le système, initialement dans l'état |ϕ_i⟩ de H₀, subit une perturbation W(t) à partir de t=0. La probabilité de transition vers |ϕ_f⟩ au premier ordre :

```
P_{i→f}(t) = (1/ℏ²) |∫₀ᵗ e^(iω_fi t') W_fi(t') dt'|²          (B-24)

avec ω_fi = (E_f − E_i)/ℏ   (pulsation de Bohr)
     W_fi(t) = ⟨ϕ_f|W(t)|ϕ_i⟩
```
**Source** : [file:1, Chap. XIII, § B-3]

#### 1.6.2 Perturbation sinusoïdale

**Énoncé** : Pour W(t) = W sin(ωt) ou W cos(ωt), la probabilité de transition fait intervenir une fonction de résonance :

```
P_{i→f}(t) ∝ |W_fi|² * F(t, ω_fi ± ω)

F(t, Ω) = sin²(Ωt/2) / (Ω/2)²
```

Le maximum de F est atteint pour Ω = 0, soit ω = ±ω_fi (condition de résonance).
**Source** : [file:1, Chap. XIII, § C-1]

**Propriétés** :
- Largeur du pic de résonance : Δω ≈ 2π/t (s'affine aux temps longs)
- Pour ω ≈ ω_fi > 0 : absorption (absorption d'énergie ℏω)
- Pour ω ≈ −ω_fi : émission stimulée

#### 1.6.3 Couplage à un continuum — Règle d'or de Fermi

**Énoncé** : Quand l'état discret |ϕ_i⟩ est couplé à un continuum d'états finals de densité ρ(E_f), le taux de transition constant aux temps longs :

```
Γ_{i→f} = dP/dt = (2π/ℏ) |W_fi|² ρ(E_f = E_i + ℏω)            (C-39)
```
**Source** : [file:1, Chap. XIII, § C-3]

**Propriétés** :
- Le taux est proportionnel à |W_fi|² et à la densité d'états au niveau de résonance
- Valide aux temps longs (t ≫ 1/ΔE où ΔE est la largeur du continuum)
- La probabilité croît linéairement : P(t) = Γ * t

#### 1.6.4 Perturbation aléatoire

**Énoncé** : Pour une perturbation aléatoire caractérisée par sa fonction de corrélation :

```
g(τ) = ⟨W(t) W(t+τ)⟩    (moyenne d'ensemble)
```

Aux temps courts, le taux de transition :
```
Γ_{i→f} = (1/ℏ²) ∫₋∞^∞ g_fi(τ) e^(iω_fi τ) dτ
```
**Source** : [file:1, Chap. XIII, § D]

#### 1.6.5 Relaxation (Complément E_XIII)

**Énoncé** : Pour un ensemble de spins 1/2 soumis à une perturbation aléatoire, l'évolution de l'opérateur densité fait apparaître deux temps de relaxation :

```
T₁ : temps de relaxation longitudinale (retour à l'équilibre de ⟨S_z⟩)
T₂ : temps de relaxation transversale (amortissement de ⟨S_x⟩, ⟨S_y⟩)
```
**Source** : [file:1, Chap. XIII, Complément E_XIII]

**LIMITE** : Le traitement complet de la relaxation nécessite la condition de rétrécissement par le mouvement (ω_c * τ_c ≪ 1 où τ_c est le temps de corrélation).

---

### 1.7 Systèmes de particules identiques (Chapitre XIV)

#### 1.7.1 Postulat de symétrisation

**Énoncé** : Les kets physiques d'un système de N particules identiques sont :
- **Complètement symétriques** pour les **bosons** (spin entier)
- **Complètement antisymétriques** pour les **fermions** (spin demi-entier)

```
Bosons : |ψ_phys⟩ ∈ E_S    (S|ψ_phys⟩ = |ψ_phys⟩)
Fermions : |ψ_phys⟩ ∈ E_A   (P₂₁|ψ_phys⟩ = −|ψ_phys⟩)
```

Opérateurs de symétrisation/antisymétrisation :
```
S = (1/N!) Σ_α P_α                                              (B-49)
A = (1/N!) Σ_α ε_α P_α                                          (B-50)
```
**Source** : [file:1, Chap. XIV, § C-1]

**Règle spin-statistique** : particules de spin demi-entier → fermions, spin entier → bosons.

#### 1.7.2 Principe d'exclusion de Pauli

**Énoncé** : Deux fermions identiques ne peuvent occuper le même état quantique individuel.

Pour N fermions, le ket physique est un déterminant de Slater :
```
|ψ⟩ = (1/√N!) * det[|i : ϕ_α⟩]     (i = particule, α = état)
```
Si deux états ϕ_α coïncident, le déterminant est nul.
**Source** : [file:1, Chap. XIV, § C-3]

#### 1.7.3 Effet de l'identité sur les observables

**Énoncé** : Seules les observables symétriques (commutant avec toutes les permutations) sont physiquement observables.

```
∀α : [O, P_α] = 0     (O observable physique)
```

Pour deux particules identiques dans les états |ϕ⟩ et |χ⟩ orthogonaux :
```
Bosons  : |ψ⟩ = (1/√2)(|1:ϕ; 2:χ⟩ + |1:χ; 2:ϕ⟩)
Fermions: |ψ⟩ = (1/√2)(|1:ϕ; 2:χ⟩ − |1:χ; 2:ϕ⟩)
```

L'effet d'échange modifie la section efficace de diffusion de deux particules identiques :
```
σ_bosons(θ) ∝ |f(θ) + f(π−θ)|²       (interférence constructive)
σ_fermions(θ) ∝ |f(θ) − f(π−θ)|²     (interférence destructive)
σ_classique(θ) ∝ |f(θ)|² + |f(π−θ)|²  (pas d'interférence)
```
**Source** : [file:1, Chap. XIV, § D-2]

### 1.8 Notations et conventions

| Symbole | Signification |
|---------|---------------|
| ℏ | Constante de Planck réduite |
| μ | Masse réduite μ = m₁m₂/(m₁+m₂) |
| k | Vecteur d'onde, |k| = √(2μE)/ℏ |
| δ_l | Déphasage de l'onde partielle l |
| Y_l^m(θ,ϕ) | Harmonique sphérique |
| σ_i | Matrices de Pauli |
| α = e²/(ℏc) ≈ 1/137 | Constante de structure fine |
| μ_B = eℏ/(2m_e) | Magnéton de Bohr |
| a₀ = ℏ²/(m_e e²) | Rayon de Bohr |
| ⟨j₁,m₁;j₂,m₂\|J,M⟩ | Coefficient de Clebsch-Gordan |
| S, A | Symétriseur, antisymétriseur |
| ε_α | Signature de la permutation P_α |

---

## 2. Règles physiques implémentables

### R6.x : Diffusion quantique (Chapitre VIII)

**R6.1** — Section efficace depuis amplitude de diffusion
- **Énoncé** : σ(θ,ϕ) = |f_k(θ,ϕ)|²
- **Formulation** : `sigma[i,j] = abs(f_k[i,j])**2`
- **Source** : [file:1, Chap. VIII, § B-2, eq. B-24]
- **Contrainte numérique** : σ ≥ 0, intégrale sur dΩ donne σ_tot
- **Invariant** : théorème optique σ_tot = (4π/k) Im(f_k(θ=0))

**R6.2** — Approximation de Born
- **Énoncé** : f_k^(B) = −(μ/(2πℏ²)) * TF[V](K)
- **Formulation** : `f_Born = -(mu/(2*pi*hbar**2)) * FFT(V)(K)` avec |K| = 2k sin(θ/2)
- **Source** : [file:1, Chap. VIII, § B-4, eq. B-47/B-48]
- **Contrainte numérique** : Validité : énergie suffisamment grande ou potentiel faible
- **Invariant** : pour V réel symétrique, f_Born est réel ⇒ théorème optique violé (correction d'ordre supérieur)

**R6.3** — Déphasages et section efficace
- **Énoncé** : σ_tot = (4π/k²) Σ_l (2l+1) sin²(δ_l)
- **Formulation** : `sigma_tot = (4*pi/k**2) * sum((2*l+1)*sin(delta_l)**2 for l in range(l_max))`
- **Source** : [file:1, Chap. VIII, § C-4, eq. C-58]
- **Contrainte numérique** : Troncature à l_max tel que δ_l ≈ 0 pour l > l_max ; vérifier convergence
- **Invariant** : 0 ≤ sin²(δ_l) ≤ 1 pour tout l ; limite unitaire σ_l ≤ 4π(2l+1)/k²

**R6.4** — Calcul des déphasages (équation radiale)
- **Énoncé** : La partie radiale u_l(r) = r R_l(r) vérifie l'équation radiale avec conditions aux limites
- **Formulation** :
```
u_l''(r) + [k² − l(l+1)/r² − U(r)] u_l(r) = 0
u_l(r) →_{r→∞} A_l sin(kr − lπ/2 + δ_l)
```
- **Source** : [file:1, Chap. VIII, § C-3]
- **Contrainte numérique** : u_l(0) = 0 ; intégration depuis r=0 ; extraction de δ_l par raccordement asymptotique

### R7.x : Spin (Chapitre IX)

**R7.1** — Représentation matricielle du spin 1/2
- **Énoncé** : S = (ℏ/2)σ avec σ les matrices de Pauli
- **Formulation** :
```
S_x = (hbar/2)*np.array([[0,1],[1,0]])
S_y = (hbar/2)*np.array([[0,-1j],[1j,0]])
S_z = (hbar/2)*np.array([[1,0],[0,-1]])
```
- **Source** : [file:1, Chap. IX, § B]
- **Contrainte numérique** : S² = (3/4)ℏ² * I ; hermiticité ; Tr(S_i) = 0

**R7.2** — Opérateur densité de spin
- **Énoncé** : ρ = (1/2)(I + P·σ) avec P le vecteur de polarisation
- **Source** : [file:1, Chap. IX, Complément E_IV rappelé via Chap. IV]
- **Contrainte numérique** : |P| ≤ 1 ; Tr(ρ) = 1 ; ρ ≥ 0

### R8.x : Composition des moments cinétiques (Chapitre X)

**R8.1** — Clebsch-Gordan et changement de base
- **Énoncé** : |j₁,j₂;J,M⟩ = Σ_{m₁+m₂=M} C(j₁,m₁;j₂,m₂;J,M) |j₁,m₁⟩⊗|j₂,m₂⟩
- **Source** : [file:1, Chap. X, § C, eq. C-28]
- **Contrainte numérique** : Unitarité du changement de base ; |j₁−j₂| ≤ J ≤ j₁+j₂ ; M = m₁+m₂

**R8.2** — Relations de récurrence pour les Clebsch-Gordan
- **Énoncé** : Application de J± aux deux membres de la relation de couplage
- **Source** : [file:1, Chap. X, § C-3]
- **Contrainte numérique** : Normalisation Σ_{m₁,m₂} |C|² = 1 ; coefficients réels (convention Condon-Shortley)

**R8.3** — Facteur de Landé
- **Énoncé** : g_J = 1 + [J(J+1)+S(S+1)−L(L+1)]/(2J(J+1))
- **Source** : [file:1, Chap. X, Complément D_X]
- **Contrainte numérique** : g_J ∈ [0, 2] typiquement

### R9.x : Perturbations stationnaires (Chapitre XI)

**R9.1** — Correction énergie 1er ordre (non dégénéré)
- **Énoncé** : E_n^(1) = ⟨ϕ_n|W|ϕ_n⟩
- **Source** : [file:1, Chap. XI, § B-1, eq. B-5]
- **Contrainte numérique** : Réel (W hermitique)

**R9.2** — Correction énergie 2e ordre (non dégénéré)
- **Énoncé** : E_n^(2) = Σ_{p≠n} |⟨ϕ_p|W|ϕ_n⟩|²/(E_n⁰−E_p⁰)
- **Source** : [file:1, Chap. XI, § B-2, eq. B-15]
- **Contrainte numérique** : E_n⁰ − E_p⁰ ≠ 0 (niveau non dégénéré) ; somme convergente

**R9.3** — Correction état 1er ordre (non dégénéré)
- **Énoncé** : |ψ_n^(1)⟩ = |ϕ_n⟩ + Σ_{p≠n} [⟨ϕ_p|W|ϕ_n⟩/(E_n⁰−E_p⁰)] |ϕ_p⟩
- **Source** : [file:1, Chap. XI, § B-1, eq. B-11]
- **Contrainte numérique** : Re-normaliser le vecteur corrigé

**R9.4** — Perturbation dégénérée (1er ordre)
- **Énoncé** : Diagonaliser la matrice W_{ij} = ⟨ϕ_n^i|W|ϕ_n^j⟩ dans le sous-espace dégénéré
- **Source** : [file:1, Chap. XI, § C]
- **Contrainte numérique** : Matrice hermitique → valeurs propres réelles

**R9.5** — Méthode des variations
- **Énoncé** : Minimiser ⟨ψ(α)|H|ψ(α)⟩ par rapport aux paramètres variationnels α
- **Source** : [file:1, Chap. XI, Complément E_XI]
- **Contrainte numérique** : ⟨ψ|H|ψ⟩ ≥ E₀ (borne inférieure) ; gradient = 0 au minimum

### R10.x : Structure fine et hyperfine (Chapitre XII)

**R10.1** — Structure fine de H (n quelconque)
- **Énoncé** : E_SF(n,j) = E_n⁰ + (α²/n²)E_n⁰ [n/(j+1/2) − 3/4]     (formule de Dirac approchée)
- **Source** : [file:1, Chap. XII, § C]
- **Contrainte numérique** : j = l ± 1/2 ; la correction ne dépend que de n et j (pas de l individuellement)

**R10.2** — Structure hyperfine du niveau 1s
- **Énoncé** : E_hf = A/2 [F(F+1) − I(I+1) − S(S+1)] avec F = I + S
- **Source** : [file:1, Chap. XII, § D]
- **Contrainte numérique** : F = 0 ou F = 1 pour l'hydrogène (I = S = 1/2)

**R10.3** — Effet Zeeman (champ faible)
- **Énoncé** : δE = g_F μ_B m_F B₀
- **Source** : [file:1, Chap. XII, § E]

### R11.x : Perturbations dépendant du temps (Chapitre XIII)

**R11.1** — Probabilité de transition (1er ordre)
- **Énoncé** : P_{i→f}(t) = (1/ℏ²)|∫₀ᵗ e^{iω_fi t'} W_fi(t') dt'|²
- **Source** : [file:1, Chap. XIII, § B-3, eq. B-24]
- **Contrainte numérique** : P ∈ [0,1] ; si P → 1, l'approximation du 1er ordre est violée

**R11.2** — Règle d'or de Fermi
- **Énoncé** : Γ = (2π/ℏ)|W_fi|² ρ(E_f)
- **Source** : [file:1, Chap. XIII, § C-3, eq. C-39]
- **Contrainte numérique** : ρ(E_f) ≥ 0 ; Γ * t ≪ 1 pour rester dans le cadre perturbatif

**R11.3** — Oscillations de Rabi (résonance exacte)
- **Énoncé** : P_{1→2}(t) = sin²(Ω_R t/2) avec Ω_R = |W₁₂|/ℏ (pulsation de Rabi)
- **Source** : [file:1, Chap. XIII, Complément C_XIII]
- **Contrainte numérique** : Valide hors perturbation ; P oscille entre 0 et 1

### R12.x : Particules identiques (Chapitre XIV)

**R12.1** — Symétrisation des états
- **Énoncé** : |ψ_phys⟩ = N * S|ψ⟩ (bosons) ou N * A|ψ⟩ (fermions)
- **Source** : [file:1, Chap. XIV, § C-3]
- **Contrainte numérique** : vérifier que P₂₁|ψ⟩ = ε|ψ⟩ avec ε = +1 (bosons) ou −1 (fermions)

**R12.2** — Déterminant de Slater
- **Énoncé** : Pour N fermions : |ψ⟩ = (1/√N!) det[ϕ_α(r_i)]
- **Source** : [file:1, Chap. XIV, § C-3-c]
- **Contrainte numérique** : Orthogonalité des états individuels → normalisation automatique

**R12.3** — Section efficace avec particules identiques
- **Énoncé** : σ = |f(θ) ± f(π−θ)|² (+: bosons, −: fermions)
- **Source** : [file:1, Chap. XIV, § D-2]
- **Contrainte numérique** : σ(θ=π/2) = 0 pour fermions identiques de même spin

---

## 3. Traduction logicielle des concepts physiques

### 3.1 Correspondances fondamentales

| Concept physique | Module | Classe | Responsabilité |
|---|---|---|---|
| Diffusion élastique | `systems/` | `ScatteringSystem` | Potentiel V(r), masse réduite μ, énergie E |
| Amplitude de diffusion | `systems/` | `ScatteringAmplitude` | f_k(θ,ϕ), calcul via Born ou déphasages |
| Section efficace | `systems/` | `CrossSection` | σ(θ,ϕ), σ_tot, σ_transport |
| Déphasage δ_l | `dynamics/` | `PhaseShiftSolver` | Résolution équation radiale, extraction δ_l |
| Approximation de Born | `dynamics/` | `BornApproximation` | TF du potentiel, amplitude 1er/2e ordre |
| Spin 1/2 | `core/` | `SpinHalf` | Matrices de Pauli, opérateurs de spin |
| Spineur | `core/` | `Spinor` | État spin+espace, produit tensoriel |
| Clebsch-Gordan | `core/` | `ClebschGordan` | Coefficients CG, changement de base |
| Moment cinétique total | `core/` | `AngularMomentumCoupling` | J = J₁ + J₂, bases couplée/non couplée |
| Perturbation stationnaire | `dynamics/` | `StationaryPerturbation` | Corrections ordre 1 et 2, dégénéré/non dégénéré |
| Méthode des variations | `dynamics/` | `VariationalMethod` | Minimisation énergie, paramètres variationnels |
| Structure fine H | `systems/` | `HydrogenFineSructure` | W_mv, W_SO, W_D, niveaux (n,l,j) |
| Structure hyperfine H | `systems/` | `HydrogenHyperfine` | W_hf, niveaux (n,l,j,F), fréquence 21 cm |
| Effet Zeeman | `systems/` | `ZeemanEffect` | W_Z, diagramme Zeeman, 3 régimes |
| Perturbation temporelle | `dynamics/` | `TimeDependentPerturbation` | P_{i→f}(t), perturbation sinusoïdale/constante |
| Règle d'or de Fermi | `dynamics/` | `FermiGoldenRule` | Taux Γ, densité d'états, couplage au continuum |
| Relaxation | `dynamics/` | `Relaxation` | T₁, T₂, équations de Bloch, opérateur densité |
| Symétrisation | `core/` | `Symmetrizer` | S, A, construction kets physiques |
| Déterminant de Slater | `core/` | `SlaterDeterminant` | N fermions, antisymétrisation, Pauli |
| Particules identiques | `systems/` | `IdenticalParticles` | Correction section efficace, échange |

### 3.2 Hypothèses numériques fondamentales

| Décision | Paramètre YAML | Justification | Impact | Test |
|---|---|---|---|---|
| Troncature ondes partielles | `scattering.l_max` | δ_l → 0 pour l ≫ k*r₀ | Précision σ_tot | Convergence σ_tot vs l_max |
| Grille radiale pour déphasages | `scattering.r_max, dr` | r_max ≫ portée potentiel | Extraction δ_l | Comparer δ_l(r_max) et δ_l(2*r_max) |
| Grille angulaire FFT (Born) | `scattering.N_theta` | Résolution angulaire σ(θ) | Intégrale dΩ | σ_tot par intégration = Σ partielle |
| Dimension base Clebsch-Gordan | `angular.j_max` | j₁+j₂ ≤ j_max | Taille mémoire | Unitarité matrice CG |
| Troncature somme perturbative | `perturbation.n_states` | Convergence correction 2e ordre | Précision E^(2) | Comparer avec valeur exacte (si connue) |
| Pas de temps intégration | `time_dep.dt` | dt ≪ ℏ/|W_fi| | Stabilité P(t) | P(t) ∈ [0,1] ; conservation probabilité |
| Nombre d'orbitales (Slater) | `identical.n_orbitals` | N fermions dans N orbitales | Taille déterminant | Normalisation ⟨ψ|ψ⟩ = 1 |

### 3.3 Séparation des responsabilités

**`core/`** : Objets mathématiques fondamentaux
- Contient : SpinHalf, Spinor, ClebschGordan, AngularMomentumCoupling, Symmetrizer, SlaterDeterminant
- Ne contient PAS : systèmes physiques spécifiques, évolution temporelle
- Dépendances : numpy, scipy.special (harmoniques sphériques, fonctions de Bessel)

**`dynamics/`** : Méthodes de résolution et évolution
- Contient : PhaseShiftSolver, BornApproximation, StationaryPerturbation, VariationalMethod, TimeDependentPerturbation, FermiGoldenRule, Relaxation
- Ne contient PAS : potentiels physiques concrets, constantes physiques
- Dépendances : core/, scipy.integrate, scipy.linalg

**`systems/`** : Systèmes physiques concrets
- Contient : ScatteringSystem, HydrogenFineStructure, HydrogenHyperfine, ZeemanEffect, IdenticalParticles
- Ne contient PAS : algorithmes génériques
- Dépendances : core/, dynamics/

**`experiments/`** : Protocoles de simulation complets
- Contient : ScatteringExperiment, PerturbationExperiment, SpinDynamicsExperiment, IdenticalParticlesExperiment
- Ne contient PAS : calculs bas niveau
- Dépendances : systems/, dynamics/, validation/

**`validation/`** : Tests physiques
- Contient : vérificateurs d'invariants physiques
- Dépendances : core/

**`utils/`** : Outils transverses
- Contient : constantes, chargement YAML, visualisation
- Dépendances : aucune vers les autres modules

---

## 4. Architecture logicielle globale

### 4.1 Organisation dossiers

```
quantum_mechanics_tome2/
├── core/
│   ├── __init__.py
│   ├── spin.py                    # SpinHalf, SpinOperators, Spinor
│   ├── angular_momentum.py        # ClebschGordan, AngularMomentumCoupling
│   ├── symmetry.py                # Symmetrizer, AntiSymmetrizer, SlaterDeterminant
│   └── operators.py               # Extension opérateurs (commutateurs, produit tensoriel)
├── dynamics/
│   ├── __init__.py
│   ├── scattering.py              # PhaseShiftSolver, BornApproximation, CrossSection
│   ├── stationary_perturbation.py # StationaryPerturbation (dégénéré + non dégénéré)
│   ├── variational.py             # VariationalMethod
│   ├── time_dependent.py          # TimeDependentPerturbation, FermiGoldenRule
│   └── relaxation.py              # Relaxation, BlochEquations
├── systems/
│   ├── __init__.py
│   ├── scattering_potentials.py   # Yukawa, sphère dure, carré, coulombien écranté
│   ├── hydrogen.py                # HydrogenAtom, HydrogenFineStructure, HydrogenHyperfine
│   ├── zeeman.py                  # ZeemanEffect (faible, fort, intermédiaire)
│   ├── stark.py                   # StarkEffect
│   ├── two_level.py               # TwoLevelSystem, RabiOscillations
│   └── identical_particles.py     # IdenticalParticles, FreeElectronGas
├── experiments/
│   ├── __init__.py
│   ├── scattering_experiment.py   # Exp. complète de diffusion
│   ├── perturbation_experiment.py # Exp. perturbation stationnaire
│   ├── zeeman_experiment.py       # Diagramme Zeeman
│   ├── rabi_experiment.py         # Oscillations de Rabi
│   └── fermi_gas_experiment.py    # Gaz d'électrons libres
├── validation/
│   ├── __init__.py
│   ├── invariants.py              # Tests physiques (normalisation, hermiticité, etc.)
│   └── known_results.py           # Résultats analytiques connus pour comparaison
├── utils/
│   ├── __init__.py
│   ├── constants.py               # Constantes physiques
│   ├── config.py                  # Chargement YAML
│   └── visualization.py           # Plots
├── parameters.yaml
├── tests/
│   ├── test_spin.py
│   ├── test_clebsch_gordan.py
│   ├── test_scattering.py
│   ├── test_perturbation.py
│   ├── test_hydrogen.py
│   ├── test_time_dependent.py
│   └── test_identical.py
└── README.md
```

### 4.2 Flux de dépendances autorisées

```
utils/  ←──────────────────────────────────────────┐
  │                                                  │
  v                                                  │
core/  ←────────────────────────────────────────┐    │
  │                                              │    │
  v                                              │    │
dynamics/ ←─────────────────────────────────┐    │    │
  │                                          │    │    │
  v                                          │    │    │
systems/  ←──────────────────────────────┐   │    │    │
  │                                      │   │    │    │
  v                                      │   │    │    │
validation/ ←──────────────────────┐     │   │    │    │
  │                                │     │   │    │    │
  v                                │     │   │    │    │
experiments/ ──────────────────────┴─────┴───┴────┴────┘
```

Règle stricte : pas de dépendance inverse (experiments/ n'importe jamais dans core/).

### 4.3 Points d'entrée

- **Script** : `python -m experiments.scattering_experiment --config parameters.yaml`
- **Interactif** : Jupyter notebook important les modules
- **Tests** : `pytest tests/`

---

## 5. État actuel de l'implémentation

### 5.1 Implémenté (structure définie)

Tous les modules sont **À IMPLÉMENTER**. Ce document fournit les spécifications complètes.

### 5.2 Prévu mais non codé

Tous les éléments listés dans la section 4.1.

### 5.3 Interfaces décidées (signatures clés)

```python
# === core/spin.py ===
import numpy as np
from typing import Tuple

class SpinHalf:
    """Spin 1/2 — Matrices de Pauli et opérateurs. [R7.1]"""
    def __init__(self, hbar: float = 1.0):
        self.hbar = hbar
    def sigma_x(self) -> np.ndarray: ...   # Matrice 2x2
    def sigma_y(self) -> np.ndarray: ...
    def sigma_z(self) -> np.ndarray: ...
    def S_plus(self) -> np.ndarray: ...
    def S_minus(self) -> np.ndarray: ...
    def S_squared(self) -> np.ndarray: ...

class Spinor:
    """Spineur à deux composantes. [R7.1]"""
    def __init__(self, up: complex, down: complex): ...
    def density_matrix(self) -> np.ndarray: ...
    def polarization_vector(self) -> np.ndarray: ...

# === core/angular_momentum.py ===
class ClebschGordan:
    """Coefficients de Clebsch-Gordan. [R8.1, R8.2]"""
    @staticmethod
    def coefficient(j1: float, m1: float, j2: float, m2: float,
                    J: float, M: float) -> float: ...
    @staticmethod
    def coupled_basis(j1: float, j2: float) -> dict: ...

class AngularMomentumCoupling:
    """Composition de deux moments cinétiques. [R8.1]"""
    def __init__(self, j1: float, j2: float): ...
    def J_range(self) -> list: ...
    def change_of_basis_matrix(self) -> np.ndarray: ...

# === dynamics/scattering.py ===
class PhaseShiftSolver:
    """Calcul des déphasages par intégration radiale. [R6.4]"""
    def __init__(self, potential: callable, mu: float, E: float,
                 r_max: float, dr: float): ...
    def compute_phase_shift(self, l: int) -> float: ...
    def compute_all_phase_shifts(self, l_max: int) -> np.ndarray: ...

class BornApproximation:
    """Approximation de Born pour la diffusion. [R6.2]"""
    def __init__(self, potential: callable, mu: float, hbar: float = 1.0): ...
    def amplitude(self, k: float, theta: float) -> complex: ...
    def cross_section(self, k: float, theta: np.ndarray) -> np.ndarray: ...
    def total_cross_section(self, k: float) -> float: ...

class CrossSection:
    """Section efficace depuis déphasages ou Born. [R6.1, R6.3]"""
    def __init__(self, k: float, phase_shifts: np.ndarray = None,
                 born_amplitude: callable = None): ...
    def differential(self, theta: np.ndarray) -> np.ndarray: ...
    def total(self) -> float: ...
    def verify_optical_theorem(self) -> float: ...

# === dynamics/stationary_perturbation.py ===
class StationaryPerturbation:
    """Théorie des perturbations stationnaires. [R9.1-R9.4]"""
    def __init__(self, H0_eigenvalues: np.ndarray,
                 H0_eigenstates: np.ndarray,
                 W_matrix: np.ndarray): ...
    def energy_correction_first_order(self, n: int) -> float: ...
    def energy_correction_second_order(self, n: int) -> float: ...
    def state_correction_first_order(self, n: int) -> np.ndarray: ...
    def degenerate_first_order(self, n: int,
                                degeneracy_indices: list) -> Tuple[np.ndarray, np.ndarray]: ...

# === dynamics/time_dependent.py ===
class TimeDependentPerturbation:
    """Perturbation dépendant du temps — 1er ordre. [R11.1]"""
    def __init__(self, H0_eigenvalues: np.ndarray, W_func: callable): ...
    def transition_probability(self, i: int, f: int, t: float) -> float: ...
    def transition_amplitude(self, i: int, f: int, t: float) -> complex: ...

class FermiGoldenRule:
    """Règle d'or de Fermi. [R11.2]"""
    def __init__(self, W_fi: complex, density_of_states: callable): ...
    def transition_rate(self, E_f: float) -> float: ...

# === core/symmetry.py ===
class Symmetrizer:
    """Symétrisation/antisymétrisation. [R12.1]"""
    @staticmethod
    def symmetrize(state: np.ndarray, n_particles: int) -> np.ndarray: ...
    @staticmethod
    def antisymmetrize(state: np.ndarray, n_particles: int) -> np.ndarray: ...

class SlaterDeterminant:
    """Déterminant de Slater pour N fermions. [R12.2]"""
    def __init__(self, orbitals: list): ...
    def evaluate(self, *positions) -> complex: ...
    def overlap(self, other: 'SlaterDeterminant') -> float: ...
```

---

## 6. Gestion des expériences et simulations

### 6.1 Cycle type d'une expérience

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ PRÉPARATION  │───>│  DÉFINITION  │───>│  ÉVOLUTION   │───>│   MESURE     │───>│  ANALYSE     │
│              │    │   SYSTÈME    │    │              │    │              │    │              │
│ Charger YAML │    │ Hamiltonien  │    │ Résoudre éq. │    │ Calculer     │    │ Comparer     │
│ Vérifier     │    │ Base, espace │    │ ou appliquer │    │ observables  │    │ avec théorie │
│ paramètres   │    │ État initial │    │ perturbation │    │ probabilités │    │ Exporter     │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### 6.2 Contraintes par étape

| Étape | Autorisé | Obligatoire | Interdit |
|-------|----------|-------------|----------|
| Préparation | Lire YAML, créer objets | Valider tous les paramètres | Calculs physiques |
| Définition système | Construire H₀, W, base | Vérifier hermiticité de H | Modifier paramètres |
| Évolution | Appliquer dynamique | Vérifier conservation énergie/norme | Changer le système |
| Mesure | Calculer ⟨O⟩, P(a_n) | Vérifier Σ P = 1 | Modifier l'état |
| Analyse | Comparer, visualiser | Appeler validation/ | Extrapoler hors cadre |

### 6.3 Exemples concrets

#### Exemple 1 : Diffusion par un potentiel de Yukawa

**Objectif** : Calculer la section efficace différentielle et totale par déphasages et Born.

**Étapes** :
1. Charger paramètres (masse, énergie, portée potentiel)
2. Définir V(r) = V₀ exp(−r/a)/r
3. Résoudre l'équation radiale pour l = 0, 1, ..., l_max → δ_l
4. Calculer σ(θ) et σ_tot via déphasages
5. Comparer avec σ_Born

```yaml
scattering:
  potential: "yukawa"
  V0: -50.0           # MeV
  range_a: 1.0         # fm
  mu: 469.0            # MeV/c² (masse réduite proton-proton)
  E: 10.0              # MeV (énergie CM)
  l_max: 10
  r_max: 20.0          # fm
  dr: 0.01             # fm
  N_theta: 180
```

#### Exemple 2 : Structure fine du niveau n=2 de l'hydrogène

**Objectif** : Calculer les corrections de structure fine au niveau n=2.

**Étapes** :
1. Construire la base |n=2, l, m_l, m_s⟩ (8 états)
2. Calculer les matrices W_mv, W_SO, W_D dans cette base
3. Diagonaliser W = W_mv + W_SO + W_D
4. Vérifier que les niveaux ne dépendent que de j

```yaml
hydrogen:
  n: 2
  fine_structure: true
  hyperfine_structure: false
  Z: 1
  corrections:
    - "mass_velocity"
    - "spin_orbit"
    - "darwin"
```

#### Exemple 3 : Oscillations de Rabi

**Objectif** : Simuler les oscillations entre deux niveaux sous perturbation sinusoïdale résonnante.

**Étapes** :
1. Système à deux niveaux, écart ℏω₀
2. Perturbation W(t) = W₀ cos(ωt), ω ≈ ω₀
3. Calculer P₁₂(t) en fonction du temps
4. Comparer avec formule de Rabi exacte

```yaml
rabi:
  E1: 0.0
  E2: 1.0              # ℏω₀ = 1
  W12: 0.1              # couplage
  omega_drive: 1.0      # résonance
  t_max: 100.0
  dt: 0.01
```

---

## 7. Configuration et paramètres (parameters.yaml)

### 7.1 Rôle du fichier

Code = logique physique et algorithmes. YAML = toutes les valeurs numériques.

### 7.2 Structure décidée

```yaml
# === CONSTANTES PHYSIQUES ===
constants:
  hbar: 1.0545718e-34      # J·s
  m_e: 9.1093837e-31       # kg
  e_charge: 1.6021766e-19  # C
  c: 2.9979246e8           # m/s
  alpha: 7.2973526e-3      # constante de structure fine
  mu_B: 9.2740100e-24      # J/T (magnéton de Bohr)
  a_0: 5.2917721e-11       # m (rayon de Bohr)
  m_p: 1.6726219e-27       # kg (masse proton)

# === PARAMÈTRES NUMÉRIQUES GÉNÉRAUX ===
numerical:
  default_hbar: 1.0        # unités naturelles par défaut
  tolerance: 1.0e-10       # tolérance convergence
  max_iterations: 1000

# === DIFFUSION (Chapitre VIII) ===
scattering:
  potential: "yukawa"       # yukawa, hard_sphere, square_well, gaussian
  V0: -50.0
  range_a: 1.0
  mu: 1.0
  E: 10.0
  l_max: 20
  r_max: 50.0
  dr: 0.01
  N_theta: 360
  born_order: 1             # 1 ou 2

# === SPIN (Chapitre IX) ===
spin:
  particle: "electron"
  s: 0.5
  g_factor: 2.0023          # facteur g de l'électron

# === COMPOSITION MOMENTS CINÉTIQUES (Chapitre X) ===
angular_momentum:
  j1: 0.5
  j2: 0.5
  j_max: 10                 # troncature pour tables CG

# === PERTURBATIONS STATIONNAIRES (Chapitre XI) ===
perturbation_stationary:
  n_states: 50              # nombre d'états dans la somme perturbative
  order: 2                  # ordre du calcul (1 ou 2)
  degenerate_threshold: 1.0e-8  # seuil pour considérer un niveau dégénéré

# === HYDROGÈNE (Chapitre XII) ===
hydrogen:
  n_max: 4
  Z: 1
  fine_structure: true
  hyperfine_structure: true
  zeeman:
    B_field: 0.0            # Tesla
    regime: "auto"          # auto, weak, strong, intermediate
  stark:
    E_field: 0.0            # V/m

# === PERTURBATIONS TEMPORELLES (Chapitre XIII) ===
time_dependent:
  dt: 0.01
  t_max: 100.0
  perturbation_type: "sinusoidal"  # sinusoidal, constant, pulse, random
  omega: 1.0
  amplitude: 0.1
  # Relaxation
  relaxation:
    T1: 1.0
    T2: 0.5
    tau_c: 0.01             # temps de corrélation

# === PARTICULES IDENTIQUES (Chapitre XIV) ===
identical_particles:
  particle_type: "fermion"  # fermion ou boson
  N: 2                      # nombre de particules
  n_orbitals: 10
  # Gaz d'électrons
  electron_gas:
    L: 1.0                  # taille boîte
    N_electrons: 10
    temperature: 0.0        # K

# === VISUALISATION ===
visualization:
  dpi: 150
  figsize: [10, 7]
  style: "seaborn-v0_8"
  save_format: "png"
```

### 7.3 Catégories

- **OBLIGATOIRE** : `constants`, `numerical.tolerance`, choix de `potential`/`particle_type` pour chaque expérience
- **OPTIONNEL** : `visualization`, paramètres de convergence (l_max, n_states...)
- **INTERDIT dans YAML** : formules physiques, logique algorithmique, chemins de fichiers

---

## 8. Limites actuelles et points ouverts

### 8.1 Limites théoriques

| Problème | Impact | Dans le tome? |
|----------|--------|---------------|
| Diffusion coulombienne exclue | Pas de diffusion Rutherford | Mentionné, non traité |
| Pas de 2e quantification | Fermions uniquement par Slater | Renvoyé au Tome III |
| Effet Lamb absent | Pas de décalage 2s₁/₂ – 2p₁/₂ | Mentionné, QED nécessaire |
| Correction ordre 3+ perturbation | Limité à l'ordre 2 | Formule générale donnée mais non développée |
| Perturbation aléatoire : validité temps longs | Condition de rétrécissement nécessaire | Traitée au § E du Chap. XIII |

### 8.2 Limites numériques

| Choix | Alternative non implémentée |
|-------|----------------------------|
| Intégration radiale Numerov | Matrice de transfert, méthode WKB |
| Diagonalisation exacte | DMRG, Monte Carlo quantique |
| FFT pour Born | Intégration adaptative |
| Runge-Kutta pour perturbation temp. | Split-operator, Crank-Nicolson |

### 8.3 Points ouverts

1. **Unités** : Travailler en SI ou unités atomiques ?
   - Options : SI (plus explicite), atomiques (ℏ=m_e=e=1, simplifie les formules)
   - Recommandation : unités atomiques en interne, conversion en sortie

2. **Couplage spin-orbite pour N électrons** : Comment généraliser W_SO au-delà de l'hydrogène ?
   - Le tome ne donne que le cas monoélectronique en détail
   - Recommandation : utiliser l'approximation du champ central (Complément A_XIV)

3. **Convergence des déphasages** : Quel critère d'arrêt pour l_max ?
   - Options : |δ_{l_max}| < ε, contribution relative < ε
   - Recommandation : |δ_l| < 10⁻⁸ rad

### 8.4 Extensions futures

| Extension | Requis | Bloqueurs | Effort |
|-----------|--------|-----------|--------|
| Diffusion inélastique | Canaux couplés, matrice S | Complexité N-corps | Élevé |
| Atome d'hélium complet | Intégrales à 6D, échange | Performance | Moyen |
| QED (Lamb shift) | Quantification du champ EM | Hors Tome II | Hors périmètre |
| Seconde quantification | Opérateurs création/annihilation | Tome III | Élevé |
| N fermions (Hartree-Fock) | Itérations SCF | Tome III | Moyen |

---

## 9. Références traçabilité

### 9.1 Tableau correspondance Règles ↔ Sources cours

| Règle | Description courte | Source cours |
|-------|-------------------|-------------|
| R6.1 | σ = \|f\|² | [file:1, Chap. VIII, § B-2, eq. B-24] |
| R6.2 | Approximation de Born | [file:1, Chap. VIII, § B-4, eq. B-47] |
| R6.3 | σ_tot via déphasages | [file:1, Chap. VIII, § C-4, eq. C-58] |
| R6.4 | Équation radiale pour δ_l | [file:1, Chap. VIII, § C-3] |
| R7.1 | Matrices de Pauli | [file:1, Chap. IX, § B] |
| R7.2 | Opérateur densité spin | [file:1, Chap. IV, Compl. E_IV] |
| R8.1 | Coefficients CG | [file:1, Chap. X, § C, eq. C-28] |
| R8.2 | Récurrence CG | [file:1, Chap. X, § C-3] |
| R8.3 | Facteur de Landé | [file:1, Chap. X, Compl. D_X] |
| R9.1 | E^(1) non dégénéré | [file:1, Chap. XI, § B-1, eq. B-5] |
| R9.2 | E^(2) non dégénéré | [file:1, Chap. XI, § B-2, eq. B-15] |
| R9.3 | État corrigé 1er ordre | [file:1, Chap. XI, § B-1, eq. B-11] |
| R9.4 | Perturbation dégénérée | [file:1, Chap. XI, § C] |
| R9.5 | Méthode des variations | [file:1, Chap. XI, Compl. E_XI] |
| R10.1 | Structure fine H | [file:1, Chap. XII, § C] |
| R10.2 | Structure hyperfine H | [file:1, Chap. XII, § D] |
| R10.3 | Effet Zeeman | [file:1, Chap. XII, § E] |
| R11.1 | P_{i→f}(t) 1er ordre | [file:1, Chap. XIII, § B-3, eq. B-24] |
| R11.2 | Règle d'or de Fermi | [file:1, Chap. XIII, § C-3, eq. C-39] |
| R11.3 | Oscillations de Rabi | [file:1, Chap. XIII, Compl. C_XIII] |
| R12.1 | Symétrisation états | [file:1, Chap. XIV, § C-3] |
| R12.2 | Déterminant de Slater | [file:1, Chap. XIV, § C-3-c] |
| R12.3 | σ particules identiques | [file:1, Chap. XIV, § D-2] |

### 9.2 Tableau correspondance Classes ↔ Règles

| Classe | Règles implémentées | Tests requis |
|--------|---------------------|-------------|
| SpinHalf | R7.1 | σ²=I, Tr=0, [σ_i,σ_j]=2iε_{ijk}σ_k |
| Spinor | R7.1, R7.2 | Normalisation, ρ≥0, Tr(ρ)=1 |
| ClebschGordan | R8.1, R8.2 | Unitarité, règle triangle, M=m₁+m₂ |
| PhaseShiftSolver | R6.4 | u_l(0)=0, convergence δ_l |
| BornApproximation | R6.2 | Comparaison déphasages à haute E |
| CrossSection | R6.1, R6.3 | Théorème optique, σ≥0 |
| StationaryPerturbation | R9.1-R9.4 | E réelle, état normé, λ→0 retrouve H₀ |
| VariationalMethod | R9.5 | E_var ≥ E₀ |
| HydrogenFineStructure | R10.1 | Dépendance en j seul, ordre α² |
| HydrogenHyperfine | R10.2 | ν(21cm) ≈ 1420 MHz |
| TimeDependentPerturbation | R11.1 | 0≤P≤1, P=0 si W_fi=0 |
| FermiGoldenRule | R11.2 | Γ≥0, linéarité en t |
| Symmetrizer | R12.1 | S²=S, A²=A, SA=0 |
| SlaterDeterminant | R12.2 | Antisymétrie, Pauli |

---

## 10. Checklist implémentation future

### Pour chaque nouvelle classe/méthode
- [ ] Traçabilité : docstring mentionne règle(s) R*.* et source [file:1, Chap., §]
- [ ] Invariants : tests unitaires vérifient propriétés physiques
- [ ] Paramètres : valeurs numériques viennent de parameters.yaml
- [ ] Exceptions : erreur explicite si préconditions violées (hermiticité, normalisation, etc.)
- [ ] Documentation : hypothèses numériques explicitées

### Pour chaque expérience
- [ ] Config YAML dédiée
- [ ] Cycle complet : Préparation → Évolution → Mesure → Analyse
- [ ] Validation : appel méthodes validation/
- [ ] Résultats : export structuré avec métadonnées
- [ ] Visualisation : au moins un plot résumant résultats

### Tests physiques obligatoires
- [ ] **Normalisation** : ⟨ψ|ψ⟩ = 1 à chaque étape
- [ ] **Hermiticité** : H = H†, W = W†
- [ ] **Théorème optique** : σ_tot = (4π/k)Im(f(0))
- [ ] **Unitarité CG** : Σ|C|² = 1
- [ ] **Anti-commutation Pauli** : {σ_i, σ_j} = 2δ_{ij}
- [ ] **Règle triangle** : |j₁−j₂| ≤ J ≤ j₁+j₂
- [ ] **Conservation énergie** : ⟨H⟩ constant si H indépendant de t
- [ ] **Symétrie/Antisymétrie** : P₁₂|ψ⟩ = ±|ψ⟩ selon bosons/fermions
- [ ] **Pauli** : A|ψ⟩ = 0 si deux fermions dans le même état
- [ ] **Convergence perturbative** : |E^(2)| ≪ |E^(1)| ≪ E⁰
- [ ] **P ∈ [0,1]** pour toutes les probabilités de transition
- [ ] **Γ ≥ 0** pour tous les taux de transition

---

## 11. Glossaire des symboles

| Symbole | Signification | Unité SI |
|---------|---------------|----------|
| ℏ | Constante de Planck réduite | J·s |
| m_e | Masse de l'électron | kg |
| e | Charge élémentaire | C |
| c | Vitesse de la lumière | m/s |
| α | Constante de structure fine (≈1/137) | sans dimension |
| μ_B | Magnéton de Bohr | J/T |
| a₀ | Rayon de Bohr | m |
| μ | Masse réduite | kg |
| k | Nombre d'onde | m⁻¹ |
| f_k(θ,ϕ) | Amplitude de diffusion | m |
| σ(θ,ϕ) | Section efficace différentielle | m² |
| σ_tot | Section efficace totale | m² |
| δ_l | Déphasage de l'onde partielle l | rad |
| Y_l^m | Harmonique sphérique | sans dimension |
| S, S_x, S_y, S_z | Opérateur de spin | J·s |
| σ_x, σ_y, σ_z | Matrices de Pauli | sans dimension |
| J | Moment cinétique total | J·s |
| J₁, J₂ | Moments cinétiques partiels | J·s |
| ⟨j₁,m₁;j₂,m₂\|J,M⟩ | Coefficient de Clebsch-Gordan | sans dimension |
| g_J | Facteur de Landé | sans dimension |
| H₀ | Hamiltonien non perturbé | J |
| W | Perturbation | J |
| λ | Paramètre de perturbation | sans dimension |
| E_n⁰ | Énergie non perturbée du niveau n | J |
| ε₁, ε₂ | Corrections 1er, 2e ordre | J |
| W_mv | Correction masse-vitesse | J |
| W_SO | Couplage spin-orbite | J |
| W_D | Terme de Darwin | J |
| W_hf | Hamiltonien hyperfin | J |
| ω_fi | Pulsation de Bohr (E_f−E_i)/ℏ | rad/s |
| P_{i→f} | Probabilité de transition | sans dimension |
| Γ | Taux de transition | s⁻¹ |
| ρ(E) | Densité d'états | J⁻¹ |
| T₁, T₂ | Temps de relaxation | s |
| S, A | Symétriseur, antisymétriseur | sans dimension |
| P_α | Opérateur de permutation | sans dimension |
| ε_α | Signature de permutation (±1) | sans dimension |
| E_S, E_A | Sous-espaces symétrique, antisymétrique | — |

---

## 12. Synthèse finale pour l'agent de code

1. **Traçabilité** : Chaque équation implémentée DOIT référencer une règle R*.* et une source `[file:1, Chap. X, § Y]`. Aucune formule ne doit être inventée.

2. **Aucune extrapolation** : Si une formule manque dans ce document ou dans le cours, l'agent DOIT demander avant de coder. Ne jamais deviner un facteur numérique ou un signe.

3. **Structure modulaire** : Respecter strictement l'arborescence `core/ → dynamics/ → systems/ → experiments/`. Pas de dépendance inverse. Chaque fichier a une responsabilité unique.

4. **Paramètres externes** : Toute valeur numérique (constantes physiques, paramètres numériques, configuration) provient de `parameters.yaml`. Aucune constante en dur dans le code.

5. **Tests physiques obligatoires** : Chaque classe doit être accompagnée de tests vérifiant les invariants physiques listés à la section 10 (normalisation, hermiticité, théorème optique, Pauli, etc.).

6. **Documentation complète** : Chaque classe/méthode a une docstring mentionnant (a) la règle implémentée, (b) la source dans le cours, (c) les hypothèses numériques, (d) les limitations connues.
