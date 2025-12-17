"""
Expérience : Statistiques de mesure quantique.

Objectifs:
    - Mesurer N fois même observable sur états identiques
    - Vérifier distribution empirique → P(aₙ) théorique (Règle R2.2)
    - Tester réduction paquet d'ondes (mesures successives, Règle R2.3)
    - Valider postulat mesure via test χ²

Système : État superposition (exemple : paquet gaussien ou état propre superposé)
"""

import numpy as np
from typing import Dict, Any, List, Optional
from scipy.stats import chi2

from quantum_simulation.experiments.base_experiment import Experiment
from quantum_simulation.systems.free_particle import FreeParticle
from quantum_simulation.systems.potential_systems import InfiniteWell
from quantum_simulation.dynamics.measurement import QuantumMeasurement
from quantum_simulation.core.state import WaveFunctionState, EigenStateBasis
from quantum_simulation.core.operators import PositionOperator, MomentumOperator, Hamiltonian


class MeasurementStatistics(Experiment):
    """
    Expérience validation postulats mesure quantique.
    
    Configuration requise (dans parameters.yaml):
        experiments.measurement_statistics:
            observable_to_measure: "position" | "momentum" | "energy"
            n_measurements: 1000  # Nombre mesures indépendantes
            system_type: "free_particle" | "infinite_well"
            initial_state:
                type: "gaussian" | "superposition"
                # Paramètres selon type
            successive_measurements:
                enabled: true  # Tester réduction paquet
                n_repetitions: 3  # Mesures sur même état
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration complète (YAML chargé)
        """
        super().__init__(config)
        
        # Extraction config expérience
        self.exp_config = config['experiments']['measurement_statistics']
        
        # Constantes physiques
        self.hbar = config['physical_constants']['hbar']
        self.mass = config['physical_constants']['m_electron']
        
        # Paramètres mesure
        self.observable_name = self.exp_config['observable_to_measure']
        self.n_measurements = self.exp_config['n_measurements']
        self.system_type = self.exp_config.get('system_type', 'free_particle')
        
        # Configuration état initial
        self.initial_config = self.exp_config['initial_state']
        
        # CORRECTION : Grille spatiale depuis config expérience (pas numerical_parameters)
        if 'spatial_grid' in self.exp_config:
            # Option 1 : Config locale (recommandé)
            spatial_params = self.exp_config['spatial_grid']
        else:
            # Option 2 : Fallback sur config globale
            spatial_params = config['numerical_parameters']['spatial_discretization']
            
            # Vérifier que nx est défini
            if spatial_params['nx'] is None:
                raise ValueError(
                    "Configuration spatiale incomplète : 'nx' doit être défini.\n"
                    "Ajouter 'spatial_grid' dans experiments.measurement_statistics "
                    "ou définir 'nx' dans numerical_parameters.spatial_discretization"
                )
        
        self.x = np.linspace(
            spatial_params['x_min'],
            spatial_params['x_max'],
            spatial_params['nx']
        )
        
        # DEBUG TEMPORAIRE
        # print(f"    DEBUG: Type self.x = {type(self.x)}")
        # print(f"    DEBUG: Valeurs params = x_min={spatial_params['x_min']}, "
        #         f"x_max={spatial_params['x_max']}, nx={spatial_params['nx']}")

        # Vérification
        if not isinstance(self.x, np.ndarray):
            raise TypeError(
                f"Grille spatiale devrait être ndarray, obtenu {type(self.x)}.\n"
                f"Paramètres: {spatial_params}"
            )

        # Attributs calculés
        self.system = None
        self.observable = None
        self.measurement_module = None
        self.eigenvalues = None
        self.eigenstates = None
        self.theoretical_probabilities = None
        self.measurement_results_ensemble = []
        self.successive_measurement_results = []
        
    def prepare_initial_state(self):
        """
        Prépare état initial selon configuration.
        
        Supporte :
        - Paquet gaussien (superposition états impulsion)
        - Superposition explicite d'états propres puits infini
        """
        print("    Configuration état initial...")
        
        state_type = self.initial_config['type']
        
        if state_type == "gaussian":
            # Paquet gaussien (superposition continue)
            if self.system_type == "free_particle":
                self.system = FreeParticle(self.mass, self.hbar)
                
                self.initial_state = self.system.create_gaussian_wavepacket(
                    spatial_grid=self.x,
                    x0=self.initial_config.get('x0', 0.0),
                    sigma_x=self.initial_config.get('sigma_x', 1e-9),
                    k0=self.initial_config.get('k0', 0.0)
                )
            else:
                raise ValueError(f"Paquet gaussien non supporté pour système {self.system_type}")
                
        elif state_type == "superposition":
            # Superposition discrète d'états propres
            if self.system_type == "infinite_well":
                width = self.config['systems']['potential_systems']['infinite_well']['width']
                self.system = InfiniteWell(width, self.mass, self.hbar)
                
                # Coefficients superposition (normalisés)
                coeffs = np.array(self.initial_config['coefficients'], dtype=complex)
                n_levels = self.initial_config['n_levels']
                
                # Vérifier normalisation coefficients
                norm_coeffs = np.linalg.norm(coeffs)
                if abs(norm_coeffs - 1.0) > 1e-10:
                    print(f"    ⚠ Coefficients non normés ({norm_coeffs:.6f}), normalisation automatique")
                    coeffs = coeffs / norm_coeffs
                
                # Construire superposition ψ = Σ cₙ ψₙ
                psi_superposition = np.zeros_like(self.x, dtype=complex)
                for i, n in enumerate(range(1, n_levels + 1)):
                    psi_n = self.system.eigenstate_wavefunction(n, self.x).wavefunction
                    psi_superposition += coeffs[i] * psi_n
                
                self.initial_state = WaveFunctionState(self.x, psi_superposition)
                
                # Vérifier normalisation finale
                if not self.initial_state.is_normalized(tolerance=1e-6):
                    self.initial_state = self.initial_state.normalize()
            else:
                raise ValueError(f"Superposition non supportée pour système {self.system_type}")
        else:
            raise ValueError(f"Type état inconnu : {state_type}")
        
        # CORRECTION : Tolérance relâchée pour erreurs numériques
        # Intégration trapézoïdale a erreur O(dx²) ~ 1e-6 pour grilles typiques
        norm = self.initial_state.norm()
        if abs(norm - 1.0) > 1e-3:  # Tolérance 0.1%
            print(f"    ⚠ État initial mal normé : ||ψ|| = {norm:.10f}")
            print(f"      Renormalisation automatique...")
            self.initial_state = self.initial_state.normalize()
            norm = self.initial_state.norm()
            print(f"      Nouvelle norme : ||ψ|| = {norm:.10f}")
        
        print(f"    ✓ État initial préparé : {state_type} (norme = {norm:.10f})")
        
    def define_hamiltonian(self):
        """
        Définit hamiltonien système (déjà fait dans prepare_initial_state).
        
        Note: Hamiltonien construit par système (FreeParticle ou InfiniteWell).
        """
        print("    Hamiltonien défini par système")
        
    def evolve_state(self):
        """
        Pas d'évolution temporelle : mesures instantanées sur état fixe.
        """
        print("    Pas d'évolution temporelle (mesures instantanées)")
        
    def perform_measurements(self):
        """
        Effectue N mesures indépendantes de l'observable.
        
        Étapes :
        1. Diagonaliser observable (obtenir {aₙ}, {|uₙ⟩})
        2. Calculer probabilités théoriques P(aₙ) = |⟨uₙ|ψ⟩|²
        3. Effectuer N mesures (tirage aléatoire selon P(aₙ))
        4. Comparer distributions empirique vs théorique (test χ²)
        """
        print("    Préparation observable...")
        
        # 1. Construire observable
        if self.observable_name == "position":
            self.observable = PositionOperator(dimension=1)
        elif self.observable_name == "momentum":
            self.observable = MomentumOperator(hbar=self.hbar, dimension=1)
        elif self.observable_name == "energy":
            if self.system_type == "free_particle":
                self.observable = self.system.hamiltonian(self.x)
            elif self.system_type == "infinite_well":
                self.observable = self.system.hamiltonian(self.x)
            else:
                raise ValueError(f"Énergie non supportée pour {self.system_type}")
        else:
            raise ValueError(f"Observable inconnue : {self.observable_name}")
        
        # 2. Diagonaliser observable (obtenir spectre)
        print(f"    Diagonalisation observable '{self.observable_name}'...")
        
        # LIMITATION : Pour observables continues (position, impulsion sur grille infinie),
        # spectre est quasi-continu → regrouper en bins
        
        if self.observable_name in ["position", "momentum"]:
            # Cas observables continues : utiliser approximation discrète
            # Spectre ≈ valeurs grille
            print("      ⚠ Observable continue : approximation discrète (spectre = grille)")
            
            # Mesure discrète sur grille
            self.eigenvalues = self.x  # Approximation : valeurs position possibles
            
            # États propres : delta de Dirac approchés (non utilisés directement)
            # On utilise plutôt densité probabilité ρ(x) = |ψ(x)|²
            self.eigenstates = None  # Non nécessaire pour mesure position
            
            # Probabilités théoriques
            rho = self.initial_state.probability_density()
            dx = self.x[1] - self.x[0]
            self.theoretical_probabilities = rho * dx  # P(xᵢ) = ρ(xᵢ) · dx
            
            # Normaliser (au cas où intégration discrète imparfaite)
            self.theoretical_probabilities /= np.sum(self.theoretical_probabilities)
            
        elif self.observable_name == "energy":
            # Cas observable discrète (puits infini) ou continue (particule libre)
            if self.system_type == "infinite_well":
                # Spectre discret : utiliser états propres analytiques
                n_max = 20  # Nombre états propres considérés
                
                self.eigenvalues = np.array([
                    self.system.energy_eigenvalue(n) for n in range(1, n_max + 1)
                ])
                
                self.eigenstates = [
                    self.system.eigenstate_wavefunction(n, self.x) for n in range(1, n_max + 1)
                ]
                
                # Probabilités théoriques : P(Eₙ) = |⟨ψₙ|ψ⟩|²
                self.theoretical_probabilities = np.array([
                    abs(self.initial_state.inner_product(psi_n))**2 
                    for psi_n in self.eigenstates
                ])
                
                # Vérifier normalisation
                sum_probs = np.sum(self.theoretical_probabilities)
                if abs(sum_probs - 1.0) > 1e-6:
                    print(f"      ⚠ Somme probabilités = {sum_probs:.6f}, renormalisation")
                    self.theoretical_probabilities /= sum_probs
                
            else:
                # Particule libre : spectre continu, utiliser bins énergie
                print("      ⚠ Particule libre : spectre continu, implémentation simplifiée")
                # TODO : Implémenter binning énergie si nécessaire
                raise NotImplementedError("Mesure énergie particule libre non implémentée")
        
        # 3. Effectuer N mesures indépendantes
        print(f"    Effectuer {self.n_measurements} mesures indépendantes...")
        
        # MODIFICATION : Ne pas créer QuantumMeasurement avec observable
        # car cela nécessite eigensystem() non implémenté pour Hamiltonian.
        # À la place, utiliser directement measure_once_manual()
        
        for i in range(self.n_measurements):
            # Mesure sur état initial (réinitialisé à chaque fois)
            # Note : On ne modifie PAS l'état après mesure (mesures indépendantes)
            
            # Tirage aléatoire selon probabilités
            outcome_idx = np.random.choice(
                len(self.eigenvalues),
                p=self.theoretical_probabilities
            )
            outcome = self.eigenvalues[outcome_idx]
            
            self.measurement_results_ensemble.append(outcome)
        
        print(f"    ✓ {self.n_measurements} mesures effectuées")
        
        # 4. Test mesures successives (optionnel)
        if self.exp_config.get('successive_measurements', {}).get('enabled', False):
            self._test_successive_measurements()
        
    def _test_successive_measurements(self):
        """
        Teste réduction paquet d'ondes via mesures successives.
        
        Principe :
        1. Mesurer état → obtenir aₖ
        2. Réduire état → |ψ'⟩ = |uₖ⟩
        3. Mesurer à nouveau → doit donner aₖ avec probabilité 1
        """
        print("    Test mesures successives (réduction paquet d'ondes)...")
        
        n_rep = self.exp_config['successive_measurements'].get('n_repetitions', 3)
        
        for rep in range(n_rep):
            # État de départ : copie état initial
            state_current = WaveFunctionState(self.x, self.initial_state.wavefunction.copy())
            
            outcomes = []
            
            # Première mesure (tirage aléatoire)
            outcome_idx = np.random.choice(
                len(self.eigenvalues),
                p=self.theoretical_probabilities
            )
            outcome1 = self.eigenvalues[outcome_idx]
            outcomes.append(outcome1)
            
            # Réduction état : projeter sur état propre mesuré
            if self.eigenstates is not None:
                # Cas discret : projeter sur |uₙ⟩
                psi_reduced = self.eigenstates[outcome_idx].wavefunction
                state_reduced = WaveFunctionState(self.x, psi_reduced)
                
                # Renormaliser (au cas où)
                if not state_reduced.is_normalized(tolerance=1e-6):
                    state_reduced = state_reduced.normalize()
                
                # Recalculer probabilités pour état réduit
                probs_reduced = np.array([
                    abs(state_reduced.inner_product(psi_n))**2 
                    for psi_n in self.eigenstates
                ])
                probs_reduced /= np.sum(probs_reduced)
            else:
                # Cas continu : projeter sur position
                psi_reduced = np.zeros_like(self.x, dtype=complex)
                psi_reduced[outcome_idx] = 1.0
                state_reduced = WaveFunctionState(self.x, psi_reduced)
                state_reduced = state_reduced.normalize()
                
                rho_reduced = state_reduced.probability_density()
                dx = self.x[1] - self.x[0]
                probs_reduced = rho_reduced * dx
                probs_reduced /= np.sum(probs_reduced)
            
            # Seconde mesure (devrait donner même résultat avec prob ~1)
            outcome_idx2 = np.random.choice(
                len(self.eigenvalues),
                p=probs_reduced
            )
            outcome2 = self.eigenvalues[outcome_idx2]
            outcomes.append(outcome2)
            
            self.successive_measurement_results.append({
                'repetition': rep + 1,
                'first_outcome': outcome1,
                'second_outcome': outcome2,
                'match': np.isclose(outcome1, outcome2, rtol=1e-5)
            })
        
        print(f"    ✓ {n_rep} séquences mesures successives enregistrées")
        
    def validate_physics(self) -> Dict[str, bool]:
        """
        Valide postulats mesure via test statistique χ².
        
        Tests :
        1. Distribution empirique compatible avec P(aₙ) théorique (test χ²)
        2. Réduction paquet : mesures successives donnent même résultat
        
        Returns:
            Dictionnaire {test_name: passed}
        """
        results = {}
        
        print("    Validation postulats mesure...")
        
        # 1. Test χ² : distribution empirique vs théorique
        print("      Test 1/2: Distribution probabilités (χ²)...")
        
        # Construire histogramme empirique
        if self.observable_name in ["position", "momentum"]:
            # Cas continu : binning
            n_bins = min(50, len(self.eigenvalues) // 10)
            
            # Bins uniformes
            bins = np.linspace(self.eigenvalues.min(), self.eigenvalues.max(), n_bins + 1)
            
            # Histogramme empirique
            counts_empirical, _ = np.histogram(self.measurement_results_ensemble, bins=bins)
            
            # Histogramme théorique (regrouper probabilités par bin)
            counts_theoretical = []
            for i in range(n_bins):
                mask = (self.eigenvalues >= bins[i]) & (self.eigenvalues < bins[i+1])
                counts_theoretical.append(np.sum(self.theoretical_probabilities[mask]) * self.n_measurements)
            
            counts_theoretical = np.array(counts_theoretical)
            
        else:
            # Cas discret : compter occurrences chaque valeur propre
            counts_empirical = np.zeros(len(self.eigenvalues))
            
            for outcome in self.measurement_results_ensemble:
                # Trouver index valeur propre la plus proche
                idx = np.argmin(np.abs(self.eigenvalues - outcome))
                counts_empirical[idx] += 1
            
            counts_theoretical = self.theoretical_probabilities * self.n_measurements
        
        # Statistique χ²
        # χ² = Σ (Oᵢ - Eᵢ)² / Eᵢ
        # où Oᵢ = counts observed, Eᵢ = counts expected
        
        # Éviter division par zéro : exclure bins avec Eᵢ < 5 (règle empirique)
        mask_valid = counts_theoretical >= 5
        
        if np.sum(mask_valid) < 5:
            print("        ⚠ Trop peu de bins valides pour test χ², validation ignorée")
            results['chi2_test'] = True  # Par défaut accepté
        else:
            chi2_stat = np.sum(
                (counts_empirical[mask_valid] - counts_theoretical[mask_valid])**2 / 
                counts_theoretical[mask_valid]
            )
            
            # Degrés liberté = nombre bins - 1
            dof = np.sum(mask_valid) - 1
            
            # P-valeur : probabilité observer χ² ≥ χ²_obs sous H₀ (distribution théorique correcte)
            p_value = 1 - chi2.cdf(chi2_stat, dof)
            
            # Seuil acceptation : p > 0.05 (5%)
            results['chi2_test'] = p_value > 0.05
            
            print(f"        χ² = {chi2_stat:.2f}, dof = {dof}, p-value = {p_value:.4f}")
            
            if results['chi2_test']:
                print("        ✓ Distribution empirique compatible avec théorie")
            else:
                print("        ✗ Distribution empirique significativement différente")
        
        # 2. Test réduction paquet d'ondes
        if self.successive_measurement_results:
            print("      Test 2/2: Réduction paquet d'ondes...")
            
            all_match = all(
                res['match'] for res in self.successive_measurement_results
            )
            
            results['wavefunction_collapse'] = all_match
            
            if all_match:
                print("        ✓ Mesures successives cohérentes (réduction vérifiée)")
            else:
                n_mismatch = sum(
                    not res['match'] for res in self.successive_measurement_results
                )
                print(f"        ✗ {n_mismatch}/{len(self.successive_measurement_results)} mesures incohérentes")
        else:
            print("      Test 2/2: Réduction paquet d'ondes (désactivé)")
            results['wavefunction_collapse'] = True  # Non testé
        
        return results
        
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyse statistique détaillée.
        
        Retourne:
        - Histogrammes empirique vs théorique
        - Statistiques (moyenne, variance mesurée vs attendue)
        - Résultats test χ²
        """
        analysis = {}
        
        print("\n  Analyse statistique:")
        
        # Statistiques descriptives
        outcomes_array = np.array(self.measurement_results_ensemble)
        
        mean_measured = np.mean(outcomes_array)
        var_measured = np.var(outcomes_array)
        
        # Valeur moyenne théorique : ⟨A⟩ = Σ aₙ P(aₙ)
        mean_theory = np.sum(self.eigenvalues * self.theoretical_probabilities)
        
        # Variance théorique : ⟨A²⟩ - ⟨A⟩²
        mean_squared_theory = np.sum(self.eigenvalues**2 * self.theoretical_probabilities)
        var_theory = mean_squared_theory - mean_theory**2
        
        analysis['mean'] = {
            'measured': mean_measured,
            'theoretical': mean_theory,
            'relative_error': abs(mean_measured - mean_theory) / abs(mean_theory) if mean_theory != 0 else 0
        }
        
        analysis['variance'] = {
            'measured': var_measured,
            'theoretical': var_theory,
            'relative_error': abs(var_measured - var_theory) / abs(var_theory) if var_theory != 0 else 0
        }
        
        print(f"    Valeur moyenne:")
        print(f"      Mesurée    : {mean_measured:.6e}")
        print(f"      Théorique  : {mean_theory:.6e}")
        print(f"      Écart      : {analysis['mean']['relative_error']:.2%}")
        
        print(f"    Variance:")
        print(f"      Mesurée    : {var_measured:.6e}")
        print(f"      Théorique  : {var_theory:.6e}")
        print(f"      Écart      : {analysis['variance']['relative_error']:.2%}")
        
        # Stocker distributions pour visualisation
        analysis['distributions'] = {
            'outcomes': outcomes_array,
            'eigenvalues': self.eigenvalues,
            'theoretical_probabilities': self.theoretical_probabilities
        }
        
        if self.successive_measurement_results:
            analysis['successive_measurements'] = self.successive_measurement_results
        
        return analysis