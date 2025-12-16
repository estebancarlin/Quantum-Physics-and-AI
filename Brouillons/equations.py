import math
import numpy as np
import yaml

# Load constants from YAML file
with open('Constants.yaml', 'r') as file:
    constants = yaml.safe_load(file)

# Relations de Planck-Einstein
def Energie(fréquence):
    """Calculer l'énergie d'un photon donné sa fréquence."""
    h = constants['Constante de Planck']
    return h * fréquence

def Fréquence(énergie):
    """Calculer la fréquence d'un photon donné son énergie."""
    h = constants['Constante de Planck']
    return énergie / h

def QuantitéDeMouvement(vecteurDonde):
    """Calculer la quantité de mouvement d'un photon donné son vecteur d'onde."""
    h = constants['Constante de Planck']
    return h * vecteurDonde / (2 * math.pi)

def vecteurDonde(quantitéDeMouvement):
    """Calculer le vecteur d'onde d'un photon donné sa quantité de mouvement."""
    h = constants['Constante de Planck']
    return (2 * math.pi / h) * quantitéDeMouvement

# Onde lumineuse
class OndeLumineuse:
    def __init__(self, amplitude, fréquence, phase=0):
        self.amplitude = amplitude
        self.fréquence = fréquence
        self.phase = phase
        
    def Intensité(self):
        """Calculer l'intensité de l'onde lumineuse."""
        c = constants['Vitesse de la lumière']
        ε0 = constants['Permittivité du vide']
        return 0.5 * c * ε0 * (self.amplitude ** 2)

class OndePlane(OndeLumineuse):
    def __init__(self, amplitude, fréquence, direction, phase=0):
        super().__init__(amplitude, fréquence, phase)
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.vecteurOnde = (2 * math.pi * fréquence / constants['Vitesse de la lumière']) * self.direction
        self.pulsation = 2 * math.pi * fréquence
        
class OndePlanePolariséeMonochromatique(OndePlane):
    def __init__(self, amplitude, fréquence, direction, polarisation, phase=0):
        super().__init__(amplitude, fréquence, direction, phase)
        self.polarisation = np.array(polarisation) / np.linalg.norm(polarisation)
        
    def ChampÉlectrique(self, position, temps):
        """Calculer le champ électrique en un point donné et à un instant donné."""
        k_dot_r = np.dot(self.vecteurOnde, position)
        ωt = self.pulsation * temps
        return self.amplitude * math.exp(1j * (k_dot_r - ωt + self.phase)) * self.polarisation
    
    def Intensité(self):
        return super().Intensité() * math.cos(self.polarisation) ** 2
    
# Louis de Broglie
class Photon:
    def __init__(self):