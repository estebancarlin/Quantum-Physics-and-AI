"""
Gestionnaire accélération GPU avec détection automatique.

Fonctionnalités:
    - Détection CuPy/Numba
    - Sélection automatique CPU/GPU selon taille grille
    - Gestion mémoire VRAM
    - Profiling performances optionnel
"""

import os
import warnings
from typing import Tuple, Optional
import numpy as np

# ==================== Détection GPU ====================

# Variable contrôle utilisateur
USE_GPU = os.getenv('QUANTUM_USE_GPU', 'true').lower() == 'true'

# Détection CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    
    # Infos GPU
    device = cp.cuda.Device()
    gpu_name = device.attributes['Name'].decode('utf-8')
    total_memory = device.mem_info[1] / 1e9  # GB
    
    print(f"✓ GPU détecté : {gpu_name}")
    print(f"  VRAM totale : {total_memory:.1f} GB")
    print(f"  CuPy version : {cp.__version__}")
    
except ImportError:
    cp = np  # Fallback NumPy
    CUPY_AVAILABLE = False
    warnings.warn(
        "CuPy non installé. Installer avec: pip install cupy-cuda12x",
        ImportWarning
    )

# Détection Numba CUDA
try:
    from numba import cuda
    NUMBA_AVAILABLE = cuda.is_available()
    
    if NUMBA_AVAILABLE:
        print(f"✓ Numba CUDA disponible")
        print(f"  Compute Capability : {cuda.get_current_device().compute_capability}")
except ImportError:
    NUMBA_AVAILABLE = False

# État final
GPU_AVAILABLE = USE_GPU and CUPY_AVAILABLE


# ==================== Sélection Automatique CPU/GPU ====================

class GPUConfig:
    """Configuration GPU globale."""
    
    # Seuils activation GPU (éviter overhead petites grilles)
    MIN_GRID_SIZE_1D = 1024
    MIN_GRID_SIZE_2D = 256  # 256×256 = 65k points
    
    # Limites mémoire
    MAX_MEMORY_FRACTION = 0.8  # 80% VRAM max
    
    # Profiling
    PROFILE_TRANSFERS = os.getenv('QUANTUM_GPU_PROFILE', 'false').lower() == 'true'


def should_use_gpu(nx: int, ny: Optional[int] = None, nz: Optional[int] = None) -> bool:
    """
    Décide CPU vs GPU selon taille grille.
    
    Args:
        nx, ny, nz: Dimensions grille
        
    Returns:
        True si GPU recommandé
        
    Example:
        >>> should_use_gpu(512)  # 1D
        False  # Trop petit, overhead GPU
        >>> should_use_gpu(2048, 2048)  # 2D
        True  # Grille assez grande
    """
    if not GPU_AVAILABLE:
        return False
    
    if nz is not None:  # 3D
        n_total = nx * ny * nz
        return n_total > GPUConfig.MIN_GRID_SIZE_2D ** 3
    
    elif ny is not None:  # 2D
        n_total = nx * ny
        return n_total > GPUConfig.MIN_GRID_SIZE_2D ** 2
    
    else:  # 1D
        return nx > GPUConfig.MIN_GRID_SIZE_1D


def estimate_gpu_memory(nx: int, ny: Optional[int] = None, 
                        nz: Optional[int] = None,
                        dtype=np.complex128) -> float:
    """
    Estime mémoire requise (bytes).
    
    Args:
        nx, ny, nz: Dimensions
        dtype: Type données
        
    Returns:
        Mémoire estimée (bytes)
    """
    itemsize = np.dtype(dtype).itemsize
    
    if nz is not None:
        return nx * ny * nz * itemsize
    elif ny is not None:
        return nx * ny * itemsize
    else:
        return nx * itemsize


def check_gpu_capacity(nx: int, ny: Optional[int] = None, 
                        nz: Optional[int] = None) -> Tuple[bool, str]:
    """
    Vérifie si grille tient en VRAM.
    
    Returns:
        (can_fit, message)
        
    Raises:
        MemoryError: Si grille trop grande
    """
    if not GPU_AVAILABLE:
        return False, "GPU non disponible"
    
    required = estimate_gpu_memory(nx, ny, nz)
    available = cp.cuda.Device().mem_info[0]  # Bytes libres
    max_allowed = available * GPUConfig.MAX_MEMORY_FRACTION
    
    if required > max_allowed:
        msg = (
            f"Grille trop grande pour GPU:\n"
            f"  Requis : {required/1e9:.2f} GB\n"
            f"  Disponible : {available/1e9:.2f} GB ({max_allowed/1e9:.2f} GB utilisables)\n"
            f"  Réduire nx/ny/nz ou utiliser CPU (QUANTUM_USE_GPU=false)"
        )
        return False, msg
    
    return True, f"OK ({required/1e6:.1f} MB / {available/1e9:.1f} GB)"


# ==================== Utilitaires Transfert ====================

def to_gpu(array: np.ndarray) -> 'cp.ndarray':
    """Transfert CPU → GPU avec gestion erreurs."""
    if not GPU_AVAILABLE:
        return array
    
    try:
        return cp.asarray(array)
    except cp.cuda.memory.OutOfMemoryError as e:
        warnings.warn(f"VRAM saturée, fallback CPU: {e}", RuntimeWarning)
        return array


def to_cpu(array) -> np.ndarray:
    """Transfert GPU → CPU."""
    if GPU_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def get_array_module(array):
    """Retourne numpy ou cupy selon type array."""
    if GPU_AVAILABLE:
        return cp.get_array_module(array)
    return np


# ==================== Profiling Optionnel ====================

class TransferProfiler:
    """Mesure temps transferts CPU↔GPU."""
    
    def __init__(self):
        self.transfers = []
        self.enabled = GPUConfig.PROFILE_TRANSFERS
    
    def log_transfer(self, direction: str, size_mb: float, time_ms: float):
        """Enregistre transfert."""
        if self.enabled:
            self.transfers.append({
                'direction': direction,
                'size_mb': size_mb,
                'time_ms': time_ms,
                'bandwidth_gb_s': size_mb / (time_ms / 1000) / 1024
            })
    
    def print_summary(self):
        """Affiche statistiques."""
        if not self.transfers:
            return
        
        import pandas as pd
        df = pd.DataFrame(self.transfers)
        
        print("\n=== Profiling Transferts GPU ===")
        print(f"Total transferts : {len(df)}")
        print(f"Bande passante moyenne : {df['bandwidth_gb_s'].mean():.2f} GB/s")
        print(df.groupby('direction').agg({
            'size_mb': 'sum',
            'time_ms': 'sum',
            'bandwidth_gb_s': 'mean'
        }))


# Instance globale profiler
profiler = TransferProfiler()


# ==================== Tests ====================

if __name__ == "__main__":
    print("="*70)
    print(" Test Configuration GPU")
    print("="*70)
    
    # Test 1 : Détection
    print(f"\n1. Détection:")
    print(f"   USE_GPU = {USE_GPU}")
    print(f"   GPU_AVAILABLE = {GPU_AVAILABLE}")
    print(f"   CUPY_AVAILABLE = {CUPY_AVAILABLE}")
    print(f"   NUMBA_AVAILABLE = {NUMBA_AVAILABLE}")
    
    # Test 2 : Sélection automatique
    print(f"\n2. Sélection automatique:")
    test_cases = [
        (512, None, None),      # 1D petit
        (2048, None, None),     # 1D moyen
        (4096, None, None),     # 1D grand
        (256, 256, None),       # 2D petit
        (512, 512, None),       # 2D moyen
        (2048, 2048, None),     # 2D grand
    ]
    
    for nx, ny, nz in test_cases:
        use_gpu = should_use_gpu(nx, ny, nz)
        dim = "1D" if ny is None else ("2D" if nz is None else "3D")
        size_str = f"{nx}" if ny is None else f"{nx}×{ny}"
        print(f"   {dim} grille {size_str:15s} → {'GPU' if use_gpu else 'CPU'}")
    
    # Test 3 : Capacité mémoire
    if GPU_AVAILABLE:
        print(f"\n3. Capacité mémoire:")
        test_grids = [
            (2048, None, None),
            (512, 512, None),
            (8192, 8192, None),
            (16384, 16384, None),
        ]
        
        for nx, ny, nz in test_grids:
            can_fit, msg = check_gpu_capacity(nx, ny, nz)
            size_str = f"{nx}" if ny is None else f"{nx}×{ny}"
            status = "✓" if can_fit else "✗"
            print(f"   {status} {size_str:15s} : {msg}")
    
    # Test 4 : Transfert simple
    if GPU_AVAILABLE:
        print(f"\n4. Test transfert:")
        x = np.random.randn(1000, 1000) + 1j * np.random.randn(1000, 1000)
        
        import time
        t0 = time.time()
        x_gpu = to_gpu(x)
        t_up = time.time() - t0
        
        t0 = time.time()
        x_cpu = to_cpu(x_gpu)
        t_down = time.time() - t0
        
        print(f"   Taille : {x.nbytes / 1e6:.1f} MB")
        print(f"   CPU→GPU : {t_up*1000:.2f} ms")
        print(f"   GPU→CPU : {t_down*1000:.2f} ms")
        print(f"   Vérification : {np.allclose(x, x_cpu)}")
    
    print("\n" + "="*70)