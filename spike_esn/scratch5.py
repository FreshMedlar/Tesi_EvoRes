import numpy as np
import time

def to_one_hot(indices: np.ndarray, vocab_size: int) -> np.ndarray:
    T = len(indices)
    one_hot = np.zeros((T, vocab_size), dtype=np.float64)
    one_hot[np.arange(T), indices.astype(int)] = 1.0
    return one_hot

def encode_series_vectorized(u_series, N_sam, rng):
    T = len(u_series)
    U_max = 1.0
    U_min = 0.0
    h_kappas = N_sam * (U_max - u_series) / 1.0
    h_kappas = np.maximum(h_kappas, 1.0)
    
    kappas = rng.poisson(lam=h_kappas[:, None], size=(T, N_sam))
    kappas = np.maximum(kappas, 1)
    cumsums = np.cumsum(kappas, axis=1)
    
    valid = cumsums <= N_sam
    b_idx, i_idx = np.where(valid)
    spike_pos = cumsums[b_idx, i_idx] - 1
    
    spikes = np.zeros((T, N_sam), dtype=np.int8)
    spikes[b_idx, spike_pos] = 1
    return spikes

T = 50000
vocab_size = 65
N_sam = 50
indices = np.random.randint(0, vocab_size, size=T)

t0 = time.time()
rng = np.random.default_rng(42)

# Vectorized generation of the entire spike matrix!
one_hot = to_one_hot(indices, vocab_size) # (T, V)
flat_u = one_hot.flatten() # (T * V,)
spikes = encode_series_vectorized(flat_u, N_sam, rng) # (T * V, N_sam)
spike_matrix = spikes.reshape(T, vocab_size * N_sam)

t1 = time.time()
print(f"Generated complete (T=50k, V=65) spike matrix in {t1-t0:.4f}s")
print("Shape:", spike_matrix.shape)
