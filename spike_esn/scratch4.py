import numpy as np

def encode_series_vectorized(u_series, N_sam, rng):
    T = len(u_series)
    U_max = np.max(u_series)
    U_min = np.min(u_series)
    denom = U_max - U_min
    if denom == 0:
        h_kappas = np.full(T, N_sam / 2.0)
    else:
        h_kappas = N_sam * (U_max - u_series) / denom
    
    h_kappas = np.maximum(h_kappas, 1.0)
    
    # Generate poisson matrix
    kappas = rng.poisson(lam=h_kappas[:, None], size=(T, N_sam))
    kappas = np.maximum(kappas, 1)
    cumsums = np.cumsum(kappas, axis=1)
    
    valid = cumsums <= N_sam
    b_idx, i_idx = np.where(valid)
    spike_pos = cumsums[b_idx, i_idx] - 1
    
    spikes = np.zeros((T, N_sam), dtype=np.int8)
    spikes[b_idx, spike_pos] = 1
    return spikes

u = np.linspace(0, 1, 1000)
rng = np.random.default_rng(42)
spikes_vec = encode_series_vectorized(u, 50, rng)
print("Vectorized shape:", spikes_vec.shape)
