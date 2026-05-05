import numpy as np
import time

Batch = 50000
N_sam = 50
h_kappa = 1.0  # For u=1.0

t0 = time.time()
rng = np.random.default_rng()

kappas = rng.poisson(lam=h_kappa, size=(Batch, N_sam))
kappas = np.maximum(kappas, 1)
cumsums = np.cumsum(kappas, axis=1)

valid = cumsums <= N_sam
b_idx, i_idx = np.where(valid)
spike_pos = cumsums[b_idx, i_idx] - 1

spike_seq = np.zeros((Batch, N_sam), dtype=np.int8)
spike_seq[b_idx, spike_pos] = 1

t1 = time.time()
print(f"Vectorized generated {Batch} sequences in {t1-t0:.4f}s")
