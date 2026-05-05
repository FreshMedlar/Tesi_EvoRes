import numpy as np
import time

Batch = 3200000
N_sam = 50
h_kappa = 50.0  # For u=0.0

t0 = time.time()
rng = np.random.default_rng()

# Generate max possible intervals (N_sam) for the whole batch
# For h_kappa=50, we actually only need 1 or 2 intervals because sum will exceed 50 immediately!
# But let's be safe and generate shape (Batch, 2)
kappas = rng.poisson(lam=h_kappa, size=(Batch, 2))
kappas = np.maximum(kappas, 1)
cumsums = np.cumsum(kappas, axis=1)

# We want to set spike_seq[b, cumsums[b, i] - 1] = 1 for all i where cumsums[b, i] <= N_sam
valid = cumsums <= N_sam

# Get the coordinates
b_idx, i_idx = np.where(valid)
spike_pos = cumsums[b_idx, i_idx] - 1

spike_seq = np.zeros((Batch, N_sam), dtype=np.int8)
spike_seq[b_idx, spike_pos] = 1

t1 = time.time()
print(f"Vectorized generated {Batch} sequences in {t1-t0:.4f}s")
