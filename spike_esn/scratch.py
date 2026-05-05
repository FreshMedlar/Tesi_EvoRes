import numpy as np

N_sam = 50
psi = 2000.0
spikes_2d = np.random.randint(0, 2, size=(3, N_sam)).astype(np.int8)

# Old way
f_spike_2d_old = np.zeros((3, N_sam), dtype=np.float64)
t_seq = np.arange(1, N_sam + 1, dtype=np.float64)
for c in range(3):
    spike_positions = np.where(spikes_2d[c] == 1)[0] + 1
    if len(spike_positions) > 0:
        diffs = t_seq[:, None] - spike_positions[None, :]
        f_spike_2d_old[c] = np.sum(np.exp(-diffs / psi), axis=1)

# New way
_spike_kernel = np.exp(-(t_seq[:, None] - t_seq[None, :]) / psi).T
f_spike_2d_new = spikes_2d @ _spike_kernel

print("Max difference:", np.max(np.abs(f_spike_2d_old - f_spike_2d_new)))
