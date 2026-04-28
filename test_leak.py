from spike_esn.train_shakespeare import *
import numpy as np
import argparse

# Monkey-patch SpikeReservoir
from spike_esn.reservoir import SpikeReservoir
original_update = SpikeReservoir.update_state

def new_update_state(self, f_spike, x_prev):
    alpha = 0.2
    return (1 - alpha) * x_prev + alpha * np.tanh(self.W_in @ f_spike + self.W_res @ x_prev)

SpikeReservoir.update_state = new_update_state

if __name__ == "__main__":
    import sys
    sys.argv = ["train_shakespeare.py", "--N-res", "2000", "--train-len", "10000", "--test-len", "1000", "--encoding", "one-hot", "--no-baseline"]
    main()
