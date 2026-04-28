"""
train_shakespeare.py — Train a Spike-ESN on the Tiny Shakespeare dataset.

Task: next-character prediction (character-level language modelling via
regression — the normalised character index at time t predicts t+1).

Usage (from project root):
    PYTHONPATH=. .venv/bin/python spike_esn/train_shakespeare.py [options]

Options:
    --train-len   INT    Training characters (default: 5000)
    --test-len    INT    Test characters      (default: 1000)
    --washout     INT    Reservoir warm-up    (default: 200)
    --N-res       INT    Reservoir neurons    (default: 500)
    --N-sam       INT    Spike sampling times (default: 50)
    --rho         FLOAT  Spectral radius      (default: 0.9)
    --eta         FLOAT  Reservoir sparsity   (default: 0.1)
    --mu          FLOAT  Ridge regularisation (default: 1e-4)
    --psi         FLOAT  Synaptic time const  (default: 2000)
    --input-scaling FLOAT W_in scaling         (default: 0.8)
    --seed        INT    Random seed          (default: 42)
    --gen-len     INT    Characters to gen    (default: 200)
    --encoding    STR    Input encoding: 'scalar' or 'one-hot' (default: scalar)
    --no-baseline        Skip baseline ESN comparison
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(filepath: str):
    """Load tinyshakespeare and build a vocabulary from the FULL corpus.

    The vocabulary is built before any slicing so that every character has a
    consistent integer index regardless of which slice is used for
    training / testing / generation.

    Returns
    -------
    data         : float64 array of character indices for the whole corpus
    chars        : sorted list of unique characters
    char_to_int  : dict mapping character → index
    int_to_char  : dict mapping index → character
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}

    data = np.array([char_to_int[c] for c in text], dtype=np.float64)
    return data, chars, char_to_int, int_to_char


def normalize(data: np.ndarray, vocab_size: int) -> np.ndarray:
    """Map integer character indices from [0, vocab_size-1] to [0, 1].

    This gives the SpikeEncoder a consistent U_max=1 / U_min=0 across every
    series (training, testing, generation), avoiding the subtle bug where
    encode_series() derives U_max/U_min from a local slice that may not
    cover the full vocabulary.
    """
    return data / (vocab_size - 1)


def denormalize(values: np.ndarray, vocab_size: int) -> np.ndarray:
    """Inverse of normalize — maps [0, 1] back to character index range."""
    return values * (vocab_size - 1)


# ---------------------------------------------------------------------------
# One-hot encoding helpers
# ---------------------------------------------------------------------------

def to_one_hot(indices: np.ndarray, vocab_size: int) -> np.ndarray:
    """Convert integer character indices to a one-hot matrix.

    Parameters
    ----------
    indices   : integer ndarray of shape (T,) with values in [0, vocab_size).
    vocab_size: number of unique characters.

    Returns
    -------
    one_hot : float64 ndarray of shape (T, vocab_size), each row is a
              standard basis vector e_i.
    """
    T = len(indices)
    one_hot = np.zeros((T, vocab_size), dtype=np.float64)
    one_hot[np.arange(T), indices.astype(int)] = 1.0
    return one_hot


def build_one_hot_spike_matrix(
    indices: np.ndarray,
    vocab_size: int,
    encoder,
    rng: np.random.Generator,
    deterministic: bool = False,
) -> np.ndarray:
    """Spike-encode a sequence of character indices using one-hot representation."""
    T = len(indices)
    N_sam = encoder.N_sam
    spike_matrix = np.zeros((T, vocab_size * N_sam), dtype=np.int8)

    for t in range(T):
        char_idx = int(indices[t])
        for v in range(vocab_size):
            channel_val = 1.0 if v == char_idx else 0.0
            spike_seq = encoder.encode_scalar(
                channel_val, 1.0, 0.0, rng=rng, deterministic=deterministic
            )
            spike_matrix[t, v * N_sam:(v + 1) * N_sam] = spike_seq

    return spike_matrix


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def char_accuracy(y_true_norm: np.ndarray, y_pred: np.ndarray,
                  vocab_size: int) -> float:
    """Fraction of time steps where the predicted character is correct."""
    if y_pred.ndim == 2:
        pred_idx = np.argmax(y_pred, axis=1)
    else:
        pred_idx = np.clip(np.round(denormalize(y_pred, vocab_size)),
                           0, vocab_size - 1).astype(int)
    
    true_idx = np.round(denormalize(y_true_norm, vocab_size)).astype(int)
    return float(np.mean(pred_idx == true_idx))


def top_k_accuracy(y_true_norm: np.ndarray, y_pred: np.ndarray,
                   vocab_size: int, k: int = 3) -> float:
    """Fraction where the true index is within k-best predictions."""
    true_idx = np.round(denormalize(y_true_norm, vocab_size)).astype(int)
    
    if y_pred.ndim == 2:
        top_k_indices = np.argsort(y_pred, axis=1)[:, -k:]
        correct = np.any(top_k_indices == true_idx[:, None], axis=1)
        return float(np.mean(correct))
    else:
        pred_cont = denormalize(y_pred, vocab_size)
        return float(np.mean(np.abs(pred_cont - true_idx) <= k / 2.0))


# ---------------------------------------------------------------------------
# Autoregressive text generation
# ---------------------------------------------------------------------------

def generate_text(model, seed_indices: np.ndarray, n_chars: int,
                  vocab_size: int, int_to_char: dict,
                  temperature: float = 0.0,
                  encoding: str = "scalar",
                  deterministic: bool = False) -> str:
    """Generate text autoregressively using the trained model.

    The reservoir is first warmed up on `seed_indices` and then rolled out
    one character at a time. The predicted value is rounded to the nearest
    valid character index and fed back as the next input.

    Parameters
    ----------
    model        : trained SpikeESN instance
    seed_indices : raw integer character indices for the seed (shape (S,))
    n_chars      : number of new characters to generate
    vocab_size   : total vocabulary size
    int_to_char  : index → character mapping
    temperature  : std of Gaussian noise added to output (0 = greedy)
    encoding     : 'scalar' or 'one-hot'
    deterministic: whether to use deterministic spike encoding
    """
    encoder   = model.encoder
    reservoir = model.reservoir
    rng = np.random.default_rng(0)

    x = np.zeros(model.N_res)

    # Warm up the reservoir on the seed
    for char_idx in seed_indices.astype(int):
        if encoding == "one-hot":
            # One-hot channels concatenated into a single spike row
            spike_row = np.zeros(vocab_size * encoder.N_sam, dtype=np.int8)
            for v in range(vocab_size):
                channel_val = 1.0 if v == char_idx else 0.0
                spike_row[v * encoder.N_sam:(v + 1) * encoder.N_sam] = \
                    encoder.encode_scalar(channel_val, 1.0, 0.0, rng=rng, 
                                          deterministic=deterministic)
            f_spike = reservoir.compute_spike_current(spike_row)
        else:
            val = normalize(np.array([char_idx], dtype=np.float64), vocab_size)[0]
            spike_seq = encoder.encode_scalar(val, 1.0, 0.0, rng=rng,
                                              deterministic=deterministic)
            f_spike   = reservoir.compute_spike_current(spike_seq)
        x = reservoir.update_state(f_spike, x)

    generated = []
    current_idx = int(seed_indices[-1])

    for _ in range(n_chars):
        if encoding == "one-hot":
            spike_row = np.zeros(vocab_size * encoder.N_sam, dtype=np.int8)
            for v in range(vocab_size):
                channel_val = 1.0 if v == current_idx else 0.0
                spike_row[v * encoder.N_sam:(v + 1) * encoder.N_sam] = \
                    encoder.encode_scalar(channel_val, 1.0, 0.0, rng=rng,
                                          deterministic=deterministic)
            f_spike = reservoir.compute_spike_current(spike_row)
        else:
            val = normalize(np.array([current_idx], dtype=np.float64), vocab_size)[0]
            spike_seq = encoder.encode_scalar(val, 1.0, 0.0, rng=rng,
                                              deterministic=deterministic)
            f_spike   = reservoir.compute_spike_current(spike_seq)

        x = reservoir.update_state(f_spike, x)
        y_hat = (model.W_out @ x).flatten()

        if temperature > 0:
            y_hat += rng.normal(0, temperature, size=y_hat.shape)

        # Clamp and decode
        if encoding == "one-hot":
            current_idx = int(np.argmax(y_hat))
        else:
            current_idx = int(np.clip(round(denormalize(y_hat[0], vocab_size)),
                                      0, vocab_size - 1))
        generated.append(current_idx)

    return "".join(int_to_char[i] for i in generated)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Spike-ESN character-level language model on Tiny Shakespeare",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-len",   type=int,   default=5000,   help="Training chars")
    p.add_argument("--test-len",    type=int,   default=1000,   help="Test chars")
    p.add_argument("--washout",     type=int,   default=200,    help="Reservoir washout steps")
    p.add_argument("--N-res",       type=int,   default=500,    help="Reservoir size")
    p.add_argument("--N-sam",       type=int,   default=50,     help="Spike sampling times")
    p.add_argument("--rho",         type=float, default=0.9,    help="Spectral radius")
    p.add_argument("--eta",         type=float, default=0.1,    help="Reservoir sparsity")
    p.add_argument("--mu",          type=float, default=1e-4,   help="Ridge regularisation")
    p.add_argument("--psi",         type=float, default=2000.0, help="Synaptic time constant")
    p.add_argument("--input-scaling", type=float, default=0.8,    help="Input weight scaling")
    p.add_argument("--seed",        type=int,   default=42,     help="Random seed")
    p.add_argument("--gen-len",     type=int,   default=200,    help="Characters to generate")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Generation noise std in normalised space (0 = greedy)")
    p.add_argument("--encoding", choices=["scalar", "one-hot"],
                   default="scalar",
                   help="Input encoding: 'scalar' (normalised index) or "
                        "'one-hot' (orthogonal categorical vectors)")
    p.add_argument("--deterministic", action="store_true",
                   help="Use deterministic spike encoding (stable patterns)")
    p.add_argument("--no-baseline", action="store_true",
                   help="Skip baseline ESN")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Locate data file relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "tinyshakespeare.txt")

    if not os.path.exists(filepath):
        print(f"[ERROR] Data file not found at {filepath}")
        return

    # -----------------------------------------------------------------------
    # Data preparation
    # -----------------------------------------------------------------------
    print("Loading data...")
    data, chars, char_to_int, int_to_char = load_data(filepath)
    vocab_size = len(chars)

    total_needed = args.train_len + args.test_len + 1
    if total_needed > len(data):
        raise ValueError(
            f"Dataset too small ({len(data)} chars) for requested split "
            f"({total_needed} needed)."
        )

    # Raw integer indices (needed for one-hot and for decoding)
    data_int = data.astype(int)

    # Normalised indices in [0, 1] — used for scalar encoding and targets
    data_norm = normalize(data, vocab_size)

    # Targets are always scalar-normalised (regression output)
    y_train = data_norm[1:args.train_len + 1]
    y_test  = data_norm[args.train_len + 1:args.train_len + args.test_len + 1]

    print(f"  Vocab size      : {vocab_size}")
    print(f"  Train / Test    : {args.train_len} / {args.test_len} chars")
    print(f"  Washout         : {args.washout}")
    print(f"  Encoding        : {args.encoding}")
    print(f"  Deterministic   : {args.deterministic}")

    # -----------------------------------------------------------------------
    # Spike-ESN
    # -----------------------------------------------------------------------
    from spike_esn.model import SpikeESN

    # When using one-hot encoding, each input step is represented by
    # vocab_size independent spike channels → W_in grows accordingly.
    effective_N_sam = (
        vocab_size * args.N_sam
        if args.encoding == "one-hot"
        else args.N_sam
    )

    print(f"\nFitting SpikeESN  (N_res={args.N_res}, N_sam={args.N_sam}, "
          f"effective_input_dim={effective_N_sam})...")
    t0 = time.perf_counter()
    # Always construct with per-channel N_sam so encoder.N_sam stays correct.
    # For one-hot mode, W_in is then resized to (N_res, effective_N_sam).
    model = SpikeESN(
        N_res=args.N_res, N_sam=args.N_sam,
        rho=args.rho, eta=args.eta, mu=args.mu, psi=args.psi,
        input_scaling=args.input_scaling, seed=args.seed,
    )
    if args.encoding == "one-hot":
        # Resize W_in: (N_res, effective_N_sam) — keeps same RNG stream seed
        rng_win = np.random.default_rng(args.seed)
        model.reservoir.W_in = (
            rng_win.uniform(-1, 1, size=(args.N_res, effective_N_sam))
            * args.input_scaling
        )

    if args.encoding == "one-hot":
        # Build the full one-hot spike matrix outside of model.fit(),
        # then pass the pre-computed spike matrix to the reservoir directly.
        rng_enc = np.random.default_rng(args.seed)
        print("  Building one-hot spike matrix (train)...")
        spike_matrix_train = build_one_hot_spike_matrix(
            data_int[:args.train_len], vocab_size,
            model.encoder, rng_enc,
            deterministic=args.deterministic
        )
        # Harvest reservoir states and fit readout
        X_train = model.reservoir.harvest_states(
            spike_matrix_train, washout=args.washout
        )
        T_eff = X_train.shape[1]
        y_train_one_hot = to_one_hot(data_int[1:args.train_len + 1], vocab_size).T
        y_target = y_train_one_hot[:, args.washout:args.washout + T_eff]
        XXT = X_train @ X_train.T
        reg = model.mu * np.eye(model.N_res)
        model.W_out = y_target @ X_train.T @ np.linalg.inv(XXT + reg)
    else:
        u_train = data_norm[:args.train_len]
        model.fit(u_train, y_train, washout=args.washout)

    t_fit = time.perf_counter() - t0
    print(f"  Done in {t_fit:.1f}s")

    if args.encoding == "one-hot":
        rng_enc2 = np.random.default_rng(args.seed)
        print("  Building one-hot spike matrix (test)...")
        spike_matrix_test = build_one_hot_spike_matrix(
            data_int[args.train_len:args.train_len + args.test_len],
            vocab_size, model.encoder, rng_enc2,
            deterministic=args.deterministic
        )
        X_test = model.reservoir.harvest_states(spike_matrix_test, washout=0)
        y_pred = (model.W_out @ X_test).T
    else:
        u_test = data_norm[args.train_len:args.train_len + args.test_len]
        y_pred = model.predict(u_test, washout=0)

    if args.encoding == "one-hot":
        y_test_one_hot = to_one_hot(data_int[args.train_len + 1:args.train_len + args.test_len + 1], vocab_size)
        rmse     = model.rmse(y_test_one_hot, y_pred)
    else:
        rmse     = model.rmse(y_test, y_pred)

    acc      = char_accuracy(y_test, y_pred, vocab_size)
    top3     = top_k_accuracy(y_test, y_pred, vocab_size, k=3)
    print(f"  RMSE (norm)     : {rmse:.5f}")
    print(f"  Char accuracy   : {acc * 100:.2f}%")
    print(f"  Top-3 accuracy  : {top3 * 100:.2f}%")

    # -----------------------------------------------------------------------
    # Autoregressive generation
    # -----------------------------------------------------------------------
    seed_len     = min(50, args.washout)
    seed_indices = data_int[:seed_len]
    seed_text    = "".join(int_to_char[i] for i in seed_indices)

    print(f"\n--- Text Generation (seed: {repr(seed_text)}) ---")
    gen_text = generate_text(
        model, seed_indices, n_chars=args.gen_len,
        vocab_size=vocab_size, int_to_char=int_to_char,
        temperature=args.temperature,
        encoding=args.encoding,
        deterministic=args.deterministic
    )
    print(gen_text)

    # -----------------------------------------------------------------------
    # Baseline ESN comparison
    # -----------------------------------------------------------------------
    if not args.no_baseline:
        from spike_esn.baseline_esn import ESN

        # Baseline ESN always uses scalar encoding
        u_train_scalar = data_norm[:args.train_len]
        u_test_scalar  = data_norm[args.train_len:args.train_len + args.test_len]

        print(f"\nFitting Baseline ESN (N_res={args.N_res})...")
        t0 = time.perf_counter()
        baseline = ESN(N_res=args.N_res, rho=args.rho, eta=args.eta,
                       mu=args.mu, input_scaling=args.input_scaling,
                       seed=args.seed)
        baseline.fit(u_train_scalar, y_train, washout=args.washout)
        t_base = time.perf_counter() - t0
        print(f"  Done in {t_base:.1f}s")

        y_pred_base = baseline.predict(u_test_scalar, washout=0)
        rmse_base   = baseline.rmse(y_test, y_pred_base)
        acc_base    = char_accuracy(y_test, y_pred_base, vocab_size)
        top3_base   = top_k_accuracy(y_test, y_pred_base, vocab_size, k=3)
        print(f"  RMSE (norm)     : {rmse_base:.5f}")
        print(f"  Char accuracy   : {acc_base * 100:.2f}%")
        print(f"  Top-3 accuracy  : {top3_base * 100:.2f}%")

        # Summary table
        print("\n" + "=" * 50)
        print(f"{'Model':<14} {'RMSE':>9} {'Acc%':>8} {'Top3%':>8} {'Time':>7}")
        print("-" * 50)
        print(f"{'SpikeESN':<14} {rmse:>9.5f} {acc*100:>8.2f} {top3*100:>8.2f} {t_fit:>6.1f}s")
        print(f"{'Baseline ESN':<14} {rmse_base:>9.5f} {acc_base*100:>8.2f} {top3_base*100:>8.2f} {t_base:>6.1f}s")
        print("=" * 50)


if __name__ == "__main__":
    main()
