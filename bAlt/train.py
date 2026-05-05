"""
bAlt/train.py — Train an all-to-all connected network on Tiny Shakespeare 
using the "Contribute to balance, wire in accordance" Backpropagation Alternative.
"""

import argparse
import os
import time
import numpy as np

def load_data(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    data = np.array([char_to_int[c] for c in text], dtype=np.float64)
    return data, chars, char_to_int, int_to_char

def to_one_hot(indices: np.ndarray, vocab_size: int) -> np.ndarray:
    T = len(indices)
    one_hot = np.zeros((T, vocab_size), dtype=np.float64)
    one_hot[np.arange(T), indices.astype(int)] = 1.0
    return one_hot

def char_accuracy(y_true_idx: np.ndarray, y_pred: np.ndarray) -> float:
    pred_idx = np.argmax(y_pred, axis=1)
    return float(np.mean(pred_idx == y_true_idx))

def top_k_accuracy(y_true_idx: np.ndarray, y_pred: np.ndarray, k: int = 3) -> float:
    top_k_indices = np.argsort(y_pred, axis=1)[:, -k:]
    correct = np.any(top_k_indices == y_true_idx[:, None], axis=1)
    return float(np.mean(correct))

def rmse_metric(y_true_one_hot: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true_one_hot - y_pred)**2)))

class BAltNet:
    def __init__(self, N: int, vocab_size: int, learning_rate: float, prune_p: float = 0.0, rewire_threshold: float = 0.0, seed: int = 42):
        self.N = N
        self.V = vocab_size
        self.lr = learning_rate
        self.prune_p = prune_p
        self.rewire_threshold = rewire_threshold
        if self.N <= 2 * self.V:
            raise ValueError(f"N must be > 2*V ({2*self.V}) to have hidden neurons.")
            
        rng = np.random.default_rng(seed)
        # Initialize W with scale 0.1
        self.W = rng.normal(0, 0.1 / np.sqrt(N), size=(N, N))
        self.x = np.zeros(N)
        self.c = np.zeros(N)

    def train_step(self, u: np.ndarray, y: np.ndarray, accumulate=False):
        # 1. Forward Firing Dynamic
        s = self.W @ self.x
        x_new = np.tanh(s)
        x_new[:self.V] = u  # Input clamping
        self.x = x_new
        
        # 2. Backward Credit Dynamic
        s_post = self.W @ self.x
        f_s = 1.0 - np.tanh(s_post)**2
        c_new = self.W.T @ (self.c * f_s)
        c_new[-self.V:] = y - self.x[-self.V:]  # Output clamping
        self.c = c_new
        
        # 3. Plasticity Rule
        if accumulate:
            return self.c * f_s, self.x
            
        dW = np.outer(self.c * f_s, self.x)
        self.W += self.lr * dW
        self.prune_weights()
        self.rewire_weights()
        return self.x[-self.V:]
        
    def apply_update(self, C_batch: np.ndarray, X_batch: np.ndarray):
        # C_batch: (B, N), X_batch: (B, N)
        dW = C_batch.T @ X_batch
        self.W += self.lr * dW
        self.prune_weights()
        self.rewire_weights()

    def prune_weights(self):
        if self.prune_p <= 0.0:
            return
        abs_W = np.abs(self.W)
        threshold = np.percentile(abs_W, self.prune_p * 100)
        self.W[abs_W < threshold] = 0.0

    def rewire_weights(self):
        if self.rewire_threshold <= 0.0:
            return
        out_sums = np.sum(np.abs(self.W), axis=0)
        low_neurons = np.where(out_sums < self.rewire_threshold)[0]
        
        for j in low_neurons:
            zeros = np.where(self.W[:, j] == 0.0)[0]
            if len(zeros) > 0:
                target_i = np.random.choice(zeros)
                self.W[target_i, j] = np.random.normal(0, 0.1 / np.sqrt(self.N))

    def predict_step(self, u: np.ndarray):
        s = self.W @ self.x
        self.x = np.tanh(s)
        self.x[:self.V] = u
        return self.x[-self.V:]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-len", type=int, default=5000)
    p.add_argument("--test-len", type=int, default=1000)
    p.add_argument("--washout", type=int, default=200)
    p.add_argument("--N", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gen-len", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=50, help="Steps to accumulate before updating W")
    p.add_argument("--prune-p", type=float, default=0.0, help="Fraction of weights to prune (e.g., 0.1 for 10%)")
    p.add_argument("--rewire-threshold", type=float, default=0.0, help="Add outgoing weight if sum is below this")
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "tinyshakespeare.txt")

    if not os.path.exists(filepath):
        print(f"[ERROR] Data file not found at {filepath}")
        return

    print("Loading data...")
    data, chars, char_to_int, int_to_char = load_data(filepath)
    vocab_size = len(chars)

    data_int = data.astype(int)
    
    print(f"  Vocab size      : {vocab_size}")
    print(f"  Train / Test    : {args.train_len} / {args.test_len} chars")
    print(f"  Washout         : {args.washout}")
    print(f"  N               : {args.N}")
    print(f"  Learning Rate   : {args.lr}")
    print(f"  Pruning fraction: {args.prune_p}")
    print(f"  Rewire threshold: {args.rewire_threshold}")

    model = BAltNet(N=args.N, vocab_size=vocab_size, learning_rate=args.lr, 
                    prune_p=args.prune_p, rewire_threshold=args.rewire_threshold, seed=args.seed)

    # Convert to one-hot for fast access
    train_int = data_int[:args.train_len+1]
    train_oh = to_one_hot(train_int, vocab_size)
    
    print("\nTraining BAltNet...")
    t0 = time.perf_counter()
    C_acc, X_acc = [], []
    for t in range(args.train_len):
        u = train_oh[t]
        y = train_oh[t+1]
        
        if args.batch_size > 1:
            c_eff, x_state = model.train_step(u, y, accumulate=True)
            C_acc.append(c_eff)
            X_acc.append(x_state)
            
            if len(C_acc) == args.batch_size:
                model.apply_update(np.array(C_acc), np.array(X_acc))
                C_acc, X_acc = [], []
        else:
            model.train_step(u, y)
            
        if (t + 1) % 1000 == 0:
            print(f"  Trained {t+1}/{args.train_len} chars...", end='\r')
            
    if len(C_acc) > 0:
        model.apply_update(np.array(C_acc), np.array(X_acc))
            
    t_fit = time.perf_counter() - t0
    print(f"\n  Done in {t_fit:.1f}s")

    print("\nEvaluating...")
    test_int = data_int[args.train_len : args.train_len + args.test_len + 1]
    test_oh = to_one_hot(test_int, vocab_size)
    
    # Washout before testing
    for t in range(args.washout):
        model.predict_step(test_oh[t])
        
    y_preds = []
    for t in range(args.washout, args.test_len):
        y_pred = model.predict_step(test_oh[t])
        y_preds.append(y_pred)
        
    y_preds = np.array(y_preds)
    y_true_oh = test_oh[args.washout+1 : args.test_len+1]
    y_true_idx = test_int[args.washout+1 : args.test_len+1]
    
    rmse = rmse_metric(y_true_oh, y_preds)
    acc = char_accuracy(y_true_idx, y_preds)
    top3 = top_k_accuracy(y_true_idx, y_preds, k=3)
    
    print(f"  RMSE (one-hot)  : {rmse:.5f}")
    print(f"  Char accuracy   : {acc * 100:.2f}%")
    print(f"  Top-3 accuracy  : {top3 * 100:.2f}%")

    print(f"\n--- Text Generation ---")
    seed_len = max(1, min(50, args.washout))
    seed_indices = data_int[:seed_len]
    seed_text = "".join(int_to_char[i] for i in seed_indices)
    print(f"Seed: {repr(seed_text)}")
    
    model.x = np.zeros(model.N)
    for char_idx in seed_indices:
        u = to_one_hot(np.array([char_idx]), vocab_size)[0]
        model.predict_step(u)
        
    generated = []
    current_idx = int(seed_indices[-1])
    
    rng = np.random.default_rng(args.seed)
    
    for _ in range(args.gen_len):
        u = to_one_hot(np.array([current_idx]), vocab_size)[0]
        y_hat = model.predict_step(u)
        
        if args.temperature > 0:
            y_hat += rng.normal(0, args.temperature, size=y_hat.shape)
            
        current_idx = int(np.argmax(y_hat))
        generated.append(current_idx)
        
    gen_text = "".join(int_to_char[i] for i in generated)
    print(gen_text)

if __name__ == "__main__":
    main()
