import argparse
import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------

class AERC(nn.Module):
    """
    Attention-Enhanced Reservoir Computing (AERC) model.
    """
    def __init__(self, N, H, vocab_size):
        super().__init__()
        self.N = N
        self.H = H
        self.vocab_size = vocab_size
        
        # Attention Network F: Computes dynamic attention weights based on r_l
        # Consists of a single hidden layer equipped with a ReLU activation
        self.fc1 = nn.Linear(N, H)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(H, H * N)
        
        # Final Output Mapping
        self.W_out = nn.Linear(H, vocab_size)
        
    def forward(self, r_l):
        """
        r_l shape can be (Batch, SeqLen, N) or (Batch, N)
        """
        original_shape = r_l.shape
        if len(original_shape) == 3:
            B, L, N_dim = original_shape
            r_l_flat = r_l.view(B * L, N_dim)
        else:
            r_l_flat = r_l
            
        # 1. Attention Network (F)
        x = self.fc1(r_l_flat)
        x = self.relu(x)
        # dynamic attention weights W_att,l
        W_att = self.fc2(x) # shape: (B*L, H * N)
        
        # Reshape to (B*L, H, N)
        W_att = W_att.view(-1, self.H, self.N)
        
        # 2. Intermediate Projection: r_ol = W_att,l * r_l
        # Add a dummy dimension to r_l_flat for matrix multiplication
        r_l_unsqueeze = r_l_flat.unsqueeze(-1) # shape: (B*L, N, 1)
        
        # bmm( (B*L, H, N), (B*L, N, 1) ) -> (B*L, H, 1)
        r_ol = torch.bmm(W_att, r_l_unsqueeze).squeeze(-1) # shape: (B*L, H)
        
        # 3. Final Output Mapping
        logits = self.W_out(r_ol) # shape: (B*L, vocab_size)
        
        if len(original_shape) == 3:
            logits = logits.view(B, L, self.vocab_size)
            
        return logits

# ---------------------------------------------------------------------------
# Data and Reservoir Setup
# ---------------------------------------------------------------------------

class AERCDataset(Dataset):
    def __init__(self, r_states, targets, seq_len=32):
        self.r_states = r_states
        self.targets = targets
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.r_states) - self.seq_len
        
    def __getitem__(self, idx):
        return (
            self.r_states[idx : idx + self.seq_len],
            self.targets[idx : idx + self.seq_len]
        )

def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    data = np.array([char_to_int[c] for c in text], dtype=np.int64)
    return data, chars, char_to_int, int_to_char

def precompute_reservoir_states(data_indices, vocab_size, N=75, d=16, rho=0.9, input_scale=0.1, seed=42):
    """
    Computes all reservoir responses for the data shard to conserve compute power.
    r_t = tanh(r_{t-1} W_res + x_t W_in)
    """
    torch.manual_seed(seed)
    
    # 1. Fixed Embedding
    embedding = nn.Embedding(vocab_size, d)
    embedding.requires_grad_(False)
    
    # 2. Fixed Reservoir Matrix W_res
    W_res = torch.randn(N, N) / math.sqrt(N)
    # Scale to desired spectral radius
    eigenvalues = torch.linalg.eigvals(W_res)
    max_eig = torch.max(torch.abs(eigenvalues)).item()
    W_res = W_res * (rho / max_eig)
    
    # 3. Fixed Input Matrix W_in
    W_in = torch.randn(d, N) * input_scale
    
    # Move to GPU if available for faster precomputation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding = embedding.to(device)
    W_res = W_res.to(device)
    W_in = W_in.to(device)
    
    T = len(data_indices)
    # Calculate target indices (next character)
    # if data_indices is [c0, c1, ..., c_T]
    # x_t is c_t. target for r_t is c_{t+1}.
    # We will compute r_t for t=0 to T-2.
    inputs = torch.tensor(data_indices[:-1], dtype=torch.long, device=device)
    targets = torch.tensor(data_indices[1:], dtype=torch.long, device=device)
    
    T_eff = len(inputs)
    
    print(f"Precomputing reservoir states for {T_eff} steps on {device}...")
    t0 = time.time()
    
    # We can precompute all embeddings at once
    # Then iterate to compute reservoir states
    batch_x = embedding(inputs) # (T_eff, d)
    
    # Store results on CPU to save GPU memory if dataset is huge
    states = torch.zeros((T_eff, N), dtype=torch.float32)
    
    r = torch.zeros(N, device=device)
    for t in range(T_eff):
        r = torch.tanh(r @ W_res + batch_x[t] @ W_in)
        # Occasionally copy to CPU
        states[t] = r.cpu()
        
    t1 = time.time()
    print(f"Precomputation finished in {t1 - t0:.2f} seconds.")
    
    return states, targets.cpu()

# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Attention-Enhanced Reservoir Computing (AERC)")
    parser.add_argument("--N", type=int, default=75, help="Reservoir size (N)")
    parser.add_argument("--H", type=int, default=13, help="Attention hidden size (H)")
    parser.add_argument("--d", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length for sampling")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (Adam)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--train-len", type=int, default=100000, help="Number of characters to use for training")
    parser.add_argument("--val-len", type=int, default=10000, help="Number of characters for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Locate data file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "tinyshakespeare.txt")
    
    if not os.path.exists(filepath):
        print(f"[ERROR] Data file not found at {filepath}")
        return
        
    print("Loading data...")
    data_indices, chars, char_to_int, int_to_char = load_data(filepath)
    vocab_size = len(chars)
    
    total_needed = args.train_len + args.val_len + 1
    if len(data_indices) < total_needed:
        print(f"[WARNING] Requested {total_needed} characters but only {len(data_indices)} available.")
        # Adjust lengths if dataset is smaller
        args.train_len = int(len(data_indices) * 0.9)
        args.val_len = len(data_indices) - args.train_len - 1
        
    train_data = data_indices[:args.train_len+1]
    val_data = data_indices[args.train_len : args.train_len + args.val_len + 1]
    
    print(f"Vocab size: {vocab_size}")
    print(f"Train chars: {args.train_len}, Val chars: {args.val_len}")
    
    # Precompute reservoir states
    # Computational Efficiency: To conserve compute power, all reservoir responses for a data shard 
    # are computed and stored in memory before running epochs on that shard.
    print("\n--- Processing Training Data ---")
    r_states_train, targets_train = precompute_reservoir_states(
        train_data, vocab_size, N=args.N, d=args.d, seed=args.seed
    )
    
    print("\n--- Processing Validation Data ---")
    r_states_val, targets_val = precompute_reservoir_states(
        val_data, vocab_size, N=args.N, d=args.d, seed=args.seed + 1
    )
    
    train_dataset = AERCDataset(r_states_train, targets_train, seq_len=args.seq_len)
    val_dataset = AERCDataset(r_states_val, targets_val, seq_len=args.seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    model = AERC(N=args.N, H=args.H, vocab_size=vocab_size).to(device)
    print(f"Model parameters: {count_parameters(model)}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        t0 = time.time()
        for batch_r, batch_targets in train_loader:
            batch_r = batch_r.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch_r) # (B, L, vocab_size)
            
            # Cross entropy loss expects (N, C)
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = batch_targets.view(-1)
            
            loss = criterion(logits_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_r, batch_targets in val_loader:
                batch_r = batch_r.to(device)
                batch_targets = batch_targets.to(device)
                
                logits = model(batch_r)
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = batch_targets.view(-1)
                
                loss = criterion(logits_flat, targets_flat)
                val_loss += loss.item()
                
                # Accuracy
                preds = torch.argmax(logits_flat, dim=1)
                correct += (preds == targets_flat).sum().item()
                total += targets_flat.size(0)
                
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        t1 = time.time()
        print(f"Epoch {epoch}/{args.epochs} | Time: {t1-t0:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

if __name__ == "__main__":
    main()
