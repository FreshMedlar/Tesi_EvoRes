import os
import time
import math
import torch
import torch.nn.functional as F

from paral_esn import ParalESN

def load_text(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, vocab, stoi, itos

def make_loader(data: torch.Tensor, seq_len: int, batch_size: int, vocab_size: int, device: str):
    """
    Returns x_oh: [batch_size, seq_len, vocab_size]
            y:    [batch_size, seq_len]
    """
    def loader():
        ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
        x_idx = torch.stack([data[i:i+seq_len] for i in ix])
        y_idx = torch.stack([data[i+1:i+seq_len+1] for i in ix])

        x_oh = F.one_hot(x_idx, vocab_size).float()
        y = y_idx
        return x_oh.to(device), y.to(device)

    return loader

def evaluate(model, loader, n_batches):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = loader()
            logits = model(x) # (B, T, V)
            
            logits_flat = logits.reshape(-1, logits.size(-1))
            y_flat = y.reshape(-1)
            
            loss = F.cross_entropy(logits_flat, y_flat)
            total_loss += loss.item()
            
            preds = torch.argmax(logits_flat, dim=-1)
            correct += (preds == y_flat).sum().item()
            total += y_flat.size(0)
            
    return total_loss / n_batches, correct / total

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "./tinyshakespeare.txt"
    
    seq_len = 128
    batch_size = 32
    train_batches = 200 # number of batches to accumulate for Ridge Regression
    val_batches = 50
    
    # Model hyperparameters
    hidden_size = 1024
    num_layers = 2
    lambda_reg = 1e-3
    
    print(f"Device: {device}")
    print(f"Loading data from '{data_path}' ...")
    data, vocab_size, stoi, itos = load_text(data_path)
    
    n_train = int(0.9 * len(data))
    train_raw = data[:n_train]
    val_raw = data[n_train:]
    
    print(f"Vocab size: {vocab_size}")
    print(f"Train tokens: {len(train_raw):,}")
    print(f"Val tokens: {len(val_raw):,}")
    
    train_loader = make_loader(train_raw, seq_len, batch_size, vocab_size, device)
    val_loader = make_loader(val_raw, seq_len, batch_size, vocab_size, device)
    
    # Initialize model
    model = ParalESN(
        input_size=vocab_size,
        hidden_size=hidden_size,
        output_size=vocab_size,
        num_layers=num_layers
    ).to(device)
    
    print(f"\nTraining readout via batched Ridge Regression ({train_batches} batches) ...")
    start_time = time.time()
    
    ZTZ = torch.zeros(hidden_size * num_layers + 1, hidden_size * num_layers + 1, device=device)
    ZTY = torch.zeros(hidden_size * num_layers + 1, vocab_size, device=device)
    
    model.eval() # no train mode needed for ESN
    
    for i in range(train_batches):
        x, y = train_loader()
        
        with torch.no_grad():
            _, states = model(x, return_states=True)
            
        y_oh = F.one_hot(y, vocab_size).float()
        
        states_flat = states.reshape(-1, states.size(-1))
        y_flat = y_oh.reshape(-1, vocab_size)
        
        states_bias = torch.cat([states_flat, torch.ones(states_flat.size(0), 1, device=device)], dim=-1)
        
        ZTZ += torch.matmul(states_bias.T, states_bias)
        ZTY += torch.matmul(states_bias.T, y_flat)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{train_batches} batches")
            
    print("Solving Ridge Regression...")
    reg_matrix = lambda_reg * torch.eye(ZTZ.size(0), device=device)
    reg_matrix[-1, -1] = 0.0 # Don't regularize bias
    
    W_out = torch.linalg.solve(ZTZ + reg_matrix, ZTY).T
    
    model.readout.weight.data = W_out[:, :-1]
    model.readout.bias.data = W_out[:, -1]
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # Evaluate
    train_loss, train_acc = evaluate(model, train_loader, val_batches)
    val_loss, val_acc = evaluate(model, val_loader, val_batches)
    
    print("\nResults:")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")

if __name__ == "__main__":
    main()
