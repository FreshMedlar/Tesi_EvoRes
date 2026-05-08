import numpy as np
from model import PlasticRNN
import os

# ---------------------------------------------------------------------------
# Data loading (adapted from spike_esn/train_shakespeare.py)
# ---------------------------------------------------------------------------

def load_data(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    data = np.array([char_to_int[c] for c in text], dtype=np.float64)
    return data, chars, char_to_int, int_to_char

def generate_shakespeare_episode(data, seq_len=50):
    """Pick a random slice of the data."""
    start_idx = np.random.randint(0, len(data) - seq_len - 1)
    # Input is sequence, target is sequence shifted by 1
    input_indices = data[start_idx : start_idx + seq_len].astype(int)
    target_indices = data[start_idx + 1 : start_idx + seq_len + 1].astype(int)
    return input_indices, target_indices

def run_inner_loop_shakespeare(rnn, data, vocab_size, H=5, T=50):
    """
    Run H episodes of character prediction.
    """
    episode_data = []
    rewards = []
    
    for h in range(H):
        input_indices, target_indices = generate_shakespeare_episode(data, seq_len=T)
        rnn.reset_state()
        
        correct_count = 0
        for t in range(T):
            # One-hot input
            u = np.zeros(vocab_size)
            u[input_indices[t]] = 1.0
            
            # Prediction
            out = rnn.step(u)
            
            # Check prediction (at the end of response period or every step?)
            # The paper framework is for sparse feedback. Let's give reward at the end.
            if np.argmax(out) == target_indices[t]:
                correct_count += 1
        
        # Reward = accuracy during the episode
        reward = correct_count / T
        rewards.append(reward)
        
        # Reward error before update
        reward_error = reward - rnn.expected_reward
        
        # Capture z at the end of episode (before weight update)
        z_end = rnn.z.copy()
        
        # Update weights and get noise
        noise, mu = rnn.update_weights(reward)
        
        episode_data.append({
            'reward': reward,
            'reward_error': reward_error,
            'z_end': z_end,
            'noise': noise
        })
        
    return rewards, episode_data

def train_meta_shakespeare():
    # Load data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "tinyshakespeare.txt")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return
    
    data, chars, char_to_int, int_to_char = load_data(filepath)
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")

    # Network parameters
    N = 100 # Larger network for Shakespeare
    N_in = vocab_size
    N_out = vocab_size
    H = 10  # Episodes per inner loop
    T = 30  # Sequence length
    Meta_Steps = 50
    meta_lr = 0.0001
    
    rnn = PlasticRNN(N, N_in, N_out, eta=0.01, sigma_w=0.01)
    
    # Initialize with Hebbian-like term
    rnn.theta = np.zeros((6, 6))
    rnn.theta[1, 1] = 0.01 
    
    print(f"Starting meta-training on Shakespeare...")
    
    for step in range(Meta_Steps):
        S = 2 # Fewer sessions due to computation cost
        batch_grad = np.zeros((6, 6))
        total_reward = 0
        
        for s in range(S):
            # Reset weights for each session
            rnn.W = np.random.randn(N, N) * (1.0 / np.sqrt(N))
            rnn.U = np.zeros((N, N, 6, 6))
            rnn.expected_reward = 0.0
            
            rewards, data_batch = run_inner_loop_shakespeare(rnn, data, vocab_size, H=H, T=T)
            total_reward += sum(rewards)
            
            # Meta-gradient calculation
            for h in range(H-1):
                future_reward_error_sum = sum(data_batch[hp]['reward_error'] for hp in range(h+1, H))
                noise = data_batch[h]['noise']
                z = data_batch[h]['z_end']
                delta_R_h = data_batch[h]['reward_error']
                
                dmu_dtheta = rnn.eta * delta_R_h * z
                grad_h = np.einsum('ij,ijkl->kl', noise, dmu_dtheta)
                batch_grad += (future_reward_error_sum / (rnn.sigma_w**2)) * grad_h
        
        avg_grad = batch_grad / S
        grad_norm = np.linalg.norm(avg_grad)
        if grad_norm > 1.0:
            avg_grad = avg_grad / grad_norm
            
        rnn.theta += meta_lr * avg_grad
        
        avg_step_reward = total_reward / (S * H)
        
        if step % 1 == 0:
            print(f"Step {step:2d} | Avg Acc: {avg_step_reward:.4f} | Theta[1,1]: {rnn.theta[1,1]:.6f} | Grad Norm: {grad_norm:.4f}")

    print("Meta-training complete.")

if __name__ == "__main__":
    train_meta_shakespeare()
