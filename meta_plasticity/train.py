import numpy as np
from model import PlasticRNN
import os

def generate_task_episode():
    """
    Generate a simple delayed association task.
    Input: 2 bits (one-hot).
    Delay: 30 steps.
    Response: 10 steps.
    """
    T = 50
    stim_idx = np.random.randint(2)
    inputs = np.zeros((T, 2))
    inputs[10, stim_idx] = 1.0 # Pulse stimulus at t=10
    
    target = np.zeros(2)
    target[stim_idx] = 1.0
    
    return inputs, target

def run_inner_loop(rnn, H=10, T=50):
    """
    Run H episodes of the task.
    rnn: PlasticRNN instance
    H: Number of episodes
    """
    episode_data = []
    rewards = []
    
    for h in range(H):
        inputs, target = generate_task_episode()
        rnn.reset_state()
        
        # Capture z BEFORE steps update it
        # Actually z is updated during steps. We need z at the END of the episode.
        
        outputs = []
        for t in range(T):
            out = rnn.step(inputs[t])
            if t >= 40: # Response period
                outputs.append(out)
        
        # Calculate reward
        avg_output = np.mean(outputs, axis=0)
        reward = 1.0 if np.argmax(avg_output) == np.argmax(target) else 0.0
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

def train_meta():
    N = 20
    N_in = 2
    N_out = 2
    H = 20 # Episodes per inner loop
    Meta_Steps = 50
    meta_lr = 0.0001
    
    rnn = PlasticRNN(N, N_in, N_out, eta=0.1, sigma_w=0.01)
    
    # Initialize theta: Hebbian term (k=1, l=1) could be a good start
    # theta_kl: k is power of r_j (pre), l is power of dx_i (post)
    rnn.theta = np.zeros((6, 6))
    rnn.theta[1, 1] = 0.1 # Hebbian-like
    
    history_rewards = []
    
    print(f"Starting meta-training for {Meta_Steps} steps...")
    
    for step in range(Meta_Steps):
        S = 4 # Sessions
        batch_grad = np.zeros((6, 6))
        total_reward = 0
        
        for s in range(S):
            # Reset weights for each session to evaluate "learnability"
            rnn.W = np.random.randn(N, N) * (1.0 / np.sqrt(N))
            rnn.U = np.zeros((N, N, 6, 6))
            rnn.expected_reward = 0.0
            
            rewards, data = run_inner_loop(rnn, H=H)
            total_reward += sum(rewards)
            
            # Compute meta-gradient for this session
            for h in range(H-1):
                # Sum of future reward errors
                future_reward_error_sum = sum(data[hp]['reward_error'] for hp in range(h+1, H))
                
                noise = data[h]['noise']
                z = data[h]['z_end']
                delta_R_h = data[h]['reward_error']
                
                # d_mu/d_theta = eta * delta_R_h * z
                dmu_dtheta = rnn.eta * delta_R_h * z
                
                # grad_h = (1/sigma_w^2) * sum_ij (noise_ij * dmu_dtheta_ij)
                grad_h = np.einsum('ij,ijkl->kl', noise, dmu_dtheta)
                
                batch_grad += (future_reward_error_sum / (rnn.sigma_w**2)) * grad_h
        
        # Average gradient over sessions
        avg_grad = batch_grad / S
        
        # Clip gradient for stability
        grad_norm = np.linalg.norm(avg_grad)
        if grad_norm > 1.0:
            avg_grad = avg_grad / grad_norm
            
        rnn.theta += meta_lr * avg_grad
        
        avg_step_reward = total_reward / (S * H)
        
        if step % 5 == 0:
            print(f"Step {step:3d} | Avg Reward: {avg_step_reward:.3f} | Theta[1,1]: {rnn.theta[1,1]:.4f} | Grad Norm: {grad_norm:.4f}")

    print("Meta-training complete.")

if __name__ == "__main__":
    train_meta()
