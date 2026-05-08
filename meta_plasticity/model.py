import numpy as np

class PlasticRNN:
    """
    Recurrent Neural Network with meta-learnable three-factor plasticity rules.
    Implemented as described in "Meta-learning three-factor plasticity rules for 
    structured credit assignment with sparse feedback".
    """
    def __init__(self, N, N_in, N_out, dt=0.1, tau=1.0, tau_e=5.0, eta=0.01, sigma_w=0.01, alpha_x=0.9):
        self.N = N
        self.N_in = N_in
        self.N_out = N_out
        self.dt = dt
        self.tau = tau
        self.tau_e = tau_e
        self.eta = eta
        self.sigma_w = sigma_w
        self.alpha_x = alpha_x # Decay for running average x_bar
        
        # Meta-parameters: theta_{k,l} for 0 <= k, l <= 5
        self.theta = np.zeros((6, 6))
        
        # Initial weights
        self.W = np.random.randn(N, N) * (1.0 / np.sqrt(N))
        self.W_in = np.random.randn(N, N_in) * (1.0 / np.sqrt(N_in))
        self.W_out = np.random.randn(N_out, N) * (1.0 / np.sqrt(N))
        
        # Inter-trial weight matrix tangent U = dW/dtheta (N, N, 6, 6)
        self.U = np.zeros((N, N, 6, 6))
        
        # Reward tracking
        self.expected_reward = 0.0
        self.reward_alpha = 0.1

        self.reset_state()

    def reset_state(self):
        """Reset internal state for a new episode."""
        self.x = np.zeros(self.N)
        self.r = np.tanh(self.x)
        self.x_bar = np.zeros(self.N)
        self.e = np.zeros((self.N, self.N))
        
        # Within-trial tangents
        # chi: dx/dtheta (N, 6, 6)
        # psi: dx_bar/dtheta (N, 6, 6)
        # z: de/dtheta (N, N, 6, 6)
        self.chi = np.zeros((self.N, 6, 6))
        self.psi = np.zeros((self.N, 6, 6))
        self.z = np.zeros((self.N, self.N, 6, 6))

    def step(self, u):
        """
        Perform one time step of the network dynamics and update eligibility/tangents.
        u: input vector (N_in,)
        """
        alpha = self.dt / self.tau
        
        # Current activations and derivatives
        r_t = self.r
        phi_prime = 1.0 - r_t**2 # tanh'(x) = 1 - tanh^2(x)
        
        # 1. Update State x (Eq 209)
        dx = -self.x + self.W @ r_t + self.W_in @ u
        self.x = self.x + alpha * dx
        self.r = np.tanh(self.x)
        
        # 2. Update Running Average x_bar
        # Note: Eq 220 doesn't specify x_bar dynamics, but Sec C mentions psi = d x_bar / d theta.
        # It says psi^{t+1} = alpha_x psi^t + (1-alpha_x) chi^{t+1}.
        # This implies x_bar^{t+1} = alpha_x x_bar^t + (1-alpha_x) x^{t+1}.
        old_x_bar = self.x_bar
        self.x_bar = self.alpha_x * old_x_bar + (1.0 - self.alpha_x) * self.x
        
        # 3. Update Eligibility Trace e (Eq 220)
        # de/dt = H_theta(r_j, x_i) - e/tau_e
        # H_theta = sum theta_kl (r_j)^k (x_bar_i - x_i)^l
        delta_x = self.x_bar - self.x
        
        # Precompute powers
        r_powers = np.array([r_t**k for k in range(6)]) # (6, N)
        dx_powers = np.array([delta_x**l for l in range(6)]) # (6, N)
        
        # H_theta(r_j, x_i)
        # We can compute this as sum_{k,l} theta_kl * (dx_powers[l, i] * r_powers[k, j])
        # This is a bit slow if not vectorized.
        # H is (N, N). H[i, j] = sum_kl theta_kl * dx[l, i] * r[k, j]
        # H = sum_kl theta_kl * (dx_powers[l][:, None] @ r_powers[k][None, :])
        H = np.zeros((self.N, self.N))
        for k in range(6):
            for l in range(6):
                if self.theta[k, l] != 0:
                    H += self.theta[k, l] * np.outer(dx_powers[l], r_powers[k])
        
        de = H - self.e / self.tau_e
        self.e = self.e + self.dt * de
        
        # 4. Update Within-trial Tangents (Sec C)
        # chi (N, 6, 6)
        # psi (N, 6, 6)
        # z (N, N, 6, 6)
        
        chi_flat = self.chi.reshape(self.N, 36)
        U_flat = self.U.reshape(self.N, self.N, 36)
        
        # U_flat @ r_t -> (N, 36)
        U_r = np.tensordot(U_flat, r_t, axes=([1], [0])) # (N, 36)
        
        # W @ (phi_prime * chi_flat) -> (N, 36)
        W_phi_chi = self.W @ (phi_prime[:, None] * chi_flat)
        
        new_chi_flat = chi_flat + alpha * (-chi_flat + W_phi_chi + U_r)
        self.chi = new_chi_flat.reshape(self.N, 6, 6)
        
        # psi^{t+1} = alpha_x psi^t + (1 - alpha_x) chi^{t+1}
        self.psi = self.alpha_x * self.psi + (1.0 - self.alpha_x) * self.chi
        
        # z^{t+1} (Eq 380-381) optimized
        S_A = np.zeros((6, self.N))
        S_B = np.zeros((6, self.N))
        for kappa in range(6):
            for lambda_ in range(1, 6): # lambda=0 gives 0
                S_A[kappa] += self.theta[kappa, lambda_] * lambda_ * (delta_x**(lambda_-1))
        for lambda_ in range(6):
            for kappa in range(1, 6): # kappa=0 gives 0
                S_B[lambda_] += self.theta[kappa, lambda_] * kappa * (r_t**(kappa-1))
        
        vec_A = np.zeros(self.N)
        for kappa in range(6):
            vec_A += S_A[kappa] * r_powers[kappa]
            
        vec_B = np.zeros(self.N)
        for lambda_ in range(6):
            vec_B += S_B[lambda_] * dx_powers[lambda_]

        # Now update z for each (k, l)
        V1 = self.psi - self.chi # (N, 6, 6)
        V2 = phi_prime[:, None, None] * self.chi # (N, 6, 6)
        
        for k in range(6):
            for l in range(6):
                term_direct = np.outer(dx_powers[l], r_powers[k])
                term_indirect = np.outer(V1[:, k, l], vec_A) + np.outer(vec_B, V2[:, k, l])
                self.z[:, :, k, l] = self.z[:, :, k, l] * (1.0 - self.dt/self.tau_e) + \
                                     self.dt * (term_direct + term_indirect)

        return self.W_out @ self.r

    def update_weights(self, reward):
        """
        Update weights at the end of an episode (Eq 233).
        Returns the noise added to the weights.
        """
        reward_error = reward - self.expected_reward
        self.expected_reward = self.expected_reward + self.reward_alpha * reward_error
        
        # mu = eta * e * reward_error
        mu = self.eta * self.e * reward_error
        
        # Noise
        noise = np.random.randn(self.N, self.N) * self.sigma_w
        
        self.W = self.W + mu + noise
        
        # Update inter-trial tangent U (Eq 405)
        # U^{h+1} = U^h + d(mu)/dtheta
        # d(mu)/dtheta = eta * reward_error * z
        self.U += self.eta * reward_error * self.z

        return noise, mu

    def get_meta_gradient(self, rewards, delta_Ws_mean):
        """
        Compute the meta-gradient of the expected reward with respect to theta (Eq 256).
        rewards: list of rewards for each episode in the meta-batch
        delta_Ws_mean: list of the deterministic part of the weight updates for each episode
        """
        # Eq 263: Gradient is approx sum_h (sum_{h'=h+1}^H R_h') * (1/sigma_w^2) * sum_ij (delta_w_ij - mu_ij) * d_mu_ij/d_theta
        # Wait, the REINFORCE estimator in Eq 256 is:
        # grad = sum_h (sum_{h'=h+1}^H (R_h' - bar_R_h')) * grad_theta log pi(delta_W_h | theta)
        # log pi is -(1/2 sigma_w^2) * ||delta_W - mu||^2 + const
        # grad_theta log pi = (1/sigma_w^2) * (delta_W - mu)^T * (d_mu/d_theta)
        
        # Here mu_h = eta * e_h * (R_h - bar_R_h)
        # d_mu_h/d_theta = eta * (R_h - bar_R_h) * z_h
        
        # This function should be called after a meta-batch of episodes.
        # But wait, the model updates W every episode.
        pass
