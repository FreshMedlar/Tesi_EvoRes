import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParalESNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, layer_idx=1,
                 tau=1.0, rho_min=0.1, rho_max=0.9, theta_min=0.0, theta_max=math.pi,
                 omega_b=0.1, omega_mix=0.1, omega_mix_b=0.1, kernel_size=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.tau = tau
        
        # Initialize transition matrix Lambda_h (diagonal, complex-valued)
        rho = torch.rand(hidden_size) * (rho_max - rho_min) + rho_min
        theta = torch.rand(hidden_size) * (theta_max - theta_min) + theta_min
        self.lambda_h = rho * torch.exp(1j * theta)
        
        # Effective transition matrix: Lambda_bar_h = (1 - tau) * I + tau * Lambda_h
        self.lambda_bar_h = (1 - tau) + tau * self.lambda_h
        
        # Initialize W_in
        if layer_idx == 1:
            W_in_real = torch.rand(hidden_size, input_size) * 2 - 1
            W_in_imag = torch.rand(hidden_size, input_size) * 2 - 1
            W_in_complex = W_in_real + 1j * W_in_imag
            
            scale = torch.sqrt(1 - torch.abs(self.lambda_bar_h)**2).unsqueeze(1)
            self.W_in = nn.Parameter(W_in_complex * scale, requires_grad=False)
        else:
            w_real = torch.rand(hidden_size) * 2 - 1
            w_imag = torch.rand(hidden_size) * 2 - 1
            w_complex = w_real + 1j * w_imag
            
            scale = torch.sqrt(1 - torch.abs(self.lambda_bar_h)**2)
            self.w_in = nn.Parameter(w_complex * scale, requires_grad=False)
            
        self.b = nn.Parameter((torch.rand(hidden_size) * 2 - 1 + 1j * (torch.rand(hidden_size) * 2 - 1)) * omega_b, requires_grad=False)
        self.lambda_bar_h = nn.Parameter(self.lambda_bar_h, requires_grad=False)
        
        # Mixing function f_mix
        self.kernel_size = kernel_size
        self.W_mix_real = nn.Parameter((torch.rand(1, 1, kernel_size) * 2 - 1) * omega_mix, requires_grad=False)
        self.W_mix_imag = nn.Parameter((torch.rand(1, 1, kernel_size) * 2 - 1) * omega_mix, requires_grad=False)
        self.b_mix_real = nn.Parameter((torch.rand(1) * 2 - 1) * omega_mix_b, requires_grad=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        if self.layer_idx == 1:
            u = torch.matmul(x.to(torch.complex64), self.W_in.T)
        else:
            x_shifted = torch.roll(x, shifts=1, dims=-1)
            u = x_shifted * self.w_in
            
        u = self.tau * (u + self.b)
        
        # Parallel associative scan using FFT
        powers = torch.arange(seq_len, device=u.device).unsqueeze(1)
        lambda_kernel = self.lambda_bar_h.unsqueeze(0) ** powers # (seq_len, hidden_size)
        
        n_fft = 2 ** math.ceil(math.log2(2 * seq_len - 1))
        U_f = torch.fft.fft(u, n=n_fft, dim=1)
        K_f = torch.fft.fft(lambda_kernel, n=n_fft, dim=0).unsqueeze(0)
        
        H_f = U_f * K_f
        h = torch.fft.ifft(H_f, n=n_fft, dim=1)[:, :seq_len, :]
        
        # Mixing function
        h_flat = h.reshape(batch_size * seq_len, 1, self.hidden_size)
        h_real = torch.real(h_flat)
        h_imag = torch.imag(h_flat)
        
        mix_real = F.conv1d(h_real, self.W_mix_real, padding='same')
        mix_imag = F.conv1d(h_imag, self.W_mix_imag, padding='same')
        
        z_flat = torch.tanh(mix_real - mix_imag + self.b_mix_real)
        z = z_flat.reshape(batch_size, seq_len, self.hidden_size)
        
        return z

class ParalESN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 tau=1.0, rho_min=0.1, rho_max=0.9, theta_min=0.0, theta_max=math.pi,
                 omega_b=0.1, omega_mix=0.1, omega_mix_b=0.1, kernel_size=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.layers = nn.ModuleList([
            ParalESNLayer(
                input_size=input_size if i == 1 else hidden_size,
                hidden_size=hidden_size,
                layer_idx=i,
                tau=tau,
                rho_min=rho_min,
                rho_max=rho_max,
                theta_min=theta_min,
                theta_max=theta_max,
                omega_b=omega_b,
                omega_mix=omega_mix,
                omega_mix_b=omega_mix_b,
                kernel_size=kernel_size
            ) for i in range(1, num_layers + 1)
        ])
        
        # Readout layer (the only trainable part)
        self.readout = nn.Linear(hidden_size * num_layers, output_size)

    def forward(self, x, return_states=False):
        # x: (batch_size, seq_len, input_size)
        states = []
        current_input = x
        for layer in self.layers:
            z = layer(current_input)
            states.append(z)
            current_input = z
            
        # Concatenate states from all layers along the hidden dimension
        z_all = torch.cat(states, dim=-1) # (batch_size, seq_len, hidden_size * num_layers)
        
        output = self.readout(z_all)
        
        if return_states:
            return output, z_all
        return output

    def fit_readout(self, x, y, lambda_reg=1e-4):
        """
        Fits the readout layer using Ridge Regression.
        x: (batch_size, seq_len, input_size)
        y: (batch_size, seq_len, output_size)
        """
        with torch.no_grad():
            _, states = self.forward(x, return_states=True)
            
        states_flat = states.reshape(-1, states.size(-1))
        y_flat = y.reshape(-1, y.size(-1))
        
        # Add bias term for regression
        states_bias = torch.cat([states_flat, torch.ones(states_flat.size(0), 1, device=states_flat.device)], dim=-1)
        
        ZTZ = torch.matmul(states_bias.T, states_bias)
        ZTY = torch.matmul(states_bias.T, y_flat)
        
        reg_matrix = lambda_reg * torch.eye(ZTZ.size(0), device=ZTZ.device)
        reg_matrix[-1, -1] = 0.0 # Don't regularize bias
        
        W_out = torch.linalg.solve(ZTZ + reg_matrix, ZTY).T
        
        self.readout.weight.data = W_out[:, :-1]
        self.readout.bias.data = W_out[:, -1]
