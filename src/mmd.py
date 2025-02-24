import copy
import torch
import numpy as np


from torch import nn
from tqdm import tqdm
from itertools import product
from torch.utils.data import DataLoader, TensorDataset


class MMDKernels(nn.Module):
    def __init__(self, device, n_scales=6, mul_factor=2.0, scale=None):
        super().__init__()
        self.device = device
        self.exponent = torch.arange(n_scales) - n_scales // 2
        self.scale_multipliers = mul_factor ** (self.exponent).to(device)
        self.scale = scale

    def _compute_distances(self, X, Y):
        XY = torch.vstack([X, Y])
        distances = torch.cdist(XY, XY) ** 2
        return distances

    def _compute_scale(self, distances):
        if self.scale is not None:
            return self.scale

        n_samples = distances.shape[0]
        return distances.data.sum() / (n_samples**2 - n_samples)

    def _compute_kernel(self, distances, scale, multiplier):
        kernel = torch.exp(-distances / (scale * multiplier))
        return kernel

    def _compute_kernel_from_different_scales(self, distances, scale):
        kernel = torch.zeros_like(distances)
        for multiplier in self.scale_multipliers:
            kernel += self._compute_kernel(distances, scale, multiplier)

        return kernel

    def forward(self, X, Y):
        dists = self._compute_distances(X, Y)
        scale = self._compute_scale(dists)
        kernel = self._compute_kernel_from_different_scales(dists, scale)

        X_size = X.shape[0]
        kernel_x = kernel[:X_size, :X_size].mean()
        kernel_y = kernel[X_size:, X_size:].mean()
        kernel_xy = kernel[:X_size, X_size:].mean()

        return kernel_x, kernel_y, kernel_xy


class MMDLoss(nn.Module):
    def __init__(self, device, n_scales=5):
        super().__init__()
        self.kernel = MMDKernels(device=device, n_scales=n_scales)

    def forward(self, X, Y):
        k_x, k_y, k_xy = self.kernel(X, Y)

        return k_x - 2 * k_xy + k_y


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        out = self.layers(x)
        out += x
        return out


class MMDNet(nn.Module):
    def __init__(self, input_dim):
        super(MMDNet, self).__init__()
        self.res1 = ResidualBlock(input_dim, input_dim)

    def forward(self, x):
        out = self.res1(x)
        return out



def sign_cca(X, Y):
   X = torch.from_numpy(X).float()
   Y = torch.from_numpy(Y).float()

   dim = X.shape[1]
   signs = torch.ones(dim)  # Store signs for each column
   
   print("Start computing all permutations..")
   for i in range(dim):
       x_col = X[:, i]
       y_col = Y[:, i]

       correlation = torch.corrcoef(torch.stack((x_col, y_col)))[0, 1]
       sign = torch.sign(correlation)
       signs[i] = sign
       X[:, i] *= sign

   X = X.numpy()
   Y = Y.numpy()
   signs = signs.numpy()
   
   return X, Y, signs


def sign_mmd(X, Y, device=torch.device("cuda")):
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    
    dim = X.shape[1]
    sequences = torch.tensor(list(product([-1, 1], repeat=dim)))
    criterion = MMDLoss(device=device, n_scales=5)

    mmds = []
    min_seq = torch.tensor(sequences[0])
    min_loss = 1_000_000
    print("Start computing all permutations..")
    for seq in sequences:
        X_perm = X * seq
        X_perm = X_perm.to(device)
        Y = Y.to(device)
        mmd_loss = criterion(X_perm, Y)
        if mmd_loss < min_loss:
            min_loss = mmd_loss
            min_seq = seq

    X_final = X * min_seq
    X_final = X_final.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    return X_final, Y, min_seq.numpy()


def fine_tune_alignment_using_mmd_network(
    X, Y, device=torch.device("cuda"), epochs=100, batch_size=64, n_scales=5, val_split=0.1
):
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()

    # Split into train and validation sets
    n_samples = len(X)
    n_val = int(n_samples * val_split)
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    input_dim = X.shape[1]
    criterion = MMDLoss(device, n_scales=n_scales)

    # Create train and validation dataloaders
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MMDNet(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    best_model = None
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            x_output = model(x)
            loss = criterion(x_output, y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x_output = model(x)
                loss = criterion(x_output, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        # Progress update
        tqdm.write(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # Use best model for final transformation
    model = best_model
    model.eval()
    with torch.no_grad():
        X_transformed = model(X.to(device)).cpu().numpy()

    return X_transformed, Y.numpy(), model