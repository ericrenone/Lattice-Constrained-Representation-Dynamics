import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import math

# ==============================
# Synthetic Dataset (10 clusters)
# ==============================
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, num_classes=10, noise=0.1):
        self.data = []
        self.labels = []
        for c in range(num_classes):
            center = np.random.uniform(-5, 5, 2)
            points = center + np.random.randn(num_samples // num_classes, 2) * noise
            self.data.extend(points)
            self.labels.extend([c] * (num_samples // num_classes))
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

# ==============================
# Batch Augmentation: Random Rotation
# ==============================
def augment_batch(x):
    batch_size = x.size(0)
    theta = torch.rand(batch_size) * 2 * math.pi
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    rot_matrices = torch.stack([
        torch.stack([cos_t, -sin_t], dim=1),
        torch.stack([sin_t,  cos_t], dim=1)
    ], dim=1)
    return torch.bmm(x.unsqueeze(1), rot_matrices).squeeze(1)

# ==============================
# Hierarchical MLP
# ==============================
class HierarchicalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 10)

    def forward(self, x, return_reps=False):
        rep1 = F.relu(self.layer1(x))
        rep2 = F.relu(self.layer2(rep1))
        out = self.output(rep2)
        if return_reps:
            return out, rep1, rep2
        return out

# ==============================
# Contrastive Loss (PyTorch)
# ==============================
def contrastive_loss(reps, aug_reps, temperature=0.5):
    reps = F.normalize(reps, dim=1)
    aug_reps = F.normalize(aug_reps, dim=1)
    sim_matrix = torch.mm(reps, aug_reps.t()) / temperature
    exp_sim = torch.exp(sim_matrix)
    pos = torch.diag(exp_sim)
    neg = exp_sim.sum(dim=1) - pos
    return -torch.log(pos / (pos + neg)).mean()

# ==============================
# Metrics
# ==============================
def compute_avg_sim(reps, labels):
    reps = reps.detach().numpy() if torch.is_tensor(reps) else reps
    sims = []
    for lbl in np.unique(labels):
        cls = reps[np.array(labels) == lbl]
        if len(cls) > 1:
            sims.append(cosine_similarity(cls).mean())
    return np.mean(sims) if sims else 0.0

def proxy_join_dim(reps):
    reps = reps.detach().numpy() if torch.is_tensor(reps) else reps
    _, s, _ = np.linalg.svd(reps, full_matrices=False)
    return (s**2).sum() / (s**2).max()

def plot_pca(reps, labels, ax, title):
    pca = PCA(n_components=2)
    reps_2d = pca.fit_transform(reps)
    ax.cla()
    ax.scatter(reps_2d[:,0], reps_2d[:,1], c=labels, cmap='tab10', alpha=0.6)
    ax.set_title(title)

# ==============================
# Training Loop with Real-Time Visualization
# ==============================
def train_model(epochs=5, batch_size=64, lr=0.01, lambda_inv=0.1):
    dataset = SyntheticDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = HierarchicalMLP()
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    reps_l1_list, reps_l2_list = [], []
    losses_all = []

    # Real-time plot
    fig, axs = plt.subplots(1,3,figsize=(15,5))
    axs[0].scatter(dataset.data[:,0], dataset.data[:,1], c=dataset.labels, cmap='tab10', alpha=0.6)
    axs[0].set_title('Input Data')

    text_loss = axs[0].text(0.05,0.95,'Loss:N/A',transform=axs[0].transAxes)
    text_sim_l1 = axs[1].text(0.05,0.95,'Sim L1:N/A',transform=axs[1].transAxes)
    text_sim_l2 = axs[2].text(0.05,0.95,'Sim L2:N/A',transform=axs[2].transAxes)
    text_dim = axs[2].text(0.05,0.85,'Join Dim:N/A',transform=axs[2].transAxes)

    for epoch in range(epochs):
        epoch_losses = []
        for data, labels in loader:
            aug_data = augment_batch(data)
            opt.zero_grad()
            out, rep1, rep2 = model(data, return_reps=True)
            _, aug_r1, aug_r2 = model(aug_data, return_reps=True)
            pred_loss = crit(out, labels)
            inv_loss = contrastive_loss(rep2, aug_r2)
            loss = pred_loss + lambda_inv * inv_loss
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        losses_all.extend(epoch_losses)

        # Full reps for visualization
        with torch.no_grad():
            _, full_r1, full_r2 = model(torch.tensor(dataset.data), return_reps=True)

        reps_l1_list.append(full_r1)
        reps_l2_list.append(full_r2)

        # Update PCA plots
        plot_pca(full_r1, dataset.labels, axs[1], f'Layer1 Epoch {epoch+1}')
        plot_pca(full_r2, dataset.labels, axs[2], f'Layer2 Epoch {epoch+1}')

        # Update metrics
        text_loss.set_text(f'Loss:{np.mean(epoch_losses):.3f}')
        text_sim_l1.set_text(f'Sim L1:{compute_avg_sim(full_r1,dataset.labels):.3f}')
        text_sim_l2.set_text(f'Sim L2:{compute_avg_sim(full_r2,dataset.labels):.3f}')
        text_dim.set_text(f'Join Dim:{(proxy_join_dim(full_r1)+proxy_join_dim(full_r2))/2:.2f}')

        plt.pause(0.1)

    plt.show(block=False)
    return reps_l1_list, reps_l2_list, dataset.labels, losses_all

# ==============================
# Summary Popup (Main Thread)
# ==============================
def show_summary_popup(reps_l1, reps_l2, labels, losses):
    root = tk.Tk()
    root.withdraw()
    summary = f"Final Loss: {np.mean(losses[-1:]):.3f}\n"
    summary += f"L1 Sim: {compute_avg_sim(reps_l1[0], labels):.3f} → {compute_avg_sim(reps_l1[-1], labels):.3f}\n"
    summary += f"L2 Sim: {compute_avg_sim(reps_l2[0], labels):.3f} → {compute_avg_sim(reps_l2[-1], labels):.3f}\n"
    summary += f"Join Dim: {(proxy_join_dim(reps_l1[0])+proxy_join_dim(reps_l2[0]))/2:.2f} → {(proxy_join_dim(reps_l1[-1])+proxy_join_dim(reps_l2[-1]))/2:.2f}"
    messagebox.showinfo("LCRD Simulation Summary", summary)
    root.destroy()

# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    reps_l1, reps_l2, labels, losses = train_model()
    show_summary_popup(reps_l1, reps_l2, labels, losses)
