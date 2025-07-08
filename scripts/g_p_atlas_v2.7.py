#
# Description:
# This is an implementation of the G-P Atlas method, v2.7.
# This version tests the hypothesis that treating fitness as the 17th
# phenotype can lead to a more holistic and predictive model.
#
# v2.7 Final Changelog:
# - This version is now fully optimized, incorporating target normalization,
#   a learning rate scheduler, and weight_decay for a robust training process.
#
# Author: Gemini
# Date: July 7, 2025
#

import pickle as pk
import sys
import json
from argparse import ArgumentParser
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

try:
    from captum.attr import FeatureAblation
except ImportError:
    print("Captum not found. Please install it: pip install captum")
    sys.exit(1)

# --- Dataset Classes ---
class PhenoDataset(Dataset):
    def __init__(self, data_file, n_phens, pheno_mean, pheno_std):
        dataset = pk.load(open(data_file, "rb"))
        phens = torch.tensor(dataset["phenotypes"][:, :n_phens], dtype=torch.float32)
        self.phens = (phens - pheno_mean) / pheno_std
    def __len__(self): return len(self.phens)
    def __getitem__(self, idx): return self.phens[idx]

class GenoDataset(Dataset):
    def __init__(self, data_file, n_loci_measured, n_phens, pheno_mean, pheno_std):
        dataset = pk.load(open(data_file, "rb"))
        phens = torch.tensor(dataset["phenotypes"][:, :n_phens], dtype=torch.float32)
        self.norm_phens = (phens - pheno_mean) / pheno_std
        self.genotypes = torch.tensor(dataset["genotypes"], dtype=torch.float32)
        self.n_loci_measured = n_loci_measured
    def __len__(self): return len(self.genotypes)
    def __getitem__(self, idx):
        return self.norm_phens[idx], torch.flatten(self.genotypes[idx])

# --- Captum Helper Function ---
def forward_func_for_captum(genotypes, model_gq, model_p):
    latent = model_gq(genotypes)
    return model_p(latent)

# --- Model Classes ---
class Q_net(nn.Module):
    def __init__(self, vabs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vabs.n_phens, vabs.e_hidden_dim),
            nn.BatchNorm1d(vabs.e_hidden_dim), nn.LeakyReLU(0.01),
            nn.Linear(vabs.e_hidden_dim, vabs.latent_dim),
            nn.BatchNorm1d(vabs.latent_dim), nn.LeakyReLU(0.01))
    def forward(self, x): return self.encoder(x)

class P_net(nn.Module):
    def __init__(self, vabs):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(vabs.latent_dim, vabs.d_hidden_dim),
            nn.BatchNorm1d(vabs.d_hidden_dim), nn.LeakyReLU(0.01),
            nn.Linear(vabs.d_hidden_dim, vabs.n_phens))
    def forward(self, x): return self.decoder(x)

class GQ_net(nn.Module):
    def __init__(self, vabs):
        super().__init__()
        n_loci = vabs.n_loci_measured * vabs.n_alleles
        self.encoder = nn.Sequential(
            nn.Linear(n_loci, vabs.ge_hidden_dim),
            nn.BatchNorm1d(vabs.ge_hidden_dim), nn.LeakyReLU(0.01),
            nn.Linear(vabs.ge_hidden_dim, vabs.latent_dim),
            nn.BatchNorm1d(vabs.latent_dim), nn.LeakyReLU(0.01))
    def forward(self, x): return self.encoder(x)

def main():
    parser = ArgumentParser()
    parser.add_argument("--n_phens", type=int, default=17)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--calculate_importance", type=str, default="yes")
    parser.add_argument("--n_alleles", type=int, default=3)
    parser.add_argument("--n_loci_measured", type=int, default=53)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--e_hidden_dim", type=int, default=32)
    parser.add_argument("--d_hidden_dim", type=int, default=32)
    parser.add_argument("--ge_hidden_dim", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--n_epochs_gen", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sd_noise", type=float, default=0.2)
    parser.add_argument("--gen_noise", type=float, default=0.3)
    vabs = parser.parse_args()

    if not os.path.exists(vabs.results_dir):
        os.makedirs(vabs.results_dir)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_file = os.path.join(vabs.dataset_path, "train_data.pk")
    test_file = os.path.join(vabs.dataset_path, "test_data.pk")

    with open(train_file, 'rb') as f:
        train_raw = pk.load(f)
    pheno_mean = torch.tensor(train_raw['phenotypes'][:, :vabs.n_phens], dtype=torch.float32).mean(dim=0)
    pheno_std = torch.tensor(train_raw['phenotypes'][:, :vabs.n_phens], dtype=torch.float32).std(dim=0)
    pheno_std[pheno_std == 0] = 1.0
    pheno_mean, pheno_std = pheno_mean.to(device), pheno_std.to(device)

    train_loader_pheno = torch.utils.data.DataLoader(PhenoDataset(train_file, vabs.n_phens, pheno_mean, pheno_std), batch_size=vabs.batch_size, shuffle=True)
    train_loader_geno = torch.utils.data.DataLoader(GenoDataset(train_file, vabs.n_loci_measured, vabs.n_phens, pheno_mean, pheno_std), batch_size=vabs.batch_size, shuffle=True)
    test_loader_geno = torch.utils.data.DataLoader(GenoDataset(test_file, vabs.n_loci_measured, vabs.n_phens, pheno_mean, pheno_std), batch_size=vabs.batch_size, shuffle=False)
    test_loader_attr = torch.utils.data.DataLoader(GenoDataset(test_file, vabs.n_loci_measured, vabs.n_phens, pheno_mean, pheno_std), batch_size=1, shuffle=False)

    Q, P, GQ = Q_net(vabs).to(device), P_net(vabs).to(device), GQ_net(vabs).to(device)
    # FIXED: Added weight_decay to optimizers
    optim_P_autoencoder = torch.optim.Adam(list(Q.parameters()) + list(P.parameters()), lr=vabs.lr, weight_decay=1e-5)
    optim_GQ_enc = torch.optim.Adam(GQ.parameters(), lr=vabs.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optim_GQ_enc, 'min', patience=10, factor=0.5)

    print("\n--- Stage 1: Training Phenotype Autoencoder ---")
    Q.train(); P.train()
    for n in range(vabs.n_epochs):
        progress_bar = tqdm(train_loader_pheno, desc=f"P-P Epoch {n+1}/{vabs.n_epochs}")
        for norm_phens in progress_bar:
            norm_phens = norm_phens.to(device)
            noisy_phens = norm_phens + torch.randn_like(norm_phens) * vabs.sd_noise
            optim_P_autoencoder.zero_grad()
            z = Q(noisy_phens)
            recon_phens = P(z)
            loss = F.mse_loss(recon_phens, norm_phens)
            loss.backward()
            optim_P_autoencoder.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    print("\n--- Stage 2: Training Genotype Encoder ---")
    P.eval(); GQ.train()
    for n in range(vabs.n_epochs_gen):
        progress_bar = tqdm(train_loader_geno, desc=f"G-P Epoch {n+1}/{vabs.n_epochs_gen}")
        epoch_loss = 0.0
        for norm_phens, gens in progress_bar:
            norm_phens, gens = norm_phens.to(device), gens.to(device)
            pos_noise = torch.bernoulli(torch.full_like(gens, vabs.gen_noise / 2)).to(device)
            neg_noise = torch.bernoulli(torch.full_like(gens, vabs.gen_noise / 2)).to(device)
            noise_gens = torch.clamp(gens + pos_noise - neg_noise, 0, 1)
            optim_GQ_enc.zero_grad()
            z_sample = GQ(noise_gens)
            X_sample = P(z_sample)
            recon_loss = F.mse_loss(X_sample, norm_phens)
            recon_loss.backward()
            optim_GQ_enc.step()
            progress_bar.set_postfix(loss=f"{recon_loss.item():.4f}")
            epoch_loss += recon_loss.item()
        scheduler.step(epoch_loss / len(train_loader_geno))

    print("\n--- Stage 3: Evaluating Final Model ---")
    GQ.eval(); P.eval()
    
    all_true_phens, all_pred_phens, all_attributions = [], [], []
    fa = FeatureAblation(lambda x: forward_func_for_captum(x, GQ, P))

    with torch.no_grad():
        for norm_phens, gens in tqdm(test_loader_geno, desc="Evaluating Model"):
            norm_phens, gens = norm_phens.to(device), gens.to(device)
            z = GQ(gens)
            pred_phens_norm = P(z)
            pred_phens = pred_phens_norm * pheno_std + pheno_mean
            true_phens = norm_phens * pheno_std + pheno_mean
            all_pred_phens.extend(pred_phens.cpu().numpy())
            all_true_phens.extend(true_phens.cpu().numpy())
    
    true_phens_np, pred_phens_np = np.array(all_true_phens), np.array(all_pred_phens)
    per_phen_r2 = [r2_score(true_phens_np[:, i], pred_phens_np[:, i]) for i in range(vabs.n_phens)]
    
    print("\n--- Per-Phenotype R^2 Scores (Phenotype 17 is Fitness) ---")
    with open(os.path.join(vabs.results_dir, "results.txt"), "w") as f:
        f.write("--- Per-Phenotype R^2 Scores ---\n")
        for i, r2 in enumerate(per_phen_r2):
            phen_name = f"Phenotype {i+1:02d}" if i < 16 else "Fitness"
            line = f"  {phen_name}: R^2 = {r2:.4f}\n"
            print(line, end='')
            f.write(line)

    if vabs.calculate_importance == "yes":
        print("\nCalculating feature attributions (this may take a while)...")
        with torch.enable_grad():
            for _, gen_sample in tqdm(test_loader_attr, desc="Calculating Attributions"):
                gen_sample = gen_sample.to(device)
                pheno_attr_sample_list = []
                for i in range(vabs.n_phens):
                    attr = fa.attribute(gen_sample, target=i)
                    pheno_attr_sample_list.append(attr.cpu().detach().numpy())
                all_attributions.append(np.stack(pheno_attr_sample_list, axis=0))
        
        mean_attrs = np.mean(np.stack(all_attributions, axis=1), axis=1)
            
        with open(os.path.join(vabs.results_dir, "mean_attributions.pk"), "wb") as f:
            pk.dump(mean_attrs, f)
        print("Mean feature attributions saved.")

    print(f"\nEvaluation complete. Models and results saved to {vabs.results_dir}")

if __name__ == "__main__":
    main()
