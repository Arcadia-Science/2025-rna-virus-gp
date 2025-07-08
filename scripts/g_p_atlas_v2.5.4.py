#
# Description:
# This is an implementation of the G-P Atlas method, v2.5.4.
# This version is identical to v2.5.3 but adds the raw predicted
# and true fitness arrays to the output file for downstream analysis.
#
# v2.5.4 Changelog:
# - Saves 'predicted_fitness' and 'true_fitness' numpy arrays to the
#   final_results.pk file.
#
# Author: Gemini
# Date: July 8, 2025
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
    def __init__(self, data_file, pheno_mean, pheno_std):
        dataset = pk.load(open(data_file, "rb"))
        phens = torch.tensor(dataset["phenotypes"], dtype=torch.float32)
        self.phens = (phens - pheno_mean) / pheno_std
    def __len__(self): return len(self.phens)
    def __getitem__(self, idx): return self.phens[idx]

class GenoFitnessDataset(Dataset):
    def __init__(self, data_file, n_loci_measured, pheno_mean, pheno_std, fit_mean, fit_std):
        dataset = pk.load(open(data_file, "rb"))
        phens = torch.tensor(dataset["phenotypes"], dtype=torch.float32)
        fitness = torch.tensor(dataset["fitness"], dtype=torch.float32)
        self.norm_phens = (phens - pheno_mean) / pheno_std
        self.norm_fitness = (fitness - fit_mean) / fit_std
        self.genotypes = torch.tensor(dataset["genotypes"], dtype=torch.float32)
        self.n_loci_measured = n_loci_measured
    def __len__(self): return len(self.genotypes)
    def __getitem__(self, idx):
        return self.norm_phens[idx], self.norm_fitness[idx], torch.flatten(self.genotypes[idx])

# --- Captum Helper Functions ---
def forward_pheno_for_captum(genotypes, model_gq, model_p):
    latent = model_gq(genotypes)
    return model_p(latent)

def forward_fitness_for_captum(genotypes, model_gq, model_fp):
    latent = model_gq(genotypes)
    return model_fp(latent)

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

class Fitness_predictor_net(nn.Module):
    def __init__(self, vabs):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(vabs.latent_dim, vabs.fit_hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(vabs.fit_hidden_dim, 1))
    def forward(self, x): return self.predictor(x)

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--calculate_importance", type=str, default="no")
    parser.add_argument("--train_suffix", type=str, default="train_data.pk")
    parser.add_argument("--test_suffix", type=str, default="test_data.pk")
    parser.add_argument("--n_alleles", type=int, default=3)
    parser.add_argument("--n_loci_measured", type=int, default=53)
    parser.add_argument("--n_phens", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--e_hidden_dim", type=int, default=32)
    parser.add_argument("--d_hidden_dim", type=int, default=32)
    parser.add_argument("--ge_hidden_dim", type=int, default=32)
    parser.add_argument("--fit_hidden_dim", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--n_epochs_gen", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sd_noise", type=float, default=0.2)
    parser.add_argument("--gen_noise", type=float, default=0.3)
    parser.add_argument("--loss_alpha", type=float, default=0.9)
    vabs = parser.parse_args()

    if not os.path.exists(vabs.results_dir):
        os.makedirs(vabs.results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_file = os.path.join(vabs.dataset_path, vabs.train_suffix)
    test_file = os.path.join(vabs.dataset_path, vabs.test_suffix)

    with open(train_file, 'rb') as f:
        train_raw = pk.load(f)
    pheno_mean = torch.tensor(train_raw['phenotypes'], dtype=torch.float32).mean(dim=0)
    pheno_std = torch.tensor(train_raw['phenotypes'], dtype=torch.float32).std(dim=0)
    fit_mean = torch.tensor(train_raw['fitness'], dtype=torch.float32).mean()
    fit_std = torch.tensor(train_raw['fitness'], dtype=torch.float32).std()
    pheno_std[pheno_std == 0] = 1.0
    if fit_std == 0: fit_std = 1.0

    pheno_mean, pheno_std = pheno_mean.to(device), pheno_std.to(device)
    fit_mean, fit_std = fit_mean.to(device), fit_std.to(device)

    train_loader_pheno = torch.utils.data.DataLoader(PhenoDataset(train_file, pheno_mean, pheno_std), batch_size=vabs.batch_size, shuffle=True)
    train_loader_geno = torch.utils.data.DataLoader(GenoFitnessDataset(train_file, vabs.n_loci_measured, pheno_mean, pheno_std, fit_mean, fit_std), batch_size=vabs.batch_size, shuffle=True)
    test_loader_geno = torch.utils.data.DataLoader(GenoFitnessDataset(test_file, vabs.n_loci_measured, pheno_mean, pheno_std, fit_mean, fit_std), batch_size=vabs.batch_size, shuffle=False)
    test_loader_attr = torch.utils.data.DataLoader(GenoFitnessDataset(test_file, vabs.n_loci_measured, pheno_mean, pheno_std, fit_mean, fit_std), batch_size=1, shuffle=False)

    Q, P, GQ, FP = Q_net(vabs).to(device), P_net(vabs).to(device), GQ_net(vabs).to(device), Fitness_predictor_net(vabs).to(device)
    optim_P_autoencoder = torch.optim.Adam(list(Q.parameters()) + list(P.parameters()), lr=vabs.lr, weight_decay=1e-5)
    optim_G_F_models = torch.optim.Adam(list(GQ.parameters()) + list(FP.parameters()), lr=vabs.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optim_G_F_models, 'min', patience=10, factor=0.5)

    pp_loss_history, gp_loss_history = [], []

    print("\n--- Stage 1: Training Phenotype Autoencoder ---")
    Q.train(); P.train()
    for n in range(vabs.n_epochs):
        progress_bar = tqdm(train_loader_pheno, desc=f"P-P Epoch {n+1}/{vabs.n_epochs}")
        epoch_loss = 0.0
        for norm_phens in progress_bar:
            norm_phens = norm_phens.to(device)
            noisy_phens = norm_phens + torch.randn_like(norm_phens) * vabs.sd_noise
            optim_P_autoencoder.zero_grad()
            z = Q(noisy_phens)
            recon_phens = P(z)
            loss = F.mse_loss(recon_phens, norm_phens)
            loss.backward()
            optim_P_autoencoder.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        pp_loss_history.append(epoch_loss / len(train_loader_pheno))


    print("\n--- Stage 2: Training G-P Encoder & Fitness Predictor ---")
    P.eval(); GQ.train(); FP.train()
    for n in range(vabs.n_epochs_gen):
        progress_bar = tqdm(train_loader_geno, desc=f"G-P+Fit Epoch {n+1}/{vabs.n_epochs_gen}")
        epoch_loss = 0.0
        for norm_phens, norm_fitness, gens in progress_bar:
            norm_phens, norm_fitness, gens = norm_phens.to(device), norm_fitness.to(device), gens.to(device)
            pos_noise = torch.bernoulli(torch.full_like(gens, vabs.gen_noise / 2)).to(device)
            neg_noise = torch.bernoulli(torch.full_like(gens, vabs.gen_noise / 2)).to(device)
            noise_gens = torch.clamp(gens + pos_noise - neg_noise, 0, 1)
            optim_G_F_models.zero_grad()
            z_sample = GQ(noise_gens)
            X_sample = P(z_sample)
            F_sample = FP(z_sample)
            recon_loss = F.mse_loss(X_sample, norm_phens)
            fitness_loss = F.mse_loss(F_sample, norm_fitness)
            total_loss = (vabs.loss_alpha * recon_loss) + ((1 - vabs.loss_alpha) * fitness_loss)
            total_loss.backward()
            optim_G_F_models.step()
            epoch_loss += total_loss.item()
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")
        scheduler.step(epoch_loss / len(train_loader_geno))
        gp_loss_history.append(epoch_loss / len(train_loader_geno))

    print("\n--- Stage 3: Evaluating Final Model ---")
    GQ.eval(); FP.eval(); P.eval()

    all_true_phens, all_pred_phens, all_true_fitness, all_pred_fitness = [], [], [], []

    with torch.no_grad():
        for norm_phens, norm_fitness, gens in tqdm(test_loader_geno, desc="Evaluating Model"):
            norm_phens, norm_fitness, gens = norm_phens.to(device), norm_fitness.to(device), gens.to(device)
            z = GQ(gens)
            pred_phens_norm = P(z)
            pred_fitness_norm = FP(z)
            pred_phens = pred_phens_norm * pheno_std + pheno_mean
            true_phens = norm_phens * pheno_std + pheno_mean
            pred_fitness = pred_fitness_norm * fit_std + fit_mean
            true_fitness = norm_fitness * fit_std + fit_mean
            all_pred_phens.extend(pred_phens.cpu().numpy())
            all_true_phens.extend(true_phens.cpu().numpy())
            all_pred_fitness.extend(pred_fitness.cpu().numpy())
            all_true_fitness.extend(true_fitness.cpu().numpy())

    true_phens_np, pred_phens_np = np.array(all_true_phens), np.array(all_pred_phens)
    true_fitness_np, pred_fitness_np = np.array(all_true_fitness).flatten(), np.array(all_pred_fitness).flatten()

    per_phen_r2 = [r2_score(true_phens_np[:, i], pred_phens_np[:, i]) for i in range(vabs.n_phens)]
    fitness_r2 = r2_score(true_fitness_np, pred_fitness_np)

    # --- Save Results ---
    with open(os.path.join(vabs.results_dir, "results.txt"), "w") as f:
        f.write("--- Per-Phenotype R^2 Scores ---\n")
        for i, r2 in enumerate(per_phen_r2):
            line = f"  Phenotype {i+1:02d}: R^2 = {r2:.4f}\n"
            print(line, end='')
            f.write(line)
        print("\n--- Fitness Prediction Performance ---")
        f.write("\n--- Fitness Prediction Performance ---\n")
        line = f"  Fitness R^2 = {fitness_r2:.4f}\n"
        print(line, end='')
        f.write(line)

    # NEW: Save raw predictions and loss histories to the pickle file
    results_data = {
        "per_phen_r2": per_phen_r2,
        "fitness_r2": fitness_r2,
        "pp_loss_history": pp_loss_history,
        "gp_loss_history": gp_loss_history,
        "predicted_fitness": pred_fitness_np,
        "true_fitness": true_fitness_np
    }
    with open(os.path.join(vabs.results_dir, "final_results.pk"), "wb") as f:
        pk.dump(results_data, f)

    if vabs.calculate_importance == "yes":
        print("\nCalculating feature attributions (this may take a while)...")
        fa_pheno = FeatureAblation(lambda x: forward_pheno_for_captum(x, GQ, P))
        fa_fitness = FeatureAblation(lambda x: forward_fitness_for_captum(x, GQ, FP))

        all_pheno_attrs, all_fitness_attrs = [], []

        with torch.enable_grad():
            for _, _, gen_sample in tqdm(test_loader_attr, desc="Calculating Attributions"):
                gen_sample = gen_sample.to(device)

                pheno_attr_sample_list = []
                for i in range(vabs.n_phens):
                    attr = fa_pheno.attribute(gen_sample, target=i)
                    pheno_attr_sample_list.append(attr.cpu().detach().numpy())
                all_pheno_attrs.append(np.stack(pheno_attr_sample_list, axis=0))

                fitness_attr_sample = fa_fitness.attribute(gen_sample)
                all_fitness_attrs.append(fitness_attr_sample.cpu().detach().numpy())

        mean_pheno_attrs = np.mean(np.concatenate(all_pheno_attrs, axis=1), axis=1)
        mean_fitness_attrs = np.mean(np.concatenate(all_fitness_attrs, axis=0), axis=0)

        with open(os.path.join(vabs.results_dir, "pheno_attributions.pk"), "wb") as f:
            pk.dump(mean_pheno_attrs, f)
        with open(os.path.join(vabs.results_dir, "fitness_attributions.pk"), "wb") as f:
            pk.dump(mean_fitness_attrs, f)
        print("Mean feature attributions saved.")

    print(f"\nEvaluation complete. Models and results saved to {vabs.results_dir}")

if __name__ == "__main__":
    main()
