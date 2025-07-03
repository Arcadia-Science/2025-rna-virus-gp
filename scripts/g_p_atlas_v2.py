import pickle as pk
import sys
import time as tm
from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import FeatureAblation
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data.dataset import Dataset
from tqdm import tqdm # Import tqdm for progress bars

"""This is an implementation of the G-P Atlas method for mapping genotype
to phenotype described in https://doi.org/10.57844/arcadia-d316-721f.
For help type:
python3 g_p_atlas.py --help"""


# define a torch dataset object
class dataset_pheno(Dataset):
    """a class for importing simulated phenotype data.
    It expects a pickled object that is organized as a dictionary of tensors:
    phenotypes[n_animals,n_phens] float value for phenotype
    """

    def __init__(self, data_file, n_phens):
        self.datset = pk.load(open(data_file, "rb"))
        self.phens = torch.tensor(np.array(self.datset["phenotypes"]), dtype=torch.float32)
        self.data_file = data_file
        self.n_phens = n_phens

    def __len__(self):
        return len(self.phens)

    def __getitem__(self, idx):
        phenotypes = self.phens[idx][: self.n_phens]
        return phenotypes


class dataset_geno(Dataset):
    """a class for importing simulated genotype and phenotype data.
    It expects a pickled object that is organized as a dictionary of tensors:
    genotypes[n_animals, n_loci, n_alleles] (one hot at allelic state)
    phenotypes[n_animals,n_phens] float value for phenotype
    """

    def __init__(self, data_file, n_geno, n_phens):
        self.datset = pk.load(open(data_file, "rb"))
        self.phens = torch.tensor(np.array(self.datset["phenotypes"]), dtype=torch.float32)
        self.genotypes = torch.tensor(np.array(self.datset["genotypes"]), dtype=torch.float32)
        self.data_file = data_file
        self.n_geno = n_geno
        self.n_phens = n_phens

    def __len__(self):
        return len(self.genotypes)

    def __getitem__(self, idx):
        phenotypes = self.phens[idx][: self.n_phens]
        genotype = torch.flatten(self.genotypes[idx])
        return phenotypes, genotype


# helper functions
def sequential_forward_attr_gen_phen(input, phens, GQ_model, P_model, EPS, n_phens_pred):
    """puts together two models for use of captum feature
    importance in genotype-phenotype prediction"""
    mod_2_input = GQ_model(input)
    X_sample = P_model(mod_2_input)
    output = F.mse_loss(X_sample + EPS, phens[:, :n_phens_pred] + EPS)
    return output


def sequential_forward_attr_phen_phen(input, phens, Q_model, P_model, EPS, n_phens_pred):
    """puts together two models for use of captum feature
    importance in phenotype-phenotype prediction"""
    mod_2_input = Q_model(input)
    X_sample = P_model(mod_2_input)
    output = F.mse_loss(X_sample + EPS, phens[:, :n_phens_pred] + EPS)
    return output


def mean_absolute_percentage_error(y_true, y_pred, EPS):
    y_true, y_pred = np.array(y_true + EPS), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# A function to evaluate the performance of each model, saving summaries of model performance
def analyze_predictions(
    phens,
    phen_encodings,
    phen_latent,
    fa_attr,
    dataset_path,
    n_phens_pred,
    EPS,
    model_type="g_p",
):
    """Analyze predictions and save visualization results."""
    suffix = "_p" if model_type == "p_p" else ""

    # Save attributions
    if fa_attr:
        plt.hist(fa_attr, bins=20)
        plt.savefig(dataset_path + f"{model_type}_attr.svg")
        plt.close()
        pk.dump(fa_attr, open(dataset_path + f"{model_type}_attr.pk", "wb"))

    # Convert and transpose data
    phens = np.array(phens).T
    phen_encodings = np.array(phen_encodings).T

    # Save predictions data
    if model_type == "p_p":
        phen_latent = np.array(phen_latent).T
        pk.dump(
            [phens, phen_encodings, phen_latent],
            open(dataset_path + f"phens_phen_encodings_dng_attr{suffix}.pk", "wb"),
        )
    else:
        pk.dump(
            [phens, phen_encodings],
            open(dataset_path + f"phens_phen_encodings_dng_attr{suffix}.pk", "wb"),
        )

    # Plot predictions
    for n in range(len(phens[:n_phens_pred])):
        plt.plot(phens[n], phen_encodings[n], "o")
    plt.xlabel("real")
    plt.ylabel("predicted")
    plt.gca().set_aspect("equal")
    plt.savefig(dataset_path + f"phen_real_pred_dng_attr{suffix}.svg")
    plt.close()

    # Calculate and plot metrics
    stats_aggregator = []

    # Pearson correlation
    cors = [
        sc.stats.pearsonr(phens[n], phen_encodings[n])[0] for n in range(len(phens[:n_phens_pred]))
    ]
    stats_aggregator.append(cors)
    plt.hist(cors, bins=20)
    plt.savefig(dataset_path + f"phen_real_pred_pearsonsr_dng_attr{suffix}.svg")
    plt.close()

    # MSE
    errs = [
        mean_squared_error(phens[n], phen_encodings[n]) for n in range(len(phens[:n_phens_pred]))
    ]
    stats_aggregator.append(errs)
    plt.hist(errs, bins=20)
    plt.savefig(dataset_path + f"phen_real_pred_mse_dng_attr{suffix}.svg")
    plt.close()

    # MAPE
    errs = [
        mean_absolute_percentage_error(phens[n], phen_encodings[n], EPS)
        for n in range(len(phens[:n_phens_pred]))
    ]
    stats_aggregator.append(errs)
    plt.hist(errs, bins=20)
    plt.savefig(dataset_path + f"phen_real_pred_mape_dng_attr{suffix}.svg")
    plt.close()

    # R2
    errs = [r2_score(phens[n], phen_encodings[n]) for n in range(len(phens[:n_phens_pred]))]
    stats_aggregator.append(errs)
    plt.hist(errs, bins=20)
    plt.savefig(dataset_path + f"phen_real_pred_r2_dng_attr{suffix}.svg")
    plt.close()

    return stats_aggregator


# --- MODEL CLASSES ---
class Q_net(nn.Module):
    def __init__(self, vabs):
        super().__init__()
        N = vabs.e_hidden_dim
        phen_dim = vabs.n_phens_to_analyze
        batchnorm_momentum = vabs.batchnorm_momentum
        latent_dim = vabs.latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_features=phen_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(in_features=N, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),
        )
    def forward(self, x):
        return self.encoder(x)

class P_net(nn.Module):
    def __init__(self, vabs):
        super().__init__()
        N = vabs.d_hidden_dim
        out_phen_dim = vabs.n_phens_to_predict
        latent_dim = vabs.latent_dim
        batchnorm_momentum = vabs.batchnorm_momentum
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=out_phen_dim),
        )
    def forward(self, x):
        return self.decoder(x)

class GQ_net(nn.Module):
    def __init__(self, vabs):
        super().__init__()
        N = vabs.ge_hidden_dim
        n_loci = vabs.n_loci_measured * vabs.n_alleles
        batchnorm_momentum = vabs.batchnorm_momentum
        latent_dim = vabs.latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_features=n_loci, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
        )
    def forward(self, x):
        return self.encoder(x)


def main():
    # parse commandline arguments
    parser = ArgumentParser()
    parser.add_argument("--n_alleles", type=int, default=2)
    parser.add_argument("--n_locs", type=int, default=900)
    parser.add_argument("--n_env", type=int, default=3)
    parser.add_argument("--n_phens", type=int, default=30)
    parser.add_argument("--gen_lw", type=float, default=1)
    parser.add_argument("--eng_lw", type=float, default=0.1)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_r", type=float, default=0.001)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--n_cpu", type=int, default=4)
    parser.add_argument("--e_hidden_dim", type=int, default=32)
    parser.add_argument("--d_hidden_dim", type=int, default=32)
    parser.add_argument("--ge_hidden_dim", type=int, default=32)
    parser.add_argument("--batchnorm_momentum", type=float, default=0.8)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--n_phens_to_analyze", type=int, default=30)
    parser.add_argument("--sd_noise", type=float, default=0.1)
    parser.add_argument("--gen_noise", type=float, default=0.3)
    parser.add_argument("--n_phens_to_predict", type=int, default=30)
    parser.add_argument("--n_epochs_gen", type=int, default=100)
    parser.add_argument("--n_loci_measured", type=int, default=3000)
    parser.add_argument("--l1_lambda", type=float, default=0.8)
    parser.add_argument("--l2_lambda", type=float, default=0.01)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--train_suffix", type=str, default="train.pk")
    parser.add_argument("--test_suffix", type=str, default="test.pk")
    parser.add_argument("--hot_start", type=bool, default=False)
    parser.add_argument("--hot_start_path_e", type=str, default=None)
    parser.add_argument("--hot_start_path_d", type=str, default=None)
    parser.add_argument("--hot_start_path_ge", type=str, default=None)
    parser.add_argument("--calculate_importance", type=str, default="no")
    vabs = parser.parse_args()

    # G-P Atlas run
    EPS = 1e-15

    # load the training and test datasets
    dataset_path = vabs.dataset_path
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        
    params_file = open(os.path.join(dataset_path, "run_params.txt"), "w")
    params_file.write(" ".join(sys.argv[:]))
    params_file.close()

    train_dat = vabs.train_suffix
    test_dat = vabs.test_suffix

    train_data_pheno = dataset_pheno(os.path.join(dataset_path, train_dat), n_phens=vabs.n_phens_to_analyze)
    test_data_pheno = dataset_pheno(os.path.join(dataset_path, test_dat), n_phens=vabs.n_phens_to_analyze)

    train_data_geno = dataset_geno(
        os.path.join(dataset_path, train_dat), n_geno=vabs.n_loci_measured, n_phens=vabs.n_phens_to_analyze
    )
    test_data_geno = dataset_geno(
        os.path.join(dataset_path, test_dat), n_geno=vabs.n_loci_measured, n_phens=vabs.n_phens_to_analyze
    )

    # setting device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # prepare data loaders
    train_loader_pheno = torch.utils.data.DataLoader(
        dataset=train_data_pheno, batch_size=vabs.batch_size, num_workers=vabs.n_cpu, shuffle=True
    )
    test_loader_pheno = torch.utils.data.DataLoader(
        dataset=test_data_pheno, batch_size=1, num_workers=vabs.n_cpu, shuffle=True
    )
    train_loader_geno = torch.utils.data.DataLoader(
        dataset=train_data_geno, batch_size=vabs.batch_size, num_workers=vabs.n_cpu, shuffle=True
    )
    test_loader_geno = torch.utils.data.DataLoader(
        dataset=test_data_geno, batch_size=1, num_workers=vabs.n_cpu, shuffle=True
    )

    # define encoders and decoders
    Q = Q_net(vabs).to(device)
    P = P_net(vabs).to(device)
    GQ = GQ_net(vabs).to(device)

    if vabs.hot_start:
        Q.load_state_dict(torch.load(vabs.hot_start_path_e, weights_only=True), strict=False)
        P.load_state_dict(torch.load(vabs.hot_start_path_d, weights_only=True), strict=False)
        GQ.load_state_dict(torch.load(vabs.hot_start_path_ge, weights_only=True), strict=False)

    # Set up feature importance measure
    fa = FeatureAblation(lambda x, y: sequential_forward_attr_gen_phen(x, y, GQ, P, EPS, vabs.n_phens_to_predict))
    fa_p = FeatureAblation(lambda x, y: sequential_forward_attr_phen_phen(x, y, Q, P, EPS, vabs.n_phens_to_predict))

    # Set optimizers
    optim_P = torch.optim.Adam(P.parameters(), lr=vabs.lr_r, betas=(vabs.b1, vabs.b2))
    optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=vabs.lr_r, betas=(vabs.b1, vabs.b2))
    optim_GQ_enc = torch.optim.Adam(GQ.parameters(), lr=vabs.lr_r, betas=(vabs.b1, vabs.b2))

    # --- Train P-P Model ---
    print("\n--- Training Phenotype-Phenotype Autoencoder ---")
    rcon_loss = []
    for n in range(vabs.n_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader_pheno, desc=f"P-P Epoch {n+1}/{vabs.n_epochs}", leave=False)
        for phens in progress_bar:
            phens = phens[:, :vabs.n_phens_to_analyze].to(device)
            noise_phens = phens + (vabs.sd_noise**0.5) * torch.randn(phens.shape).to(device)
            
            P.zero_grad()
            Q.zero_grad()
            
            z_sample = Q(noise_phens)
            X_sample = P(z_sample)
            
            recon_loss = F.mse_loss(X_sample + EPS, phens[:, :vabs.n_phens_to_predict] + EPS)
            epoch_loss += recon_loss.item()
            
            recon_loss.backward()
            optim_P.step()
            optim_Q_enc.step()
            
            progress_bar.set_postfix(loss=f"{recon_loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader_pheno)
        rcon_loss.append(avg_epoch_loss)
        print(f"P-P Epoch {n+1} complete. Average Loss: {avg_epoch_loss:.4f}")


    # --- Train G-P Model ---
    print("\n--- Training Genotype-Phenotype Encoder ---")
    P.requires_grad_(False)
    P.eval()
    g_rcon_loss = []
    for n in range(vabs.n_epochs_gen):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader_geno, desc=f"G-P Epoch {n+1}/{vabs.n_epochs_gen}", leave=False)
        for phens, gens in progress_bar:
            phens = phens.to(device)
            gens = gens[:, :vabs.n_loci_measured * vabs.n_alleles]
            
            pos_noise = np.random.binomial(1, vabs.gen_noise / 2, gens.shape)
            neg_noise = np.random.binomial(1, vabs.gen_noise / 2, gens.shape)
            noise_gens = torch.tensor(np.where((gens + pos_noise - neg_noise) > 0, 1, 0), dtype=torch.float32).to(device)
            
            GQ.zero_grad()
            
            z_sample = GQ(noise_gens)
            X_sample = P(z_sample)
            
            g_recon_loss = F.mse_loss(X_sample + EPS, phens[:, :vabs.n_phens_to_predict] + EPS)
            l1_reg = torch.linalg.norm(torch.sum(GQ.encoder[0].weight, axis=0), 1)
            l2_reg = torch.linalg.norm(torch.sum(GQ.encoder[0].weight, axis=0), 2)
            total_loss = g_recon_loss + l1_reg * vabs.l1_lambda + l2_reg * vabs.l2_lambda
            
            epoch_loss += g_recon_loss.item()
            
            total_loss.backward()
            optim_GQ_enc.step()

            progress_bar.set_postfix(loss=f"{g_recon_loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader_geno)
        g_rcon_loss.append(avg_epoch_loss)
        print(f"G-P Epoch {n+1} complete. Average Loss: {avg_epoch_loss:.4f}")


    # --- Analysis ---
    print("\n--- Analyzing Model Performance ---")
    plt.plot(rcon_loss)
    plt.plot(g_rcon_loss)
    plt.title("Reconstruction Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend(["P-P Training", "G-P Training"])
    plt.savefig(os.path.join(dataset_path, "reconstruction_loss.svg"))
    plt.close()

    stats_aggregator = []
    torch.save(Q.state_dict(), os.path.join(dataset_path, "phen_encoder_state.pt"))
    torch.save(P.state_dict(), os.path.join(dataset_path, "phen_decoder_state.pt"))
    torch.save(GQ.state_dict(), os.path.join(dataset_path, "gen_encoder_state.pt"))

    # G-P prediction analysis
    GQ.eval()
    phens, phen_encodings, phen_latent, fa_attr = [], [], [], []
    for dat in test_loader_geno:
        ph, gt = dat
        gt = gt[:, :vabs.n_loci_measured * vabs.n_alleles].to(device)
        ph = ph.to(device)
        z_sample = GQ(gt)
        X_sample = P(z_sample)
        phens.extend(list(ph.detach().cpu().numpy()))
        phen_encodings.extend(list(X_sample.detach().cpu().numpy()))
        phen_latent.extend(list(z_sample.detach().cpu().numpy()))
        if vabs.calculate_importance == "yes":
            fa_attr.append(list(fa.attribute(inputs=(gt, ph))[0].squeeze().detach().cpu().numpy()))
    stats_aggregator.extend(
        analyze_predictions(phens, phen_encodings, phen_latent, fa_attr, dataset_path, vabs.n_phens_to_predict, EPS, "g_p")
    )

    # P-P prediction analysis
    Q.eval()
    phens, phen_encodings, phen_latent, fa_attr = [], [], [], []
    for dat in test_loader_pheno:
        ph = dat.to(device)
        z_sample = Q(ph)
        X_sample = P(z_sample)
        phens.extend(list(ph.detach().cpu().numpy()))
        phen_encodings.extend(list(X_sample.detach().cpu().numpy()))
        phen_latent.extend(list(z_sample.detach().cpu().numpy()))
        if vabs.calculate_importance == "yes":
            fa_attr.append(list(fa_p.attribute(inputs=(ph, ph))[0].squeeze().detach().cpu().numpy()))
    stats_aggregator.extend(
        analyze_predictions(phens, phen_encodings, phen_latent, fa_attr, dataset_path, vabs.n_phens_to_predict, EPS, "p_p")
    )

    with open(os.path.join(dataset_path, "test_stats.pk"), "wb") as out_stats:
        pk.dump(stats_aggregator, out_stats)
    
    print("\nAnalysis complete. Results saved to:", dataset_path)


if __name__ == "__main__":
    main()
