#
# G-P Atlas Version 4: Fixed Graph-Aware Architecture
#
# This script fixes the major issues in v3 by adopting v2's successful training strategy
# while keeping the graph-aware benefits. Key improvements:
# 1. Sequential training (like v2): P-P autoencoder → GAT encoder → fitness predictor
# 2. Proper graph construction from JSON G-P map
# 3. Explicit node type tracking (no fragile pooling assumptions)
# 4. Learnable phenotype node embeddings
# 5. Simplified, robust architecture
#
# Author: Claude
# Date: July 7, 2025
#

import pickle as pk
import sys
import os
import json
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

# --- Captum for Interpretability ---
try:
    from captum.attr import FeatureAblation
except ImportError:
    print("Captum not found. Please install it: pip install captum")
    sys.exit(1)

# --- PyTorch Geometric Imports ---
try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:
    print("PyTorch Geometric not found. Please install it to run this script.")
    print("Installation command: pip install torch_geometric")
    sys.exit(1)

def construct_graph_from_map(gp_map):
    """
    Construct graph structure from G-P map JSON.
    Returns edge_index tensor and node type information.
    """
    n_loci = gp_map['n_loci']
    n_phens = gp_map['n_phens']
    
    # Node indices: 0 to n_loci-1 are genotype nodes, n_loci to n_loci+n_phens-1 are phenotype nodes
    edges = []
    
    # Create phenotype name to index mapping
    phen_names = list(gp_map['phenotypes'].keys())
    phen_name_to_idx = {name: i for i, name in enumerate(phen_names)}
    
    for phen_idx, (phen_name, phen_data) in enumerate(gp_map['phenotypes'].items()):
        phen_node_idx = n_loci + phen_idx
        
        # Additive loci -> phenotype edges
        for locus in phen_data.get('additive_loci', []):
            locus_idx = locus - 1  # Convert from 1-indexed to 0-indexed
            edges.append([locus_idx, phen_node_idx])
        
        # Interactive loci -> phenotype edges
        for locus_pair in phen_data.get('interactive_loci', []):
            for locus in locus_pair:
                locus_idx = locus - 1  # Convert from 1-indexed to 0-indexed
                edges.append([locus_idx, phen_node_idx])
        
        # Dependent phenotypes -> current phenotype edges
        for dep_phen_name in phen_data.get('dependent_phens', []):
            dep_phen_idx = phen_name_to_idx[dep_phen_name]
            dep_phen_node_idx = n_loci + dep_phen_idx
            edges.append([dep_phen_node_idx, phen_node_idx])
    
    # Convert to tensor and make undirected
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # Add reverse edges for undirected graph
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Node type masks
    loci_mask = torch.zeros(n_loci + n_phens, dtype=torch.bool)
    loci_mask[:n_loci] = True
    
    pheno_mask = torch.zeros(n_loci + n_phens, dtype=torch.bool)  
    pheno_mask[n_loci:] = True
    
    return edge_index, loci_mask, pheno_mask

# --- Interpretability Helper Functions ---
def forward_pheno_for_captum(genotypes, gat_encoder, p_decoder, edge_index, loci_mask, pheno_mask, device):
    """Forward function for Captum phenotype attribution"""
    # Create graph data for single sample
    batch_size = genotypes.shape[0]
    
    # Create node features
    n_loci = loci_mask.sum().item()
    n_phens = pheno_mask.sum().item()
    n_alleles = genotypes.shape[1] // n_loci
    
    genotypes_reshaped = genotypes.view(batch_size, n_loci, n_alleles)
    geno_features = genotypes_reshaped.view(batch_size * n_loci, n_alleles)
    pheno_features = torch.zeros(batch_size * n_phens, n_alleles, device=device)
    
    # Create batched graph
    node_features = torch.cat([geno_features, pheno_features], dim=0)
    
    # Create batch indices
    batch_indices = torch.repeat_interleave(torch.arange(batch_size, device=device), n_loci + n_phens)
    
    # Expand edge indices for batch
    edge_indices_expanded = []
    for i in range(batch_size):
        offset = i * (n_loci + n_phens)
        edge_indices_expanded.append(edge_index + offset)
    edge_index_batch = torch.cat(edge_indices_expanded, dim=1)
    
    # Create masks for batch
    loci_mask_batch = loci_mask.repeat(batch_size)
    pheno_mask_batch = pheno_mask.repeat(batch_size)
    
    # Create data object
    data = type('Data', (), {
        'x': node_features,
        'edge_index': edge_index_batch,
        'loci_mask': loci_mask_batch,
        'pheno_mask': pheno_mask_batch,
        'batch': batch_indices
    })()
    
    # Forward pass
    z = gat_encoder(data)
    return p_decoder(z)

def forward_fitness_for_captum(genotypes, gat_encoder, p_decoder, f_predictor, edge_index, loci_mask, pheno_mask, device):
    """Forward function for Captum fitness attribution"""
    pheno_pred = forward_pheno_for_captum(genotypes, gat_encoder, p_decoder, edge_index, loci_mask, pheno_mask, device)
    return f_predictor(pheno_pred)

def extract_attention_weights(gat_encoder, data_loader, device, n_samples=100):
    """Extract attention weights from GAT for interpretability"""
    gat_encoder.eval()
    attention_weights = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= n_samples:
                break
                
            batch = batch.to(device)
            
            # Hook to capture attention weights
            attention_scores = []
            
            def attention_hook(module, input, output):
                if hasattr(output, 'attention_weights'):
                    attention_scores.append(output.attention_weights.cpu())
            
            # Register hooks on GAT layers
            handles = []
            for module in gat_encoder.modules():
                if isinstance(module, GATv2Conv):
                    handles.append(module.register_forward_hook(attention_hook))
            
            # Forward pass
            _ = gat_encoder(batch)
            
            # Remove hooks
            for handle in handles:
                handle.remove()
            
            if attention_scores:
                attention_weights.append(attention_scores)
    
    return attention_weights

# --- Dataset Classes ---
class PhenoDataset(Dataset):
    """Simple phenotype dataset for P-P autoencoder training"""
    def __init__(self, data_file, pheno_mean=None, pheno_std=None):
        dataset = pk.load(open(data_file, "rb"))
        phens = torch.tensor(dataset["phenotypes"], dtype=torch.float32)
        
        if pheno_mean is not None and pheno_std is not None:
            self.phens = (phens - pheno_mean) / pheno_std
        else:
            self.phens = phens

    def __len__(self):
        return len(self.phens)

    def __getitem__(self, idx):
        return self.phens[idx]

class GraphGenotypeDataset(Dataset):
    """Graph dataset for GAT training"""
    def __init__(self, data_file, edge_index, loci_mask, pheno_mask, pheno_mean, pheno_std, fit_mean=None, fit_std=None):
        dataset = pk.load(open(data_file, "rb"))
        
        # Load and normalize data
        phens = torch.tensor(dataset["phenotypes"], dtype=torch.float32)
        self.norm_phens = (phens - pheno_mean) / pheno_std
        
        if "fitness" in dataset and fit_mean is not None and fit_std is not None:
            fitness = torch.tensor(dataset["fitness"], dtype=torch.float32)
            self.norm_fitness = (fitness - fit_mean) / fit_std
            self.has_fitness = True
        else:
            self.norm_fitness = None
            self.has_fitness = False
        
        self.genotypes = torch.tensor(dataset["genotypes"], dtype=torch.float32)
        self.edge_index = edge_index
        self.loci_mask = loci_mask
        self.pheno_mask = pheno_mask
        
        # Graph structure info
        self.n_loci = loci_mask.sum().item()
        self.n_phens = pheno_mask.sum().item()
        self.n_alleles = self.genotypes.shape[2]

    def __len__(self):
        return len(self.genotypes)

    def __getitem__(self, idx):
        # Create node features: genotype nodes get one-hot, phenotype nodes get learnable embeddings
        geno_features = self.genotypes[idx].view(self.n_loci, -1)  # [n_loci, n_alleles]
        
        # Phenotype nodes start as zero - will be learned by the model
        pheno_features = torch.zeros(self.n_phens, self.n_alleles, dtype=torch.float32)
        
        # Combine node features
        node_features = torch.cat([geno_features, pheno_features], dim=0)
        
        # Create graph data
        graph_data = Data(
            x=node_features,
            edge_index=self.edge_index,
            loci_mask=self.loci_mask,
            pheno_mask=self.pheno_mask,
            y_pheno=self.norm_phens[idx],
        )
        
        if self.has_fitness:
            graph_data.y_fit = self.norm_fitness[idx]
            
        return graph_data

# --- Model Classes ---
class PhenoEncoder(nn.Module):
    """Phenotype encoder (same as v2)"""
    def __init__(self, vabs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vabs.n_phens, vabs.e_hidden_dim),
            nn.BatchNorm1d(vabs.e_hidden_dim), 
            nn.LeakyReLU(0.01),
            nn.Linear(vabs.e_hidden_dim, vabs.latent_dim),
            nn.BatchNorm1d(vabs.latent_dim), 
            nn.LeakyReLU(0.01)
        )
    
    def forward(self, x): 
        return self.encoder(x)

class PhenoDecoder(nn.Module):
    """Phenotype decoder (same as v2)"""
    def __init__(self, vabs):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(vabs.latent_dim, vabs.d_hidden_dim),
            nn.BatchNorm1d(vabs.d_hidden_dim), 
            nn.LeakyReLU(0.01),
            nn.Linear(vabs.d_hidden_dim, vabs.n_phens)
        )
    
    def forward(self, x): 
        return self.decoder(x)

class ImprovedGATEncoder(nn.Module):
    """
    Fixed GAT encoder with proper node handling and simplified pooling.
    Key improvements:
    1. Learnable phenotype node embeddings
    2. Explicit node type tracking
    3. Robust pooling strategy
    4. Similar architecture complexity to v2
    """
    def __init__(self, vabs):
        super().__init__()
        
        # Learnable embeddings for phenotype nodes
        self.pheno_embedding = nn.Parameter(torch.randn(vabs.n_phens, vabs.n_alleles) * 0.1)
        
        # GAT layers - simpler than v3
        self.conv1 = GATv2Conv(vabs.n_alleles, vabs.ge_hidden_dim, heads=vabs.gat_heads, concat=True)
        self.conv2 = GATv2Conv(vabs.ge_hidden_dim * vabs.gat_heads, vabs.latent_dim, heads=1, concat=False)
        
        # Projection layer to match latent dimension
        self.projection = nn.Linear(vabs.latent_dim, vabs.latent_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        loci_mask, pheno_mask = data.loci_mask, data.pheno_mask
        
        # Replace phenotype node features with learnable embeddings
        x_updated = x.clone()
        
        # Handle batched data: repeat phenotype embeddings for each graph in batch
        if hasattr(data, 'batch'):
            # Get number of graphs in batch
            n_graphs = data.batch.max().item() + 1
            # Repeat phenotype embeddings for each graph
            repeated_embeddings = self.pheno_embedding.repeat(n_graphs, 1)
            x_updated[pheno_mask] = repeated_embeddings
        else:
            # Single graph case
            x_updated[pheno_mask] = self.pheno_embedding
        
        # Apply GAT layers
        x_updated = F.leaky_relu(self.conv1(x_updated, edge_index))
        x_updated = self.conv2(x_updated, edge_index)
        
        # Pool over phenotype nodes only (robust strategy)
        pheno_embeddings = x_updated[pheno_mask]  # Extract phenotype node embeddings
        
        # Handle batched data properly
        if hasattr(data, 'batch'):
            # For batched data, we need to group phenotype embeddings by graph
            pheno_batch = data.batch[pheno_mask]
            graph_embedding = global_mean_pool(pheno_embeddings, pheno_batch)
        else:
            # For single graph, just take mean
            graph_embedding = pheno_embeddings.mean(dim=0, keepdim=True)
        
        # Project to latent dimension
        return self.projection(graph_embedding)

class FitnessPredictor(nn.Module):
    """Simple fitness predictor (same as v3)"""
    def __init__(self, vabs):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(vabs.n_phens, vabs.fit_hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(vabs.fit_hidden_dim, 1)
        )
    
    def forward(self, x): 
        return self.predictor(x)

# --- Main Execution ---
def main():
    parser = ArgumentParser(description="G-P Atlas v4: Fixed Graph-Aware Architecture")
    parser.add_argument("--map_file", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    
    # Training parameters
    parser.add_argument("--n_epochs_pheno", type=int, default=100)
    parser.add_argument("--n_epochs_geno", type=int, default=200)
    parser.add_argument("--n_epochs_fitness", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    
    # Architecture parameters
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--e_hidden_dim", type=int, default=64)
    parser.add_argument("--d_hidden_dim", type=int, default=64)
    parser.add_argument("--ge_hidden_dim", type=int, default=32)
    parser.add_argument("--fit_hidden_dim", type=int, default=32)
    parser.add_argument("--gat_heads", type=int, default=4)
    
    # Noise parameters
    parser.add_argument("--pheno_noise", type=float, default=0.1)
    parser.add_argument("--geno_noise", type=float, default=0.1)
    
    # Interpretability parameters
    parser.add_argument("--calculate_importance", type=str, default="no", help="Calculate feature attributions (yes/no)")
    parser.add_argument("--n_attribution_samples", type=int, default=500, help="Number of samples for attribution calculation")
    
    vabs = parser.parse_args()

    if not os.path.exists(vabs.results_dir):
        os.makedirs(vabs.results_dir)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load G-P map and construct graph
    print("Loading G-P map and constructing graph...")
    gp_map = json.load(open(vabs.map_file, 'r'))
    vabs.n_loci = gp_map['n_loci']
    vabs.n_phens = gp_map['n_phens']
    vabs.n_alleles = gp_map['n_alleles']
    
    edge_index, loci_mask, pheno_mask = construct_graph_from_map(gp_map)
    print(f"Graph constructed: {edge_index.shape[1]} edges, {loci_mask.sum()} loci nodes, {pheno_mask.sum()} pheno nodes")

    # Load datasets and compute normalization
    train_file = os.path.join(vabs.dataset_path, 'train_data.pk')
    test_file = os.path.join(vabs.dataset_path, 'test_data.pk')

    with open(train_file, 'rb') as f:
        train_data_raw = pk.load(f)
    
    pheno_mean = torch.tensor(train_data_raw['phenotypes'], dtype=torch.float32).mean(dim=0)
    pheno_std = torch.tensor(train_data_raw['phenotypes'], dtype=torch.float32).std(dim=0)
    pheno_std[pheno_std == 0] = 1.0
    
    fit_mean = fit_std = None
    if 'fitness' in train_data_raw:
        fit_mean = torch.tensor(train_data_raw['fitness'], dtype=torch.float32).mean()
        fit_std = torch.tensor(train_data_raw['fitness'], dtype=torch.float32).std()
        if fit_std == 0: fit_std = 1.0
    
    pheno_mean, pheno_std = pheno_mean.to(device), pheno_std.to(device)
    if fit_mean is not None:
        fit_mean, fit_std = fit_mean.to(device), fit_std.to(device)

    # Create data loaders
    pheno_train_loader = torch.utils.data.DataLoader(
        PhenoDataset(train_file, pheno_mean, pheno_std), 
        batch_size=vabs.batch_size, shuffle=True
    )
    
    graph_train_loader = PyGDataLoader(
        GraphGenotypeDataset(train_file, edge_index, loci_mask, pheno_mask, pheno_mean, pheno_std, fit_mean, fit_std), 
        batch_size=vabs.batch_size, shuffle=True
    )
    
    graph_test_loader = PyGDataLoader(
        GraphGenotypeDataset(test_file, edge_index, loci_mask, pheno_mask, pheno_mean, pheno_std, fit_mean, fit_std), 
        batch_size=vabs.batch_size, shuffle=False
    )

    # Initialize models
    p_encoder = PhenoEncoder(vabs).to(device)
    p_decoder = PhenoDecoder(vabs).to(device)
    gat_encoder = ImprovedGATEncoder(vabs).to(device)
    if fit_mean is not None:
        f_predictor = FitnessPredictor(vabs).to(device)

    # Optimizers
    optim_P_autoencoder = torch.optim.Adam(
        list(p_encoder.parameters()) + list(p_decoder.parameters()), 
        lr=vabs.lr, weight_decay=1e-5
    )
    optim_GAT = torch.optim.Adam(gat_encoder.parameters(), lr=vabs.lr, weight_decay=1e-5)
    if fit_mean is not None:
        optim_fitness = torch.optim.Adam(f_predictor.parameters(), lr=vabs.lr, weight_decay=1e-5)

    # --- Stage 1: Train Phenotype Autoencoder (same as v2) ---
    print("\n--- Stage 1: Training Phenotype Autoencoder ---")
    p_encoder.train()
    p_decoder.train()
    
    for epoch in range(vabs.n_epochs_pheno):
        progress_bar = tqdm(pheno_train_loader, desc=f"P-P Epoch {epoch+1}/{vabs.n_epochs_pheno}")
        epoch_loss = 0.0
        
        for norm_phens in progress_bar:
            norm_phens = norm_phens.to(device)
            noisy_phens = norm_phens + torch.randn_like(norm_phens) * vabs.pheno_noise
            
            optim_P_autoencoder.zero_grad()
            z = p_encoder(noisy_phens)
            recon_phens = p_decoder(z)
            loss = F.mse_loss(recon_phens, norm_phens)
            
            loss.backward()
            optim_P_autoencoder.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        print(f"P-P Epoch {epoch+1} complete. Average Loss: {epoch_loss/len(pheno_train_loader):.4f}")

    # --- Stage 2: Train GAT Encoder (freeze P-P autoencoder) ---
    print("\n--- Stage 2: Training GAT Encoder ---")
    p_encoder.eval()
    p_decoder.eval()
    p_decoder.requires_grad_(False)  # Freeze decoder like in v2
    gat_encoder.train()
    
    for epoch in range(vabs.n_epochs_geno):
        progress_bar = tqdm(graph_train_loader, desc=f"GAT Epoch {epoch+1}/{vabs.n_epochs_geno}")
        epoch_loss = 0.0
        
        for batch in progress_bar:
            batch = batch.to(device)
            # Reshape target_phens to handle PyTorch Geometric batching
            target_phens = batch.y_pheno.view(-1, vabs.n_phens)
            
            # Add genotype noise (similar to v2)
            if vabs.geno_noise > 0:
                # Apply noise to genotype nodes only
                noise_mask = torch.rand(batch.x[batch.loci_mask].shape[0], device=device) < vabs.geno_noise
                if noise_mask.any():
                    noisy_alleles = torch.randint(0, vabs.n_alleles, (noise_mask.sum(),), device=device)
                    new_one_hot = F.one_hot(noisy_alleles, num_classes=vabs.n_alleles).float()
                    batch.x[batch.loci_mask][noise_mask] = new_one_hot
            
            optim_GAT.zero_grad()
            z = gat_encoder(batch)
            recon_phens = p_decoder(z)
            
            loss = F.mse_loss(recon_phens, target_phens)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(gat_encoder.parameters(), 1.0)
            optim_GAT.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        print(f"GAT Epoch {epoch+1} complete. Average Loss: {epoch_loss/len(graph_train_loader):.4f}")

    # --- Stage 3: Train Fitness Predictor (if fitness data available) ---
    if fit_mean is not None:
        print("\n--- Stage 3: Training Fitness Predictor ---")
        gat_encoder.eval()
        p_decoder.eval()
        f_predictor.train()
        
        # Freeze GAT and decoder
        gat_encoder.requires_grad_(False)
        
        for epoch in range(vabs.n_epochs_fitness):
            progress_bar = tqdm(graph_train_loader, desc=f"Fitness Epoch {epoch+1}/{vabs.n_epochs_fitness}")
            epoch_loss = 0.0
            
            for batch in progress_bar:
                batch = batch.to(device)
                # Reshape target_fitness to handle PyTorch Geometric batching  
                target_fitness = batch.y_fit.view(-1, 1)
                
                optim_fitness.zero_grad()
                
                with torch.no_grad():
                    z = gat_encoder(batch)
                    recon_phens = p_decoder(z)
                
                pred_fitness = f_predictor(recon_phens)
                loss = F.mse_loss(pred_fitness, target_fitness)
                
                loss.backward()
                optim_fitness.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            print(f"Fitness Epoch {epoch+1} complete. Average Loss: {epoch_loss/len(graph_train_loader):.4f}")

    # --- Evaluation ---
    print("\n--- Evaluating Final Model ---")
    gat_encoder.eval()
    p_decoder.eval()
    if fit_mean is not None:
        f_predictor.eval()
    
    all_true_phens, all_pred_phens = [], []
    all_true_fitness, all_pred_fitness = [], []

    with torch.no_grad():
        for batch in graph_test_loader:
            batch = batch.to(device)
            
            z = gat_encoder(batch)
            recon_phens_norm = p_decoder(z)
            
            # Denormalize predictions
            recon_phens = recon_phens_norm * pheno_std + pheno_mean
            true_phens = batch.y_pheno.view(-1, vabs.n_phens) * pheno_std + pheno_mean
            
            all_true_phens.append(true_phens.cpu().numpy())
            all_pred_phens.append(recon_phens.cpu().numpy())
            
            if fit_mean is not None and hasattr(batch, 'y_fit'):
                pred_fitness_norm = f_predictor(recon_phens_norm)
                pred_fitness = pred_fitness_norm * fit_std + fit_mean
                true_fitness = batch.y_fit.view(-1, 1) * fit_std + fit_mean
                
                all_true_fitness.append(true_fitness.cpu().numpy())
                all_pred_fitness.append(pred_fitness.cpu().numpy())

    # Calculate metrics
    true_phens_np = np.vstack(all_true_phens)
    pred_phens_np = np.vstack(all_pred_phens)
    
    per_phen_r2 = [r2_score(true_phens_np[:, i], pred_phens_np[:, i]) for i in range(vabs.n_phens)]
    
    print("\n--- Per-Phenotype R^2 Scores ---")
    for i, r2 in enumerate(per_phen_r2):
        print(f"  Phenotype {i+1:02d}: R^2 = {r2:.4f}")
    
    if fit_mean is not None and all_true_fitness:
        true_fitness_np = np.vstack(all_true_fitness).flatten()
        pred_fitness_np = np.vstack(all_pred_fitness).flatten()
        fitness_r2 = r2_score(true_fitness_np, pred_fitness_np)
        
        print("\n--- Fitness Prediction Performance ---")
        print(f"  Fitness R^2 = {fitness_r2:.4f}")
    else:
        fitness_r2 = None

    # Save models and results
    torch.save(p_encoder.state_dict(), os.path.join(vabs.results_dir, "pheno_encoder.pt"))
    torch.save(p_decoder.state_dict(), os.path.join(vabs.results_dir, "pheno_decoder.pt"))
    torch.save(gat_encoder.state_dict(), os.path.join(vabs.results_dir, "gat_encoder.pt"))
    if fit_mean is not None:
        torch.save(f_predictor.state_dict(), os.path.join(vabs.results_dir, "fitness_predictor.pt"))
    
    results_data = {
        "per_phen_r2": per_phen_r2, 
        "fitness_r2": fitness_r2,
        "mean_pheno_r2": np.mean(per_phen_r2),
        "graph_info": {
            "n_edges": edge_index.shape[1],
            "n_loci": loci_mask.sum().item(),
            "n_phens": pheno_mask.sum().item()
        }
    }
    
    with open(os.path.join(vabs.results_dir, "final_results.pk"), "wb") as f:
        pk.dump(results_data, f)

    print(f"\nEvaluation complete. Models and results saved to {vabs.results_dir}")
    print(f"Mean phenotype R^2: {np.mean(per_phen_r2):.4f}")
    if fitness_r2 is not None:
        print(f"Fitness R^2: {fitness_r2:.4f}")

    print(f"\nEvaluation complete. Models and results saved to {vabs.results_dir}")
    print(f"Mean phenotype R^2: {np.mean(per_phen_r2):.4f}")
    if fitness_r2 is not None:
        print(f"Fitness R^2: {fitness_r2:.4f}")

    # --- Interpretability Analysis ---
    if vabs.calculate_importance == "yes":
        print("\n--- Calculating Feature Attributions ---")
        
        # Create single-sample data loader for attribution
        graph_attr_loader = PyGDataLoader(
            GraphGenotypeDataset(test_file, edge_index, loci_mask, pheno_mask, pheno_mean, pheno_std, fit_mean, fit_std), 
            batch_size=1, shuffle=False
        )
        
        # Flatten genotype function for Captum
        def create_genotype_input(batch):
            """Convert graph batch to flattened genotype for Captum"""
            genotype_nodes = batch.x[batch.loci_mask].view(1, -1)  # [1, n_loci * n_alleles]
            return genotype_nodes
        
        # Feature attribution for phenotypes
        print("Calculating phenotype attributions...")
        fa_pheno = FeatureAblation(
            lambda x: forward_pheno_for_captum(x, gat_encoder, p_decoder, edge_index, loci_mask, pheno_mask, device)
        )
        
        all_pheno_attrs = []
        sample_count = 0
        
        with torch.enable_grad():
            for batch in tqdm(graph_attr_loader, desc="Phenotype Attribution"):
                if sample_count >= vabs.n_attribution_samples:
                    break
                    
                batch = batch.to(device)
                genotype_input = create_genotype_input(batch)
                
                # Calculate attribution for each phenotype
                pheno_attrs_sample = []
                for pheno_idx in range(vabs.n_phens):
                    attr = fa_pheno.attribute(genotype_input, target=pheno_idx)
                    pheno_attrs_sample.append(attr.cpu().detach().numpy())
                
                all_pheno_attrs.append(np.stack(pheno_attrs_sample, axis=0))
                sample_count += 1
        
        # Average attributions across samples
        mean_pheno_attrs = np.mean(np.concatenate(all_pheno_attrs, axis=1), axis=1)  # [n_phens, n_features]
        
        # Feature attribution for fitness (if available)
        all_fitness_attrs = []
        if fit_mean is not None:
            print("Calculating fitness attributions...")
            fa_fitness = FeatureAblation(
                lambda x: forward_fitness_for_captum(x, gat_encoder, p_decoder, f_predictor, edge_index, loci_mask, pheno_mask, device)
            )
            
            sample_count = 0
            with torch.enable_grad():
                for batch in tqdm(graph_attr_loader, desc="Fitness Attribution"):
                    if sample_count >= vabs.n_attribution_samples:
                        break
                        
                    batch = batch.to(device)
                    genotype_input = create_genotype_input(batch)
                    
                    attr = fa_fitness.attribute(genotype_input)
                    all_fitness_attrs.append(attr.cpu().detach().numpy())
                    sample_count += 1
            
            mean_fitness_attrs = np.mean(np.concatenate(all_fitness_attrs, axis=0), axis=0)  # [n_features]
        
        # Extract GAT attention weights
        print("Extracting GAT attention weights...")
        attention_weights = extract_attention_weights(gat_encoder, graph_test_loader, device, n_samples=100)
        
        # Save interpretability results
        interpretability_results = {
            'phenotype_attributions': mean_pheno_attrs,
            'fitness_attributions': mean_fitness_attrs if fit_mean is not None else None,
            'attention_weights': attention_weights,
            'feature_names': [f'locus_{i+1}_allele_{j}' for i in range(vabs.n_loci) for j in range(vabs.n_alleles)],
            'phenotype_names': [f'P{i+1:02d}' for i in range(vabs.n_phens)],
            'graph_structure': {
                'edge_index': edge_index.cpu().numpy(),
                'loci_mask': loci_mask.cpu().numpy(),
                'pheno_mask': pheno_mask.cpu().numpy()
            }
        }
        
        with open(os.path.join(vabs.results_dir, "interpretability_results.pk"), "wb") as f:
            pk.dump(interpretability_results, f)
        
        # Save individual attribution files for compatibility with other versions
        with open(os.path.join(vabs.results_dir, "pheno_attributions.pk"), "wb") as f:
            pk.dump(mean_pheno_attrs, f)
        
        if fit_mean is not None:
            with open(os.path.join(vabs.results_dir, "fitness_attributions.pk"), "wb") as f:
                pk.dump(mean_fitness_attrs, f)
        
        print("Interpretability analysis complete. Results saved.")
        
        # Print top important features
        print("\n--- Top Important Genetic Features ---")
        feature_names = [f'L{i+1}_A{j}' for i in range(vabs.n_loci) for j in range(vabs.n_alleles)]
        
        # Overall importance (sum across all phenotypes)
        overall_importance = np.sum(np.abs(mean_pheno_attrs), axis=0)
        top_features = np.argsort(overall_importance)[-10:][::-1]
        
        print("Top 10 most important genetic features (overall):")
        for i, feat_idx in enumerate(top_features):
            print(f"  {i+1:2d}. {feature_names[feat_idx]}: {overall_importance[feat_idx]:.4f}")

if __name__ == "__main__":
    main()