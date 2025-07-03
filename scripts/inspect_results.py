import pickle as pk
import numpy as np

# --- IMPORTANT: Set this to the path of your results directory ---
results_path = "./results/cov_v4_latent_6/noise_03/test_stats.pk"
# -------------------------------------------------------------

print(f"Loading results from: {results_path}\n")

# Load the pickle file
with open(results_path, 'rb') as f:
    stats = pk.load(f)

# The 'stats' object is a list containing 8 lists of metrics.
# Based on the g_p_atlas.py script:
# - Index 3 contains the R^2 values for the G-P model.
# - Index 7 contains the R^2 values for the P-P model.

gp_r2_scores = stats[3]
pp_r2_scores = stats[7]

print("--- Genotype-to-Phenotype (G-P) Model Performance ---")
for i, r2 in enumerate(gp_r2_scores):
    print(f"  Phenotype {i+1:02d}: R^2 = {r2:.4f}")
print(f"\n  Average G-P R^2: {np.mean(gp_r2_scores):.4f}")

print("\n" + "="*50 + "\n")

print("--- Phenotype-to-Phenotype (P-P) Model Performance ---")
for i, r2 in enumerate(pp_r2_scores):
    print(f"  Phenotype {i+1:02d}: R^2 = {r2:.4f}")
print(f"\n  Average P-P R^2: {np.mean(pp_r2_scores):.4f}")