#
# Description:
# This script generates a synthetic genotype-phenotype dataset based on a
# deterministic, user-defined Genotype-Phenotype map provided in a JSON file.
#
# Version 2.2: Corrected output dictionary key for phenotypes to 'phenotypes'
# to ensure exact compatibility with the specific g_p_atlas.py training script.
#
# Author: Gemini
# Date: July 1, 2025
#

import json
import pickle as pk
import numpy as np
import argparse
import os

def load_gp_map(map_path):
    """Loads the G-P map from a JSON file."""
    print(f"Loading G-P map from: {map_path}")
    with open(map_path, 'r') as f:
        gp_map = json.load(f)
    return gp_map

def generate_genotypes(n_individuals, n_loci, n_alleles):
    """Generates a population of random genotypes."""
    print(f"Generating {n_individuals} random genotypes...")
    genotype_indices = np.random.randint(0, n_alleles, size=(n_individuals, n_loci))
    genotypes = np.zeros((n_individuals, n_loci, n_alleles))
    for i in range(n_individuals):
        for j in range(n_loci):
            genotypes[i, j, genotype_indices[i, j]] = 1
    return genotypes, genotype_indices

def get_allele_weights(gp_map):
    """
    Generates a 3D matrix of allele weights, applying custom weights from the map.
    Shape: (n_phens, n_loci, n_alleles)
    """
    n_phens = gp_map['n_phens']
    n_loci = gp_map['n_loci']
    n_alleles = gp_map['n_alleles']
    phen_to_idx = {name: i for i, name in enumerate(gp_map['phenotypes'].keys())}

    # 1. Initialize with baseline weights
    allele_weights = np.random.normal(loc=1.0, scale=0.5, size=(n_phens, n_loci, n_alleles))
    allele_weights[:, :, 0] = 1.0

    # 2. Apply custom weights from the map
    if 'custom_weights' in gp_map:
        print("Applying custom allele weights...")
        for p_name, rules in gp_map['custom_weights'].items():
            if p_name not in phen_to_idx: continue
            p_idx = phen_to_idx[p_name]
            for rule in rules:
                locus_idx_0 = rule['locus'] - 1
                weight = rule['weight']
                if rule['allele'] == '*':
                    allele_weights[p_idx, locus_idx_0, 1:] = weight
                else:
                    allele_idx = rule['allele']
                    allele_weights[p_idx, locus_idx_0, allele_idx] = weight
                print(f"  - Set Phenotype '{p_name}', Locus {rule['locus']}, Allele {rule['allele']} -> Weight {weight}")
    return allele_weights

def calculate_phenotypes(genotype_indices, gp_map, allele_weights):
    """Calculates phenotypes deterministically based on the G-P map and pre-computed weights."""
    print("Calculating phenotypes based on the deterministic map...")
    n_individuals = genotype_indices.shape[0]
    phenotype_names = list(gp_map['phenotypes'].keys())
    n_phens = len(phenotype_names)
    phenotypes = np.zeros((n_individuals, n_phens))
    phen_to_idx = {name: i for i, name in enumerate(phenotype_names)}

    for i, p_name in enumerate(phenotype_names):
        rules = gp_map['phenotypes'][p_name]
        p_values = np.zeros(n_individuals)

        # 1. Additive effects
        if 'additive_loci' in rules:
            for locus_idx in rules['additive_loci']:
                locus_idx_0 = locus_idx - 1
                alleles_at_locus = genotype_indices[:, locus_idx_0]
                p_values += allele_weights[i, locus_idx_0, alleles_at_locus] - 1.0

        # 2. Interactive effects
        if 'interactive_loci' in rules:
            for interaction in rules['interactive_loci']:
                locus1_idx_0, locus2_idx_0 = interaction[0] - 1, interaction[1] - 1
                effects1 = allele_weights[i, locus1_idx_0, genotype_indices[:, locus1_idx_0]]
                effects2 = allele_weights[i, locus2_idx_0, genotype_indices[:, locus2_idx_0]]
                p_values += (effects1 * effects2) - 1.0

        # 3. Phenotype-Phenotype dependencies
        if 'dependent_phens' in rules:
             for dep_p_name in rules['dependent_phens']:
                 dep_p_idx = phen_to_idx[dep_p_name]
                 p_values += phenotypes[:, dep_p_idx]

        phenotypes[:, i] = p_values
    return phenotypes

def main(args):
    """Main function to generate and save the dataset."""
    gp_map = load_gp_map(args.map_file)
    allele_weights = get_allele_weights(gp_map)
    genotypes, genotype_indices = generate_genotypes(args.n_individuals, gp_map['n_loci'], gp_map['n_alleles'])
    phenotypes_no_noise = calculate_phenotypes(genotype_indices, gp_map, allele_weights)

    print(f"Adding noise (level: {args.noise})...")
    noise_matrix = np.random.randn(*phenotypes_no_noise.shape) * np.std(phenotypes_no_noise, axis=0) * args.noise
    noisy_phenotypes = phenotypes_no_noise + noise_matrix

    n_train = int(args.n_individuals * 0.8)
    
    # CRITICAL FIX: The output dictionary keys must match what the g_p_atlas.py script expects.
    # The phenotype data key must be 'phenotypes'.
    train_data = {
        'genotypes': genotypes[:n_train],
        'phenotypes': noisy_phenotypes[:n_train], # Corrected Key
        'true_phens_no_noise': phenotypes_no_noise[:n_train],
        'gp_map': gp_map
    }
    test_data = {
        'genotypes': genotypes[n_train:],
        'phenotypes': noisy_phenotypes[n_train:], # Corrected Key
        'true_phens_no_noise': phenotypes_no_noise[n_train:],
        'gp_map': gp_map
    }

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    train_path = os.path.join(args.out_dir, 'train_data.pk')
    test_path = os.path.join(args.out_dir, 'test_data.pk')

    with open(train_path, 'wb') as f: pk.dump(train_data, f)
    with open(test_path, 'wb') as f: pk.dump(test_data, f)
        
    print("\nDataset generation complete!")
    print(f"Training data saved to: {train_path}")
    print(f"Testing data saved to: {test_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic data from a deterministic G-P map.")
    parser.add_argument('--map_file', type=str, required=True, help='Path to the JSON file containing the G-P map.')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save the output pickle files.')
    parser.add_argument('--n_individuals', type=int, default=10000, help='Total number of individuals to simulate.')
    parser.add_argument('--noise', type=float, default=0.1, help='Fraction of noise to add to the phenotype values.')
    
    args = parser.parse_args()
    main(args)
