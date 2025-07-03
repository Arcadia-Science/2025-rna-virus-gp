#
# Description:
# This script generates two synthetic genotype-phenotype datasets based on the
# detailed simulation plans for an Influenza-like virus (IAV) and a
# Coronavirus-like virus (CoV).
#
# It uses the provided `tools_for_phen_gen_creation.py` module to create the
# datasets and saves them as pickle files for later use with the G-P Atlas model.
#
# Usage:
# Ensure `tools_for_phen_gen_creation.py` is in the same directory or in the
# Python path, then run this script from the command line:
# python3 generate_datasets.py
#

import pickle as pk
import tools_for_phen_gen_creation as sim_tools
import numpy as np
import os

print("Starting dataset generation...")

# ---
# Plan 1: Generate the Influenza A Virus (IAV) Simulated Dataset
# ---
print("\nGenerating IAV-like dataset...")

# Parameters are derived directly from our "Simulation Plan: IAV Dataset v1"
iav_params = {
    "n_animals": 10000,
    "n_phens": 12,
    "n_loci": 30,
    "n_as": 3,
    "noise": 0.1,
    "n_loci_ip": 6,      # Approximate number of loci influencing a typical phenotype
    "p_pleio": 0.4,      # Probability of a gene affecting more than one trait
    "p_interact": 0.25,  # Probability of epistatic interactions
    "downsample": True   # This is the default, splits into train/test
}

# The make_genotype function returns two dictionaries, one for training and one for testing.
iav_train_data, iav_test_data = sim_tools.make_genotype(**iav_params)

# FIX: The source script returns genotypes and phenotypes as lists/tuples.
# They must be converted to NumPy arrays to have a .shape attribute and for
# compatibility with machine learning libraries.
iav_train_data['genotypes'] = np.array(iav_train_data['genotypes'])
iav_test_data['genotypes'] = np.array(iav_test_data['genotypes'])
iav_train_data['phens'] = np.array(iav_train_data['phens'])
iav_test_data['phens'] = np.array(iav_test_data['phens'])


# Define output directory and filenames
output_dir = "simulated_datasets"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

iav_train_filename = os.path.join(output_dir, "iav_simulation_train.pk")
iav_test_filename = os.path.join(output_dir, "iav_simulation_test.pk")

# Save the datasets to pickle files
with open(iav_train_filename, "wb") as f:
    pk.dump(iav_train_data, f)
with open(iav_test_filename, "wb") as f:
    pk.dump(iav_test_data, f)

print(f"IAV dataset generated successfully.")
print(f"  - Training data saved to: {iav_train_filename}")
print(f"  - Testing data saved to: {iav_test_filename}")
print(f"  - Training genotypes shape: {iav_train_data['genotypes'].shape}")
print(f"  - Training phenotypes shape: {iav_train_data['phens'].shape}")


# ---
# Plan 2: Generate the Coronavirus (SARS-CoV-2-like) Simulated Dataset
# ---
print("\nGenerating Coronavirus-like dataset...")

# Parameters are derived directly from our "Simulation Plan: Coronavirus Dataset v1"
cov_params = {
    "n_animals": 10000,
    "n_phens": 15,
    "n_loci": 50,
    "n_as": 3,
    "noise": 0.1,
    "n_loci_ip": 10,     # Higher number of loci per trait for more complexity
    "p_pleio": 0.6,      # Higher pleiotropy, a key feature of our CoV plan
    "p_interact": 0.4,   # Higher interaction rate for complex epistasis
    "downsample": True
}

# Generate the dataset
cov_train_data, cov_test_data = sim_tools.make_genotype(**cov_params)

# FIX: Convert lists/tuples to NumPy arrays for both genotypes and phenotypes.
cov_train_data['genotypes'] = np.array(cov_train_data['genotypes'])
cov_test_data['genotypes'] = np.array(cov_test_data['genotypes'])
cov_train_data['phens'] = np.array(cov_train_data['phens'])
cov_test_data['phens'] = np.array(cov_test_data['phens'])


# Define filenames
cov_train_filename = os.path.join(output_dir, "cov_simulation_train.pk")
cov_test_filename = os.path.join(output_dir, "cov_simulation_test.pk")

# Save the datasets to pickle files
with open(cov_train_filename, "wb") as f:
    pk.dump(cov_train_data, f)
with open(cov_test_filename, "wb") as f:
    pk.dump(cov_test_data, f)

print(f"Coronavirus dataset generated successfully.")
print(f"  - Training data saved to: {cov_train_filename}")
print(f"  - Testing data saved to: {cov_test_filename}")
print(f"  - Training genotypes shape: {cov_train_data['genotypes'].shape}")
print(f"  - Training phenotypes shape: {cov_train_data['phens'].shape}")

print("\nDataset generation complete.")
