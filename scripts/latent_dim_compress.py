# This script runs the G-P Atlas training pipeline on the deterministic
# Coronavirus dataset (v4) with different latent space dimensions to
# test the effect of this hyperparameter on model performance.

# --- Experiment 1: Latent Dimension = 8 ---

"--- Starting Run: Latent Dimension = 8 ---"

# 1. Define the new results directory
RESULTS_DIR_8="./results/cov_v4_latent_8/"

# 2. Create the directory
mkdir -p $RESULTS_DIR_8

# 3. Copy the dataset into the new directory
cp ./deterministic_datasets/cov_v4/cov_v4_noise_01_latent10/train_data.pk $RESULTS_DIR_8
cp ./deterministic_datasets/cov_v4/cov_v4_noise_01_latent10/test_data.pk $RESULTS_DIR_8

# 4. Run the training command, pointing to the new directory and setting latent_dim=8
python scripts/g_p_atlas_v2.py \
    --dataset_path $RESULTS_DIR_8 \
    --train_suffix train_data.pk \
    --test_suffix test_data.pk \
    --n_alleles 3 \
    --n_loci_measured 53 \
    --n_phens 16 \
    --n_phens_to_analyze 16 \
    --n_phens_to_predict 16 \
    --latent_dim 8 \
    --n_epochs 200 \
    --n_epochs_gen 200 \
    --batch_size 64 \
    --sd_noise 0.2

"--- Run with Latent Dimension = 8 Complete ---"


# --- Experiment 2: Latent Dimension = 6 ---

"--- Starting Run: Latent Dimension = 6 ---"

# 1. Define the new results directory
RESULTS_DIR_6="./results/cov_v4_latent_6/"

# 2. Create the directory
mkdir -p $RESULTS_DIR_6

# 3. Copy the dataset into the new directory
cp ./deterministic_datasets/cov_v4/cov_v4_noise_01_latent10/train_data.pk $RESULTS_DIR_6
cp ./deterministic_datasets/cov_v4/cov_v4_noise_01_latent10/test_data.pk $RESULTS_DIR_6

# 4. Run the training command, pointing to the new directory and setting latent_dim=6
python scripts/g_p_atlas_v2.py \
    --dataset_path $RESULTS_DIR_6 \
    --train_suffix train_data.pk \
    --test_suffix test_data.pk \
    --n_alleles 3 \
    --n_loci_measured 53 \
    --n_phens 16 \
    --n_phens_to_analyze 16 \
    --n_phens_to_predict 16 \
    --latent_dim 6 \
    --n_epochs 200 \
    --n_epochs_gen 200 \
    --batch_size 64 \
    --sd_noise 0.2

"--- Run with Latent Dimension = 6 Complete ---"

