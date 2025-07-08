# TODO: Replace with the name of the repo

[![run with conda](https://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/projects/miniconda/en/latest/)

## Purpose

This repository contains the complete workflow for validating the G-P Atlas framework on complex, simulated RNA virus datasets. The core motivation is to move beyond statistical validation and rigorously test the model's ability to learn specific, biologically-grounded causal mechanisms.

A central challenge in modern biology is deciphering the nonlinear mapping between genotype and phenotype. The G-P Atlas was developed to address this by modeling epistasis and pleiotropy in a two-tiered neural network architecture. This project builds upon that foundation by developing a custom, deterministic simulation framework to generate synthetic viral populations with precisely known G-P maps. We then tested multiple architectures on this dataset.

The core analyses performed in this repository are:

1. Deterministic Data Generation: Creation of biologically realistic SARS-CoV-2 and Influenza A virus datasets where the ground-truth G-P map (including epistasis, pleiotropy, and phenotype-phenotype dependencies) is explicitly defined.

2. Model Training & Hyperparameter Sweeps: Training the G-P Atlas model on these deterministic datasets under a range of experimental conditions (e.g., varying noise levels and latent dimension sizes).

3. Performance Analysis & Interpretation: A deep analysis of the model's performance to understand its architectural strengths and weaknesses, specifically its ability to learn different types of biological complexity.

The goals of this work are: 

1. To rigorously validate and improve the original G-P Atlas framework on a complex, deterministic simulation of viral evolution.

2. Optimize the G-P Atlas architecture and extend it to predict an emergent fitness score for each individual in the dataset.

3. Demonstrate the utility of a denoising autoencoder for viral fitness prediction.

The final results and the scripts to reproduce them are all contained within this repository.


## Installation and Setup

This repository uses conda to manage software environments and installations. You can find operating system-specific instructions for installing miniconda [here](https://docs.conda.io/projects/miniconda/en/latest/). After installing conda and [mamba](https://mamba.readthedocs.io/en/latest/), run the following command to create the pipeline run environment.

```{bash}
TODO: Replace <NAME> with the name of your environment
mamba env create -n rna_virus_gp_env --file envs/dev.yml
conda activate rna_virus_gp_env
```

<details><summary>Developer Notes (click to expand/collapse)</summary>

1. Install your pre-commit hooks:

    ```{bash}
    pre-commit install
    ```

    This installs the pre-commit hooks defined in your config (`./.pre-commit-config.yaml`).

2. Export your conda environment before sharing:

    As your project develops, the number of dependencies in your environment may increase. Whenever you install new dependencies (using either `pip install` or `mamba install`), you should update the environment file using the following command.

    ```{bash}
    conda env export --no-builds > envs/dev.yml
    ```

    `--no-builds` removes build specification from the exported packages to increase portability between different platforms.
</details>

3. You will also need to install captum for the interpretability analysis:
pip install captum

## Data

This project utilizes a custom data generation pipeline. The primary input is the G-P Map Definition File, maps/cov_map_v4.json, which declaratively defines the entire causal network for the simulation. This map is a biologically grounded, synthetic SARS-CoV-2 genotype-phenotype map. This serves as proof of principle for genotype-phenotype mapping of an RNA virus using a denoising autoencoder (DAE) architecture.

The workflow generates two main types of output data:

Generated Datasets:

The map_to_phenotype_v2.py script produces training and testing datasets from the cov_map_v4.json file. This is adapted from the base G-P Atlas tools_for_phen_gen_creation.py script that produces a simulated dataset. The new script implements a hybrid weight assignment approach, so that custom weights can be fixed for certain loci, while allowing random weights to be set for remaining loci.

The map_to_phenotype_v2.1.py adds an emergent fitness score to the G-P map, defined as a multiplicative function of a critical subset of phenotypes.

Model Training Results: 

The g_p_atlas_v2.5.3.py script saves its outputs to a user-specified results directory, which contains saved model weights (.pt) and final performance metrics (results.txt, final_results.pk, and attribution files). This version also generates several plots, including real vs. predicted scatter plots, metric histograms, and training loss curves.

## Overview

### Description of the folder structure

### Methods

The project is structured as a pipeline, moving from simulation design to final analysis. The core of this work was the iterative development and optimization of the G-P Atlas model.

Adaptation of base G-P Atlas
The Baseline: The v2.0 Architecture
The original G-P Atlas script uses a two-stage training paradigm:

Stage 1: Learning the Phenotype Manifold: A denoising autoencoder (Q_net and P_net) was trained exclusively on phenotype data to learn a compressed, meaningful latent representation (z) of the relationships between phenotypes.

Stage 2: Mapping Genotype to the Latent Space: The phenotype decoder (P_net) was frozen, and a new genotype encoder (GQ_net) was trained to map a flattened genotype vector onto the pre-trained latent space z.

Key Limitation: While architecturally sound, the v2.0 script performed poorly on some complex phenotypes, due to lacking modern training optimizations like target normalization and learning rate scheduling.

The Evolution: The v2.5 Simultaneous, Multi-Task Architecture
The g_p_atlas_v2.5.3.py script represents a significant evolution in both capability and training methodology.

Architectural Additions:

New Data (map_to_phenotype_v2.1.py): The data pipeline was modified to generate a new, separate fitness score for each individual.

New Module (Fitness_predictor_net): A new, small MLP "head" was added. This module runs in parallel to the P_net decoder, taking the same latent vector z as input but trained specifically to predict the scalar fitness score.

The Shift to Simultaneous, Multi-Task Training:
The most significant change is in Stage 2. We now train the GQ_net and the Fitness_predictor_net simultaneously. A single latent vector z is used for both phenotype reconstruction and fitness prediction, and the GQ_net learns from a composite loss function derived from both tasks.

How Training Optimizations Improved Performance
The success of the more complex simultaneous architecture was entirely dependent on three key improvements to the training pipeline:

Target Normalization (The Stabilizer): We now calculate the mean and standard deviation for the phenotypes and fitness score independently and scale both sets of targets to have a mean of 0 and a standard deviation of 1. This puts both tasks on an equal footing in the loss calculation.

Optimizer Regularization (weight_decay): We added a small weight_decay (1e-5) to the Adam optimizer to prevent overfitting and push the model towards a more robust solution.

Learning Rate Scheduling (ReduceLROnPlateau): We implemented a scheduler that monitors the training loss and automatically reduces the learning rate if the loss plateaus, allowing for more precise optimization.

The loss_alpha Optimization: A Key Insight
The final piece of the puzzle was tuning the loss_alpha parameter. Our experiments revealed that the best fitness prediction (R² ~0.69) was achieved with --loss_alpha 0.9. This counter-intuitively tells the model to focus 90% of its effort on accurately reconstructing the phenotypes. This works because the Fitness_predictor_net's performance is entirely dependent on the quality of the latent space z. By forcing the GQ_net to create a high-fidelity latent space, we provide the fitness predictor with a clean, information-rich input.

Final Workflow
This workflow uses the final, optimized scripts to train the model, evaluate its performance, and analyze the feature importances.

Step 1: Generate the Fitness-Aware Dataset

python scripts/map_to_phenotype_v2.1.py \
    --map_file ./maps/cov_map_v4.json \
    --out_dir ./deterministic_datasets/cov_v4_fitness \
    --noise 0.2

Step 2: Train the Model and Calculate Attributions

python scripts/g_p_atlas_v2.5.3.py \
    --dataset_path ./deterministic_datasets/cov_v4_fitness \
    --results_dir ./results/cov_v4_fitness/v2.5.3_sim_lr_0005_run_1 \
    --n_epochs 200 \
    --n_epochs_gen 200 \
    --latent_dim 8 \
    --sd_noise 0.2 \
    --loss_alpha 0.9 \
    --calculate_importance yes

Step 3: Analyze the Interpretability Results

python scripts/analyze_attributions_v2.py \
    --results_dir ./results/v2.5_simultaneous_run_final \
    --map_file ./maps/cov_map_v4.json \
    --top_n 10

Analysis and Visualization
 
All post-training analysis is performed in the CoV_v4_analysis.ipynb Jupyter Notebook. This notebook loads the results from all experimental runs and generates the final plots and summary statistics, including:

Noise and latent dimension sensitivity analyses.

Per-phenotype performance plots.

A quantitative comparison of our deterministic map's complexity vs. the original stochastic generator.

A network visualization of the G-P map.

Additional architectures:

We also experimented with predicting fitness as a 17th phenotype (g_p_atlas_v2.8.py) and using a graph attention network for the G-P training (g_p_atlas_v2.4_fixed.py) but these performed poorly and need further work. Fitness as Phenotype performs well on phenotypes 1-16 but poorly predicting fitness. The GAT is generally less effective across the board.

### Compute Specifications

The analyses were performed on a MacBook Pro with an Apple M3 Max chip and 36GB of RAM. Training time for a single experimental run (200 epochs) was approximately 15-25 minutes.

## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide--credit-for-contributions.md).

---
## For Developers

This section contains information for developers who are working off of this template. Please adjust or edit this section as appropriate when you're ready to share your repo.

### GitHub templates
This template uses GitHub templates to provide checklists when making new pull requests. These templates are stored in the [.github/](./.github/) directory.

### VSCode
This template includes recommendations to VSCode users for extensions, particularly the `ruff` linter. These recommendations are stored in `.vscode/extensions.json`. When you open the repository in VSCode, you should see a prompt to install the recommended extensions.

### `.gitignore`
This template uses a `.gitignore` file to prevent certain files from being committed to the repository.

### `pyproject.toml`
`pyproject.toml` is a configuration file to specify your project's metadata and to set the behavior of other tools such as linters, type checkers etc. You can learn more [here](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)

### Linting
This template automates linting and formatting using GitHub Actions and the `ruff` linter. When you push changes to your repository, GitHub will automatically run the linter and report any errors, blocking merges until they are resolved.
