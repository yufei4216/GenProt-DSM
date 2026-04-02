# GenProt-DSM

GenProt-DSM is a cross-modal framework for pathogenicity prediction of psychiatric-disorder-associated missense variants. It integrates DNA-level evolutionary constraints and protein-level sequence semantics using pretrained language models and performs downstream prediction through sequence encoders and gated fusion.

## Related Paper

GenProt-DSM: Leveraging Pretrained Language Models to Integrate Evolutionary Constraints and Protein Semantics for Psychiatric Missense Variant Pathogenicity Prediction

## Repository Overview

This repository provides the code for the downstream experiments of GenProt-DSM, including:

- main model training and evaluation on the balanced test set
- challenging evaluation on rare-variant and unseen-gene test sets
- ablation experiments
- fusion-strategy experiments
- dimensionality experiments
- encoder module experiments
- interpretability and visualization analyses

## Repository Structure

GenProt-DSM/
README.md
requirements.txt
src/
data/
results/

## Data and Features

The processed dataset and feature files used in this study are publicly available at:

https://www.kaggle.com/datasets/yufei4216/processed-dataset-for-genprot-dsm

The Kaggle dataset contains:

- train.csv
- test.csv
- train_gpn_ref.npy
- train_gpn_alt.npy
- test_gpn_ref.npy
- test_gpn_alt.npy
- train_t5_wt.npy
- train_t5_mut.npy
- test_t5_wt.npy
- test_t5_mut.npy

These files correspond to the processed training/test dataset and the pre-extracted DNA-level and protein-level feature arrays used in the downstream experiments.

## Expected Local Data Layout

After downloading the dataset files, the local data structure should be organized as follows:

data/
train.csv
test.csv
merged_memmap_raw/
train_gpn_ref.npy
train_gpn_alt.npy
test_gpn_ref.npy
test_gpn_alt.npy
train_t5_wt.npy
train_t5_mut.npy
test_t5_wt.npy
test_t5_mut.npy

## Important Note on Pretrained Models

This repository does not include the original pretrained model code or weights for:

- GPN-MSA
- ProtT5-XL

The Kaggle dataset already provides the pre-extracted feature arrays required for downstream training and evaluation. Therefore, users do not need to reproduce the upstream feature extraction step in order to run the code in this repository.

## Environment

Tested with Python 3.10 on Windows.

Install dependencies with:

pip install -r requirements.txt

## Main Scripts

Main model training and evaluation:
python src/model.py

Challenge-set evaluation:
python src/model_challenge_tools.py

Baseline tool evaluation:
python src/tool_metrics.py

Ablation study:
python src/ablation.py

Fusion-strategy experiment:
python src/fusion.py

Dimensionality experiments:
python src/fusion_d.py
python src/fusion_dd.py

Encoder module experiments:
python src/GPNMSA_enhanced.py
python src/protT5_enhanced.py

Interpretability and visualization:
python src/interpretability.py

## Reproducibility

To reproduce the downstream workflow:

1. download the processed dataset and feature files from the Kaggle link above
2. organize the files under the expected local directory structure
3. update local paths in the scripts if necessary
4. run model.py for the main experiment
5. run the other scripts for ablation, challenge-set evaluation, and interpretability analyses

## Data Source Statement

The downstream dataset was curated from public and licensed resources, including HGMD, ClinVar, and gnomAD. Users should ensure compliance with the original data-source licenses and usage policies.

## Data and Code Availability

The processed dataset and feature files used in this study are publicly available at https://www.kaggle.com/datasets/yufei4216/processed-dataset-for-genprot-dsm, and the source code is publicly available in this repository.

## Citation

If you use this repository, please cite the corresponding paper.

## Contact

For questions regarding this repository, please contact the corresponding author.
