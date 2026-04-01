# GenProt-DSM
Code and processed data for GenProt-DSM

GenProt-DSM is a cross-modal framework for pathogenicity prediction of psychiatric-disorder-associated missense variants. It integrates DNA-level evolutionary constraints and protein-level sequence semantics using pretrained language models and performs downstream prediction through sequence encoders and gated fusion.

## Related Paper

**GenProt-DSM: Leveraging Pretrained Language Models to Integrate Evolutionary Constraints and Protein Semantics for Psychiatric Missense Variant Pathogenicity Prediction**

## Repository Overview

This repository provides the code and processed feature files required to run the downstream experiments of GenProt-DSM, including:

- main model training and evaluation on the balanced test set
- challenging evaluation on rare-variant and unseen-gene test sets
- ablation experiments
- fusion-strategy experiments
- dimensionality experiments
- encoder module experiments
- interpretability and visualization analyses

## Repository Structure

```text
GenProt-DSM/
├─ README.md
├─ requirements.txt
├─ src/
├─ data/
└─ results/

Data and Features

The repository contains:
	•	data/train.csv: training set
	•	data/test.csv: test set
	•	data/merged_memmap_raw/: pre-extracted feature arrays used for downstream training and evaluation

The feature directory contains eight .npy files:
	•	train_gpn_ref.npy
	•	train_gpn_alt.npy
	•	test_gpn_ref.npy
	•	test_gpn_alt.npy
	•	train_t5_wt.npy
	•	train_t5_mut.npy
	•	test_t5_wt.npy
	•	test_t5_mut.npy

These files correspond to DNA-level features extracted by GPN-MSA and protein-level features extracted by ProtT5-XL.

Important Note on Pretrained Models

This repository does not include the original pretrained model code or weights for:
	•	GPN-MSA
	•	ProtT5-XL

Users who wish to reproduce the upstream feature extraction process should obtain these resources from their official sources.

This repository focuses on the downstream training, evaluation, and interpretability pipeline of GenProt-DSM.

Environment
Tested with Python 3.13 on Windows.
Install dependencies with:
pip install -r requirements.txt
Main Scripts
Main model training and evaluation
python src/model.py
Challenge-set evaluation
python src/model_challenge_tools.py
Baseline tool evaluation
