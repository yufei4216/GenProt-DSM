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



