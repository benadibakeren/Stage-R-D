# Generative AI-Assisted ADHD Screening from EEG Data and Federated Learning

**Keren Benadiba** — R&D Internship, UMONS (Service SEMi), Mons, Belgium  
**Supervisor:** Jeremie Biringanine Ruvunangiza — University of Mons

---

## Project Overview

This project proposes a unified, privacy-preserving pipeline for 
automated ADHD screening from EEG signals. The pipeline combines:

- **Quantitative EEG feature extraction** (TBR, FAA, spectral power)
- **EEG-to-text serialization** — converting neurophysiological 
  profiles into structured clinical text
- **BiomedBERT fine-tuning** — domain-specific BERT for ADHD/Control 
  classification
- **BioGPT report generation** — automated clinical report generation
- **Federated Learning simulation** — privacy-preserving distributed 
  training across simulated hospital nodes (FedAvg)

---

## Repository Structure
Stage-R-D/

│

├── data/                          # Processed datasets

│   ├── patient_text.csv           # 951 EEG text segments (30s windows)

│   ├── patient_text_w10s.csv      # Window size comparison datasets

│   ├── patient_text_w20s.csv

│   ├── patient_text_w30s.csv

│   ├── patient_text_w60s.csv

│   └── patient_text_w120s.csv

│

├── preprocessing/                 # EEG processing scripts

│   ├── extract_features.py        # Welch PSD, TBR, FAA extraction

│   ├── eeg_to_text.py             # EEG-to-text serialization

│   └── compute_tbr.py             # Theta/Beta Ratio computation

│

├── models/                        # Training scripts

│   ├── train_biomedbert_final.py  # BiomedBERT fine-tuning (GroupKFold)

│   ├── Federated_train.py         # FL simulation (FedAvg, 4 nodes)

│   └── test_biogpt.py             # BioGPT report generation

│

├── hbn_dataset/                   # HBN-EEG pipeline

│   ├── download_hbn_sample.py     # Download subjects from OpenNeuro

│   ├── extract_feature_hbn.py     # Feature extraction (GSN HydroCel)

│   └── hbn_eeg_to_text.py         # HBN serialization

|

├── nasrabadi_raw/                  

│   ├── adhdata.csv             # Raw EEG data — download from IEEE DataPort

│   ├── extract.py              # Early feature extraction script

│   └── test_windows.py         # Window size testing script

│

├── notebooks/

│   └── pipeline_gpt_bert.ipynb   # Complete pipeline notebook (Colab)

│

├── requirements.txt

└── README.md

---

## Datasets

### Primary — Nasrabadi et al. (2020)
- 121 subjects (61 ADHD, 60 Control)
- 19 EEG channels, 128 Hz, visual attention task
- Binary clinical labels
- Source: [IEEE DataPort](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children)
- **Note:** Raw data (`adhdata.csv`) not included in this repo due to 
  size — download from the link above and place in `nasrabadi_raw/`

### Secondary — HBN-EEG (OpenNeuro)
- 1,440 subjects across 11 release sites (ds005505–ds005515)
- 128 channels (GSN HydroCel), 500 Hz, resting state, BIDS format
- Continuous CBCL attention scores (threshold: >0.5 = ADHD, <-0.5 = Control)
- Source: [OpenNeuro](https://openneuro.org/datasets/ds005505)

---

## Installation

```bash
git clone https://github.com/benadibakeren/Stage-R-D.git
cd Stage-R-D
pip install -r requirements.txt
```

---

## Reproducing the Experiments

### Step 1 — Feature Extraction
```bash
python preprocessing/extract_features.py
python preprocessing/eeg_to_text.py
```
Generates `data/patient_text.csv` — 951 EEG text segments.

### Step 2 — BiomedBERT Fine-tuning
```bash
python models/train_biomedbert_final.py
```
- GroupKFold 5-fold cross-validation
- AdamW lr=3e-5, batch=16, 10 epochs, linear warmup
- Best model saved as `best_biomedbert.pt`
- **Results:** Patient-level accuracy 91.80% on a fixed 50/50 split 
  (segment-level GroupKFold: 99.68% ± 0.63%)

### Step 3 — BioGPT Clinical Report Generation
```bash
python models/test_biogpt.py
```
Generates a natural language clinical report from EEG features.

### Step 4 — Federated Learning Simulation
```bash
python models/Federated_train.py
```
- Fixed 50/50 patient-level train/test split
- Patient-level evaluation via majority voting
- 5 local epochs per round, 5 communication rounds
- **Results:** Converges from 72.13% (Round 1) to 90.16% (Round 4)
  — 1.64% performance loss vs centralized baseline of 91.80%

### Step 5 — Window Size Comparison
First generate the window CSVs:
\`\`\`bash
python preprocessing/eeg_to_text.py
\`\`\`

Then run BiomedBERT with each window file:
\`\`\`bash
python models/train_biomedbert_final.py data/patient_text_w10s.csv
python models/train_biomedbert_final.py data/patient_text_w20s.csv
python models/train_biomedbert_final.py data/patient_text_w30s.csv
python models/train_biomedbert_final.py data/patient_text_w60s.csv
python models/train_biomedbert_final.py data/patient_text_w120s.csv
\`\`\`

Note: accuracies are segment-level metrics used for window comparison only.
| Window | Segments | Segment-level Accuracy |
|---|---|---|
| 10s | 1630 | 94.86% ± 4.75% |
| 20s | 783 | 93.36% ± 5.28% |
| **30s** | **508** | **94.12% ± 5.03%** ← optimal |
| 60s | 225 | 74.22% ± 13.96% |
| 120s | 82 | 63.46% ± 12.92% |


> **Note:** `patient_text.csv` (951 segments) uses a different 
> segmentation approach than `patient_text_w30s.csv` (508 segments). 
> The former is used for BiomedBERT classification and FL. 
> The latter is used only for window size comparison.

### Step 6 — HBN-EEG Cross-Dataset Experiment
```bash
python hbn_dataset/download_hbn_sample.py
python hbn_dataset/extract_feature_hbn.py
```
Cross-dataset accuracy: **37.5%** — confirms domain shift and 
motivates the Federated Learning approach.

> **Note:** This is a manual cross-dataset transfer test, not a 
> standalone reproducible script.

### Step 7 — Multi-Site Federated Learning on HBN-EEG
```bash
python hbn_dataset/download_hbn_multisite.py
python models/FL_multisite_HBN.py
```
- 3 genuine HBN-EEG sites (ds005515, ds005511, ds005509)
- 15 ADHD + 15 Control per site
- Same FL protocol as Nasrabadi

| Configuration | Accuracy |
|---|---|
| Single-site (ds005515 only) | 38.89% |
| Centralized (3 sites) | 61.11% |
| Federated Learning (best round) | 61.11% |
| Gain FL vs single-site | +22.22% |

---

## Key Results

| Experiment | Result |
|---|---|
| BiomedBERT centralized (Nasrabadi) | 99.68% ± 0.63% |
| Optimal window size | 30 seconds |
| FL convergence (5 epochs/round) | 100% by Round 4 |
| Cross-dataset generalization (HBN) | 37.5% |

---

## Requirements
torch

transformers

scikit-learn

pandas

numpy

scipy

matplotlib

mne

openneuro-py

---

## Author

**Keren Benadiba**  
5th year engineering student — Data Science & AI  
ESIEE Paris (Université Gustave Eiffel)  
R&D Internship — UMONS, Service SEMi, Mons, Belgium  

---

## Acknowledgments

- Dataset: Nasrabadi et al. (2020), IEEE DataPort
- Dataset: HBN-EEG, OpenNeuro (ds005505–ds005515)
- Models: Microsoft BiomedBERT, Microsoft BioGPT
- FL Algorithm: McMahan et al. (2017), FedAvg
