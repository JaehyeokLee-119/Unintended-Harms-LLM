# Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights

This repository contains the implementation for the paper "Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights" accepted to ACL 2025.

## Overview

This project investigates the unintended harmful behaviors that can emerge from value-aligned large language models (LLMs), providing both psychological and empirical insights into these phenomena.

## Requirements

- Python 3.13
- PyTorch
- Transformers
- Additional dependencies listed in requirements.txt (if available)

## Data Setup

### Data Preprocessing

Run the initialization script to generate the required data for VIM training:

```bash
bash scripts/Initialize.sh
```

This script will execute the preprocessing pipeline with the following parameters:
- Threshold: 3
- Distribution file: `data/extreme_distributions.csv`
- ValueEval file: `data/valueEval_10.csv`
- Output directory: `data/values`

### HEx-PHI Dataset

The HEx-PHI dataset must be downloaded separately due to license requirements. Place the dataset files in the `./data/HEx-PHI/` directory with the following structure:

```
./data/HEx-PHI/
├── category_1_illegal_activity.csv
├── category_2_child_abuse_content.csv
├── category_3_hate_speech.csv
├── ...
└── category_11_tailored_financial_advice.csv
```

## Project Structure

```
├── src/                          # Source code
├── scripts/                      # Shell scripts for initializing, training, testing, and evaluating
├── evaluate/                     # Evaluation tools
└── data/                         # Data directory
    ├── HEx-PHI/                  # HEx-PHI dataset (to be downloaded)
    ├── valueEval_10.csv          # Value evaluation data
    ├── extreme_distributions.csv # Extreme distributions
    └── ...                       # Other datasets
```

## Usage

### Training

```bash
### VIM Training
bash scripts/train_vim.sh

#### Benign Fine-tuning
bash scripts/benign_train.sh
```

### Inference

```bash
### VIM model Inference
bash scripts/test_vim.sh

### Vanilla Model Inference
bash scripts/vanilla_inference.sh

### Fine-tuned Model Inference
bash scripts/fine_tune_model_inference.sh

### In-Context Learning Inference
bash scripts/ICL_inference.sh
```

## Evaluation

The project includes multiple evaluation approaches:
- **Bias Scoring**: Multiple bias assessment methods (`bias_score_0.py`, `bias_score_1.py`, `bias_score_2.py`)
- **GPT-4 Evaluation**: Automated evaluation using GPT-4 (`gpt4_eval.py`)
- **RealToxicityPrompts**: Evaluation on toxic prompt datasets (`eval_RTP.py`)
- **HolisticBiasR**: Comprehensive bias evaluation (`eval_HBR.py`)
- **Regard-based Evaluation**: Evaluation based on regard metrics (`eval_by_regard.py`)

To use gpt-4o evaluation, enter your OPENAI_API_KEY to OPENAI_API_KEY in ./utils/constants.py
You can find evaluation scripts in ```scripts/inference_evaluation.sh```

## Citation
To be available

## Contact
For questions or issues, please contact [contact information] or open an issue on this repository. 