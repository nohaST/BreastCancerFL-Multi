
# Federated Learning for Multi-Class Breast Cancer Ultrasound Classification

This repository contains the implementation for the paper: **"Federated Learning for Multi-Class Breast Cancer Classification Using Ultrasound Images: A Comparative Analysis"**.

This project investigates the application of different learning paradigms—centralized, local, and federated—for classifying breast ultrasound images into multiple classes (normal, benign, malignant). The goal is to provide a framework for reproducing the experiments.

---

## Models and Methods Explored

*   **Learning Paradigms:**
    *   **Centralized:** A single model trained on data pooled from all sources.
    *   **Local:** Individual models trained independently for each data source.
    *   **Federated:** A global model trained collaboratively across multiple clients without sharing raw data.

*   **Federated Learning Algorithms:**
    *   Federated Averaging (FedAvg)
    *   FedProx

*   **Deep Learning Architectures:**
    *   MobileNetV2
    *   ResNet50V2
    *   InceptionV3

*   **Loss Functions:** 
    *   Categorial Cross Entropy
    *   Tversky Loss
    *   Combined Loss


---

## Installation

This project is developed using Python 3.8+. We recommend using a virtual environment.

**1. Clone the repository:**
```bash
git clone https://github.com/[YourUsername]/[YourRepoName].git
cd [YourRepoName]
## Dataset Setup

**Important:** The datasets are not included in this repository. You must download them manually from the links below and organize them as described.
```
### 1. Download Links

*   **BUSI:** [Dataset on Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
*   **BUS-UCLM:** [Dataset Kaggle](https://www.kaggle.com/datasets/orvile/bus-uclm-breast-ultrasound-dataset)
*   **BCMID:** [Dataset on Zenodo](https://zenodo.org/records/14970848)

### 2. Required Directory Structure
```
[YourRepoName]/
├── data/
│   ├── BUSI/
│   │   ├── train/
│   │   │   ├── benign/
│   │   │   ├── malignant/
│   │   │   └── normal/
│   │   ├── validation/
│   │   │   ├── benign/
│   │   │   ├── malignant/
│   │   │   └── normal/
│   │   └── test/
│   │       ├── benign/
│   │       ├── malignant/
│   │       └── normal/
│   │
│   ├── BUS-UCLM/
│   │   └── (structure repeated for train/validation/test)
│   │
│   └── BCMID/
│       └── (structure repeated for train/validation/test)
│
├── centralized.py
├── local.py
└── federated/
```
To prevent data leakage and ensure a fair evaluation, these splits must be performed at the patient level. 

## How to Run the Experiments

You can run each learning paradigm using the dedicated scripts.

1. Centralized Training

Trains a single model on all datasets combined.

```bash
python run_centralized_local.py --model ResNet50V2 --dataset BCMID BUSI BUSUCLM
```

2. Local Training
Trains and evaluates a separate model for each dataset individually.

```bash
python run_centralized_local.py --model ResNet50V2 --dataset BCMID
```
3. Federated Learning
Runs a full federated learning simulation with clients representing each dataset.

Step 1: Start the Aggregation Server**
```bash
python federated/server.py --algorithm FedProx --num_clients 3 --rounds 100
```
Step 2: Start the Clients
```bash
python federated/client.py --model InceptionV3 --dataset BUSI

python federated/client.py --model InceptionV3 --dataset BUS-UCLM

python federated/client.py --model InceptionV3 --dataset BCMID
```
