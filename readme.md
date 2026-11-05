# RELIO: Concept Drift Detection with Optimal Transport

## Overview

Concept drift occurs when the statistical properties of data streams change over time, potentially making machine learning models less reliable. Detecting these changes is essential for monitoring deployed models and deciding when adaptation is needed.

**RELIO** is a concept drift detection framework based on **Optimal Transport (OT)**. It monitors changes in data distributions by comparing reference and incoming data windows, rather than relying exclusively on changes in predictive performance.

RELIO uses OT distances to quantify distributional changes and applies configurable warning and detection thresholds to identify potential and confirmed drift. It supports supervised and unsupervised learning scenarios.

The project includes a Python implementation and a Streamlit application for exploring and evaluating drift detection.

## Key Features

- **Optimal Transport-based detection:** Uses Wasserstein-1, Wasserstein-2, and Sinkhorn distances.
- **Multivariate distribution comparison:** Compares joint data distributions rather than evaluating features independently.
- **Flexible cost functions:** Supports Euclidean, Standardized Euclidean, and Mahalanobis distances.
- **Configurable detection:** Provides warning, detection, and stability thresholds.
- **Drift characterization:** Includes mechanisms for identifying sudden, gradual, incremental, and recurrent drift.
- **Supervised and unsupervised scenarios:** Supports different learning settings.
- **Experimental evaluation:** Includes synthetic and real-world datasets.
- **Interactive interface:** Provides a Streamlit application for running simulations and comparing RELIO with existing methods.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MeriemAfafHaddou/RELIO.git
cd RELIO
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run main.py
```

## Experimental Evaluation

RELIO is evaluated using synthetic and real-world datasets under different drift scenarios, including sudden, gradual, incremental, and recurrent changes.

The experiments investigate:

- Detection behavior under different types of concept drift.
- The effect of OT metrics and cost functions on detection.
- The use of warning and detection thresholds to identify distributional changes.
- The impact of drift detection on subsequent model performance.

## Research Manuscript

This repository accompanies the research work:

**Concept Drift Detection using Optimal Transport**

A draft manuscript is available in:

`paper/relio_concept_drift_paper_draft.pdf`