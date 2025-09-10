# RELIO: Concept Drift Detection with Optimal Transport

## 📖 Overview
Concept drift is one of the main challenges in deploying machine learning models on data streams. As data evolves, models can quickly lose accuracy, requiring continuous monitoring and retraining.

This project introduces **RELIO**, a drift detection system based on **Optimal Transport (OT)**. Unlike reactive approaches that detect drift only after model performance deteriorates, RELIO takes a **proactive approach** by comparing evolving data distributions directly.

RELIO is implemented in **Python** and comes with a **Streamlit demo** to explore drift detection in action.

---

## ✨ Key Features
- **Concept Drift Detection** using Optimal Transport metrics
- Works with **multidimensional data streams**
- **Proactive detection** before performance collapse
- Tested on **synthetic and real-world datasets**
- Streamlit demo (`main.py`)

---

## 🚀 Getting Started
#### 1. Clone the repository
```bash
git https://github.com/MeriemAfafHaddou/RELIO.git
cd RELIO
```
#### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # on Linux/Mac
.venv\Scripts\activate      # on Windows
```
#### 3. Install dependencies
```bash
pip install -r requirements.txt
```
#### 4. Run the Streamlit demo
```bash
streamlit run main.py
```

## 📊 Results
- Robust performance across synthetic and real-world benchmarks
- Fast and reliable detection of drift before model failure
- Effective in multidimensional, non-stationary environments