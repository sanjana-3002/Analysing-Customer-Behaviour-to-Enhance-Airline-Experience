## Analysing-Customer-Behaviour-to-Enhance-Airline-Experience (ML+NLP)
From messy flight data to actionable CX: I used ML+NLP to predict satisfaction/loyalty on 129k rows (best RBF-SVM ~81.9% acc) and mine 64k reviews (TF-IDF) into detractor/passive/promoter themes. 

Story: reliability hurts; service lifts (Wi-Fi, boarding, crew, cleanliness) → retention wins.

## 🚀 Overview:
- Purpose: Identify what drives satisfaction & loyalty and surface themes from reviews that teams can act on.
- How it works:
  (i) Tabular ML → feature selection → train & evaluate classifiers for Satisfaction and Loyalty.
  (ii) NLP → clean reviews → TF-IDF → multi-class classification (Detractor/Passive/Promoter) + theme mining.
- Project modes
(i) Satisfaction (tabular)
(ii) Loyalty (tabular): Economy-only and Business-only subsets
(iii) Reviews (NLP): Multi-class classification + word themes

## ✨ Features:
- ML pipelines for satisfaction & loyalty with baseline models (LogReg/DT/RF/XGB/SVM)
- NLP pipeline (clean → tokenize → lemmatize → TF-IDF → classify)
- Interpretable outputs: feature importance (tree models), confusion matrices, class reports
- CX playbook mapping: reliability, boarding, Wi-Fi, crew/service, cleanliness

## 📊 Data & Results (notebooks)
- Tabular dataset: ~129,880 rows, ~24–26 columns
- Reviews dataset: 64,017 texts (NPS → Detractor/Passive/Promoter)

Results (best per notebook)
- Satisfaction: accuracy ≈ 0.807
- Loyalty – Economy: accuracy ≈ 0.920 (RandomForest, n_estimators=500)
- Loyalty – Business: accuracy ≈ 0.948 (RandomForest, n_estimators=100)
- Reviews (NLP): accuracy ≈ 0.82 on test (TF-IDF baseline)

Signals often ranked high: Online boarding, Inflight Wi-Fi, Seat comfort, Inflight entertainment, Gate location, Departure/Arrival time convenience, Age (context), and overall Satisfaction when predicting Loyalty.

## 🧱 Architecture
[Raw Tabular] ─┐    Clean/EDA → Feature Selection → Train (DT/RF/XGB/SVM) → Metrics → Insights
                └─> 
[Reviews Text] ──> Clean → Tokenize/Lemmatize → TF-IDF → Classify (NB/SVM) → Themes → Insights

## 📦 Tech Stack

Python, pandas, scikit-learn, XGBoost, NLTK, wordcloud, Matplotlib/Seaborn, Jupyter

## 📁 Repository Structure
.
├── notebooks/
│   ├── satisfaction.ipynb
│   ├── Only Eco_Loyalty.ipynb
│   ├── Only Business_Loyalty.ipynb
│   └── Text_Classification.ipynb
├── data/            # (add your CSVs here; paths referenced in notebooks)
├── figures/         # generated plots/wordclouds
└── README.md

## ⚙️ Setup

Requirements: Python 3.10+, Jupyter

**Create & activate env**
python -m venv .venv
**Windows:**
.venv\Scripts\activate
**macOS/Linux:**
source .venv/bin/activate

**Install deps**
pip install -U pandas numpy scikit-learn xgboost nltk wordcloud matplotlib seaborn jupyter

**NLTK (one-time)**
python - <<'PY'
import nltk
for p in ["punkt","stopwords","wordnet","omw-1.4"]:
    nltk.download(p)
PY

## ▶️ Run

- Place your datasets in data/ (update paths in the first cell of each notebook if needed).

- Launch notebooks:

  - jupyter notebook notebooks/


Run cells top-to-bottom to reproduce metrics and figures.

## 🧪 Tips & Testing

- Check class imbalance (e.g., Passive in NLP) → try stratified splits, class weights, or resampling.

- Add precision/recall/F1, ROC-AUC, and probability calibration for production-grade evaluation.

- Compare Linear SVM vs Multinomial NB for TF-IDF; try BERT for a stronger baseline.

## 🗺️ Roadmap

- SHAP for model explanations

- Class rebalancing & threshold tuning

- Streamlit dashboard for CX readouts

- Upgrade NLP to transformer baselines (e.g., DistilBERT)

