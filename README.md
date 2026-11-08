# Hospital AI Diagnostic & Cost Prediction System

Graduation project (Data Science & AI) built from two business requirements documents (BRDs):

1. **Hospital Multi-Model AI System** — a Streamlit app with a login gate and two tabs:
   - **Tab 1 — Insurance Cost Prediction** (regression)
   - **Tab 2 — Chest Disease Detection** (YOLO object detection on chest X-rays)
2. **Smart Healthcare Dashboard** — cleaning an 800K-row dirty hospital dataset and building a 2-page Power BI dashboard.

> This is a decision-support prototype, not a certified diagnostic device. Predictions must not replace a doctor's or actuary's judgment.

## Repository structure

```
hospital-ai-diagnostic-system/
├── README.md
├── .gitignore
├── app/                      # Streamlit app + model files it loads
│   ├── app.py
│   ├── requirements.txt
│   ├── insurance_cost_model.joblib
│   ├── insurance_cost_model_metadata.json
│   ├── insurance_form_config.json
│   ├── chexdet_yolo11s_best.pt
│   └── chexdet_deployment_metadata.json
├── notebooks/
│   ├── insurance_cost_prediction.ipynb
│   └── chest_detection_v2.ipynb
├── powerbi/                  # dashboard file / screenshots + DAX measures
├── presentation/
│   └── Hospital_AI_Project_Overview.pptx
└── docs/                     # optional: BRDs / reports
```

## Components

### Tab 1 — Insurance Cost Prediction
- Target: `annual_medical_cost` (100,000 patient records, 54 raw columns).
- Premium and claims columns (`annual_premium`, `monthly_premium`, `total_claims_paid`, `avg_claim_amount`) were dropped as **target leakage** (correlation 0.63–0.97 with the target).
- Models compared with GridSearchCV: Linear, Ridge, Lasso, Polynomial, Random Forest, XGBoost — on both raw and log-transformed targets.
- Final model: **raw Linear Regression** (simplest model, best test score).

| Metric | Value |
|---|---|
| RMSE | $2,840 |
| MAE | $1,769 |
| R² | 0.180 |

The signal in this (synthetic) dataset is weak; all models converged to a similar ceiling, so the simplest one was kept. Per-prediction explainability is shown in the app via model coefficients.

### Tab 2 — Chest Disease Detection
- Dataset: ChestX-Det (3,025 training / 553 test images, 13 disease classes, multi-label bounding boxes).
- Model: **YOLO11s** (Ultralytics), trained on a Kaggle GPU runtime.
- Class imbalance (~16x between the most and least common class) handled with photometric-augmentation oversampling on the training set only. An earlier version that duplicated raw images produced oversized, imprecise boxes and was replaced.
- Test metrics for the final YOLO11s run:
- Precision: 0.5654
Recall:    0.3984
mAP50:     0.4025.

### Smart Healthcare Dashboard (Power BI)
- 800K rows / 20 columns: missing values, inconsistent categories (Gender, Department, Readmission), numbers stored as text, out-of-range values.
- Cleaning in Power Query (standardize categories, tag blanks as `Unspecified`, validate ranges), then a 2-page dashboard (Operational + Financial/Clinical Quality) with DAX measures for the KPIs.
- The Power BI file and measure definitions live in `powerbi/`.

## Run the app locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

First run creates a local `users.db` (SQLite, SHA-256 hashed passwords) with a demo account: `admin` / `ChangeMe123!`. Change it before any real use.

## Data sources
Datasets are not included in this repository. Download them from Kaggle:
- Medical insurance cost dataset used for Tab 1
- CheXDet Image and Annotations (`metformin/chexdet-image-and-annotations`)
- Smart Healthcare dirty dataset (`Smart_Healthcare_Dirty_Data_800K.csv`)

## Known limitations
- Tab 1 form asks for fields such as `risk_score` and `claims_count` that a real intake clerk would not know; the form mirrors the dataset schema, not a real intake workflow.
- Doctor data in the healthcare dataset (8 doctors for 800K visits) suggests synthetic data; per-doctor analysis is indicative only.

## Author
Yossef Abdellatif Elshamey — Computer Science (AI & ML), Benha National University
