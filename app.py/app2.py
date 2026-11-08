"""
Hospital Multi-Model AI Diagnostic & Cost Prediction System
Graduation Project — Prepared by: Yossef Elshamey

Tab 1: Insurance Cost Prediction (Linear Regression)
Tab 2: Chest Disease Detection (YOLO11n object detection)
"""

import streamlit as st
import sqlite3
import hashlib
import json
import os
import joblib
import pandas as pd
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Hospital AI System", layout="wide")

# ---------------------------------------------------------------------------
# Paths — all artifact files are expected next to this app.py
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSURANCE_MODEL_PATH = os.path.join(BASE_DIR, "insurance_cost_model.joblib")
INSURANCE_METADATA_PATH = os.path.join(BASE_DIR, "insurance_cost_model_metadata.json")
INSURANCE_FORM_CONFIG_PATH = os.path.join(BASE_DIR, "insurance_form_config.json")
CHEXDET_MODEL_PATH = os.path.join(BASE_DIR, "chexdet_yolo11s_best.pt")
CHEXDET_METADATA_PATH = os.path.join(BASE_DIR, "chexdet_deployment_metadata.json")
DB_PATH = os.path.join(BASE_DIR, "users.db")

DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASSWORD = "ChangeMe123!"  # change this after first login


# ---------------------------------------------------------------------------
# Authentication — FR-1.1 to FR-1.5: hashed passwords, session persistence, logout
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        )
        """
    )
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (DEFAULT_ADMIN_USER, hash_password(DEFAULT_ADMIN_PASSWORD)),
        )
    conn.commit()
    conn.close()


def verify_login(username: str, password: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return False
    return row[0] == hash_password(password)


def show_login():
    st.markdown("## 🏥 Hospital AI System")
    st.caption("Sign in to continue — Authorized hospital staff only")

    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="e.g. staff_ahmed")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("LOG IN", use_container_width=True)
            if submitted:
                if verify_login(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        st.caption(f"First run? Default account: `{DEFAULT_ADMIN_USER}` / `{DEFAULT_ADMIN_PASSWORD}`")


# ---------------------------------------------------------------------------
# Cached model loading — Non-Functional: models load once at startup (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_insurance_model():
    model = joblib.load(INSURANCE_MODEL_PATH)
    with open(INSURANCE_METADATA_PATH, "r") as f:
        metadata = json.load(f)
    with open(INSURANCE_FORM_CONFIG_PATH, "r") as f:
        form_config = json.load(f)
    return model, metadata, form_config


@st.cache_resource
def load_chexdet_model():
    model = YOLO(CHEXDET_MODEL_PATH)
    with open(CHEXDET_METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, metadata


# ---------------------------------------------------------------------------
# Tab 1 — Insurance Cost Prediction
# ---------------------------------------------------------------------------
def get_top_contributors(pipeline, input_df, top_n=5):
    """Coefficient x transformed-value explainability for the Linear Regression
    pipeline, aggregated back to the original (pre-encoding) column name.
    Satisfies FR-3.4 (breakdown of which inputs most influenced the prediction)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["model"]

    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed_row = transformed[0]

    feature_names = preprocessor.get_feature_names_out()
    contributions = transformed_row * regressor.coef_

    agg = {}
    for name, contrib in zip(feature_names, contributions):
        original_col = name.split("__", 1)[1].split("_")[0] if "__" in name else name
        agg[original_col] = agg.get(original_col, 0.0) + contrib

    ranked = sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)
    return ranked[:top_n]


def render_insurance_tab():
    model, metadata, form_config = load_insurance_model()
    numeric_features = metadata["numeric_features"]
    ordinal_features = metadata["ordinal_features"]
    nominal_features = metadata["nominal_features"]
    ordinal_orders = metadata["ordinal_orders"]
    categorical_options = form_config["categorical_options"]
    numeric_ranges = form_config["numeric_ranges"]

    st.subheader("Insurance Cost Prediction")
    st.caption(
        "Estimate a patient's expected annual medical cost from demographic, "
        "clinical, and policy attributes."
    )

    with st.form("insurance_form"):
        inputs = {}

        with st.expander("Patient Details", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                inputs["age"] = st.number_input(
                    "Age", min_value=0, max_value=110,
                    value=int(numeric_ranges["age"]["mean"]), step=1,
                )
                inputs["sex"] = st.selectbox("Sex", categorical_options["sex"])
                inputs["region"] = st.selectbox("Region", categorical_options["region"])
                inputs["urban_rural"] = st.selectbox("Urban / Rural", categorical_options["urban_rural"])
            with c2:
                inputs["education"] = st.selectbox("Education", ordinal_orders["education"])
                inputs["marital_status"] = st.selectbox("Marital Status", categorical_options["marital_status"])
                inputs["employment_status"] = st.selectbox(
                    "Employment Status", categorical_options["employment_status"]
                )

        with st.expander("Lifestyle & Clinical Vitals", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                inputs["bmi"] = st.number_input(
                    "BMI", min_value=10.0, max_value=60.0,
                    value=round(numeric_ranges["bmi"]["mean"], 1), step=0.1,
                )
                inputs["smoker"] = st.selectbox("Smoker", categorical_options["smoker"])
                inputs["alcohol_freq"] = st.selectbox("Alcohol Frequency", categorical_options["alcohol_freq"])
            with c2:
                inputs["systolic_bp"] = st.number_input(
                    "Systolic BP (mmHg)", min_value=60.0, max_value=220.0,
                    value=round(numeric_ranges["systolic_bp"]["mean"], 1),
                )
                inputs["diastolic_bp"] = st.number_input(
                    "Diastolic BP (mmHg)", min_value=30.0, max_value=140.0,
                    value=round(numeric_ranges["diastolic_bp"]["mean"], 1),
                )
                inputs["hba1c"] = st.number_input(
                    "HbA1c (%)", min_value=3.0, max_value=15.0,
                    value=round(numeric_ranges["hba1c"]["mean"], 1),
                )

        with st.expander("Chronic Conditions"):
            binary_clinical = [
                "hypertension", "diabetes", "asthma", "copd",
                "cardiovascular_disease", "cancer_history",
                "kidney_disease", "liver_disease", "arthritis", "mental_health",
            ]
            cols = st.columns(2)
            for i, field in enumerate(binary_clinical):
                with cols[i % 2]:
                    label = field.replace("_", " ").title()
                    choice = st.selectbox(label, ["No", "Yes"], key=f"cc_{field}")
                    inputs[field] = 1 if choice == "Yes" else 0
            inputs["chronic_count"] = st.number_input(
                "Number of chronic conditions diagnosed", min_value=0, max_value=10,
                value=int(round(numeric_ranges["chronic_count"]["mean"])), step=1,
            )

        with st.expander("Healthcare Utilization"):
            c1, c2 = st.columns(2)
            with c1:
                inputs["visits_last_year"] = st.number_input(
                    "Provider visits last year", min_value=0, max_value=50, value=0, step=1
                )
                inputs["days_hospitalized_last_3yrs"] = st.number_input(
                    "Days hospitalized (last 3 years)", min_value=0, max_value=60, value=0, step=1
                )
                inputs["medication_count"] = st.number_input(
                    "Active medications", min_value=0, max_value=20, value=0, step=1
                )
                inputs["claims_count"] = st.number_input(
                    "Insurance claims filed this year", min_value=0, max_value=30, value=0, step=1
                )
            with c2:
                inputs["proc_imaging_count"] = st.number_input(
                    "Imaging procedures (X-ray/MRI/CT)", min_value=0, max_value=10, value=0, step=1
                )
                inputs["proc_surgery_count"] = st.number_input(
                    "Surgical procedures", min_value=0, max_value=10, value=0, step=1
                )
                inputs["proc_consult_count"] = st.number_input(
                    "Specialist consultations", min_value=0, max_value=10, value=0, step=1
                )
                inputs["proc_lab_count"] = st.number_input(
                    "Laboratory tests", min_value=0, max_value=10, value=0, step=1
                )

        with st.expander("Policy & Risk Classification"):
            c1, c2 = st.columns(2)
            with c1:
                inputs["plan_type"] = st.selectbox("Plan Type", categorical_options["plan_type"])
                inputs["network_tier"] = st.selectbox("Network Tier", ordinal_orders["network_tier"])
            with c2:
                inputs["risk_score"] = st.number_input(
                    "Actuarial Risk Score (0-1)", min_value=0.0, max_value=1.0,
                    value=round(numeric_ranges["risk_score"]["mean"], 2), step=0.01,
                )
                inputs["is_high_risk"] = 1 if st.selectbox(
                    "Classified High-Risk?", ["No", "Yes"]
                ) == "Yes" else 0
                inputs["had_major_procedure"] = 1 if st.selectbox(
                    "Major medical event in the last year?", ["No", "Yes"]
                ) == "Yes" else 0

        submitted = st.form_submit_button("PREDICT", use_container_width=True)

    if submitted:
        # FR-3.2: basic input validation
        if inputs["age"] <= 0:
            st.error("Age must be greater than 0.")
            return
        if not (10.0 <= inputs["bmi"] <= 60.0):
            st.error("BMI must be within a realistic range (10-60).")
            return

        all_features = numeric_features + ordinal_features + nominal_features
        row = {col: inputs[col] for col in all_features}
        input_df = pd.DataFrame([row])[all_features]

        prediction = model.predict(input_df)[0]

        st.markdown("### Predicted Annual Medical Cost")
        st.markdown(f"## ${prediction:,.2f}")

        contributors = get_top_contributors(model, input_df, top_n=5)
        st.markdown("**Top contributing factors:**")
        for col, contrib in contributors:
            direction = "⬆️ increases" if contrib > 0 else "⬇️ decreases"
            st.write(f"- `{col}` — {direction} the estimate")

        m = metadata["test_metrics"]
        st.caption(
            f"Model: Linear Regression | Test RMSE: ${m['rmse']:,.2f} | "
            f"Test MAE: ${m['mae']:,.2f} | Test R²: {m['r2']:.3f}"
        )
        st.info(
            "This is a decision-support estimate only, based on a model with modest "
            "explanatory power (R² ≈ 0.18 on held-out data). It should support — not "
            "replace — an actuarial or clinical review."
        )


# ---------------------------------------------------------------------------
# Tab 2 — Chest Disease Detection
# ---------------------------------------------------------------------------
def render_chexdet_tab():
    model, metadata = load_chexdet_model()

    st.subheader("Chest Disease Detection")
    st.caption("Upload a chest X-ray to screen for visible abnormalities.")

    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # FR-4.2: basic file validation
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File is too large (max 10 MB).")
            return

        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)

        if st.button("ANALYZE", use_container_width=True):
            results = model.predict(
                source=image,
                conf=metadata["confidence_threshold"],
                imgsz=metadata["image_size"],
                verbose=False,
            )
            result = results[0]

            with col2:
                if len(result.boxes) == 0:
                    st.image(image, caption="No abnormality detected", use_container_width=True)
                else:
                    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
                    st.image(annotated, caption="Detection Result", use_container_width=True)

            st.markdown("### Detection Result")
            if len(result.boxes) == 0:
                # FR-4.6
                st.success("✅ No abnormal finding detected")
            else:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = metadata["class_names"].get(str(cls_id), model.names[cls_id])
                    st.write(f"- **{class_name}** — confidence {conf:.0%}")

            st.caption(
                f"Model: YOLO11s | Test Precision: {metadata.get('precision', 0):.3f} | "
                f"Recall: {metadata.get('recall', 0):.3f} | "
                f"mAP@50: {metadata.get('map50', 0):.3f} | "
                f"mAP@50-95: {metadata.get('map50_95', 0):.3f}"
            )
            st.info("Decision support only — refer to a radiologist for a confirmed diagnosis.")


# ---------------------------------------------------------------------------
# Dashboard — FR-2.1, FR-2.2: two independent tabs
# ---------------------------------------------------------------------------
def show_dashboard():
    top_left, top_right = st.columns([6, 1])
    with top_left:
        st.markdown(f"#### 🏥 Hospital AI System — {st.session_state.username}")
    with top_right:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()

    tab1, tab2 = st.tabs(["Insurance Cost Prediction", "Chest Disease Detection"])
    with tab1:
        render_insurance_tab()
    with tab2:
        render_chexdet_tab()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    init_db()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None

    if not st.session_state.authenticated:
        show_login()
    else:
        show_dashboard()


if __name__ == "__main__":
    main()