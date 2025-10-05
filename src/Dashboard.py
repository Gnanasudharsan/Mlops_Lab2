# src/Dashboard.py
import io
import json
import os
import time
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ============ Config ============
st.set_page_config(page_title="ML Lab – Iris & Wine", layout="wide")

# Read API config from Streamlit Secrets or environment
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))
API_KEY = st.secrets.get("API_KEY", os.getenv("API_KEY"))

def api_headers():
    return {"x-api-key": API_KEY} if API_KEY else {}

# ============ Header / badge ============
st.markdown(
    """
    <style>
    .status-badge {
        background:#1b4332; color:#dcfce7; padding:12px 18px; border-radius:10px;
        display:inline-flex; gap:10px; font-weight:600
    }
    .status-badge.warn { background:#4b4a22; color:#fff7c2 }
    .status-badge.err { background:#7f1d1d; color:#fee2e2 }
    </style>
    """,
    unsafe_allow_html=True,
)

hdr, badge = st.columns([3, 1.6])
with hdr:
    st.title("Iris & Wine – Interactive ML Lab")
    st.write("Enhanced Streamlit UI with FastAPI predictions, CSV/JSON uploads, batch outputs, and Plotly EDA.")

# Ping API /health for dynamic badge
api_badge_html = '<div class="status-badge warn">Backend: checking...</div>'
try:
    r = requests.get(f"{API_URL}/health", headers=api_headers(), timeout=3)
    if r.ok:
        api_badge_html = '<div class="status-badge">Backend: online</div>'
    else:
        api_badge_html = '<div class="status-badge warn">Backend: reachable, error</div>'
except Exception:
    api_badge_html = '<div class="status-badge err">Backend: offline</div>'

with badge:
    st.markdown(api_badge_html, unsafe_allow_html=True)

# ============ Sidebar ============
st.sidebar.header("Dashboard")
dataset = st.sidebar.selectbox("Dataset", ["Iris (built-in)", "Wine Quality (CSV in data/)"])
st.sidebar.caption("Wine expects data/winequality-red.csv (semicolon-separated).")

st.sidebar.subheader("Model (RandomForest)")
n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, step=50)
max_depth    = st.sidebar.slider("max_depth", 2, 20, 8, step=1)
test_size    = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, step=0.05)

st.sidebar.subheader("Predict from file")
uploaded = st.sidebar.file_uploader("Upload CSV or JSON", type=["csv", "json"])
st.sidebar.caption("• Single-row → 1 prediction • Multi-row → batch predictions")

# ============ Data loaders ============
def load_iris_df():
    iris = load_iris(as_frame=True)
    df3 = iris.frame.copy()
    df3.rename(columns={"target": "species"}, inplace=True)
    df3["species"] = df3["species"].map(dict(enumerate(iris.target_names)))
    features = iris.feature_names
    target = "species"
    classes = iris.target_names.tolist()
    return df3, features, target, classes

@st.cache_data
def load_wine_df():
    # UCI Wine Quality (red) CSV expected at data/winequality-red.csv (sep=";")
    df3 = pd.read_csv("data/winequality-red.csv", sep=";")
    features = [c for c in df3.columns if c != "quality"]
    target = "quality"
    classes = sorted(df3[target].unique().tolist())
    return df3, features, target, classes

# ============ Choose dataset ============
if dataset == "Iris (built-in)":
    df3, FEATURES, TARGET, CLASSES = load_iris_df()
else:
    try:
        df3, FEATURES, TARGET, CLASSES = load_wine_df()
    except Exception:
        st.warning("Wine CSV not found at data/winequality-red.csv. Falling back to Iris.")
        df3, FEATURES, TARGET, CLASSES = load_iris_df()

# ============ Local training for EDA/feature-importance ============
X, y = df3[FEATURES], df3[TARGET]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42))
])
pipe.fit(Xtr, ytr)
local_acc = pipe.score(Xte, yte)

# ============ Layout ============
left, right = st.columns([1, 2.2])

with left:
    st.subheader("Manual inputs")

    if dataset == "Iris (built-in)":
        # Full 4-feature row from sliders
        vals = {}
        for col in FEATURES:
            lo, hi = float(X[col].min()), float(X[col].max())
            vals[col] = st.slider(col.replace(" (cm)", ""), lo, hi, float(X[col].median()))
        sample = pd.DataFrame([vals], columns=FEATURES)
    else:
        # ---- WINE MODE: build a complete 11-feature row ----
        defaults = X.median(numeric_only=True).to_dict()
        overrides = {}
        for col in FEATURES[:5]:
            lo = float(X[col].quantile(0.01)); hi = float(X[col].quantile(0.99))
            overrides[col] = st.slider(col, lo, hi, float(X[col].median()))
        with st.expander("More features (optional)"):
            for col in FEATURES[5:]:
                lo = float(X[col].quantile(0.01)); hi = float(X[col].quantile(0.99))
                overrides[col] = st.slider(col, lo, hi, float(X[col].median()), key=f"more_{col}")
        row = {c: float(overrides.get(c, defaults[c])) for c in FEATURES}
        sample = pd.DataFrame([row], columns=FEATURES)

    predict_btn = st.button("Predict (via FastAPI)", type="primary")
    st.caption("Uses the backend API with your current sliders and hyperparameters.")

    st.divider()
    st.subheader("File predictions")

    # ---------- Robust CSV/JSON parsing helpers ----------
    def _read_csv_flex(file, prefer_semicolon=False):
        """Try multiple parsers so we don't get a single wide column."""
        try:
            df_try = pd.read_csv(file, sep=";" if prefer_semicolon else ",")
            if df_try.shape[1] > 1:
                return df_try
        except Exception:
            pass
        try:
            file.seek(0)
            df_try = pd.read_csv(file, sep="," if prefer_semicolon else ";")
            if df_try.shape[1] > 1:
                return df_try
        except Exception:
            pass
        file.seek(0)
        return pd.read_csv(file, sep=None, engine="python")

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                prefer_semicolon = dataset.startswith("Wine")
                df_in = _read_csv_flex(uploaded, prefer_semicolon=prefer_semicolon)
            else:
                # Accept {"input_test": {...}} or list[dict]
                text = uploaded.getvalue().decode("utf-8")
                raw = json.loads(text)
                if isinstance(raw, dict) and "input_test" in raw:
                    df_in = pd.DataFrame([raw["input_test"]])
                else:
                    df_in = pd.DataFrame(raw)

            # Clean quoted headers + spaces
            df_in.columns = df_in.columns.str.replace('"', '', regex=False).str.strip()

            # Normalize Iris JSON short keys like data/test.json
            if dataset == "Iris (built-in)":
                iris_keymap = {
                    "sepal_length": "sepal length (cm)",
                    "sepal_width":  "sepal width (cm)",
                    "petal_length": "petal length (cm)",
                    "petal_width":  "petal width (cm)",
                }
                if any(k in df_in.columns for k in iris_keymap):
                    df_in = df_in.rename(columns=iris_keymap)

            # Ensure numeric dtypes for model columns
            for c in FEATURES:
                if c in df_in.columns:
                    df_in[c] = pd.to_numeric(df_in[c], errors="coerce")

            st.write("Preview", df_in.head())

            # Validate columns against current dataset
            missing = [c for c in FEATURES if c not in df_in.columns]
            if missing:
                st.error(f"Missing required columns for this model: {missing}")
            else:
                # ---- CALL FASTAPI for batch predictions ----
                with st.spinner("Predicting via API..."):
                    rows = df_in[FEATURES].to_dict(orient="records")
                    payload = {
                        "dataset": "iris" if dataset.startswith("Iris") else "wine",
                        "rows": rows,
                        "hyper": {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "test_size": test_size,
                        },
                    }
                    r = requests.post(f"{API_URL}/predict", json=payload, headers=api_headers(), timeout=60)
                    r.raise_for_status()
                    data = r.json()
                out = df_in.copy()
                out["prediction"] = data["predictions"]
                st.success("Batch predictions complete (via FastAPI)")
                st.info(f"Hold-out accuracy (API): {data['accuracy']:.3f}")
                st.write(out.head())
                st.download_button(
                    "Download predictions CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not parse or predict: {e}")

with right:
    st.toast("Ready!")
    st.subheader("Model feedback")
    st.info(f"Local hold-out accuracy: {local_acc:.3f} on {dataset}")

    # ---- Manual Predict via FastAPI ----
    if predict_btn:
        try:
            with st.spinner("Predicting via API..."):
                payload = {
                    "dataset": "iris" if dataset.startswith("Iris") else "wine",
                    "rows": sample.to_dict(orient="records"),
                    "hyper": {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "test_size": test_size,
                    },
                }
                r = requests.post(f"{API_URL}/predict", json=payload, headers=api_headers(), timeout=30)
                r.raise_for_status()
                data = r.json()
            st.info(f"Hold-out accuracy (API): {data['accuracy']:.3f}")
            st.success(f"The predicted class is: {data['predictions'][0]}")
        except Exception as e:
            st.error(f"API error: {e}")

    # ---- EDA Tabs (local DF) ----
    tab1, tab2 = st.tabs(["Explore", "Feature importance"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            xcol = st.selectbox("X", FEATURES, index=0)
            ycol = st.selectbox("Y", FEATURES, index=min(1, len(FEATURES) - 1))
            st.plotly_chart(
                px.scatter(df3, x=xcol, y=ycol, color=TARGET, opacity=0.85),
                use_container_width=True,
            )
        with c2:
            hist_col = st.selectbox("Histogram column", FEATURES, index=0)
            st.plotly_chart(
                px.histogram(df3, x=hist_col, color=TARGET, barmode="overlay"),
                use_container_width=True,
            )
    with tab2:
        try:
            imp = pipe.named_steps["rf"].feature_importances_
            imp_df = pd.DataFrame({"feature": FEATURES, "importance": imp}).sort_values(
                "importance", ascending=False
            )
            st.dataframe(imp_df, use_container_width=True)
            st.plotly_chart(px.bar(imp_df, x="feature", y="importance"), use_container_width=True)
        except Exception:
            st.warning("Feature importances not available for this classifier.")