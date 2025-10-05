# api/main.py
from typing import List, Optional, Literal
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------------- Config ----------------
API_KEY = os.getenv("API_KEY")  # set this in your shell or hosting platform
WINE_PATH = os.getenv("ML_WINE_PATH", "data/winequality-red.csv")

app = FastAPI(title="ML Lab API (Iris & Wine)")

# CORS (safe defaults; tighten origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # e.g. ["http://localhost:8501", "https://your-streamlit-app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------- Auth -------------------
def require_api_key(x_api_key: str = Header(default=None, alias="x-api-key")):
    # If API_KEY not set, allow all (dev); if set, require exact match.
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --------------- Schemas ----------------
class HyperParams(BaseModel):
    n_estimators: int = Field(200, ge=10, le=1000)
    max_depth: Optional[int] = Field(8, ge=1)
    test_size: float = Field(0.2, ge=0.05, le=0.5)

class PredictRequest(BaseModel):
    dataset: Literal["iris", "wine"]
    rows: List[dict]
    hyper: HyperParams = HyperParams()

class PredictResponse(BaseModel):
    accuracy: float
    predictions: List[str]


# ------------- Data loaders -------------
def load_iris_df():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    features = iris.feature_names
    target = "target"
    return df, features, target

def load_wine_df():
    if not os.path.exists(WINE_PATH):
        raise HTTPException(status_code=400, detail=f"Wine file not found at {WINE_PATH}")
    df = pd.read_csv(WINE_PATH, sep=";")
    features = [c for c in df.columns if c != "quality"]
    target = "quality"
    return df, features, target


# -------------- Training ----------------
def train_pipeline(df: pd.DataFrame, features: List[str], target: str, hyper: HyperParams):
    X, y = df[features], df[target]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=hyper.test_size, random_state=42, stratify=y
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=hyper.n_estimators,
            max_depth=hyper.max_depth,
            random_state=42
        )),
    ])
    pipe.fit(Xtr, ytr)
    acc = pipe.score(Xte, yte)
    return pipe, acc


# --------------- Endpoints --------------
@app.get("/health", dependencies=[Depends(require_api_key)])
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest):
    # select dataset
    if req.dataset == "iris":
        df, FEATURES, TARGET = load_iris_df()
    else:
        df, FEATURES, TARGET = load_wine_df()

    # train quick model (stateless for lab simplicity)
    pipe, acc = train_pipeline(df, FEATURES, TARGET, req.hyper)

    # build input frame
    rows_df = pd.DataFrame(req.rows)

    # map common iris short keys -> sklearn names
    if req.dataset == "iris":
        iris_keymap = {
            "sepal_length": "sepal length (cm)",
            "sepal_width":  "sepal width (cm)",
            "petal_length": "petal length (cm)",
            "petal_width":  "petal width (cm)",
        }
        rows_df = rows_df.rename(columns=iris_keymap)

    # validate columns
    missing = [c for c in FEATURES if c not in rows_df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    # coerce to numeric and check NaNs
    for c in FEATURES:
        rows_df[c] = pd.to_numeric(rows_df[c], errors="coerce")

    if rows_df[FEATURES].isna().any().any():
        raise HTTPException(status_code=400, detail="Non-numeric or null values after coercion in input rows")

    preds = pipe.predict(rows_df[FEATURES]).tolist()
    return PredictResponse(accuracy=float(acc), predictions=[str(p) for p in preds])