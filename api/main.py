# src/api/main.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

app = FastAPI()

# 🔑 API Key for authentication
API_KEY = "33e54e29bf2eecf340199d9ae883f2b9f2ed1a5f81e6308cd4bc61508b2b3b66"

# Request schema
class PredictRequest(BaseModel):
    dataset: str
    rows: list
    hyper: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest, x_api_key: str = Header(None)):
    # 🔑 Validate API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Select dataset
    if req.dataset == "iris":
        iris = load_iris(as_frame=True)
        df = iris.frame.copy()
        X, y = df[iris.feature_names], df["target"]
        classes = iris.target_names.tolist()
    elif req.dataset == "wine":
        df = pd.read_csv("data/winequality-red.csv", sep=";")
        X, y = df.drop(columns=["quality"]), df["quality"]
        classes = sorted(y.unique())
    else:
        raise HTTPException(status_code=400, detail="Unsupported dataset")

    # Train simple model
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=req.hyper.get("n_estimators", 100),
            max_depth=req.hyper.get("max_depth", None),
            random_state=42,
        )),
    ])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=req.hyper.get("test_size", 0.2),
        random_state=42,
        stratify=y
    )
    pipe.fit(Xtr, ytr)
    acc = pipe.score(Xte, yte)

    # Predict
    rows_df = pd.DataFrame(req.rows, columns=X.columns)
    preds = pipe.predict(rows_df)

    return {"predictions": preds.tolist(), "accuracy": acc}