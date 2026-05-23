"""
Auto-EDA FastAPI backend
"""
import io
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from core.preprocessor import AutoPreprocessor
from core.visualizer import AutoVisualizer


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


app = FastAPI(title="Auto-EDA", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).parent


def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    content = file.file.read()
    name = file.filename.lower()

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(content))
    elif name.endswith(".json"):
        return pd.read_json(io.BytesIO(content))
    elif name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(content))
    elif name.endswith(".tsv"):
        return pd.read_csv(io.BytesIO(content), sep="\t")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")


def numpy_safe(obj):
    """Recursively convert numpy types to native Python types."""
    return json.loads(json.dumps(obj, cls=NumpyEncoder))


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    target_col: str = Form(default=""),
):
    try:
        df_original = read_uploaded_file(file)

        if df_original.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        if df_original.shape[1] < 1:
            raise HTTPException(status_code=400, detail="Dataset must have at least one column.")

        # Preprocess
        preprocessor = AutoPreprocessor()
        df_processed, preprocess_report = preprocessor.preprocess(df_original)

        # Visualize on original (for interpretability) and report processed info
        col_types = preprocessor.analyze_column_types(df_original)
        visualizer = AutoVisualizer(df_original, df_processed, col_types)
        viz_data = visualizer.generate_all(target_col=target_col.strip() if target_col else None)

        # Column stats
        col_stats = {}
        for col in df_original.columns:
            s = df_original[col]
            stat = {
                "dtype": str(s.dtype),
                "detected_type": col_types.get(col, "unknown"),
                "unique": int(s.nunique()),
                "missing": int(s.isna().sum()),
                "missing_pct": round(float(s.isna().mean() * 100), 2),
            }
            if pd.api.types.is_numeric_dtype(s):
                stat.update({
                    "mean": round(float(s.mean()), 4) if s.notna().any() else None,
                    "std": round(float(s.std()), 4) if s.notna().any() else None,
                    "min": round(float(s.min()), 4) if s.notna().any() else None,
                    "max": round(float(s.max()), 4) if s.notna().any() else None,
                    "median": round(float(s.median()), 4) if s.notna().any() else None,
                    "skew": round(float(s.skew()), 4) if s.notna().any() else None,
                    "kurtosis": round(float(s.kurtosis()), 4) if s.notna().any() else None,
                })
            col_stats[col] = stat

        payload = {
            "status": "ok",
            "filename": file.filename,
            "col_stats": col_stats,
            "preprocess_report": preprocess_report,
            "visualizations": viz_data,
            "columns": list(df_original.columns),
            "sample": df_original.head(5).fillna("NaN").astype(str).to_dict(orient="records"),
        }

        # Sanitize all numpy types before sending
        return JSONResponse(numpy_safe(payload))

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
