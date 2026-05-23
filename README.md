# Auto-EDA 🔬

A **zero-config** Exploratory Data Analysis tool for ML/DL engineers.  
Drop a dataset → get full preprocessing + visualizations automatically.

---

## Features

### Auto-Preprocessing
| Step | Details |
|---|---|
| Column type detection | numeric, categorical, datetime, boolean, text |
| Duplicate removal | Auto-detected and removed |
| Smart imputation | Mean (normal) / Median (skewed) / Mode (categorical) |
| Datetime extraction | year, month, day, dayofweek from date columns |
| Encoding | One-hot (≤10 unique) or Label encoding (high-cardinality) |
| Scaling | StandardScaler (normal) or MinMaxScaler (skewed) |
| Drop threshold | Columns with >60% missing are auto-dropped |

### Visualizations
- 📊 Dataset overview stats
- 🗺️ Missing value heatmap
- 📈 Histograms + Box plots per numeric column
- 🔥 Correlation heatmap (Pearson)
- 🔵 Pair plots (scatter matrix, sampled)
- 📦 Categorical value count bar charts
- 🚨 Outlier detection (IQR + Z-score)
- 📐 Skewness analysis
- 🎯 Target analysis (if target column provided)

---

## Supported File Formats
- `.csv` — Comma-separated
- `.tsv` — Tab-separated
- `.xlsx` / `.xls` — Excel
- `.json` — JSON (array of records or columnar)
- `.parquet` — Apache Parquet

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the server
uvicorn app:app --reload --port 8000

# 3. Open in browser
open http://localhost:8000
```

---

## Project Structure

```
auto-eda/
├── app.py                  # FastAPI server
├── requirements.txt
├── core/
│   ├── preprocessor.py     # AutoPreprocessor class
│   └── visualizer.py       # AutoVisualizer class
└── templates/
    └── index.html          # Full dashboard UI
```

---

## Usage

1. Open `http://localhost:8000`
2. Drag & drop your dataset (CSV, Excel, JSON, Parquet, TSV)
3. Optionally enter a **target column** name for target analysis
4. Click **Analyze →**

The dashboard will show:
- **Overview** — key stats + missing value map
- **Columns** — per-column statistics table
- **Preprocessing** — every transformation applied + skewness chart
- **Distributions** — histogram + box plot per numeric column
- **Correlations** — heatmap + scatter matrix
- **Categoricals** — value count bar charts
- **Outliers** — box plots + IQR/Z-score table
- **Target** — distribution + top feature correlations *(if target provided)*
- **Sample Data** — first 5 rows

---

## Extending

**Add a custom imputation strategy:**
Edit `compute_missing_strategy()` in `core/preprocessor.py`.

**Add a new visualization:**
Add a method to `AutoVisualizer` in `core/visualizer.py` and return it from `generate_all()`.

**Add file format support:**
Edit `read_uploaded_file()` in `app.py`.
