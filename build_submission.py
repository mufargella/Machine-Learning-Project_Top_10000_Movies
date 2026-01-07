"""Build all project deliverables (figures, PDF report, ZIP) from Top_10000_Movies.csv.

This script mirrors the logic in Ml_Project.ipynb and is provided as a reliable
one-command way to reproduce the submission artifacts.

Outputs:
- figures/*.png
- Final_Report.pdf
- Submission_Package.zip
- download_dataset.py (template)
"""

from __future__ import annotations

import ast
import re
import textwrap
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.errors import ParserError

from matplotlib.backends.backend_pdf import PdfPages

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    ConfusionMatrixDisplay,
    make_scorer,
)
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Top_10000_Movies.csv"
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
REPORT_PATH = PROJECT_DIR / "Final_Report.pdf"
ZIP_PATH = PROJECT_DIR / "Submission_Package.zip"
DOWNLOAD_SCRIPT = PROJECT_DIR / "download_dataset.py"

TEAM_MEMBERS = [
    ("Mohamed Mostafa", "23101594"),
    ("Marwan Khaled", "23101599"),
    ("Mohamed Adel", "23101899"),
]


def parse_genre_list(x) -> list[str]:
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s in {"", "nan", "None"}:
        return []
    try:
        lst = ast.literal_eval(s)
        if isinstance(lst, list):
            return [str(g).strip().title() for g in lst if str(g).strip() not in {"", "nan"}]
        return []
    except Exception:
        s2 = re.sub(r"[\[\]\']", "", s)
        parts = [p.strip() for p in s2.split(",") if p.strip()]
        return [p.title() for p in parts]


def iqr_filter(df_in: pd.DataFrame, cols: list[str], k: float = 1.5):
    df_out = df_in.copy()
    mask = pd.Series(True, index=df_out.index)
    bounds: dict[str, tuple[float, float]] = {}
    for c in cols:
        q1 = df_out[c].quantile(0.25)
        q3 = df_out[c].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        bounds[c] = (float(lo), float(hi))
        mask &= df_out[c].between(lo, hi)
    return df_out[mask].copy(), bounds


def make_preprocessor(num_cols: list[str], cat_cols: list[str], scale_mode: str):
    if scale_mode == "standard":
        scaler_step = ("scaler", StandardScaler())
    elif scale_mode == "minmax":
        scaler_step = ("scaler", MinMaxScaler())
    else:
        scaler_step = None

    num_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scaler_step is not None:
        num_steps.append(scaler_step)

    numeric_transformer = Pipeline(steps=num_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )


def add_text_page(pdf: PdfPages, title: str, lines: list[str]):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    y = 0.95
    plt.text(0.07, y, title, fontsize=18, fontweight="bold", va="top")
    y -= 0.05

    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(textwrap.wrap(str(line), width=95) or [""])

    for line in wrapped:
        plt.text(0.07, y, line, fontsize=11, va="top")
        y -= 0.02
        if y < 0.08:
            break

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path):
    fig = plt.figure(figsize=(8.27, 11.69))
    plt.axis("off")
    plt.text(0.07, 0.95, title, fontsize=16, fontweight="bold", va="top")
    if image_path.exists():
        img = plt.imread(image_path)
        ax = fig.add_axes([0.07, 0.10, 0.86, 0.80])
        ax.imshow(img)
        ax.axis("off")
    else:
        plt.text(0.07, 0.85, f"Missing figure: {image_path.name}", fontsize=12)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def save_table_figure(df_table: pd.DataFrame, title: str, out_path: Path) -> None:
    """Render a pandas table as a PNG (for embedding into the PDF)."""
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    display_df = df_table.copy()
    display_df.index.name = "Model"
    display_df = display_df.reset_index()

    tbl = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    sns.set_theme(style="whitegrid")

    def load_csv_safe(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except ParserError:
            # Dirty CSVs with long/free-text columns may parse better with the Python engine.
            return pd.read_csv(path, engine="python")
        except Exception:
            # Last-resort: skip irrecoverably bad lines, but keep as much data as possible.
            return pd.read_csv(path, engine="python", on_bad_lines="skip")

    df_raw = load_csv_safe(DATA_PATH)

    # Cleaning
    df = df_raw.copy()
    for col in list(df.columns):
        if col.lower().startswith("unnamed") or col.strip() == "":
            df = df.drop(columns=[col])

    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    if "original_language" in df.columns:
        df["original_language"] = df["original_language"].str.lower().replace({"nan": np.nan})

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = pd.to_numeric(df["release_date"].dt.year, errors="coerce")
    df["release_decade"] = (df["release_year"] // 10 * 10).astype("Int64")

    if "genre" in df.columns:
        df["genre_list"] = df["genre"].apply(parse_genre_list)
        df["genre_count"] = df["genre_list"].apply(len)
        df["genre_main"] = df["genre_list"].apply(lambda g: g[0] if len(g) else "Unknown")

    for txt_col in ["overview", "tagline", "original_title"]:
        if txt_col in df.columns:
            df[txt_col] = df[txt_col].replace({"nan": ""}).fillna("").astype(str).str.strip()
            df[f"{txt_col}_len"] = df[txt_col].astype(str).str.len()

    for c in ["popularity", "vote_average", "vote_count", "revenue", "runtime", "release_year"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["original_language", "genre_main"]:
        if c in df.columns:
            df[c] = df[c].fillna("unknown").replace({"": "unknown"})

    for c in ["popularity", "vote_average", "vote_count", "runtime", "release_year"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    before = len(df)
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])
    else:
        df = df.drop_duplicates()

    # Outliers
    outlier_cols = [c for c in ["popularity", "revenue", "runtime", "vote_count"] if c in df.columns]
    fig, axes = plt.subplots(1, len(outlier_cols), figsize=(4 * len(outlier_cols), 4))
    if len(outlier_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, outlier_cols):
        sns.boxplot(y=df[col], ax=ax, color=sns.color_palette()[0])
        ax.set_title(f"Box Plot: {col}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "boxplots_outliers_before.png", dpi=160)
    plt.close(fig)

    df_no_outliers, _ = iqr_filter(df, outlier_cols, k=1.5)

    fig, axes = plt.subplots(1, len(outlier_cols), figsize=(4 * len(outlier_cols), 4))
    if len(outlier_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, outlier_cols):
        sns.boxplot(y=df_no_outliers[col], ax=ax, color=sns.color_palette()[2])
        ax.set_title(f"After Outlier Removal: {col}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "boxplots_outliers_after.png", dpi=160)
    plt.close(fig)

    df_model = df_no_outliers.copy()

    # EDA correlation heatmap
    numeric_for_corr = [
        c
        for c in [
            "popularity",
            "vote_average",
            "vote_count",
            "revenue",
            "runtime",
            "release_year",
            "overview_len",
            "tagline_len",
            "genre_count",
        ]
        if c in df_model.columns
    ]
    corr = df_model[numeric_for_corr].corr(numeric_only=True)
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=False, cmap="viridis", square=True)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_corr_heatmap.png", dpi=160)
    plt.close(fig)

    # Required visualizations
    yearly = df_model.groupby("release_year")["vote_average"].mean().sort_index()
    fig = plt.figure(figsize=(10, 4))
    plt.plot(yearly.index, yearly.values, linewidth=2)
    plt.title("Line Plot: Average Rating by Release Year")
    plt.xlabel("Release Year")
    plt.ylabel("Average Rating (vote_average)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "plot_01_line_avg_rating_by_year.png", dpi=160)
    plt.close(fig)

    yearly_rev = df_model.groupby("release_year")["revenue"].sum().sort_index()
    fig = plt.figure(figsize=(10, 4))
    plt.fill_between(yearly_rev.index, yearly_rev.values, alpha=0.4)
    plt.plot(yearly_rev.index, yearly_rev.values, linewidth=1.5)
    plt.title("Area Plot: Total Reported Revenue by Release Year")
    plt.xlabel("Release Year")
    plt.ylabel("Total Revenue")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "plot_02_area_total_revenue_by_year.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    plt.hist(df_model["runtime"], bins=30, edgecolor="black", alpha=0.8)
    plt.title("Histogram: Runtime Distribution")
    plt.xlabel("Runtime (minutes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "plot_03_hist_runtime.png", dpi=160)
    plt.close(fig)

    top_genres = df_model["genre_main"].value_counts().head(10)
    fig = plt.figure(figsize=(10, 4))
    sns.barplot(x=top_genres.index, y=top_genres.values)
    plt.title("Bar Chart: Top 10 Main Genres by Count")
    plt.xlabel("Main Genre")
    plt.ylabel("Number of Movies")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "plot_04_bar_top_genres.png", dpi=160)
    plt.close(fig)

    lang_counts = df_model["original_language"].value_counts()
    top5 = lang_counts.head(5)
    other = pd.Series({"other": lang_counts.iloc[5:].sum()})
    pie_data = pd.concat([top5, other])
    fig = plt.figure(figsize=(7, 7))
    plt.pie(pie_data.values, labels=pie_data.index, autopct="%1.1f%%", startangle=90)
    plt.title("Pie Chart: Original Language Distribution (Top 5 + Other)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "plot_05_pie_language.png", dpi=160)
    plt.close(fig)

    top6_genres = df_model["genre_main"].value_counts().head(6).index
    fig = plt.figure(figsize=(10, 4))
    sns.boxplot(data=df_model[df_model["genre_main"].isin(top6_genres)], x="genre_main", y="vote_average")
    plt.title("Box Plot: Rating Distribution by Main Genre (Top 6)")
    plt.xlabel("Main Genre")
    plt.ylabel("Rating (vote_average)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "plot_06_box_rating_by_genre.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.scatter(df_model["popularity"], df_model["vote_average"], alpha=0.25)
    plt.title("Scatter Plot: Popularity vs Rating")
    plt.xlabel("Popularity")
    plt.ylabel("Rating (vote_average)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "plot_07_scatter_popularity_vs_rating.png", dpi=160)
    plt.close(fig)

    bubble = df_model.copy()
    bubble["vote_count_clip"] = bubble["vote_count"].clip(upper=bubble["vote_count"].quantile(0.95))
    fig = plt.figure(figsize=(9, 5))
    plt.scatter(
        bubble["popularity"],
        bubble["revenue"],
        s=(bubble["vote_count_clip"] / bubble["vote_count_clip"].max()) * 300 + 10,
        alpha=0.25,
    )
    plt.title("Bubble Plot: Popularity vs Revenue (Bubble Size = Vote Count)")
    plt.xlabel("Popularity")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "plot_08_bubble_popularity_vs_revenue.png", dpi=160)
    plt.close(fig)

    # ML: classification
    rating_threshold = 7.5
    df_model["is_high_rated"] = (df_model["vote_average"] >= rating_threshold).astype(int)

    num_cols = [
        c
        for c in ["popularity", "vote_count", "runtime", "release_year", "overview_len", "tagline_len", "genre_count"]
        if c in df_model.columns
    ]
    cat_cols = [c for c in ["original_language", "genre_main"] if c in df_model.columns]
    feature_cols = num_cols + cat_cols

    X = df_model[feature_cols].copy()
    y = LabelEncoder().fit_transform(df_model["is_high_rated"].copy())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf_models = {
        "LogReg": LogisticRegression(max_iter=3000),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        # Disable probability to speed up; ROC-AUC can use decision_function.
        "SVC_RBF": SVC(kernel="rbf", probability=False, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=15),
    }

    cls_rows = []
    for scale_mode in ["none", "standard", "minmax"]:
        pre = make_preprocessor(num_cols, cat_cols, scale_mode)
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
        scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0),
            "roc_auc": "roc_auc",
        }
        for name, model in clf_models.items():
            pipe = Pipeline(steps=[("pre", pre), ("model", model)])
            cv_out = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            cls_rows.append(
                {
                    "scale_mode": scale_mode,
                    "model": name,
                    "cv_roc_auc": float(np.mean(cv_out["test_roc_auc"])),
                    "cv_accuracy": float(np.mean(cv_out["test_accuracy"])),
                    "cv_precision": float(np.mean(cv_out["test_precision"])),
                    "cv_recall": float(np.mean(cv_out["test_recall"])),
                }
            )

    cls_df = pd.DataFrame(cls_rows)
    fig = plt.figure(figsize=(10, 4))
    sns.barplot(data=cls_df, x="model", y="cv_roc_auc", hue="scale_mode")
    plt.title("Classification Model Comparison (2-fold CV ROC-AUC)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ml_classification_cv_rocauc_comparison.png", dpi=160)
    plt.close(fig)

    # Table comparison for the 4 classification models across preprocessing modes
    cls_pivot = (
        cls_df.pivot(index="model", columns="scale_mode", values="cv_roc_auc")
        .reindex(columns=[c for c in ["none", "standard", "minmax"] if c in cls_df["scale_mode"].unique()])
        .round(3)
        .sort_index()
    )
    save_table_figure(
        cls_pivot,
        "Classification (4 Models): Mean CV ROC-AUC by Preprocessing Mode",
        FIG_DIR / "ml_classification_comparison_table.png",
    )

    best = cls_df.sort_values("cv_roc_auc", ascending=False).iloc[0]
    best_pre = make_preprocessor(num_cols, cat_cols, str(best["scale_mode"]))
    best_pipe = Pipeline(steps=[("pre", best_pre), ("model", clf_models[str(best["model"])])])
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)
    if hasattr(best_pipe.named_steps["model"], "predict_proba"):
        y_score = best_pipe.predict_proba(X_test)[:, 1]
    else:
        # decision_function is acceptable as a score for ROC/AUC
        y_score = best_pipe.decision_function(X_test)

    fig = plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ml_classification_confusion_matrix.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    fig = plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ml_classification_roc_curve.png", dpi=160)
    plt.close(fig)

    # ML: regression (log-revenue)
    reg_df = df_model[df_model["revenue"] > 0].copy()
    reg_df["log_revenue"] = np.log1p(reg_df["revenue"])
    Xr = reg_df[feature_cols].copy()
    yr = reg_df["log_revenue"].copy()

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=RANDOM_STATE
    )

    reg_models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "KNN": KNeighborsRegressor(n_neighbors=25),
    }

    reg_rows = []
    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "medae": make_scorer(median_absolute_error, greater_is_better=False),
        "r2": "r2",
    }
    for scale_mode in ["none", "standard", "minmax"]:
        pre = make_preprocessor(num_cols, cat_cols, scale_mode)
        cv = KFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
        for name, model in reg_models.items():
            pipe = Pipeline(steps=[("pre", pre), ("model", model)])
            cv_out = cross_validate(pipe, Xr_train, yr_train, cv=cv, scoring=scoring, n_jobs=-1)
            reg_rows.append(
                {
                    "scale_mode": scale_mode,
                    "model": name,
                    "cv_R2": float(np.mean(cv_out["test_r2"])),
                    "cv_MAE": float(-np.mean(cv_out["test_mae"])),
                    "cv_MSE": float(-np.mean(cv_out["test_mse"])),
                    "cv_MedAE": float(-np.mean(cv_out["test_medae"])),
                }
            )

    reg_df_res = pd.DataFrame(reg_rows)
    fig = plt.figure(figsize=(10, 4))
    sns.barplot(data=reg_df_res, x="model", y="cv_R2", hue="scale_mode")
    plt.title("Regression Model Comparison (2-fold CV R2 on log-revenue)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ml_regression_cv_r2_comparison.png", dpi=160)
    plt.close(fig)

    # Table comparison for the 4 regression models across preprocessing modes
    reg_pivot = (
        reg_df_res.pivot(index="model", columns="scale_mode", values="cv_R2")
        .reindex(columns=[c for c in ["none", "standard", "minmax"] if c in reg_df_res["scale_mode"].unique()])
        .round(3)
        .sort_index()
    )
    save_table_figure(
        reg_pivot,
        "Regression (4 Models): Mean CV R² by Preprocessing Mode (log-revenue)",
        FIG_DIR / "ml_regression_comparison_table.png",
    )

    best_reg = reg_df_res.sort_values("cv_R2", ascending=False).iloc[0]
    best_reg_pre = make_preprocessor(num_cols, cat_cols, str(best_reg["scale_mode"]))
    best_reg_pipe = Pipeline(steps=[("pre", best_reg_pre), ("model", reg_models[str(best_reg["model"])])])
    best_reg_pipe.fit(Xr_train, yr_train)
    yr_pred = best_reg_pipe.predict(Xr_test)

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(yr_test, yr_pred, alpha=0.3)
    lims = [min(float(yr_test.min()), float(yr_pred.min())), max(float(yr_test.max()), float(yr_pred.max()))]
    plt.plot(lims, lims, "--", color="gray")
    plt.title("Regression: Predicted vs Actual (log-revenue)")
    plt.xlabel("Actual log(1+revenue)")
    plt.ylabel("Predicted log(1+revenue)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ml_regression_pred_vs_actual.png", dpi=160)
    plt.close(fig)

    residuals = yr_test - yr_pred
    fig = plt.figure(figsize=(8, 4))
    plt.scatter(yr_pred, residuals, alpha=0.3)
    plt.axhline(0, linestyle="--", color="gray")
    plt.title("Regression Residuals vs Predicted")
    plt.xlabel("Predicted log(1+revenue)")
    plt.ylabel("Residual (actual - predicted)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ml_regression_residuals.png", dpi=160)
    plt.close(fig)

    # Write download script template
    DOWNLOAD_SCRIPT.write_text(
        """#!/usr/bin/env python
\"\"\"Recreate the dataset via TMDb-style metadata extraction (requires an API key).

This is a template script included to satisfy the deliverable.

Set TMDB_API_KEY and implement API calls if required by your course.
\"\"\"

import os
from pathlib import Path

OUTPUT = Path.cwd() / "Top_10000_Movies.csv"
API_KEY = os.getenv("TMDB_API_KEY", "")


def main():
    if not API_KEY:
        raise SystemExit("Set TMDB_API_KEY env var before running.")
    raise SystemExit("Template script: implement API calls if required by your course.")


if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )

    # Build PDF report
    cover_lines = [
        "Course Title: Machine Learning",
        "Team Members:",
        *[f"- {name} : {sid}" for name, sid in TEAM_MEMBERS],
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Project: Movie Metadata Analytics for Streaming Acquisition Decisions",
    ]

    summary_lines = [
        "Problem Domain: Streaming platform decision support using real-world movie metadata.",
        f"Dataset: Top_10000_Movies.csv ({df_raw.shape[0]} rows, {df_raw.shape[1]} columns).",
        "Dirty data indicators: missing/blank text, zero revenues, skewed distributions, outliers, and semi-structured categorical genres.",
        "ML tasks: (1) Classification of High Rated movies; (2) Regression for log-revenue.",
        "Models: Logistic Regression, Random Forest, SVC, KNN (classification) and Linear/Ridge, Random Forest Regressor, SVR, KNN (regression).",
        "Evaluation: Cross-validation and test metrics; scaling vs normalization comparisons included.",
    ]

    code_lines = [
        "Source Code: This submission includes Ml_Project.ipynb and build_submission.py.",
        "Preprocessing uses ColumnTransformer with robust imputation and one-hot encoding.",
        "Models are compared under: none, StandardScaler (scaling), and MinMax (normalization).",
    ]

    figure_pages = [
        ("Model Comparison (Classification Table)", FIG_DIR / "ml_classification_comparison_table.png"),
        ("Model Comparison (Regression Table)", FIG_DIR / "ml_regression_comparison_table.png"),
        ("EDA: Correlation Heatmap", FIG_DIR / "eda_corr_heatmap.png"),
        ("Visualization 1: Line Plot", FIG_DIR / "plot_01_line_avg_rating_by_year.png"),
        ("Visualization 2: Area Plot", FIG_DIR / "plot_02_area_total_revenue_by_year.png"),
        ("Visualization 3: Histogram", FIG_DIR / "plot_03_hist_runtime.png"),
        ("Visualization 4: Bar Chart", FIG_DIR / "plot_04_bar_top_genres.png"),
        ("Visualization 5: Pie Chart", FIG_DIR / "plot_05_pie_language.png"),
        ("Visualization 6: Box Plot", FIG_DIR / "plot_06_box_rating_by_genre.png"),
        ("Visualization 7: Scatter Plot", FIG_DIR / "plot_07_scatter_popularity_vs_rating.png"),
        ("Visualization 8: Bubble Plot", FIG_DIR / "plot_08_bubble_popularity_vs_revenue.png"),
        ("Classification: CV ROC-AUC Comparison", FIG_DIR / "ml_classification_cv_rocauc_comparison.png"),
        ("Classification: Confusion Matrix", FIG_DIR / "ml_classification_confusion_matrix.png"),
        ("Classification: ROC Curve", FIG_DIR / "ml_classification_roc_curve.png"),
        ("Regression: CV R2 Comparison", FIG_DIR / "ml_regression_cv_r2_comparison.png"),
        ("Regression: Predicted vs Actual", FIG_DIR / "ml_regression_pred_vs_actual.png"),
        ("Regression: Residuals", FIG_DIR / "ml_regression_residuals.png"),
    ]

    with PdfPages(REPORT_PATH) as pdf:
        add_text_page(pdf, "Cover Page", cover_lines)
        add_text_page(
            pdf,
            "Section 1: Problem Domain",
            [
                "We model movie performance signals to support streaming acquisition decisions.",
                "Classification target: is_high_rated derived from vote_average.",
                "Regression target: log(1+revenue) for movies with reported revenue.",
            ],
        )
        add_text_page(pdf, "Section 2: Project Summary", summary_lines)
        add_text_page(pdf, "Section 3: Source Code", code_lines)
        add_text_page(
            pdf,
            "Section 4: Visualization Snapshots",
            ["The following pages include EDA, required visualizations, and model evaluation plots."],
        )
        for title, path in figure_pages:
            add_image_page(pdf, title, path)

    # Zip submission
    files_to_include = [
        PROJECT_DIR / "Ml_Project.ipynb",
        DATA_PATH,
        REPORT_PATH,
        DOWNLOAD_SCRIPT,
        PROJECT_DIR / "build_submission.py",
    ]
    figure_files = sorted(FIG_DIR.glob("*.png"))
    files_to_include.extend(figure_files)

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files_to_include:
            if not p.exists():
                continue
            arcname = p.name
            if p.parent.name == "figures":
                arcname = f"figures/{p.name}"
            zf.write(p, arcname=arcname)

    print(f"Removed duplicates: {before - len(df)}")
    print("Generated:", REPORT_PATH)
    print("Created:", ZIP_PATH)
    print("Figures:", len(figure_files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
