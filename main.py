from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for weather rain-forecast training and inference pipeline."""

    train_data_path: Path = Path("dataset/test_train_scaled/weather_train_2009_2020_scaled.csv")
    test_data_path: Path = Path("dataset/test_train_scaled/weather_test_2021_scaled.csv")
    target_column: str = "rain"
    random_seed: int = 42
    validation_size: float = 0.2
    output_base_dir: Path = Path("output")
    model_filename: str = "xgb_weather_model.joblib"
    artifacts_filename: str = "preprocessing_artifacts.joblib"
    max_invalid_date_ratio: float = 0.05
    lag_steps: Tuple[int, ...] = (1, 2, 3, 7)
    rolling_windows: Tuple[int, ...] = (3, 7)
    model_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "early_stopping_rounds": 30,
            "n_jobs": -1,
            "random_state": 42,
        }
    )


@dataclass
class PreprocessArtifacts:
    """Artifacts required to apply train-time preprocessing in inference."""

    target_column: str
    datetime_columns: List[str]
    feature_columns: List[str]
    numeric_fill_values: Dict[str, float]
    group_column: Optional[str]
    sort_columns: List[str]
    lag_source_columns: List[str]
    lag_steps: List[int]
    rolling_source_columns: List[str]
    rolling_windows: List[int]


def create_output_dir(base_dir: Path) -> Path:
    """Create timestamped output directory for current runtime."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_sort_and_group_columns(df: pd.DataFrame) -> Tuple[List[str], Optional[str]]:
    if "date" not in df.columns:
        raise ValueError("Missing required 'date' column for time-series processing.")

    if "province_encoded" in df.columns:
        return ["province_encoded", "date"], "province_encoded"
    return ["date"], None


def _parse_and_validate_date(
    df: pd.DataFrame,
    name: str,
    max_invalid_date_ratio: float,
) -> pd.DataFrame:
    parsed_df = df.copy()
    if "date" not in parsed_df.columns:
        raise ValueError(f"{name} data missing required 'date' column.")

    parsed_df["date"] = pd.to_datetime(parsed_df["date"], errors="coerce")
    invalid_dates = int(parsed_df["date"].isna().sum())
    invalid_ratio = invalid_dates / max(len(parsed_df), 1)

    if invalid_ratio > max_invalid_date_ratio:
        raise ValueError(
            f"{name} data has too many invalid 'date' values: "
            f"{invalid_dates}/{len(parsed_df)} ({invalid_ratio:.2%})."
        )

    if invalid_dates > 0:
        LOGGER.warning("Dropping %d rows with invalid date in %s data.", invalid_dates, name)
        parsed_df = parsed_df.dropna(subset=["date"]).copy()

    if parsed_df.empty:
        raise ValueError(f"{name} data became empty after dropping invalid dates.")

    return parsed_df


def load_data(config: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSV files, parse date, and sort by time (and province if available)."""
    try:
        train_df = pd.read_csv(config.train_data_path)
        test_df = pd.read_csv(config.test_data_path)

        for name, df in [("train", train_df), ("test", test_df)]:
            if config.target_column not in df.columns:
                raise ValueError(f"{name} data missing target column '{config.target_column}'.")

        train_df = _parse_and_validate_date(train_df, "train", config.max_invalid_date_ratio)
        test_df = _parse_and_validate_date(test_df, "test", config.max_invalid_date_ratio)

        train_sort_cols, _ = _resolve_sort_and_group_columns(train_df)
        test_sort_cols, _ = _resolve_sort_and_group_columns(test_df)

        train_df = train_df.sort_values(train_sort_cols).reset_index(drop=True)
        test_df = test_df.sort_values(test_sort_cols).reset_index(drop=True)

        LOGGER.info("Loaded train shape: %s", train_df.shape)
        LOGGER.info("Loaded test shape: %s", test_df.shape)
        LOGGER.info("Train sort columns: %s", train_sort_cols)
        LOGGER.info("Test sort columns: %s", test_sort_cols)
        return train_df, test_df
    except Exception as exc:
        raise RuntimeError(f"Failed to load data: {exc}") from exc


def _detect_datetime_columns(df: pd.DataFrame, target_column: str) -> List[str]:
    datetime_columns: List[str] = []
    for col in df.columns:
        if col == target_column:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_columns.append(col)
            continue
        if "date" in col.lower():
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                datetime_columns.append(col)
    return datetime_columns


def _apply_datetime_transform(df: pd.DataFrame, datetime_columns: List[str]) -> pd.DataFrame:
    transformed = df.copy()
    for col in datetime_columns:
        if col not in transformed.columns:
            continue
        parsed = pd.to_datetime(transformed[col], errors="coerce")
        parsed_int64 = parsed.astype("int64", copy=False)
        ordinal = (parsed_int64 // (24 * 60 * 60 * 1_000_000_000)).astype("float64")
        ordinal[parsed.isna()] = np.nan
        transformed[f"{col}_ordinal"] = ordinal
        transformed.drop(columns=[col], inplace=True)
    return transformed


def create_lag_features(
    df: pd.DataFrame,
    target_column: str,
    group_column: Optional[str],
    lag_steps: List[int],
    rolling_windows: List[int],
    lag_source_columns: List[str],
    rolling_source_columns: List[str],
    for_training: bool,
) -> pd.DataFrame:
    """Create leakage-safe lag and rolling features; shift target to t+1 for training."""
    transformed = df.copy()

    if "date" in transformed.columns:
        sort_columns = ["date"]
        if group_column and group_column in transformed.columns:
            sort_columns = [group_column, "date"]
        transformed = transformed.sort_values(sort_columns).copy()

    lag_created_cols: List[str] = []
    rolling_created_cols: List[str] = []

    lag_sources = [col for col in lag_source_columns if col in transformed.columns]
    rolling_sources = [col for col in rolling_source_columns if col in transformed.columns]

    if group_column and group_column in transformed.columns:
        grouped = transformed.groupby(group_column, sort=False)

        for col in lag_sources:
            for lag in lag_steps:
                lag_col = f"{col}_lag_{lag}"
                transformed[lag_col] = grouped[col].shift(lag)
                lag_created_cols.append(lag_col)

        for col in rolling_sources:
            shifted = grouped[col].shift(1)
            for window in rolling_windows:
                roll_col = f"{col}_roll_mean_{window}"
                transformed[roll_col] = (
                    shifted.groupby(transformed[group_column])
                    .rolling(window=window, min_periods=window)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                rolling_created_cols.append(roll_col)

        if for_training:
            transformed[target_column] = grouped[target_column].shift(-1)
    else:
        for col in lag_sources:
            for lag in lag_steps:
                lag_col = f"{col}_lag_{lag}"
                transformed[lag_col] = transformed[col].shift(lag)
                lag_created_cols.append(lag_col)

        for col in rolling_sources:
            shifted = transformed[col].shift(1)
            for window in rolling_windows:
                roll_col = f"{col}_roll_mean_{window}"
                transformed[roll_col] = shifted.rolling(window=window, min_periods=window).mean()
                rolling_created_cols.append(roll_col)

        if for_training:
            transformed[target_column] = transformed[target_column].shift(-1)

    if for_training:
        # Drop rows that cannot have valid t+1 target or complete lag/rolling history.
        required_cols = [target_column] + lag_created_cols + rolling_created_cols
        required_cols = [col for col in required_cols if col in transformed.columns]
        transformed = transformed.dropna(subset=required_cols).copy()

    transformed = transformed.reset_index(drop=True)
    return transformed


def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, PreprocessArtifacts]:
    """Preprocess train/test data into leakage-safe model-ready matrices."""
    try:
        if config.target_column not in train_df.columns or config.target_column not in test_df.columns:
            raise ValueError(f"Target column '{config.target_column}' not found in both train/test.")

        _, group_column = _resolve_sort_and_group_columns(train_df)
        sort_columns = ["date"] if group_column is None else [group_column, "date"]

        lag_source_candidates = [
            "rain",
            "max_temp",
            "min_temp",
            "wind",
            "humidi",
            "cloud",
            "pressure",
            "temp_diff",
        ]
        rolling_source_candidates = [
            "rain",
            "humidi",
            "cloud",
            "pressure",
            "max_temp",
            "min_temp",
        ]

        train_lagged = create_lag_features(
            df=train_df,
            target_column=config.target_column,
            group_column=group_column,
            lag_steps=list(config.lag_steps),
            rolling_windows=list(config.rolling_windows),
            lag_source_columns=lag_source_candidates,
            rolling_source_columns=rolling_source_candidates,
            for_training=True,
        )
        test_lagged = create_lag_features(
            df=test_df,
            target_column=config.target_column,
            group_column=group_column,
            lag_steps=list(config.lag_steps),
            rolling_windows=list(config.rolling_windows),
            lag_source_columns=lag_source_candidates,
            rolling_source_columns=rolling_source_candidates,
            for_training=True,
        )

        datetime_columns = _detect_datetime_columns(train_lagged, config.target_column)
        LOGGER.info("Detected datetime columns: %s", datetime_columns if datetime_columns else "None")

        train_proc = _apply_datetime_transform(train_lagged, datetime_columns)
        test_proc = _apply_datetime_transform(test_lagged, datetime_columns)

        y_train_full = pd.to_numeric(train_proc[config.target_column], errors="coerce")
        y_test = pd.to_numeric(test_proc[config.target_column], errors="coerce")

        x_train_full = train_proc.drop(columns=[config.target_column])
        x_test = test_proc.drop(columns=[config.target_column])

        x_train_full = pd.get_dummies(x_train_full, drop_first=False)
        x_test = pd.get_dummies(x_test, drop_first=False)
        x_test = x_test.reindex(columns=x_train_full.columns, fill_value=0)

        x_train_full = x_train_full.replace([np.inf, -np.inf], np.nan)
        x_test = x_test.replace([np.inf, -np.inf], np.nan)

        artifacts = PreprocessArtifacts(
            target_column=config.target_column,
            datetime_columns=datetime_columns,
            feature_columns=list(x_train_full.columns),
            numeric_fill_values={},
            group_column=group_column,
            sort_columns=sort_columns,
            lag_source_columns=lag_source_candidates,
            lag_steps=list(config.lag_steps),
            rolling_source_columns=rolling_source_candidates,
            rolling_windows=list(config.rolling_windows),
        )

        LOGGER.info(
            "Preprocessed features complete. Train shape: %s, Test shape: %s, Feature count: %d",
            x_train_full.shape,
            x_test.shape,
            len(artifacts.feature_columns),
        )

        return x_train_full, y_train_full, x_test, y_test, artifacts
    except Exception as exc:
        raise RuntimeError(f"Failed to preprocess data: {exc}") from exc


def split_data(
    x_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split train set into train/validation using chronological order."""
    try:
        n_rows = len(x_train_full)
        val_size = max(1, int(n_rows * config.validation_size))
        train_size = n_rows - val_size
        if train_size <= 0:
            raise ValueError("Validation size is too large for available training rows.")

        x_train = x_train_full.iloc[:train_size].copy()
        y_train = y_train_full.iloc[:train_size].copy()
        x_val = x_train_full.iloc[train_size:].copy()
        y_val = y_train_full.iloc[train_size:].copy()

        LOGGER.info("Split train shape: %s, val shape: %s", x_train.shape, x_val.shape)
        return x_train, x_val, y_train, y_val
    except Exception as exc:
        raise RuntimeError(f"Failed to split data: {exc}") from exc


def build_model(config: PipelineConfig) -> XGBRegressor:
    """Construct an XGBRegressor with configured hyperparameters."""
    try:
        LOGGER.info("Building XGBRegressor with params: %s", config.model_params)
        return XGBRegressor(**config.model_params)
    except Exception as exc:
        raise RuntimeError(f"Failed to build model: {exc}") from exc


def train_model(
    model: XGBRegressor,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> XGBRegressor:
    """Fit model and monitor validation set during training."""
    try:
        LOGGER.info("Training started. Train rows: %d, Validation rows: %d", len(x_train), len(x_val))
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        LOGGER.info("Training finished.")
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to train model: {exc}") from exc


def predict(model: XGBRegressor, x_data: pd.DataFrame) -> np.ndarray:
    """Generate predictions from fitted model."""
    try:
        return model.predict(x_data)
    except Exception as exc:
        raise RuntimeError(f"Failed to generate predictions: {exc}") from exc


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, dataset_name: str) -> Dict[str, float]:
    """Compute regression metrics for a prediction set."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {
        "dataset": dataset_name,
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": rmse,
        "R2": float(r2),
    }


def evaluate_rainy_days_only(y_true: pd.Series, y_pred: np.ndarray, name: str) -> Dict[str, Any]:
    """Evaluate only rows with rain > 0 to diagnose rainy-day performance."""
    y_true_series = pd.Series(y_true)
    rainy_mask = y_true_series > 0
    rainy_count = int(rainy_mask.sum())

    if rainy_count == 0:
        return {
            "dataset": f"{name}_rainy_days_only",
            "samples": 0,
            "MAE": float("nan"),
            "MSE": float("nan"),
            "RMSE": float("nan"),
            "R2": float("nan"),
        }

    rainy_metrics = evaluate_model(
        y_true_series[rainy_mask],
        np.asarray(y_pred)[rainy_mask.to_numpy()],
        f"{name}_rainy_days_only",
    )
    rainy_metrics["samples"] = rainy_count
    return rainy_metrics


def debug_baseline(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evaluate two simple baselines: all-zero and train-mean prediction."""
    train_mean = float(pd.Series(y_train).mean()) if len(y_train) > 0 else 0.0

    zero_val_pred = np.zeros(len(y_val), dtype=float)
    zero_test_pred = np.zeros(len(y_test), dtype=float)

    mean_val_pred = np.full(len(y_val), train_mean, dtype=float)
    mean_test_pred = np.full(len(y_test), train_mean, dtype=float)

    results = {
        "all_zero": {
            "validation": evaluate_model(pd.Series(y_val), zero_val_pred, "baseline_zero_validation"),
            "test": evaluate_model(pd.Series(y_test), zero_test_pred, "baseline_zero_test"),
        },
        "train_mean": {
            "validation": evaluate_model(pd.Series(y_val), mean_val_pred, "baseline_mean_validation"),
            "test": evaluate_model(pd.Series(y_test), mean_test_pred, "baseline_mean_test"),
        },
    }

    LOGGER.info("Baseline all-zero validation RMSE: %.6f", results["all_zero"]["validation"]["RMSE"])
    LOGGER.info("Baseline all-zero test RMSE: %.6f", results["all_zero"]["test"]["RMSE"])
    LOGGER.info("Baseline train-mean validation RMSE: %.6f", results["train_mean"]["validation"]["RMSE"])
    LOGGER.info("Baseline train-mean test RMSE: %.6f", results["train_mean"]["test"]["RMSE"])

    return results


def _distribution_summary(y: pd.Series) -> Dict[str, Any]:
    y_series = pd.Series(y)
    total = max(len(y_series), 1)
    describe = y_series.describe()
    return {
        "count": int(describe.get("count", 0)),
        "mean": float(describe.get("mean", np.nan)),
        "std": float(describe.get("std", np.nan)),
        "min": float(describe.get("min", np.nan)),
        "25%": float(describe.get("25%", np.nan)),
        "50%": float(describe.get("50%", np.nan)),
        "75%": float(describe.get("75%", np.nan)),
        "max": float(describe.get("max", np.nan)),
        "zero_ratio": float((y_series == 0).sum() / total),
        "non_zero_ratio": float((y_series > 0).sum() / total),
    }


def debug_target_distribution(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
    """Summarize target distribution for train/val/test splits."""
    distribution = {
        "train": _distribution_summary(pd.Series(y_train)),
        "validation": _distribution_summary(pd.Series(y_val)),
        "test": _distribution_summary(pd.Series(y_test)),
    }

    for split_name, stats in distribution.items():
        LOGGER.info(
            "Target distribution [%s] | count=%d mean=%.6f zero_ratio=%.4f non_zero_ratio=%.4f",
            split_name,
            stats["count"],
            stats["mean"],
            stats["zero_ratio"],
            stats["non_zero_ratio"],
        )

    return distribution


def save_model(model: XGBRegressor, model_path: Path) -> None:
    """Persist trained model to disk."""
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to save model: {exc}") from exc


def load_model(model_path: Path) -> XGBRegressor:
    """Load trained model from disk."""
    try:
        model = joblib.load(model_path)
        if not isinstance(model, XGBRegressor):
            raise TypeError("Loaded object is not an XGBRegressor instance.")
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc


def _save_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def _plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(9, 7))
    sample_idx = np.arange(len(y_true))
    if len(sample_idx) > 15_000:
        sample_idx = np.random.default_rng(42).choice(sample_idx, size=15_000, replace=False)

    y_true_sample = np.asarray(y_true)[sample_idx]
    y_pred_sample = np.asarray(y_pred)[sample_idx]

    plt.scatter(y_true_sample, y_pred_sample, alpha=0.35, s=16, label="Predicted")
    min_val = float(min(y_true_sample.min(), y_pred_sample.min()))
    max_val = float(max(y_true_sample.max(), y_pred_sample.max()))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linewidth=2, label="Ideal")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_residual_distribution(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    plt.figure(figsize=(9, 6))
    plt.hist(residuals, bins=50, alpha=0.85, edgecolor="black")
    plt.title("Residual Distribution")
    plt.xlabel("Residual (actual - predicted)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_feature_importance(
    model: XGBRegressor,
    feature_names: List[str],
    output_path: Path,
    top_n: int = 20,
) -> None:
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    top_df = feature_importance_df.head(top_n)

    plt.figure(figsize=(10, 7))
    plt.barh(top_df["feature"][::-1], top_df["importance"][::-1])
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_excel_results(
    output_path: Path,
    val_predictions_df: pd.DataFrame,
    test_predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        val_predictions_df.to_excel(writer, sheet_name="validation_predictions", index=False)
        test_predictions_df.to_excel(writer, sheet_name="test_predictions", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)


def run_training_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """Run end-to-end training, evaluation, export and artifact persistence."""
    LOGGER.info("Training pipeline started.")
    output_dir = create_output_dir(config.output_base_dir)
    model_dir = output_dir / "model"
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"

    model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Created output folders at: %s", output_dir)

    train_df, test_df = load_data(config)
    x_train_full, y_train_full, x_test, y_test, artifacts = preprocess_data(train_df, test_df, config)
    x_train, x_val, y_train, y_val = split_data(x_train_full, y_train_full, config)

    # Fit imputation values only on train split to prevent leakage.
    numeric_fill_values = x_train.median(numeric_only=True).to_dict()

    x_train = x_train.fillna(value=numeric_fill_values).fillna(0.0)
    x_val = x_val.fillna(value=numeric_fill_values).fillna(0.0)
    x_test = x_test.fillna(value=numeric_fill_values).fillna(0.0)

    train_target_fill = float(y_train.median()) if y_train.notna().any() else 0.0
    y_train = y_train.fillna(train_target_fill)
    y_val = y_val.fillna(train_target_fill)
    y_test = y_test.fillna(train_target_fill)

    artifacts.numeric_fill_values = {k: float(v) for k, v in numeric_fill_values.items()}
    artifacts.feature_columns = list(x_train.columns)

    target_distribution = debug_target_distribution(y_train, y_val, y_test)
    baseline_results = debug_baseline(y_train, y_val, y_test)

    model = build_model(config)
    model = train_model(model, x_train, y_train, x_val, y_val)

    val_pred = predict(model, x_val)
    test_pred = predict(model, x_test)

    val_metrics = evaluate_model(y_val, val_pred, "validation")
    test_metrics = evaluate_model(y_test, test_pred, "test")
    rainy_val_metrics = evaluate_rainy_days_only(y_val, val_pred, "validation")
    rainy_test_metrics = evaluate_rainy_days_only(y_test, test_pred, "test")

    LOGGER.info("Validation metrics: %s", val_metrics)
    LOGGER.info("Test metrics: %s", test_metrics)
    LOGGER.info("Validation rainy-only metrics: %s", rainy_val_metrics)
    LOGGER.info("Test rainy-only metrics: %s", rainy_test_metrics)

    metrics_df = pd.DataFrame([val_metrics, test_metrics, rainy_val_metrics, rainy_test_metrics])

    val_predictions_df = pd.DataFrame(
        {
            "actual": y_val.values,
            "predicted": val_pred,
            "residual": y_val.values - val_pred,
            "is_rainy_day": (y_val.values > 0).astype(int),
        }
    )
    test_predictions_df = pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": test_pred,
            "residual": y_test.values - test_pred,
            "is_rainy_day": (y_test.values > 0).astype(int),
        }
    )

    model_path = model_dir / config.model_filename
    artifacts_path = model_dir / config.artifacts_filename

    LOGGER.info("Saving model and preprocessing artifacts...")
    save_model(model, model_path)
    joblib.dump(artifacts, artifacts_path)

    LOGGER.info("Saving plots...")
    _plot_actual_vs_predicted(y_test, test_pred, plots_dir / "actual_vs_predicted_test.png")
    _plot_residual_distribution(y_test, test_pred, plots_dir / "residual_distribution_test.png")
    _plot_feature_importance(model, artifacts.feature_columns, plots_dir / "feature_importance.png")

    LOGGER.info("Saving CSV reports...")
    val_predictions_df.to_csv(reports_dir / "validation_predictions.csv", index=False)
    test_predictions_df.to_csv(reports_dir / "test_predictions.csv", index=False)
    metrics_df.to_csv(reports_dir / "metrics.csv", index=False)

    LOGGER.info("Saving Excel report...")
    _save_excel_results(
        reports_dir / "results.xlsx",
        val_predictions_df=val_predictions_df,
        test_predictions_df=test_predictions_df,
        metrics_df=metrics_df,
    )

    config_dict = asdict(config)
    config_dict["train_data_path"] = str(config.train_data_path)
    config_dict["test_data_path"] = str(config.test_data_path)
    config_dict["output_base_dir"] = str(config.output_base_dir)
    _save_json(config_dict, output_dir / "config_used.json")
    _save_json(
        {
            "validation": val_metrics,
            "test": test_metrics,
            "validation_rainy_days_only": rainy_val_metrics,
            "test_rainy_days_only": rainy_test_metrics,
            "baseline": baseline_results,
            "target_distribution": target_distribution,
        },
        reports_dir / "metrics.json",
    )

    LOGGER.info("Training pipeline completed. Outputs saved in: %s", output_dir)

    return {
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "artifacts_path": str(artifacts_path),
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
            "validation_rainy_days_only": rainy_val_metrics,
            "test_rainy_days_only": rainy_test_metrics,
        },
        "baseline": baseline_results,
        "target_distribution": target_distribution,
    }


def _preprocess_new_samples(
    new_df: pd.DataFrame,
    artifacts: PreprocessArtifacts,
) -> pd.DataFrame:
    transformed = new_df.copy()

    if "date" in transformed.columns:
        transformed["date"] = pd.to_datetime(transformed["date"], errors="coerce")

    available_sort_cols = [col for col in artifacts.sort_columns if col in transformed.columns]
    if available_sort_cols:
        transformed = transformed.sort_values(available_sort_cols).reset_index(drop=True)

    transformed = create_lag_features(
        df=transformed,
        target_column=artifacts.target_column,
        group_column=artifacts.group_column,
        lag_steps=artifacts.lag_steps,
        rolling_windows=artifacts.rolling_windows,
        lag_source_columns=artifacts.lag_source_columns,
        rolling_source_columns=artifacts.rolling_source_columns,
        for_training=False,
    )

    transformed = _apply_datetime_transform(transformed, artifacts.datetime_columns)

    if artifacts.target_column in transformed.columns:
        transformed = transformed.drop(columns=[artifacts.target_column])

    x_new = pd.get_dummies(transformed, drop_first=False)
    x_new = x_new.reindex(columns=artifacts.feature_columns, fill_value=0)
    x_new = x_new.replace([np.inf, -np.inf], np.nan)
    x_new = x_new.fillna(value=artifacts.numeric_fill_values).fillna(0.0)
    return x_new


def run_inference(
    new_data_path: Path,
    model_path: Path,
    artifacts_path: Path,
    output_base_dir: Path = Path("output"),
) -> Path:
    """Run inference on new samples and save prediction report."""
    try:
        LOGGER.info("Inference started.")
        model = load_model(model_path)
        artifacts = joblib.load(artifacts_path)
        if not isinstance(artifacts, PreprocessArtifacts):
            raise TypeError("Invalid preprocessing artifacts file.")

        LOGGER.info("Loaded model and artifacts. Reading new data from: %s", new_data_path)
        new_df = pd.read_csv(new_data_path)
        x_new = _preprocess_new_samples(new_df, artifacts)
        y_pred = predict(model, x_new)

        output_dir = create_output_dir(output_base_dir) / "inference"
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Inference output directory created: %s", output_dir)

        result_df = new_df.copy()
        result_df["prediction"] = y_pred

        csv_path = output_dir / "inference_predictions.csv"
        xlsx_path = output_dir / "inference_predictions.xlsx"

        result_df.to_csv(csv_path, index=False)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            result_df.to_excel(writer, sheet_name="predictions", index=False)

        LOGGER.info("Inference predictions exported to CSV and Excel.")
        LOGGER.info("Inference finished. Results saved to: %s", output_dir)
        return output_dir
    except Exception as exc:
        raise RuntimeError(f"Inference failed: {exc}") from exc


def predict_new_samples(
    new_data_path: Path,
    model_path: Path,
    artifacts_path: Path,
    output_base_dir: Path = Path("output"),
) -> Path:
    """Alias for run_inference to satisfy different integration styles."""
    return run_inference(new_data_path, model_path, artifacts_path, output_base_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weather rain forecasting pipeline with XGBoost")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--train-path", type=str, default="dataset/test_train_scaled/weather_train_2009_2020_scaled.csv")
    parser.add_argument("--test-path", type=str, default="dataset/test_train_scaled/weather_test_2021_scaled.csv")
    parser.add_argument("--target", type=str, default="rain")
    parser.add_argument("--output-dir", type=str, default="output")

    parser.add_argument("--new-data-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--artifacts-path", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.mode == "train":
        config = PipelineConfig(
            train_data_path=Path(args.train_path),
            test_data_path=Path(args.test_path),
            target_column=args.target,
            output_base_dir=Path(args.output_dir),
        )
        result = run_training_pipeline(config)

        print("=== TRAINING COMPLETED ===")
        print(f"Output directory: {result['output_dir']}")
        print(f"Model path: {result['model_path']}")
        print(f"Artifacts path: {result['artifacts_path']}")
        print("Validation metrics:")
        for key, value in result["metrics"]["validation"].items():
            if key == "dataset":
                continue
            print(f"  {key}: {value:.6f}")
        print("Test metrics:")
        for key, value in result["metrics"]["test"].items():
            if key == "dataset":
                continue
            print(f"  {key}: {value:.6f}")

    if args.mode == "infer":
        if not args.new_data_path:
            raise ValueError("--new-data-path is required for infer mode.")
        if not args.model_path:
            raise ValueError("--model-path is required for infer mode.")
        if not args.artifacts_path:
            raise ValueError("--artifacts-path is required for infer mode.")

        output_dir = run_inference(
            new_data_path=Path(args.new_data_path),
            model_path=Path(args.model_path),
            artifacts_path=Path(args.artifacts_path),
            output_base_dir=Path(args.output_dir),
        )
        print("=== INFERENCE COMPLETED ===")
        print(f"Inference output directory: {output_dir}")


if __name__ == "__main__":
    main()
