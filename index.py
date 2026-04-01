import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


def build_lagged_features_for_province(
    df_province: pd.DataFrame,
    features: list[str],
    target_col: str,
    window_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    targets = []

    df_province = df_province.sort_values("date").reset_index(drop=True)

    for i in range(window_size, len(df_province)):
        row = {}

        for lag in range(1, window_size + 1):
            prev_row = df_province.iloc[i - lag]
            for feat in features:
                row[f"{feat}_lag_{lag}"] = prev_row[feat]

        rows.append(row)
        targets.append(df_province.iloc[i][target_col])

    if not rows:
        return pd.DataFrame(), np.array([])

    return pd.DataFrame(rows), np.array(targets, dtype=float)


def split_time_series(
    X: pd.DataFrame,
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    n = len(X)
    if n < 3:
        return None

    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))

    if val_end >= n:
        val_end = n - 1
    if train_end >= val_end:
        train_end = max(1, val_end - 1)

    if train_end <= 0 or val_end <= train_end or val_end >= n:
        return None

    return (
        X.iloc[:train_end].copy(),
        y[:train_end].copy(),
        X.iloc[train_end:val_end].copy(),
        y[train_end:val_end].copy(),
        X.iloc[val_end:].copy(),
        y[val_end:].copy(),
    )


def prepare_dataset(df: pd.DataFrame, window_size: int):
    df = df.copy()
    df = df[df["rain"] < 100]
    df = df.sort_values(["province", "date"]).reset_index(drop=True)

    wind_encoder = LabelEncoder()
    province_encoder = LabelEncoder()

    df["wind_d"] = wind_encoder.fit_transform(df["wind_d"].astype(str))
    df["province_le"] = province_encoder.fit_transform(df["province"].astype(str))

    base_features = ["province_le", "max", "min", "humidi", "cloud", "wind", "wind_d"]

    X_train_parts, y_train_parts = [], []
    X_val_parts, y_val_parts = [], []
    X_test_parts, y_test_parts = [], []

    for province in sorted(df["province"].unique()):
        df_p = df[df["province"] == province].copy()

        X_p, y_p = build_lagged_features_for_province(
            df_province=df_p,
            features=base_features,
            target_col="rain",
            window_size=window_size,
        )

        if X_p.empty or len(X_p) < 10:
            continue

        split_result = split_time_series(X_p, y_p)
        if split_result is None:
            continue

        X_tr, y_tr, X_val, y_val, X_te, y_te = split_result

        X_train_parts.append(X_tr)
        y_train_parts.append(y_tr)

        X_val_parts.append(X_val)
        y_val_parts.append(y_val)

        X_test_parts.append(X_te)
        y_test_parts.append(y_te)

    if not X_train_parts:
        raise ValueError("Không tạo được dataset train/val/test. Hãy kiểm tra dữ liệu đầu vào.")

    X_train = pd.concat(X_train_parts, axis=0, ignore_index=True)
    X_val = pd.concat(X_val_parts, axis=0, ignore_index=True)
    X_test = pd.concat(X_test_parts, axis=0, ignore_index=True)

    y_train = np.concatenate(y_train_parts)
    y_val = np.concatenate(y_val_parts)
    y_test = np.concatenate(y_test_parts)

    return X_train, y_train, X_val, y_val, X_test, y_test, wind_encoder, province_encoder


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "dataset": name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_true, y_pred)),
    }


def plot_predictions(
    y_train,
    y_val,
    y_test,
    train_pred,
    val_pred,
    test_pred,
    output_dir: Path,
):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=200)

    axes[0].plot(y_train, label="Actual")
    axes[0].plot(train_pred, label="Predicted")
    axes[0].set_title("Train Results")
    axes[0].legend()

    axes[1].plot(y_val, label="Actual")
    axes[1].plot(val_pred, label="Predicted")
    axes[1].set_title("Validation Results")
    axes[1].legend()

    axes[2].plot(y_test, label="Actual")
    axes[2].plot(test_pred, label="Predicted")
    axes[2].set_title("Test Results")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "prediction_plot.png")
    plt.close(fig)


def plot_feature_importance(model: XGBRegressor, feature_names: list[str], output_dir: Path):
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False).head(20)

    plt.figure(figsize=(12, 8), dpi=200)
    plt.barh(importance.index[::-1], importance.values[::-1])
    plt.title("Top 20 Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model for weather rain prediction")
    parser.add_argument("--data", type=str, default="dataset/init/weather.csv", help="Path to weather.csv")
    parser.add_argument("--window", type=int, default=15, help="Number of lag days")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--n-estimators", type=int, default=500, help="Number of boosting trees")
    parser.add_argument("--max-depth", type=int, default=6, help="Max depth")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample ratio")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="Column sample ratio")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)

    plt.figure(figsize=(8, 4))
    plt.hist(df["rain"], density=True, bins=20)
    plt.title("Rain Histogram")
    plt.tight_layout()
    plt.savefig(output_dir / "rain_histogram.png", dpi=200)
    plt.close()

    X_train, y_train, X_val, y_val, X_test, y_test, wind_encoder, province_encoder = prepare_dataset(
        df=df,
        window_size=args.window,
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    metrics = {
        "train": evaluate_regression(y_train, train_pred, "train"),
        "validation": evaluate_regression(y_val, val_pred, "validation"),
        "test": evaluate_regression(y_test, test_pred, "test"),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_predictions(y_train, y_val, y_test, train_pred, val_pred, test_pred, output_dir)
    plot_feature_importance(model, X_train.columns.tolist(), output_dir)

    joblib.dump(model, output_dir / "xgb_rain_model.joblib")
    joblib.dump(wind_encoder, output_dir / "wind_labelencoder.pkl")
    joblib.dump(province_encoder, output_dir / "province_labelencoder.pkl")
    joblib.dump(X_train.columns.tolist(), output_dir / "feature_columns.pkl")

    print("Training complete. Outputs saved to:", output_dir.resolve())
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()