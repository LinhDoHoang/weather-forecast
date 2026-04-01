import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBRegressor


def preprocess_from_scratch(
    input_path: Path,
    repreprocess_dir: Path,
    target_col: str,
    test_year: int,
):
    df = pd.read_csv(input_path)

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if "date" not in df.columns:
        raise ValueError("Missing required column: date")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if target_col == "rain":
        df = df[df[target_col] < 100]

    sort_cols = ["date"]
    if "province" in df.columns:
        sort_cols = ["province", "date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    encoders = {}
    for cat_col in ["wind_d", "region"]:
        if cat_col in df.columns and df[cat_col].dtype == "object":
            le = LabelEncoder()
            df[cat_col] = le.fit_transform(df[cat_col].astype(str))
            encoders[cat_col] = le

    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    train_df = df[df["year"] < test_year].copy()
    test_df = df[df["year"] == test_year].copy()

    if train_df.empty or test_df.empty:
        split_index = int(len(df) * 0.8)
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()

    non_feature_cols = {"date"}
    feature_cols = [
        c
        for c in train_df.columns
        if c not in non_feature_cols
        and c != target_col
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    scaled_cols = []
    for col in feature_cols + [target_col]:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            scaled_cols.append(col)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[scaled_cols])
    test_scaled = scaler.transform(test_df[scaled_cols])

    for idx, col in enumerate(scaled_cols):
        train_df[col] = train_scaled[:, idx].astype(float)
        test_df[col] = test_scaled[:, idx].astype(float)

    repreprocess_dir.mkdir(parents=True, exist_ok=True)
    train_out = repreprocess_dir / "weather_train_repreprocess_scaled.csv"
    test_out = repreprocess_dir / "weather_test_repreprocess_scaled.csv"
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    return train_df, test_df, feature_cols, scaler, encoders, train_out, test_out


def split_train_val(train_df: pd.DataFrame, feature_cols: list[str], target_col: str, val_ratio: float):
    split_index = int(len(train_df) * (1 - val_ratio))
    split_index = max(1, min(split_index, len(train_df) - 1))

    x_train = train_df.iloc[:split_index][feature_cols].copy()
    y_train = train_df.iloc[:split_index][target_col].copy()
    x_val = train_df.iloc[split_index:][feature_cols].copy()
    y_val = train_df.iloc[split_index:][target_col].copy()
    return x_train, x_val, y_train, y_val


def evaluate(y_true: pd.Series, y_pred: np.ndarray, name: str):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "dataset": name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_plots(y_true: pd.Series, y_pred: np.ndarray, feature_names: list[str], model: XGBRegressor, output_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true.values, label="actual", linewidth=1.0)
    ax.plot(y_pred, label="predicted", linewidth=1.0)
    ax.set_title("Test: Actual vs Predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted_test.png", dpi=200)
    plt.close(fig)

    importance = model.feature_importances_
    order = np.argsort(importance)[::-1][:20]
    top_features = [feature_names[i] for i in order]
    top_values = importance[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features[::-1], top_values[::-1])
    ax.set_title("Top 20 Feature Importances")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model with full re-preprocessing")
    parser.add_argument("--data", type=str, default="dataset/init/weather.csv", help="Path to raw weather CSV")
    parser.add_argument(
        "--repreprocess-dir",
        type=str,
        default="dataset/repreprocess",
        help="Directory to log re-preprocessed datasets",
    )
    parser.add_argument("--target", type=str, default="rain", help="Target column")
    parser.add_argument("--test-year", type=int, default=2021, help="Year used as test set")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio from train set")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="output/repreprocess_xgboost")
    args = parser.parse_args()

    input_path = Path(args.data)
    repreprocess_dir = Path(args.repreprocess_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        train_df,
        test_df,
        feature_cols,
        scaler,
        encoders,
        train_out,
        test_out,
    ) = preprocess_from_scratch(
        input_path=input_path,
        repreprocess_dir=repreprocess_dir,
        target_col=args.target,
        test_year=args.test_year,
    )

    x_train, x_val, y_train, y_val = split_train_val(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=args.target,
        val_ratio=args.val_ratio,
    )
    x_test = test_df[feature_cols].copy()
    y_test = test_df[args.target].copy()

    model = XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    val_metrics = evaluate(y_val, val_pred, "validation")
    test_metrics = evaluate(y_test, test_pred, "test")

    metrics = {"validation": val_metrics, "test": test_metrics}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([val_metrics, test_metrics]).to_csv(output_dir / "metrics.csv", index=False)

    pd.DataFrame(
        {
            "actual": y_val.values,
            "predicted": val_pred,
            "residual": y_val.values - val_pred,
        }
    ).to_csv(output_dir / "validation_predictions.csv", index=False)

    pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": test_pred,
            "residual": y_test.values - test_pred,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    save_plots(y_test, test_pred, feature_cols, model, output_dir)

    artifacts = {
        "target": args.target,
        "feature_columns": feature_cols,
        "repreprocessed_train_path": str(train_out),
        "repreprocessed_test_path": str(test_out),
        "test_year": args.test_year,
    }

    joblib.dump(model, output_dir / "xgb_weather_model.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    joblib.dump(encoders, output_dir / "label_encoders.joblib")
    joblib.dump(artifacts, output_dir / "preprocessing_artifacts.joblib")

    print("=== XGBOOST TRAINING COMPLETE ===")
    print(f"Re-preprocessed train: {train_out}")
    print(f"Re-preprocessed test: {test_out}")
    print(f"Model output directory: {output_dir.resolve()}")
    print("Validation RMSE:", f"{val_metrics['rmse']:.6f}")
    print("Test RMSE:", f"{test_metrics['rmse']:.6f}")


if __name__ == "__main__":
    main()
