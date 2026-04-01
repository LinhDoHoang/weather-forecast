from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pipeline_v3 as p3


EXTRA_FEATURES = [
    "rain_lag_2d",
    "rain_lag_3d",
    "rain_lag_5d",
    "rain_roll_mean_3d",
    "rain_roll_max_3d",
    "rain_roll_std_3d",
    "rain_ewm_7d",
    "rain_ewm_14d",
    "heavy_rain_yesterday",
    "dry_to_wet",
    "wet_to_dry",
    "wet_streak_days",
    "pressure_roll_mean_3d",
    "wind_roll_mean_3d",
]


def engineer_features_v4(df: pd.DataFrame, cfg: p3.PipelineConfig) -> pd.DataFrame:
    out = p3.engineer_features(df, cfg)

    out = out.sort_values(["province_encoded", "date"]).reset_index(drop=True)
    g_rain = out.groupby("province_encoded")["rain"]
    g_pressure = out.groupby("province_encoded")["pressure"]
    g_wind = out.groupby("province_encoded")["wind"]

    for lag in (2, 3, 5):
        out[f"rain_lag_{lag}d"] = g_rain.shift(lag).fillna(0)

    out["rain_roll_mean_3d"] = g_rain.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)
    out["rain_roll_max_3d"] = g_rain.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).max()
    ).fillna(0)
    out["rain_roll_std_3d"] = g_rain.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).std().fillna(0)
    ).fillna(0)

    out["rain_ewm_7d"] = g_rain.transform(
        lambda s: s.shift(1).ewm(span=7, adjust=False).mean()
    ).fillna(0)
    out["rain_ewm_14d"] = g_rain.transform(
        lambda s: s.shift(1).ewm(span=14, adjust=False).mean()
    ).fillna(0)

    prev_rain = g_rain.shift(1).fillna(0)
    is_wet_today = (out["rain"] >= 1.0).astype(int)
    is_wet_yesterday = (prev_rain >= 1.0).astype(int)

    out["dry_to_wet"] = ((is_wet_yesterday == 0) & (is_wet_today == 1)).astype(int)
    out["wet_to_dry"] = ((is_wet_yesterday == 1) & (is_wet_today == 0)).astype(int)
    out["heavy_rain_yesterday"] = (prev_rain >= 20.0).astype(int)

    out["wet_streak_days"] = 0
    for prov in out["province_encoded"].unique():
        mask = out["province_encoded"] == prov
        wet = is_wet_today.loc[mask].values
        counts = np.zeros(len(wet), dtype=int)
        for i in range(1, len(wet)):
            counts[i] = (counts[i - 1] + 1) if wet[i] == 1 else 0
        out.loc[mask, "wet_streak_days"] = counts

    out["pressure_roll_mean_3d"] = g_pressure.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)
    out["wind_roll_mean_3d"] = g_wind.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)

    old_features = list(p3.FEATURE_COLS)
    p3.FEATURE_COLS = old_features + [f for f in EXTRA_FEATURES if f not in old_features]

    for col in EXTRA_FEATURES:
        out[col] = out[col].fillna(0)

    p3.log.info(f"v4 extra features added: {len(EXTRA_FEATURES)}")
    p3.log.stat("Total features v4", len(p3.FEATURE_COLS))
    return out


def result_row(result: dict, model_name: str, config: str) -> dict:
    return {
        "model": model_name,
        "config": config,
        "r2_log": result["r2_log"],
        "rmse_log": result["rmse_log"],
        "mae_log": result["mae_log"],
        "accuracy_log_percent": float(result["r2_log"]) * 100.0,
        "r2_mm": result["r2_mm"],
        "rmse_mm": result["rmse_mm"],
        "mae_mm": result["mae_mm"],
        "accuracy_mm_percent": float(result["r2_mm"]) * 100.0,
        "train_time_s": result["train_time"],
    }


def save_comparison_outputs(
    out_dir: Path,
    xgb: dict,
    rf: dict,
    prophet: dict,
    lstm: dict,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        result_row(xgb, "XGBoost", "unscaled"),
        result_row(rf, "RandomForest", "unscaled"),
        result_row(prophet, "Prophet", "unscaled (per-province time series)"),
        result_row(lstm, "LSTM/GRU", "scaled (RobustScaler)"),
    ]

    metrics_df = pd.DataFrame(rows).sort_values("r2_mm", ascending=False).reset_index(drop=True)
    metrics_path = out_dir / "comparison_metrics.csv"
    report_path = out_dir / "comparison_report.md"
    metrics_df.to_csv(metrics_path, index=False)

    best_row = metrics_df.iloc[0]
    md_lines = [
        "# Model Comparison",
        "",
        "Models: XGBoost, Random Forest, Prophet, LSTM/GRU.",
        "Accuracy follows the current repository style: Accuracy (%) = R2 * 100.",
        "",
        f"Best by R2 (mm): **{best_row['model']}** ({best_row['r2_mm']:.4f})",
        "",
        "| Model | Config | R2 (log1p) | RMSE (log1p) | MAE (log1p) | Accuracy log1p (%) | R2 (mm) | RMSE (mm) | MAE (mm) | Accuracy mm (%) | Train time (s) |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]

    for _, row in metrics_df.iterrows():
        md_lines.append(
            "| {model} | {config} | {r2_log:.4f} | {rmse_log:.4f} | {mae_log:.4f} | {acc_log:.2f} | {r2_mm:.4f} | {rmse_mm:.4f} | {mae_mm:.4f} | {acc_mm:.2f} | {train_time:.2f} |".format(
                model=row["model"],
                config=row["config"],
                r2_log=row["r2_log"],
                rmse_log=row["rmse_log"],
                mae_log=row["mae_log"],
                acc_log=row["accuracy_log_percent"],
                r2_mm=row["r2_mm"],
                rmse_mm=row["rmse_mm"],
                mae_mm=row["mae_mm"],
                acc_mm=row["accuracy_mm_percent"],
                train_time=row["train_time_s"],
            )
        )

    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    p3.log.info(f"Comparison metrics -> {metrics_path}")
    p3.log.info(f"Comparison report  -> {report_path}")
    return metrics_df


def save_prediction_tables(
    out_dir: Path,
    test_df: pd.DataFrame,
    xgb: dict,
    rf: dict,
    prophet: dict,
    lstm: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = {
        "xgboost_pred_mm": xgb["predictions_mm"],
        "random_forest_pred_mm": rf["predictions_mm"],
        "prophet_pred_mm": prophet["predictions_mm"],
        "lstm_pred_mm": lstm["predictions_mm"],
    }

    combined = test_df.reset_index(drop=True).copy()
    for col, values in predictions.items():
        combined[col] = values

    combined_path = out_dir / "predictions_all_models_2021.csv"
    combined.to_csv(combined_path, index=False)

    detail_columns = ["date", "province", "region", "rain", "rain_log1p"]
    for short_name, col_name in [
        ("xgboost", "xgboost_pred_mm"),
        ("random_forest", "random_forest_pred_mm"),
        ("prophet", "prophet_pred_mm"),
        ("lstm", "lstm_pred_mm"),
    ]:
        path = out_dir / f"predictions_{short_name}_2021.csv"
        combined[detail_columns + [col_name]].to_csv(path, index=False)

    p3.log.info(f"Predictions table -> {combined_path}")


def save_prediction_chart(
    out_dir: Path,
    test_df: pd.DataFrame,
    xgb: dict,
    rf: dict,
    prophet: dict,
    lstm: dict,
) -> None:
    """Enhanced multi-plot visualization inspired by Walmart analysis style."""
    out_dir.mkdir(parents=True, exist_ok=True)

    actual = test_df[p3.TARGET_ORIG_COL].to_numpy()
    dates = pd.to_datetime(test_df["date"].values)
    
    model_specs = [
        ("XGBoost", xgb["predictions_mm"], xgb["r2_mm"], xgb["rmse_mm"], xgb["mae_mm"], "#1f77b4"),
        ("Random Forest", rf["predictions_mm"], rf["r2_mm"], rf["rmse_mm"], rf["mae_mm"], "#ff7f0e"),
        ("Prophet", prophet["predictions_mm"], prophet["r2_mm"], prophet["rmse_mm"], prophet["mae_mm"], "#2ca02c"),
        ("LSTM/GRU", lstm["predictions_mm"], lstm["r2_mm"], lstm["rmse_mm"], lstm["mae_mm"], "#d62728"),
    ]

    # ───────────────────────────────────────────────────────────────────────
    # Plot 1: Scatter plot - Actual vs Predicted (improved visibility)
    # ───────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12), dpi=150)
    
    # Create 3x2 grid for 6 plots
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Scatter plots (2x2)
    min_val = float(min(actual.min(), *(np.min(pred) for _, pred, *_ in model_specs)))
    max_val = float(max(actual.max(), *(np.max(pred) for _, pred, *_ in model_specs)))
    line = np.linspace(min_val, max_val, 200)
    
    axes_scatter = [fig.add_subplot(gs[i//2, i%2]) for i in range(4)]
    for ax, (name, pred, r2_mm, rmse_mm, mae_mm, color) in zip(axes_scatter, model_specs):
        # Create better scatter with transparency gradient
        ax.scatter(actual, pred, s=25, alpha=0.4, color=color, edgecolors="white", linewidth=0.5)
        ax.plot(line, line, color="black", linewidth=1.5, linestyle="--", alpha=0.7, label="Perfect predictions")
        ax.set_title(f"{name} (Test Set)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual Rainfall (mm)", fontsize=10)
        ax.set_ylabel("Predicted Rainfall (mm)", fontsize=10)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, alpha=0.3, linestyle="--")
        
        # Metrics box
        metrics_text = f"R² = {r2_mm:.4f}\nRMSE = {rmse_mm:.2f} mm\nMAE = {mae_mm:.2f} mm"
        ax.text(
            0.05, 0.95, metrics_text,
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": color, "alpha": 0.95, "linewidth": 2}
        )

    # Time-series plot (bottom row, spanning 2 columns)
    ax_ts = fig.add_subplot(gs[2, :])
    
    # Sample every Nth point for clearer visualization if data is very large
    sample_step = max(1, len(dates) // 365)  # Show ~daily points
    sample_idx = np.arange(0, len(dates), sample_step)
    sample_dates = dates[sample_idx]
    sample_actual = actual[sample_idx]
    
    ax_ts.plot(sample_dates, sample_actual, "o-", linewidth=2, markersize=5, 
               label="Actual", color="black", alpha=0.8, zorder=5)
    
    for name, pred, _, _, _, color in model_specs:
        sample_pred = pred[sample_idx] if len(pred) == len(actual) else pred[:len(sample_idx)]
        ax_ts.plot(sample_dates, sample_pred, "s--", linewidth=1.5, markersize=3,
                   label=f"{name}", color=color, alpha=0.7)
    
    ax_ts.set_title("Time-Series Predictions vs Actual Rainfall (2021)", fontsize=12, fontweight="bold")
    ax_ts.set_xlabel("Date", fontsize=10)
    ax_ts.set_ylabel("Rainfall (mm)", fontsize=10)
    ax_ts.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax_ts.grid(True, alpha=0.3, linestyle="-")
    ax_ts.tick_params(axis="x", rotation=45)
    
    fig.suptitle("Weather Prediction Model Evaluation — 2021 Test Performance", 
                 fontsize=14, fontweight="bold", y=0.995)
    
    chart_path = out_dir / "prediction_vs_actual_overview.png"
    fig.savefig(chart_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    p3.log.info(f"Prediction chart → {chart_path}")

    # ───────────────────────────────────────────────────────────────────────
    # Plot 2: Distribution comparison
    # ───────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    
    for ax, (name, pred, _, _, _, color) in zip(axes.flat, model_specs):
        residuals = actual - pred
        
        ax.hist(residuals, bins=50, alpha=0.6, color=color, edgecolor="black", linewidth=0.5)
        ax.axvline(x=np.mean(residuals), color=color, linestyle="--", linewidth=2, 
                   label=f"Mean: {np.mean(residuals):.2f}")
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        
        ax.set_title(f"{name} — Residual Distribution", fontsize=11, fontweight="bold")
        ax.set_xlabel("Prediction Error (mm)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Prediction Error Distribution (Actual - Predicted)", 
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    
    dist_path = out_dir / "prediction_error_distribution.png"
    fig.savefig(dist_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    p3.log.info(f"Distribution chart → {dist_path}")

    # ───────────────────────────────────────────────────────────────────────
    # Plot 3: Model comparison bar chart
    # ───────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    
    models = [name for name, *_ in model_specs]
    colors_list = [color for _, _, _, _, _, color in model_specs]
    
    # R² scores
    r2_scores = [r2 for _, _, r2, _, _, _ in model_specs]
    axes[0].bar(models, r2_scores, color=colors_list, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[0].set_title("R² Score (mm scale) - Higher is Better", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("R² Score", fontsize=10)
    axes[0].set_ylim([0, max(r2_scores) * 1.15])
    axes[0].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(r2_scores):
        axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    # RMSE scores
    rmse_scores = [rmse for _, _, _, rmse, _, _ in model_specs]
    axes[1].bar(models, rmse_scores, color=colors_list, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[1].set_title("RMSE (mm) - Lower is Better", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("RMSE (mm)", fontsize=10)
    axes[1].set_ylim([0, max(rmse_scores) * 1.15])
    axes[1].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(rmse_scores):
        axes[1].text(i, v + 0.5, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    # MAE scores
    mae_scores = [mae for _, _, _, _, mae, _ in model_specs]
    axes[2].bar(models, mae_scores, color=colors_list, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[2].set_title("MAE (mm) - Lower is Better", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("MAE (mm)", fontsize=10)
    axes[2].set_ylim([0, max(mae_scores) * 1.15])
    axes[2].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(mae_scores):
        axes[2].text(i, v + 0.3, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    fig.suptitle("Model Comparison — Performance Metrics (2021 Test Set)", 
                 fontsize=13, fontweight="bold")
    plt.setp(axes, xticklabels=models)
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    
    fig.tight_layout()
    comparison_path = out_dir / "model_performance_comparison.png"
    fig.savefig(comparison_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    p3.log.info(f"Comparison chart → {comparison_path}")


def save_feature_importance_chart(
    out_dir: Path,
    xgb: dict,
) -> None:
    """Save feature importance from XGBoost model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    feat_importance = xgb["feature_importance"].head(20)
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(feat_importance)))
    
    bars = ax.barh(range(len(feat_importance)), feat_importance["importance"].values, 
                   color=colors, edgecolor="black", linewidth=0.7)
    
    ax.set_yticks(range(len(feat_importance)))
    ax.set_yticklabels(feat_importance["feature"].values, fontsize=10)
    ax.set_xlabel("Importance Score", fontsize=11, fontweight="bold")
    ax.set_title("Top 20 Feature Importance (XGBoost Model)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, feat_importance["importance"].values)):
        ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=9)
    
    fig.tight_layout()
    importance_path = out_dir / "feature_importance.png"
    fig.savefig(importance_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    p3.log.info(f"Feature importance chart → {importance_path}")


def main():
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_dir = Path("input")
    output_dir = Path("output") / f"{run_ts}"

    cfg = p3.PipelineConfig()
    t_start = time.time()

    p3.log.header("PIPELINE v4 - WEATHER PREPROCESSING AND EVALUATION")

    cfg.output_dir = input_dir
    df = p3.load_and_clean(cfg)
    df = engineer_features_v4(df, cfg)
    train_df, val_df, test_df = p3.split_train_val_test(df, cfg)
    train_unscaled, val_unscaled, test_unscaled, feat_scaler, tgt_scaler = p3.select_scale_save(
        train_df,
        val_df,
        test_df,
        cfg,
    )

    model_train_unscaled = pd.concat([train_unscaled, val_unscaled], ignore_index=True)
    model_train_unscaled = model_train_unscaled.sort_values(["province_encoded", "date"]).reset_index(drop=True)

    cfg.output_dir = output_dir
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    xgb_results = p3.evaluate_xgboost(model_train_unscaled, test_unscaled, cfg)
    rf_results = p3.evaluate_random_forest(model_train_unscaled, test_unscaled, cfg)
    prophet_results = p3.evaluate_prophet(model_train_unscaled, test_unscaled, cfg)
    lstm_results = p3.evaluate_lstm(model_train_unscaled, test_unscaled, feat_scaler, tgt_scaler, cfg)

    pipeline_time = time.time() - t_start
    summary = p3.final_summary(xgb_results, lstm_results, test_unscaled, cfg, pipeline_time)

    summary["random_forest"] = {
        "r2_log1p": round(rf_results["r2_log"], 4),
        "rmse_log1p": round(rf_results["rmse_log"], 4),
        "mae_log1p": round(rf_results["mae_log"], 4),
        "r2_mm": round(rf_results["r2_mm"], 4),
        "rmse_mm": round(rf_results["rmse_mm"], 4),
        "mae_mm": round(rf_results["mae_mm"], 4),
        "cv_rmse": rf_results["cv_rmse"],
        "train_time_s": round(rf_results["train_time"], 1),
    }
    summary["prophet"] = {
        "r2_log1p": round(prophet_results["r2_log"], 4),
        "rmse_log1p": round(prophet_results["rmse_log"], 4),
        "mae_log1p": round(prophet_results["mae_log"], 4),
        "r2_mm": round(prophet_results["r2_mm"], 4),
        "rmse_mm": round(prophet_results["rmse_mm"], 4),
        "mae_mm": round(prophet_results["mae_mm"], 4),
        "train_time_s": round(prophet_results["train_time"], 1),
    }

    save_comparison_outputs(output_dir, xgb_results, rf_results, prophet_results, lstm_results)
    save_prediction_tables(output_dir, test_df, xgb_results, rf_results, prophet_results, lstm_results)
    save_prediction_chart(output_dir, test_df, xgb_results, rf_results, prophet_results, lstm_results)
    save_feature_importance_chart(output_dir, xgb_results)

    print("\nSaved input artifacts:")
    print(f"  {input_dir}")
    print("Saved output artifacts:")
    print(f"  {output_dir}")

    return summary


if __name__ == "__main__":
    main()