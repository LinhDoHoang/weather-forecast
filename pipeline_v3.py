"""
===============================================================================
 Pipeline v3 — Vietnam Weather: Preprocessing & Model Evaluation
===============================================================================
 Quy trình:
   1. Load & Clean       — Chuẩn hóa tên, nội suy ngày thiếu, validate
   2. Feature Engineering — Cyclical encoding, lag features, domain features
   3. Temporal Split      — Train (2009-2020) / Test (2021), no leakage
   4. Scale & Save        — Unscaled (XGBoost) + Scaled (LSTM)
   5. Evaluate XGBoost    — CV + test metrics (log1p & mm)
   6. Evaluate LSTM       — PyTorch, early stopping, test metrics
   7. Final Summary       — So sánh kết quả, feature importance, khuyến nghị
===============================================================================
"""

import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """Tất cả tham số pipeline — thay đổi tại đây, không hard-code ở nơi khác."""

    # Paths
    raw_data_path: Path = field(default_factory=lambda: Path("dataset/init/weather.csv"))
    mapping_path: Path = field(default_factory=lambda: Path("dataset/province_region_code_mapping.csv"))
    output_dir: Path = field(default_factory=lambda: Path("dataset/v3"))

    # Preprocessing
    test_year: int = 2021
    random_seed: int = 42
    lag_windows: Tuple[int, ...] = (7, 14, 30)
    province_name_fixes: Dict[str, str] = field(
        default_factory=lambda: {"Hanoi": "Ha Noi"},
    )

    # Wind direction → compass degrees
    wind_dir_degrees: Dict[str, float] = field(default_factory=lambda: {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
        "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
        "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
        "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
    })

    # Rainy season per region (domain knowledge)
    rainy_months: Dict[str, List[int]] = field(default_factory=lambda: {
        "Bac Bo":     [5, 6, 7, 8, 9, 10],
        "Trung Bo":   [9, 10, 11, 12],
        "Tay Nguyen": [5, 6, 7, 8, 9, 10],
        "Nam Bo":     [5, 6, 7, 8, 9, 10, 11],
    })

    # XGBoost hyperparameters
    xgb_n_estimators: int = 800
    xgb_max_depth: int = 7
    xgb_learning_rate: float = 0.03
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.0
    xgb_min_child_weight: int = 5
    xgb_cv_folds: int = 5

    # LSTM hyperparameters
    lstm_lookback: int = 30
    lstm_batch_size: int = 512
    lstm_epochs: int = 50
    lstm_lr: float = 5e-4
    lstm_hidden1: int = 128
    lstm_hidden2: int = 64
    lstm_dropout: float = 0.2
    lstm_patience: int = 12
    lstm_val_date: str = "2019-01-01"  # validation split for LSTM


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS: List[str] = [
    # Weather core
    "max_temp", "min_temp", "wind", "humidi", "cloud", "pressure", "temp_diff",
    # Cyclical time
    "month_sin", "month_cos", "day_sin", "day_cos",
    # Wind direction (cyclical)
    "wind_dir_sin", "wind_dir_cos",
    # Domain knowledge
    "is_rainy_season", "humid_cloud",
    # Lag features
    "rain_yesterday",
    "rain_roll_mean_7d", "rain_roll_max_7d", "rain_roll_std_7d",
    "rain_roll_mean_14d", "rain_roll_max_14d", "rain_roll_std_14d",
    "rain_roll_mean_30d", "rain_roll_max_30d", "rain_roll_std_30d",
    # Additional engineered
    "pressure_diff", "consecutive_dry_days", "temp_humid_interaction",
    # Encoded categorical
    "region_encoded", "province_encoded",
]

TARGET_COL = "rain_log1p"
TARGET_ORIG_COL = "rain"

# Columns that should NOT be scaled (integer codes)
NO_SCALE_COLS = {"region_encoded", "province_encoded"}


# ─────────────────────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────────────────────
class PipelineLogger:
    """Console logger with color-coded output."""

    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    RESET  = "\033[0m"

    def __init__(self):
        self._step = 0

    def header(self, title: str):
        sep = "═" * 70
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{self.CYAN}{sep}")
        print(f"  {self.BOLD}{title}{self.RESET}")
        print(f"  {ts}")
        print(f"{self.CYAN}{sep}{self.RESET}\n")

    def step(self, title: str):
        self._step += 1
        sep = "─" * 66
        print(f"\n{self.CYAN}{sep}")
        print(f"  STEP {self._step}: {self.BOLD}{title}{self.RESET}")
        print(f"{self.CYAN}{sep}{self.RESET}")

    def info(self, msg: str):
        print(f"  {self.GREEN}►{self.RESET} {msg}")

    def warn(self, msg: str):
        print(f"  {self.YELLOW}⚠ {msg}{self.RESET}")

    def error(self, msg: str):
        print(f"  {self.RED}✗ {msg}{self.RESET}")

    def stat(self, label: str, value: Any):
        print(f"    {str(label):<42} {str(value):>15}")

    def table(self, header: str, rows: List[str]):
        print(f"\n  {self.BOLD}{header}{self.RESET}")
        for row in rows:
            print(f"    {row}")
        print()


log = PipelineLogger()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD & CLEAN
# ═════════════════════════════════════════════════════════════════════════════
def load_and_clean(cfg: PipelineConfig) -> pd.DataFrame:
    log.step("TẢI VÀ LÀM SẠCH DỮ LIỆU THÔ")

    df = pd.read_csv(cfg.raw_data_path)
    log.info(f"Loaded {cfg.raw_data_path} — shape: {df.shape}")
    log.stat("Provinces (raw)", df["province"].nunique())
    log.stat("Date range", f"{df['date'].min()} → {df['date'].max()}")

    # 1.1 — Normalize province names
    n_before = df["province"].nunique()
    df["province"] = df["province"].replace(cfg.province_name_fixes)
    n_after = df["province"].nunique()
    log.info(f"Chuẩn hóa tên tỉnh: {n_before} → {n_after} unique")

    # 1.2 — Parse dates & rename columns
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"max": "max_temp", "min": "min_temp"})

    # 1.3 — Deduplicate (province, date) if any
    num_cols = ["max_temp", "min_temp", "wind", "rain", "humidi", "cloud", "pressure"]
    n_dup = df.duplicated(subset=["province", "date"], keep=False).sum()
    if n_dup > 0:
        log.info(f"Aggregate {n_dup} duplicate (province, date) rows")
        agg_dict = {col: "mean" for col in num_cols}
        agg_dict["wind_d"] = "first"
        df = df.groupby(["province", "date"], as_index=False).agg(agg_dict)

    # 1.4 — Interpolate missing dates per province
    total_missing = 0
    frames = []
    for prov, grp in df.groupby("province"):
        grp = grp.set_index("date").sort_index().drop(columns=["province"])
        full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
        n_miss = len(full_idx) - len(grp)
        total_missing += n_miss
        grp = grp.reindex(full_idx)
        grp.index.name = "date"
        for col in num_cols:
            grp[col] = grp[col].interpolate(method="time")
        grp["wind_d"] = grp["wind_d"].ffill().bfill()
        grp["province"] = prov
        frames.append(grp.reset_index())

    df = pd.concat(frames, ignore_index=True)
    log.info(f"Nội suy {total_missing} ngày thiếu trên {df['province'].nunique()} tỉnh")
    log.stat("Shape sau nội suy", df.shape)

    # 1.5 — Round numeric columns
    round_cols = ["max_temp", "min_temp", "wind", "humidi", "cloud", "pressure"]
    df[round_cols] = df[round_cols].round().astype(int)

    # 1.6 — Validate
    nulls = df.isnull().sum().sum()
    neg_rain = (df["rain"] < 0).sum()

    if nulls > 0:
        log.error(f"Còn {nulls} null values!")
    else:
        log.info("✓ Không có null values")

    if neg_rain > 0:
        log.error(f"{neg_rain} giá trị rain < 0!")
    else:
        log.info("✓ Rain ≥ 0")

    log.stat("Final clean shape", df.shape)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    log.step("FEATURE ENGINEERING")

    # 2.1 — Load province → region mapping
    mapping = pd.read_csv(cfg.mapping_path)
    prov2region = dict(zip(mapping["province"], mapping["region"]))
    prov2code = dict(zip(mapping["province"], mapping["province_encoded"]))
    region2code = dict(zip(mapping["region"], mapping["region_encoded"]))

    df["region"] = df["province"].map(prov2region)
    df["province_encoded"] = df["province"].map(prov2code)
    df["region_encoded"] = df["region"].map(region2code)

    unmapped = df["region"].isnull().sum()
    if unmapped > 0:
        log.error(f"{unmapped} tỉnh không map được sang vùng!")
        unmapped_provs = df.loc[df["region"].isnull(), "province"].unique()
        log.error(f"  → {list(unmapped_provs)}")
        sys.exit(1)
    log.info(f"✓ Map {df['province'].nunique()} tỉnh → {df['region'].nunique()} vùng")

    # 2.2 — Temperature difference
    df["temp_diff"] = df["max_temp"] - df["min_temp"]
    log.info("+ temp_diff")

    # 2.3 — Cyclical time encoding (sin/cos)
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["year"] = df["date"].dt.year
    day_of_year = df["date"].dt.dayofyear

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["day_cos"] = np.cos(2 * np.pi * day_of_year / 365)
    log.info("+ month_sin/cos, day_sin/cos (cyclical)")

    # 2.4 — Wind direction → compass degrees → sin/cos
    df["wind_deg"] = df["wind_d"].map(cfg.wind_dir_degrees)
    df["wind_dir_sin"] = np.sin(np.deg2rad(df["wind_deg"]))
    df["wind_dir_cos"] = np.cos(np.deg2rad(df["wind_deg"]))
    log.info("+ wind_dir_sin/cos (compass)")

    # 2.5 — Rainy season flag per region
    region_month_pairs = set()
    for region, months in cfg.rainy_months.items():
        for m in months:
            region_month_pairs.add((region, m))

    df["is_rainy_season"] = df.apply(
        lambda r: 1 if (r["region"], r["month"]) in region_month_pairs else 0,
        axis=1,
    )
    log.info("+ is_rainy_season (domain knowledge)")

    # 2.6 — Humidity × cloud interaction
    df["humid_cloud"] = df["humidi"] * df["cloud"] / 100
    log.info("+ humid_cloud interaction")

    # 2.7 — Lag features per province (shift=1 → no data leakage)
    df = df.sort_values(["province_encoded", "date"]).reset_index(drop=True)

    for w in cfg.lag_windows:
        col_mean = f"rain_roll_mean_{w}d"
        col_max  = f"rain_roll_max_{w}d"
        col_std  = f"rain_roll_std_{w}d"

        df[col_mean] = (
            df.groupby("province_encoded")["rain"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
        df[col_max] = (
            df.groupby("province_encoded")["rain"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).max())
        )
        df[col_std] = (
            df.groupby("province_encoded")["rain"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).std().fillna(0))
        )
        log.info(f"+ rain_roll_mean/max/std_{w}d (lag, shift=1)")

    # 2.8 — Yesterday's rain
    df["rain_yesterday"] = df.groupby("province_encoded")["rain"].transform(
        lambda s: s.shift(1)
    )
    df["rain_yesterday"] = df["rain_yesterday"].fillna(0)
    log.info("+ rain_yesterday (lag-1)")

    # 2.9 — Pressure change per province (day-to-day delta)
    df["pressure_diff"] = df.groupby("province_encoded")["pressure"].transform(
        lambda s: s.diff().fillna(0)
    )
    log.info("+ pressure_diff (day-to-day pressure change)")

    # 2.10 — Consecutive dry days per province
    df["is_dry"] = (df["rain"] < 0.1).astype(int)
    df["consecutive_dry_days"] = 0
    for prov in df["province_encoded"].unique():
        mask = df["province_encoded"] == prov
        dry = df.loc[mask, "is_dry"].values
        counts = np.zeros(len(dry), dtype=int)
        for i in range(1, len(dry)):
            counts[i] = (counts[i - 1] + 1) * dry[i]
        df.loc[mask, "consecutive_dry_days"] = counts
    df = df.drop(columns=["is_dry"])
    log.info("+ consecutive_dry_days")

    # 2.11 — Temperature-humidity interaction (condensation proxy)
    df["temp_humid_interaction"] = df["temp_diff"] * df["humidi"] / 100
    log.info("+ temp_humid_interaction")

    # 2.12 — Log-transform target
    df["rain_log1p"] = np.log1p(df["rain"])
    log.info("+ rain_log1p = log1p(rain)")
    log.stat("rain — mean/std/skew",
             f"{df['rain'].mean():.2f} / {df['rain'].std():.2f} / {df['rain'].skew():.2f}")
    log.stat("rain_log1p — mean/std/skew",
             f"{df['rain_log1p'].mean():.2f} / {df['rain_log1p'].std():.2f} / {df['rain_log1p'].skew():.2f}")

    # 2.13 — Fill any remaining NaN from lag/diff features
    lag_cols = [c for c in df.columns if "roll_" in c or c in ("rain_yesterday", "pressure_diff")]
    df[lag_cols] = df[lag_cols].fillna(0)

    log.stat("Total columns", df.shape[1])
    log.stat("Feature columns", len(FEATURE_COLS))
    return df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: TEMPORAL TRAIN / TEST SPLIT
# ═════════════════════════════════════════════════════════════════════════════
def split_train_test(df: pd.DataFrame, cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log.step("PHÂN CHIA TRAIN / TEST (temporal)")

    split_date = pd.Timestamp(f"{cfg.test_year}-01-01")
    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()

    log.stat("Train shape", train_df.shape)
    log.stat("Test shape", test_df.shape)
    log.stat("Train date range",
             f"{train_df['date'].min().date()} → {train_df['date'].max().date()}")
    log.stat("Test date range",
             f"{test_df['date'].min().date()} → {test_df['date'].max().date()}")
    log.stat("Train rain_log1p mean", f"{train_df[TARGET_COL].mean():.4f}")
    log.stat("Test rain_log1p mean", f"{test_df[TARGET_COL].mean():.4f}")

    if train_df["date"].max() >= test_df["date"].min():
        log.error("DATA LEAKAGE detected!")
        sys.exit(1)
    log.info("✓ No data leakage")

    return train_df, test_df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: SELECT FEATURES, SCALE & SAVE
# ═════════════════════════════════════════════════════════════════════════════
def select_scale_save(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, RobustScaler, RobustScaler]:
    log.step("CHỌN FEATURES, SCALE & LƯU FILE")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    keep_cols = ["date"] + FEATURE_COLS + [TARGET_COL, TARGET_ORIG_COL]
    train_out = train_df[keep_cols].copy()
    test_out = test_df[keep_cols].copy()

    # Save unscaled (for XGBoost — tree-based models don't need scaling)
    train_unscaled_path = cfg.output_dir / "weather_train_2009_2020.csv"
    test_unscaled_path = cfg.output_dir / "weather_test_2021.csv"
    train_out.to_csv(train_unscaled_path, index=False)
    test_out.to_csv(test_unscaled_path, index=False)
    log.info(f"Saved unscaled → {train_unscaled_path}")
    log.info(f"Saved unscaled → {test_unscaled_path}")

    # Scale for LSTM — RobustScaler (fit on train only)
    scale_cols = [c for c in FEATURE_COLS if c not in NO_SCALE_COLS]

    feat_scaler = RobustScaler()
    feat_scaler.fit(train_out[scale_cols])

    tgt_scaler = RobustScaler()
    tgt_scaler.fit(train_out[[TARGET_COL]])

    train_scaled = train_out.copy()
    test_scaled = test_out.copy()
    train_scaled[scale_cols] = feat_scaler.transform(train_out[scale_cols])
    test_scaled[scale_cols] = feat_scaler.transform(test_out[scale_cols])
    train_scaled[TARGET_COL] = tgt_scaler.transform(train_out[[TARGET_COL]])
    test_scaled[TARGET_COL] = tgt_scaler.transform(test_out[[TARGET_COL]])

    train_scaled_path = cfg.output_dir / "weather_train_2009_2020_scaled.csv"
    test_scaled_path = cfg.output_dir / "weather_test_2021_scaled.csv"
    train_scaled.to_csv(train_scaled_path, index=False)
    test_scaled.to_csv(test_scaled_path, index=False)
    log.info(f"Saved scaled   → {train_scaled_path}")
    log.info(f"Saved scaled   → {test_scaled_path}")

    # Save scalers
    scaler_path = cfg.output_dir / "scalers.pkl"
    joblib.dump({"feature_scaler": feat_scaler, "target_scaler": tgt_scaler}, scaler_path)
    log.info(f"Saved scalers  → {scaler_path}")

    log.stat("Scale method", "RobustScaler (fit on train only)")
    log.stat("Scaled feature count", len(scale_cols))
    log.stat("Unscaled (categorical)", list(NO_SCALE_COLS))

    return train_out, test_out, feat_scaler, tgt_scaler


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: EVALUATE — XGBoost
# ═════════════════════════════════════════════════════════════════════════════
def evaluate_xgboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    log.step("ĐÁNH GIÁ XGBOOST")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test_log = test_df[TARGET_COL]
    y_test_mm = test_df[TARGET_ORIG_COL].values

    model = XGBRegressor(
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample_bytree,
        reg_alpha=cfg.xgb_reg_alpha,
        reg_lambda=cfg.xgb_reg_lambda,
        min_child_weight=cfg.xgb_min_child_weight,
        random_state=cfg.random_seed,
        n_jobs=-1,
        verbosity=0,
    )

    # Cross-validation
    log.info(f"Cross-validation ({cfg.xgb_cv_folds}-fold) trên log1p space...")
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cfg.xgb_cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    cv_rmse = -cv_scores
    log.stat("CV RMSE (log1p)", f"{cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

    # Train
    log.info("Training on full train set...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    log.stat("Training time", f"{train_time:.2f}s")

    # Predict (log1p space)
    pred_log = model.predict(X_test)

    # Metrics — log1p space
    mae_log = mean_absolute_error(y_test_log, pred_log)
    mse_log = mean_squared_error(y_test_log, pred_log)
    rmse_log = np.sqrt(mse_log)
    r2_log = r2_score(y_test_log, pred_log)

    log.info("Metrics (log1p space):")
    log.stat("MAE", f"{mae_log:.4f}")
    log.stat("RMSE", f"{rmse_log:.4f}")
    log.stat("R²", f"{r2_log:.4f}")

    # Metrics — original mm scale
    pred_mm = np.clip(np.expm1(pred_log), 0, None)
    mae_mm = mean_absolute_error(y_test_mm, pred_mm)
    mse_mm = mean_squared_error(y_test_mm, pred_mm)
    rmse_mm = np.sqrt(mse_mm)
    r2_mm = r2_score(y_test_mm, pred_mm)

    log.info("Metrics (mm — original scale):")
    log.stat("MAE", f"{mae_mm:.2f} mm")
    log.stat("RMSE", f"{rmse_mm:.2f} mm")
    log.stat("R²", f"{r2_mm:.4f}")

    # Feature importance (top 10)
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    log.info("Top 10 feature importance:")
    top10 = importance.head(10)
    max_imp = top10["importance"].max()
    for _, row in top10.iterrows():
        bar = "█" * int(row["importance"] / max_imp * 25)
        log.stat(row["feature"], f"{bar} {row['importance']:.4f}")

    return {
        "model_name": "XGBoost",
        "r2_log": r2_log, "rmse_log": rmse_log, "mae_log": mae_log,
        "r2_mm": r2_mm, "rmse_mm": rmse_mm, "mae_mm": mae_mm,
        "cv_rmse": f"{cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}",
        "train_time": train_time,
        "predictions_mm": pred_mm,
        "feature_importance": importance,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6: EVALUATE — LSTM (PyTorch)
# ═════════════════════════════════════════════════════════════════════════════

class _SequenceDataset(Dataset):
    """Sliding-window sequences per province (no cross-province mixing)."""

    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int,
        features: List[str],
        target: str,
        prefix_df: Optional[pd.DataFrame] = None,
    ):
        x_list, y_list = [], []
        for prov in sorted(df["province_encoded"].unique()):
            prov_df = df[df["province_encoded"] == prov]
            if prefix_df is not None:
                pre = prefix_df[prefix_df["province_encoded"] == prov].tail(lookback)
                prov_df = pd.concat([pre, prov_df], ignore_index=True)
            x_arr = prov_df[features].values.astype(np.float32)
            y_arr = prov_df[target].values.astype(np.float32)
            for i in range(lookback, len(prov_df)):
                x_list.append(x_arr[i - lookback: i])
                y_list.append(y_arr[i])

        self.X = torch.from_numpy(np.stack(x_list))
        self.y = torch.from_numpy(np.array(y_list, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class _GRUModel(nn.Module):
    """Single-layer GRU — simpler, less overfitting, faster on CPU."""

    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            n_features, hidden, num_layers=2,
            batch_first=True, dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hn = self.gru(x)          # hn: (num_layers, B, H)
        out = self.drop(hn[-1])      # last layer hidden: (B, H)
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(1)


def evaluate_lstm(
    train_unscaled: pd.DataFrame,
    test_unscaled: pd.DataFrame,
    feat_scaler: RobustScaler,
    tgt_scaler: RobustScaler,
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    log.step("ĐÁNH GIÁ LSTM (PyTorch)")

    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | Lookback: {cfg.lstm_lookback} | Max epochs: {cfg.lstm_epochs}")

    # Scale features + target
    scale_cols = [c for c in FEATURE_COLS if c not in NO_SCALE_COLS]

    def _scale(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[scale_cols] = feat_scaler.transform(df[scale_cols])
        out[TARGET_COL] = tgt_scaler.transform(df[[TARGET_COL]])
        return out

    train_scaled = _scale(train_unscaled)
    test_scaled = _scale(test_unscaled)

    # Temporal validation split based on date (not row index)
    val_date = pd.Timestamp(cfg.lstm_val_date)
    pure_train = train_scaled[train_scaled["date"] < str(val_date)]
    val_data = train_scaled[train_scaled["date"] >= str(val_date)]
    log.info(f"Train/Val split: before {val_date.date()} / from {val_date.date()}")
    log.stat("Pure train rows", len(pure_train))
    log.stat("Validation rows", len(val_data))

    # Build sequence datasets
    log.info("Building sliding-window sequences...")
    t0 = time.time()
    train_ds = _SequenceDataset(pure_train, cfg.lstm_lookback, FEATURE_COLS, TARGET_COL)
    val_ds = _SequenceDataset(val_data, cfg.lstm_lookback, FEATURE_COLS, TARGET_COL,
                              prefix_df=pure_train)
    test_ds = _SequenceDataset(test_scaled, cfg.lstm_lookback, FEATURE_COLS, TARGET_COL,
                               prefix_df=train_scaled)
    log.stat("Train sequences", f"{len(train_ds):,}")
    log.stat("Val sequences", f"{len(val_ds):,}")
    log.stat("Test sequences", f"{len(test_ds):,}")
    log.stat("Sequence shape", f"({cfg.lstm_lookback}, {len(FEATURE_COLS)})")
    log.stat("Build time", f"{time.time() - t0:.1f}s")

    train_dl = DataLoader(train_ds, batch_size=cfg.lstm_batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.lstm_batch_size * 2, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=cfg.lstm_batch_size * 2, shuffle=False)

    # Build model (GRU — faster on CPU, less overfitting)
    model = _GRUModel(len(FEATURE_COLS), cfg.lstm_hidden1, cfg.lstm_dropout)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"GRU params: {n_params:,}")

    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lstm_lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    log.info(f"Loss: SmoothL1Loss(beta=0.5) | Optimizer: AdamW(lr={cfg.lstm_lr})")
    log.info(f"{'Epoch':>6} | {'Train':>12} | {'Val':>12} | {'LR':>10} | {'Status':>20}")
    log.info("─" * 70)

    # Training loop
    best_val = math.inf
    best_weights = None
    wait = 0
    total_t0 = time.time()
    train_losses, val_losses = [], []

    for epoch in range(1, cfg.lstm_epochs + 1):
        # — Train —
        model.train()
        ep_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(yb)
        ep_loss /= len(train_ds)

        # — Validate —
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vl += criterion(model(xb), yb).item() * len(yb)
        vl /= len(val_ds)

        scheduler.step(vl)
        lr_now = optimizer.param_groups[0]["lr"]
        train_losses.append(ep_loss)
        val_losses.append(vl)

        if vl < best_val:
            best_val = vl
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            status = "✓ best"
        else:
            wait += 1
            status = f"patience {wait}/{cfg.lstm_patience}"

        log.info(f"{epoch:>6} | {ep_loss:>12.6f} | {vl:>12.6f} | {lr_now:>10.2e} | {status:>20}")

        if wait >= cfg.lstm_patience:
            log.info(f"Early stopping at epoch {epoch}")
            break

    if best_weights:
        model.load_state_dict(best_weights)

    total_time = time.time() - total_t0
    epochs_run = len(train_losses)
    log.stat("Best val loss", f"{best_val:.6f}")
    log.stat("Epochs run", epochs_run)
    log.stat("Total training time", f"{total_time:.1f}s")

    # — Predict —
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_dl:
            all_preds.append(model(xb.to(device)).cpu().numpy())
    pred_scaled = np.concatenate(all_preds)

    # Inverse transform: scaled → log1p → mm
    pred_log = tgt_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    pred_mm = np.clip(np.expm1(pred_log), 0, None)

    y_test_log = test_unscaled[TARGET_COL].values
    y_test_mm = test_unscaled[TARGET_ORIG_COL].values

    # Metrics — log1p space
    r2_log = r2_score(y_test_log, pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_test_log, pred_log))
    mae_log = mean_absolute_error(y_test_log, pred_log)

    log.info("Metrics (log1p space):")
    log.stat("MAE", f"{mae_log:.4f}")
    log.stat("RMSE", f"{rmse_log:.4f}")
    log.stat("R²", f"{r2_log:.4f}")

    # Metrics — original mm
    mae_mm = mean_absolute_error(y_test_mm, pred_mm)
    mse_mm = mean_squared_error(y_test_mm, pred_mm)
    rmse_mm = np.sqrt(mse_mm)
    r2_mm = r2_score(y_test_mm, pred_mm)

    log.info("Metrics (mm — original scale):")
    log.stat("MAE", f"{mae_mm:.2f} mm")
    log.stat("RMSE", f"{rmse_mm:.2f} mm")
    log.stat("R²", f"{r2_mm:.4f}")

    # Save loss curve
    loss_df = pd.DataFrame({"epoch": range(1, epochs_run + 1),
                            "train_loss": train_losses, "val_loss": val_losses})
    loss_path = cfg.output_dir / "lstm_loss_curve.csv"
    loss_df.to_csv(loss_path, index=False)
    log.info(f"Loss curve → {loss_path}")

    return {
        "model_name": "LSTM",
        "r2_log": r2_log, "rmse_log": rmse_log, "mae_log": mae_log,
        "r2_mm": r2_mm, "rmse_mm": rmse_mm, "mae_mm": mae_mm,
        "train_time": total_time,
        "epochs_run": epochs_run,
        "best_val_loss": best_val,
        "predictions_mm": pred_mm,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7: FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
def _segment_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute MAE/RMSE/R² for a segment."""
    if len(y_true) < 2:
        return {"n": len(y_true), "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    return {
        "n":    int(len(y_true)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def _fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if not math.isnan(v) else "—"


def _write_markdown_report(
    path: Path,
    metrics: Dict[str, Any],
    xgb: Dict[str, Any],
    lstm: Dict[str, Any],
    region_df: pd.DataFrame,
    month_df: pd.DataFrame,
    cfg: PipelineConfig,
    pipeline_time: float,
    ts: str,
) -> None:
    xm = metrics["xgboost"]
    gm = metrics["gru_lstm"]
    best = metrics["best_model"]
    rating = metrics["rating"]

    lines = [
        "# Rainfall Prediction — Evaluation Report",
        "",
        f"**Run:** {ts}  ",
        f"**Train period:** 2009–2020  ",
        f"**Test period:** {cfg.test_year} (Jan–Jun)  ",
        f"**Features:** {metrics['n_features']}  ",
        f"**Pipeline time:** {pipeline_time:.1f}s  ",
        "",
        "---",
        "",
        "## 1. Overall Metrics",
        "",
        "| Metric | XGBoost | GRU/LSTM | Winner |",
        "|:-------|--------:|---------:|:------:|",
    ]

    rows = [
        ("R² (log1p)",   xm["r2_log1p"],   gm["r2_log1p"],   True),
        ("RMSE (log1p)", xm["rmse_log1p"],  gm["rmse_log1p"], False),
        ("MAE (log1p)",  xm["mae_log1p"],   gm["mae_log1p"],  False),
        ("R² (mm)",      xm["r2_mm"],       gm["r2_mm"],      True),
        ("RMSE (mm)",    xm["rmse_mm"],     gm["rmse_mm"],    False),
        ("MAE (mm)",     xm["mae_mm"],      gm["mae_mm"],     False),
    ]
    for label, xv, gv, higher in rows:
        winner = "XGBoost" if (xv >= gv) == higher else "GRU/LSTM"
        lines.append(f"| {label} | {_fmt(xv)} | {_fmt(gv)} | {winner} |")

    lines += [
        f"| Train time (s) | {xm['train_time_s']:.1f} | {gm['train_time_s']:.1f} | — |",
        f"| CV RMSE (log1p) | {xm['cv_rmse']} | — | — |",
        f"| Epochs run | — | {gm['epochs_run']} | — |",
        "",
        f"> **Best model:** {best} &nbsp;|&nbsp; **Rating:** {rating}",
        "",
        "---",
        "",
        "## 2. Feature Importance (XGBoost — Top 15)",
        "",
        "| Rank | Feature | Importance |",
        "|-----:|:--------|----------:|",
    ]
    top15 = xgb["feature_importance"].head(15)
    for i, row in enumerate(top15.itertuples(), 1):
        lines.append(f"| {i} | `{row.feature}` | {row.importance:.4f} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Evaluation by Region",
        "",
        "| Region | N | XGB MAE | XGB RMSE | XGB R² | GRU MAE | GRU RMSE | GRU R² |",
        "|:-------|--:|--------:|---------:|-------:|--------:|---------:|-------:|",
    ]
    for _, r in region_df.iterrows():
        lines.append(
            f"| {r['region']} | {int(r['n'])} "
            f"| {_fmt(r['xgb_mae'], 2)} | {_fmt(r['xgb_rmse'], 2)} | {_fmt(r['xgb_r2'])} "
            f"| {_fmt(r['gru_mae'], 2)} | {_fmt(r['gru_rmse'], 2)} | {_fmt(r['gru_r2'])} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Evaluation by Month",
        "",
        "| Month | N | XGB MAE | XGB RMSE | XGB R² | GRU MAE | GRU RMSE | GRU R² |",
        "|------:|--:|--------:|---------:|-------:|--------:|---------:|-------:|",
    ]
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    for _, r in month_df.iterrows():
        mo = month_names.get(int(r["month"]), str(int(r["month"])))
        lines.append(
            f"| {mo} | {int(r['n'])} "
            f"| {_fmt(r['xgb_mae'], 2)} | {_fmt(r['xgb_rmse'], 2)} | {_fmt(r['xgb_r2'])} "
            f"| {_fmt(r['gru_mae'], 2)} | {_fmt(r['gru_rmse'], 2)} | {_fmt(r['gru_r2'])} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Features Used",
        "",
        "| # | Feature |",
        "|--:|:--------|",
    ]
    for i, feat in enumerate(metrics["features"], 1):
        lines.append(f"| {i} | `{feat}` |")

    lines += [
        "",
        "---",
        "",
        "## 6. Preprocessing Notes",
        "",
        "- **Target:** `rain` (mm) → `log1p` transformed (reduces skewness)",
        "- **Scaling:** RobustScaler for GRU input features (resistant to outliers)",
        "- **Split:** strict temporal — no data leakage across train/test",
        "- **Cyclical encoding:** `sin/cos` for month, day-of-year, wind direction",
        "- **Lag features:** rain rolled mean/max/std at 7d, 14d, 30d per province",
        "- **Domain features:** `is_rainy_season`, `consecutive_dry_days`, `humid_cloud`, `pressure_diff`, `temp_humid_interaction`",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Markdown report → {path}")


def final_summary(
    xgb: Dict[str, Any],
    lstm: Dict[str, Any],
    test_df: pd.DataFrame,
    cfg: PipelineConfig,
    pipeline_time: float,
):
    log.step("FINAL SUMMARY")

    sep = "═" * 66
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Overall comparison table ───────────────────────────────────────
    print(f"\n{sep}")
    print(f"  RAINFALL PREDICTION — FINAL COMPARISON")
    print(f"  Train: 2009–2020 | Test: {cfg.test_year} | Run: {ts}")
    print(f"  Target: rain (mm) via log1p | Features: {len(FEATURE_COLS)}")
    print(f"{sep}\n")

    metric_defs = [
        ("R² (log1p)",   "r2_log",    True),
        ("RMSE (log1p)", "rmse_log",  False),
        ("MAE (log1p)",  "mae_log",   False),
        ("R² (mm)",      "r2_mm",     True),
        ("RMSE (mm)",    "rmse_mm",   False),
        ("MAE (mm)",     "mae_mm",    False),
        ("Train time",   "train_time", False),
    ]

    print(f"  {'Metric':<22} {'XGBoost':>14} {'LSTM/GRU':>14} {'Winner':>10}")
    print("  " + "─" * 62)
    for label, key, higher in metric_defs:
        xv, lv = xgb[key], lstm[key]
        winner = "XGBoost" if (xv >= lv) == higher else "LSTM/GRU"
        if key == "train_time":
            print(f"  {label:<22} {xv:>12.1f}s {lv:>12.1f}s {winner:>10}")
        elif "mm" in key and "r2" not in key:
            print(f"  {label:<22} {xv:>14.2f} {lv:>14.2f} {winner:>10}")
        else:
            print(f"  {label:<22} {xv:>14.4f} {lv:>14.4f} {winner:>10}")

    best_r2    = max(xgb["r2_log"], lstm["r2_log"])
    best_model = "XGBoost" if xgb["r2_log"] >= lstm["r2_log"] else "LSTM/GRU"
    rating     = (
        "XUẤT SẮC"     if best_r2 >= 0.8 else
        "TỐT"          if best_r2 >= 0.6 else
        "TRUNG BÌNH"   if best_r2 >= 0.4 else
        "CẦN CẢI THIỆN"
    )
    print(f"\n  Best model: {best_model}  |  R² (log1p) = {best_r2:.4f}  |  Rating: {rating}")

    # ── 2. Per-region breakdown ────────────────────────────────────────────
    mapping      = pd.read_csv(cfg.mapping_path)
    region_names = dict(zip(mapping["region_encoded"], mapping["region"]))

    df_eval = test_df[["date", TARGET_ORIG_COL, "region_encoded", "province_encoded"]].copy()
    df_eval = df_eval.reset_index(drop=True)
    df_eval["xgb_pred"]  = xgb["predictions_mm"]
    lstm_preds = lstm["predictions_mm"]
    if len(lstm_preds) < len(df_eval):
        padded = np.full(len(df_eval), np.nan)
        padded[len(df_eval) - len(lstm_preds):] = lstm_preds
        df_eval["lstm_pred"] = padded
    else:
        df_eval["lstm_pred"] = lstm_preds[: len(df_eval)]
    df_eval["month"] = pd.to_datetime(df_eval["date"]).dt.month

    region_rows = []
    print(f"\n{sep}")
    print("  EVALUATION BY REGION (mm scale)")
    print(f"{sep}")
    print(f"  {'Region':<14} {'N':>6}  {'XGB MAE':>9} {'XGB R²':>8}  {'GRU MAE':>9} {'GRU R²':>8}")
    print("  " + "─" * 60)
    for code in sorted(df_eval["region_encoded"].unique()):
        mask  = df_eval["region_encoded"] == code
        y_t   = df_eval.loc[mask, TARGET_ORIG_COL].values
        y_xgb = df_eval.loc[mask, "xgb_pred"].values
        y_gru = df_eval.loc[mask, "lstm_pred"].values
        m_xgb = _segment_metrics(y_t, y_xgb)
        valid = ~np.isnan(y_gru)
        m_gru = _segment_metrics(y_t[valid], y_gru[valid])
        name  = region_names.get(int(code), f"Region {int(code)}")
        print(f"  {name:<14} {m_xgb['n']:>6}  "
              f"{m_xgb['mae']:>9.2f} {m_xgb['r2']:>8.4f}  "
              f"{m_gru['mae']:>9.2f} {m_gru['r2']:>8.4f}")
        region_rows.append({
            "region": name, "n": m_xgb["n"],
            "xgb_mae": m_xgb["mae"], "xgb_rmse": m_xgb["rmse"], "xgb_r2": m_xgb["r2"],
            "gru_mae": m_gru["mae"], "gru_rmse": m_gru["rmse"], "gru_r2": m_gru["r2"],
        })
    region_df = pd.DataFrame(region_rows)

    # ── 3. Per-month breakdown ─────────────────────────────────────────────
    month_rows = []
    print(f"\n{sep}")
    print("  EVALUATION BY MONTH (mm scale)")
    print(f"{sep}")
    print(f"  {'Month':>6} {'N':>6}  {'XGB MAE':>9} {'XGB R²':>8}  {'GRU MAE':>9} {'GRU R²':>8}")
    print("  " + "─" * 60)
    for mo in sorted(df_eval["month"].unique()):
        mask  = df_eval["month"] == mo
        y_t   = df_eval.loc[mask, TARGET_ORIG_COL].values
        y_xgb = df_eval.loc[mask, "xgb_pred"].values
        y_gru_raw = df_eval.loc[mask, "lstm_pred"].values
        m_xgb = _segment_metrics(y_t, y_xgb)
        valid = ~np.isnan(y_gru_raw)
        m_gru = _segment_metrics(y_t[valid], y_gru_raw[valid])
        print(f"  {mo:>6} {m_xgb['n']:>6}  "
              f"{m_xgb['mae']:>9.2f} {m_xgb['r2']:>8.4f}  "
              f"{m_gru['mae']:>9.2f} {m_gru['r2']:>8.4f}")
        month_rows.append({
            "month": int(mo), "n": m_xgb["n"],
            "xgb_mae": m_xgb["mae"], "xgb_rmse": m_xgb["rmse"], "xgb_r2": m_xgb["r2"],
            "gru_mae": m_gru["mae"], "gru_rmse": m_gru["rmse"], "gru_r2": m_gru["r2"],
        })
    month_df = pd.DataFrame(month_rows)

    # ── 4. Save all result files ───────────────────────────────────────────
    # 4a. Predictions CSV
    pred_path = cfg.output_dir / "predictions_2021.csv"
    df_eval.rename(columns={"lstm_pred": "gru_pred"}).to_csv(pred_path, index=False)

    # 4b. Feature importance CSV
    imp_path = cfg.output_dir / "feature_importance.csv"
    xgb["feature_importance"].to_csv(imp_path, index=False)

    # 4c. Region breakdown CSV
    region_path = cfg.output_dir / "eval_by_region.csv"
    region_df.to_csv(region_path, index=False)

    # 4d. Month breakdown CSV
    month_path = cfg.output_dir / "eval_by_month.csv"
    month_df.to_csv(month_path, index=False)

    # 4e. Metrics JSON (full report)
    metrics_report = {
        "run_timestamp": ts,
        "pipeline_time_s": round(pipeline_time, 1),
        "train_period": "2009-2020",
        "test_period":  f"{cfg.test_year}",
        "n_features":   len(FEATURE_COLS),
        "features":     FEATURE_COLS,
        "xgboost": {
            "r2_log1p":    round(xgb["r2_log"],  4),
            "rmse_log1p":  round(xgb["rmse_log"], 4),
            "mae_log1p":   round(xgb["mae_log"],  4),
            "r2_mm":       round(xgb["r2_mm"],    4),
            "rmse_mm":     round(xgb["rmse_mm"],  4),
            "mae_mm":      round(xgb["mae_mm"],   4),
            "cv_rmse":     xgb["cv_rmse"],
            "train_time_s": round(xgb["train_time"], 1),
        },
        "gru_lstm": {
            "r2_log1p":    round(lstm["r2_log"],  4),
            "rmse_log1p":  round(lstm["rmse_log"], 4),
            "mae_log1p":   round(lstm["mae_log"],  4),
            "r2_mm":       round(lstm["r2_mm"],    4),
            "rmse_mm":     round(lstm["rmse_mm"],  4),
            "mae_mm":      round(lstm["mae_mm"],   4),
            "epochs_run":       lstm["epochs_run"],
            "best_val_loss":    round(lstm["best_val_loss"], 6),
            "train_time_s":     round(lstm["train_time"], 1),
        },
        "best_model":  best_model,
        "rating":      rating,
    }
    metrics_path = cfg.output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)

    # 4f. Markdown report
    md_path = cfg.output_dir / "evaluation_report.md"
    _write_markdown_report(
        md_path, metrics_report, xgb, lstm,
        region_df, month_df, cfg, pipeline_time, ts,
    )

    # ── 5. Print saved files ───────────────────────────────────────────────
    print(f"\n{sep}")
    print("  OUTPUT FILES")
    print(f"{sep}")
    saved = [
        (pred_path,    "Predictions (date, actual, xgb_pred, gru_pred)"),
        (imp_path,     "Feature importance (XGBoost)"),
        (region_path,  "Evaluation by region"),
        (month_path,   "Evaluation by month"),
        (metrics_path, "Full metrics report (JSON)"),
        (md_path,      "Evaluation report (Markdown)"),
        (cfg.output_dir / "lstm_loss_curve.csv", "GRU training loss curve"),
        (cfg.output_dir / "weather_train_2009_2020.csv",    "Preprocessed train data (unscaled)"),
        (cfg.output_dir / "weather_test_2021.csv",          "Preprocessed test data (unscaled)"),
        (cfg.output_dir / "weather_train_2009_2020_scaled.csv", "Preprocessed train data (scaled)"),
        (cfg.output_dir / "weather_test_2021_scaled.csv",       "Preprocessed test data (scaled)"),
        (cfg.output_dir / "scalers.pkl",         "RobustScaler objects (joblib)"),
    ]
    for path, desc in saved:
        print(f"  ✓ {str(path):<52} {desc}")

    print(f"\n  Pipeline hoàn tất trong {pipeline_time:.1f}s")
    print(f"{sep}\n")

    return metrics_report


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    cfg = PipelineConfig()
    t_start = time.time()

    log.header("PIPELINE v3 — WEATHER PREPROCESSING & EVALUATION")

    # Step 1: Load & Clean
    df = load_and_clean(cfg)

    # Step 2: Feature Engineering
    df = engineer_features(df, cfg)

    # Step 3: Train/Test Split
    train_df, test_df = split_train_test(df, cfg)

    # Step 4: Select Features, Scale & Save
    train_unscaled, test_unscaled, feat_scaler, tgt_scaler = select_scale_save(
        train_df, test_df, cfg,
    )

    # Step 5: XGBoost (Tweedie loss — better for zero-inflated rainfall)
    xgb_results = evaluate_xgboost(train_unscaled, test_unscaled, cfg)

    # Step 6: LSTM (GRU variant)
    lstm_results = evaluate_lstm(train_unscaled, test_unscaled, feat_scaler, tgt_scaler, cfg)

    # Step 7: Final Summary
    pipeline_time = time.time() - t_start
    summary = final_summary(xgb_results, lstm_results, test_unscaled, cfg, pipeline_time)

    return summary


if __name__ == "__main__":
    main()
