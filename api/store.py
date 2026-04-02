from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import ceil
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


DEFAULT_SOURCE_PATH = Path("output/20260403_084235/predictions_all_models_2021.csv")
DEFAULT_XGBOOST_ONLY_PATH = Path("output/20260403_084235/predictions_xgboost_2021.csv")
DEFAULT_MAPPING_PATH = Path("dataset/province_region_code_mapping.csv")
DEFAULT_GROUND_TRUTH_PATH = Path("input/weather_test_2021.csv")


def _load_base_frame(source_path: Path) -> pd.DataFrame:
    if source_path.exists():
        return pd.read_csv(source_path)

    if not DEFAULT_XGBOOST_ONLY_PATH.exists() or not DEFAULT_GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            "Cannot find prediction source files. Expected either the combined "
            f"file at {source_path} or the fallback pair {DEFAULT_XGBOOST_ONLY_PATH} "
            f"and {DEFAULT_GROUND_TRUTH_PATH}."
        )

    xgb_df = pd.read_csv(DEFAULT_XGBOOST_ONLY_PATH)
    truth_df = pd.read_csv(DEFAULT_GROUND_TRUTH_PATH)
    return truth_df.merge(
        xgb_df[["date", "province", "region", "xgboost_pred_mm"]],
        on=["date", "province", "region"],
        how="left",
    )


def _ensure_code_columns(frame: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(mapping_path)

    if "province_encoded" not in frame.columns:
        frame = frame.merge(mapping[["province", "province_encoded"]], on="province", how="left")

    if "region_encoded" not in frame.columns:
        frame = frame.merge(mapping[["province", "region_encoded"]], on="province", how="left")

    if "region" not in frame.columns:
        frame = frame.merge(mapping[["province", "region"]], on="province", how="left")

    return frame


def _normalize_frame(frame: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    frame = _ensure_code_columns(frame, mapping_path).copy()

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date", "province", "province_encoded", "xgboost_pred_mm", "rain"])

    frame["province_encoded"] = frame["province_encoded"].astype(int)
    if "region_encoded" in frame.columns:
        frame["region_encoded"] = frame["region_encoded"].astype(int)

    frame = frame.rename(
        columns={
            "province": "province_name",
            "region": "region_name",
            "province_encoded": "province_code",
            "rain": "actual_rain_mm",
            "rain_log1p": "actual_rain_log1p",
            "xgboost_pred_mm": "predicted_rain_mm",
        }
    )

    keep_columns = [
        "date",
        "province_code",
        "province_name",
        "region_name",
        "actual_rain_mm",
        "predicted_rain_mm",
    ]
    if "actual_rain_log1p" in frame.columns:
        keep_columns.append("actual_rain_log1p")
    if "region_encoded" in frame.columns:
        keep_columns.append("region_encoded")

    normalized = frame[keep_columns].sort_values(["date", "province_code"]).reset_index(drop=True)
    return normalized


@dataclass(slots=True)
class WeatherPredictionStore:
    frame: pd.DataFrame
    source_path: Path
    mapping_path: Path

    @classmethod
    def load(
        cls,
        source_path: Path = DEFAULT_SOURCE_PATH,
        mapping_path: Path = DEFAULT_MAPPING_PATH,
    ) -> "WeatherPredictionStore":
        base_frame = _load_base_frame(source_path)
        frame = _normalize_frame(base_frame, mapping_path)
        return cls(frame=frame, source_path=source_path, mapping_path=mapping_path)

    def total_records(self) -> int:
        return int(len(self.frame))

    def _row_to_record(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "date": row["date"].date(),
            "province_code": int(row["province_code"]),
            "province_name": str(row["province_name"]),
            "region_name": str(row["region_name"]),
            "actual_rain_mm": float(row["actual_rain_mm"]),
            "predicted_rain_mm": float(row["predicted_rain_mm"]),
        }

    def get_by_date_and_province(self, query_date: date, province_code: int) -> Optional[Dict[str, Any]]:
        date_value = pd.Timestamp(query_date).normalize()
        mask = (self.frame["date"] == date_value) & (self.frame["province_code"] == province_code)
        if not mask.any():
            return None

        row = self.frame.loc[mask].iloc[0]
        return self._row_to_record(row)

    def compare(self, query_date: date, province_code: int) -> Optional[Dict[str, Any]]:
        record = self.get_by_date_and_province(query_date, province_code)
        if record is None:
            return None

        error_mm = record["predicted_rain_mm"] - record["actual_rain_mm"]
        record["error_mm"] = float(error_mm)
        record["absolute_error_mm"] = float(abs(error_mm))
        return record

    def list_records(
        self,
        page: int,
        limit: int,
        province_code: Optional[int] = None,
    ) -> Dict[str, Any]:
        frame = self.frame
        if province_code is not None:
            frame = frame[frame["province_code"] == province_code]

        total = int(len(frame))
        total_pages = ceil(total / limit) if total else 0
        start = (page - 1) * limit
        end = start + limit
        page_frame = frame.iloc[start:end]

        return {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": total_pages,
            "province_code": province_code,
            "items": [self._row_to_record(row) for _, row in page_frame.iterrows()],
        }
