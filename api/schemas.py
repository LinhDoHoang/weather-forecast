from __future__ import annotations

from datetime import date as Date
from typing import List, Optional

from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    success: bool = Field(True, description="Whether the request succeeded.", examples=[True])
    message: Optional[str] = Field(default=None, description="Optional human-readable message.")


class PredictionRecord(BaseModel):
    date: Date = Field(..., description="Forecast date in YYYY-MM-DD format.", examples=["2021-05-24"])
    province_code: int = Field(..., ge=0, description="Province code from the repository mapping.", examples=[13])
    province_name: str = Field(..., description="Province name.", examples=["Ho Chi Minh City"])
    region_name: str = Field(..., description="Region name.", examples=["Nam Bo"])
    actual_rain_mm: float = Field(..., description="Observed rainfall in millimeters.", examples=[12.4])
    predicted_rain_mm: float = Field(..., description="XGBoost predicted rainfall in millimeters.", examples=[11.935012])


class PredictionCompareRecord(PredictionRecord):
    error_mm: float = Field(..., description="Signed prediction error: predicted minus actual.", examples=[-0.465])
    absolute_error_mm: float = Field(..., description="Absolute prediction error in millimeters.", examples=[0.465])


class PredictionPage(BaseModel):
    page: int = Field(..., description="Current page number, starting from 1.", examples=[1])
    limit: int = Field(..., description="Maximum records per page.", examples=[20])
    total: int = Field(..., description="Total records after filtering.", examples=[6592])
    total_pages: int = Field(..., description="Total available pages after filtering.", examples=[330])
    province_code: Optional[int] = Field(
        default=None,
        description="Optional province filter used for the current query.",
        examples=[13],
    )
    items: List[PredictionRecord] = Field(..., description="Page of prediction records.")


class HealthData(BaseModel):
    status: str = Field(..., description="Service status.", examples=["ok"])
    source_file: str = Field(..., description="Canonical CSV used to serve predictions.")
    total_records: int = Field(..., description="Number of records loaded into memory.", examples=[6592])


class HealthResponse(ApiResponse):
    data: HealthData


class PredictionResponse(ApiResponse):
    data: PredictionRecord


class PredictionCompareResponse(ApiResponse):
    data: PredictionCompareRecord


class PaginatedPredictionResponse(ApiResponse):
    data: PredictionPage
