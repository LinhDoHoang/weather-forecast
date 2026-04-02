from __future__ import annotations

from datetime import date
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Query, status

from api.schemas import (
    HealthData,
    HealthResponse,
    PaginatedPredictionResponse,
    PredictionCompareRecord,
    PredictionCompareResponse,
    PredictionPage,
    PredictionRecord,
    PredictionResponse,
)
from api.store import WeatherPredictionStore


app = FastAPI(
    title="Weather Forecast API",
    version="1.0.0",
    description=(
        "FastAPI service that serves rainfall predictions from the best-performing XGBoost model. "
        "The API reads a precomputed prediction table at startup and exposes lookup, comparison, "
        "and paginated list endpoints for the frontend."
    ),
    openapi_tags=[
        {"name": "Health", "description": "Service readiness and data-loading status."},
        {"name": "Weather Predictions", "description": "Prediction lookup, comparison, and pagination endpoints."},
    ],
    contact={"name": "Weather Forecast Team"},
)


@lru_cache(maxsize=1)
def get_store() -> WeatherPredictionStore:
    return WeatherPredictionStore.load()


@app.on_event("startup")
def warm_up_store() -> None:
    get_store()


def store_dependency() -> WeatherPredictionStore:
    return get_store()


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Check service health",
    description="Returns a lightweight status payload and confirms that the prediction table is loaded in memory.",
)
def health(store: WeatherPredictionStore = Depends(store_dependency)) -> HealthResponse:
    return HealthResponse(
        success=True,
        message="Service is ready.",
        data=HealthData(
            status="ok",
            source_file=str(store.source_path),
            total_records=store.total_records(),
        ),
    )


@app.get(
    "/predictions/by-date-province",
    response_model=PredictionResponse,
    tags=["Weather Predictions"],
    summary="Get prediction by date and province code",
    description=(
        "Return the XGBoost prediction for a specific date and province code. "
        "This endpoint is designed for FE lookup use cases where the UI already knows the exact key."
    ),
)
def get_prediction_by_date_and_province(
    date_value: date = Query(..., alias="date", description="Forecast date in YYYY-MM-DD format."),
    province_code: int = Query(..., ge=0, description="Province code from the repository mapping."),
    store: WeatherPredictionStore = Depends(store_dependency),
) -> PredictionResponse:
    record = store.get_by_date_and_province(date_value, province_code)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction found for date={date_value.isoformat()} and province_code={province_code}",
        )
    return PredictionResponse(
        success=True,
        message="Prediction found.",
        data=PredictionRecord(**record),
    )


@app.get(
    "/predictions/compare",
    response_model=PredictionCompareResponse,
    tags=["Weather Predictions"],
    summary="Compare prediction with actual rainfall",
    description=(
        "Return both the XGBoost prediction and the actual rainfall for a given date and province code. "
        "The response also includes signed and absolute error values to simplify frontend comparison UI."
    ),
)
def compare_prediction_and_actual(
    date_value: date = Query(..., alias="date", description="Forecast date in YYYY-MM-DD format."),
    province_code: int = Query(..., ge=0, description="Province code from the repository mapping."),
    store: WeatherPredictionStore = Depends(store_dependency),
) -> PredictionCompareResponse:
    record = store.compare(date_value, province_code)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No record found for date={date_value.isoformat()} and province_code={province_code}",
        )
    return PredictionCompareResponse(
        success=True,
        message="Prediction and actual values found.",
        data=PredictionCompareRecord(**record),
    )


@app.get(
    "/predictions",
    response_model=PaginatedPredictionResponse,
    tags=["Weather Predictions"],
    summary="Get all predictions with pagination",
    description=(
        "Return a paginated list of prediction records. The result can be optionally filtered by province code. "
        "Pagination uses 1-based page numbering."
    ),
)
def get_all_predictions(
    page: int = Query(1, ge=1, description="Page number, starting from 1."),
    limit: int = Query(20, ge=1, le=200, description="Number of records per page."),
    province_code: int | None = Query(None, ge=0, description="Optional province code filter."),
    store: WeatherPredictionStore = Depends(store_dependency),
) -> PaginatedPredictionResponse:
    payload = store.list_records(page=page, limit=limit, province_code=province_code)
    return PaginatedPredictionResponse(
        success=True,
        message="Prediction list returned.",
        data=PredictionPage(**payload),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
