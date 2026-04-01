# 03. Hướng Dẫn Chạy FastAPI Application

## Tổng Quan

FastAPI application phục vụ dự báo mưa từ **mô hình XGBoost** (mô hình chính xác nhất). Application:

- Load predictions precomputed từ CSV vào bộ nhớ khi startup
- Expose 3 API endpoints chính (+ 1 health check)
- Auto-generate Swagger documentation
- Hỗ trợ pagination, filtering, so sánh actual vs predicted

---

## I. PROJECT STRUCTURE

```
weather-forecast/
├── api/                           # ← Production package
│   ├── __init__.py
│   ├── main.py                    # ← Canonical entrypoint (api.main:app)
│   ├── schemas.py                 # ← Pydantic models, Swagger docs
│   └── store.py                   # ← CSV loading, in-memory store
│
├── app.py                         # ← Compatibility shim (re-exports api.main:app)
├── schemas.py                     # ← Compatibility shim
├── weather_store.py               # ← Compatibility shim
│
├── requirements.txt               # ← Dependencies (FastAPI, Uvicorn, etc.)
├── main.py                        # ← ML pipeline entrypoint (training only)
├── pipeline_v3.py                 # ← Training code
│
├── output/v4_20260403_073740/
│   ├── predictions_xgboost_2021.csv    # ← CSV source for API
│   └── ...
│
├── input/
│   └── weather_test_2021.csv      # ← Actual rainfall (ground truth)
│
├── dataset/
│   └── province_region_code_mapping.csv  # ← Province codes
│
└── doc/
    └── 03_API_FASTAPI_GUIDE.md     # ← This file
```

---

## II. DEPENDENCIES

### 2.1 Requirements

File `requirements.txt` chứa các dependencies:

```
fastapi>=0.115.0,<1.0.0
uvicorn[standard]>=0.30.0,<1.0.0
```

### 2.2 Cài Đặt

```bash
# Option 1: Cài tất cả dependencies từ requirements.txt
pip install -r requirements.txt

# Option 2: Cài riêng
pip install fastapi uvicorn
```

---

## III. ARCHITECTURE

### 3.1 Canonical Entrypoint

```python
# api/main.py
from fastapi import FastAPI

app = FastAPI(
    title="Weather Forecast API",
    version="1.0.0",
    description="FastAPI service that serves rainfall predictions from XGBoost model.",
)
```

**Production usage:**

```
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3.2 Data Flow

```
┌────────────────────────────────────────┐
│ Startup Event (@app.on_event("startup"))
└────────────────────────────────────────┘
         ↓
    get_store()  ← Lazy init, cached with @lru_cache
         ↓
    WeatherPredictionStore.load()
         ↓
    Load 3 CSV files into pandas in-memory:
    1. predictions_xgboost_2021.csv
    2. weather_test_2021.csv (actual)
    3. province_region_code_mapping.csv
         ↓
    Normalize & merge into single in-memory table  ← Fast lookups
         ↓
    Ready for requests
```

### 3.3 Request Processing

```
HTTP Request (GET /predictions/by-date-province?date=2021-05-24&province_code=13)
         ↓
FastAPI Query params validation
         ↓
Dependency injection: store_dependency() → WeatherPredictionStore
         ↓
Logic: store.get_by_date_and_province(date, province_code)
         ↓
Response (Pydantic model validation & serialization)
         ↓
HTTP 200 + JSON
```

---

## IV. CHẠY SERVER

### 4.1 Development Mode (với reload)

```bash
# Kích hoạt venv
.\venv\Scripts\Activate.ps1

# Chạy server (tự reload khi code thay đổi)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Output:**

```
INFO:     Will watch for changes in these directories: ['D:\\Download\\252\\ml\\source\\weather-forecast']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started server process [12345]
INFO:     Application startup complete
```

### 4.2 Production Mode (no reload)

```bash
# Chạy server (không reload, ổn định hơn)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Cách hiểu:**

- `--host 0.0.0.0` = Listen trên tất cả interfaces (localhost + network)
- `--port 8000` = Port mặc định
- `--reload` = Tự restart khi code thay đổi (development only)
- `--workers 4` = Chạy 4 worker processes (production)

### 4.3 Custom Port

```bash
# Đổi port từ 8000 sang 8001
uvicorn api.main:app --port 8001
```

### 4.4 Chạy từ Python Script

```python
# api/main.py có main block:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
```

**Chạy:**

```bash
python api/main.py
# hoặc
python -m api.main
```

---

## V. API ENDPOINTS

### 5.1 Health Check

**Endpoint:** `GET /health`

**Purpose:** Kiểm tra service có sẵn sàng không, xác nhận predictions loaded.

**Response:**

```json
{
  "success": true,
  "message": "Service is ready.",
  "data": {
    "status": "ok",
    "source_file": "output/v4_20260403_073740/predictions_xgboost_2021.csv",
    "total_records": 6592
  }
}
```

**HTTP:** 200 OK

**Curl:**

```bash
curl http://127.0.0.1:8000/health
```

---

### 5.2 Get Prediction by Date + Province

**Endpoint:** `GET /predictions/by-date-province`

**Query Parameters:**

| Param         | Type              | Required | Description              | Example    |
| ------------- | ----------------- | -------- | ------------------------ | ---------- |
| date          | string (ISO date) | ✓        | Forecast date YYYY-MM-DD | 2021-05-24 |
| province_code | integer           | ✓        | Province code            | 13         |

**Response (Success → 200 OK):**

```json
{
  "success": true,
  "message": "Prediction found.",
  "data": {
    "date": "2021-05-24",
    "province_code": 13,
    "province_name": "Ho Chi Minh",
    "region_name": "Nam Bo",
    "actual_rain_mm": 12.4,
    "predicted_rain_mm": 11.935012
  }
}
```

**Response (Not Found → 404 Not Found):**

```json
{
  "success": false,
  "detail": "No prediction found for date=2021-05-24 and province_code=999"
}
```

**Curl:**

```bash
# Dự báo mưa tại TP. HCM (province_code=13) ngày 2021-05-24
curl "http://127.0.0.1:8000/predictions/by-date-province?date=2021-05-24&province_code=13"
```

---

### 5.3 Compare Prediction vs Actual

**Endpoint:** `GET /predictions/compare`

**Query Parameters:** Tương tự `/by-date-province`

**Response (Success → 200 OK):**

```json
{
  "success": true,
  "message": "Prediction and actual values found.",
  "data": {
    "date": "2021-05-24",
    "province_code": 13,
    "province_name": "Ho Chi Minh",
    "region_name": "Nam Bo",
    "actual_rain_mm": 12.4,
    "predicted_rain_mm": 11.935012,
    "error_mm": -0.464988,
    "absolute_error_mm": 0.464988
  }
}
```

**Chi Tiết:**

- `error_mm` = predicted - actual (negative = under-predict)
- `absolute_error_mm` = |error_mm| (dùng để measure accuracy)

**Curl:**

```bash
curl "http://127.0.0.1:8000/predictions/compare?date=2021-05-24&province_code=13"
```

---

### 5.4 Get All Predictions (Paginated)

**Endpoint:** `GET /predictions`

**Query Parameters:**

| Param         | Type    | Required | Default | Range | Description                 |
| ------------- | ------- | -------- | ------- | ----- | --------------------------- |
| page          | integer | ✗        | 1       | ≥1    | Page number (1-indexed)     |
| limit         | integer | ✗        | 20      | 1-200 | Records per page            |
| province_code | integer | ✗        | null    | ≥0    | Optional filter by province |

**Response (Success → 200 OK):**

```json
{
  "success": true,
  "message": "Prediction list returned.",
  "data": {
    "page": 1,
    "limit": 20,
    "total": 6592,
    "total_pages": 330,
    "province_code": null,
    "items": [
      {
        "date": "2021-01-01",
        "province_code": 1,
        "province_name": "Ha Noi",
        "region_name": "Bac Bo",
        "actual_rain_mm": 0.0,
        "predicted_rain_mm": 0.123
      },
      {...},
      {...}
    ]
  }
}
```

**Example Curl:**

```bash
# Page 1, 20 records per page
curl "http://127.0.0.1:8000/predictions?page=1&limit=20"

# Page 5, filter by TP. HCM (province_code=13)
curl "http://127.0.0.1:8000/predictions?page=5&limit=20&province_code=13"

# Get 50 records
curl "http://127.0.0.1:8000/predictions?limit=50"
```

**Pagination Logic:**

```python
start_index = (page - 1) * limit
end_index = start_index + limit

# Ví dụ:
# page=1, limit=20 → index 0-19
# page=2, limit=20 → index 20-39
# page=3, limit=20 → index 40-59
```

**Thông Tin:**

- Nếu `province_code=null` → trả về ALL provinces
- Nếu `province_code=13` → filter chỉ TP. HCM
- Total pages = ceil(total_records / limit)

---

## VI. RESPONSE CONTRACT

### 6.1 Envelope Pattern

Tất cả responses tuân theo envelope pattern:

```json
{
  "success": true|false,
  "message": "optional message",
  "data": {...}  // payload - có thể là object hoặc list
}
```

**Ý nghĩa:**

- `success` — Yêu cầu có thành công không
- `message` — Thông báo cho người dùng (ví dụ: "Prediction found.")
- `data` — Nội dung (tùy endpoint)

### 6.2 Error Response (4xx/5xx)

```json
{
  "detail": "No prediction found for date=2021-05-24 and province_code=999"
}
```

**HTTP Status Codes:**
| Code | Tình Huống |
|------|-----------|
| 200 | ✓ Success |
| 404 | ✗ Not found (prediction/record không tồn tại) |
| 422 | ✗ Validation error (invalid query params) |
| 500 | ✗ Internal server error |

---

## VII. SWAGGER / OPENAPI DOCUMENTATION

### 7.1 Truy Cập Swagger

**URL:** `http://127.0.0.1:8000/docs`

**Tính năng:**

- Interactive API explorer
- Try it out — chạy request trực tiếp từ browser
- Auto-generated từ Pydantic models

**Ví dụ:**

```
GET /predictions/by-date-province
├─ Parameters
│  ├─ date (query) — string, format: "2021-05-24"
│  └─ province_code (query) — integer, minimum: 0
├─ Responses
│  ├─ 200 OK — success response
│  └─ 404 Not Found — not found
└─ Try it out button
```

### 7.2 ReDoc (Alternative Documentation)

**URL:** `http://127.0.0.1:8000/redoc`

**Tính năng:**

- Formal API documentation (read-only)
- Search function
- Tốt hơn Swagger cho testing

---

## VIII. SCHEMA DEFINITIONS

### 8.1 PredictionRecord

```python
class PredictionRecord(BaseModel):
    date: Date
    province_code: int
    province_name: str
    region_name: str
    actual_rain_mm: float
    predicted_rain_mm: float
```

**Ví dụ:**

```json
{
  "date": "2021-05-24",
  "province_code": 13,
  "province_name": "Ho Chi Minh",
  "region_name": "Nam Bo",
  "actual_rain_mm": 12.4,
  "predicted_rain_mm": 11.935012
}
```

### 8.2 PredictionCompareRecord

Extends `PredictionRecord`:

```python
class PredictionCompareRecord(PredictionRecord):
    error_mm: float              # predicted - actual
    absolute_error_mm: float     # |error_mm|
```

### 8.3 PredictionPage (Paginated)

```python
class PredictionPage(BaseModel):
    page: int
    limit: int
    total: int
    total_pages: int
    province_code: Optional[int]
    items: List[PredictionRecord]
```

### 8.4 HealthData

```python
class HealthData(BaseModel):
    status: str                  # "ok"
    source_file: str             # CSV path
    total_records: int           # Number of records in memory
```

---

## IX. DATA SOURCE & LOADING

### 9.1 CSV Files Used

| File                | Path                                                   | Purpose             | Rows   |
| ------------------- | ------------------------------------------------------ | ------------------- | ------ |
| XGBoost Predictions | output/v4_20260403_073740/predictions_xgboost_2021.csv | Dự báo XGBoost      | 6,592  |
| Actual Rainfall     | input/weather_test_2021.csv                            | Ground truth        | 11,466 |
| Province Mapping    | dataset/province_region_code_mapping.csv               | Tỉnh → vùng mapping | 63     |

### 9.2 Normalization Logic

```python
# 1. Load XGBoost predictions
xgb_df = pd.read_csv("output/.../predictions_xgboost_2021.csv")

# 2. Load actual rainfall
actual_df = pd.read_csv("input/weather_test_2021.csv")

# 3. Merge on (date, province, region)
merged = actual_df.merge(
    xgb_df[["date", "province", "region", "xgboost_pred_mm"]],
    on=["date", "province", "region"],
    how="left"
)

# 4. Add codes (province_encoded, region_encoded) từ mapping
mapping = pd.read_csv("dataset/province_region_code_mapping.csv")
merged = merged.merge(mapping, on="province")

# 5. Rename để match API contract
merged = merged.rename(columns={
    "province": "province_name",
    "region": "region_name",
    "province_encoded": "province_code",
    "rain": "actual_rain_mm",  # from actual_df
    "xgboost_pred_mm": "predicted_rain_mm",  # from xgb_df
})

# 6. In-memory table (sorted by date, province_code)
final = merged.sort_values(["date", "province_code"]).reset_index(drop=True)
```

---

## X. PERFORMANCE & BENCHMARKS

### 10.1 Memory Usage

```
Total records: 6,592
Columns per record: 6 (date, province_code, province_name, region_name, actual_rain_mm, predicted_rain_mm)
Memory footprint: ~2-3 MB (DataFrame)
Startup time: ~1-2 seconds (CSV load + parsing)
```

### 10.2 Latency (Response Time)

| Endpoint                      | Query Type                | Latency  | Notes                          |
| ----------------------------- | ------------------------- | -------- | ------------------------------ |
| /health                       | Simple stats              | 1-2 ms   | Chỉ in-memory lookup           |
| /predictions/by-date-province | Exact match               | 1-5 ms   | Index lookup                   |
| /predictions/compare          | Exact match + calculation | 2-5 ms   | Same as above                  |
| /predictions (page 1)         | Paginated, unfiltered     | 5-10 ms  | Slice + serialization          |
| /predictions (filter)         | Paginated, filtered       | 10-20 ms | Filter + slice + serialization |

### 10.3 Load Testing

Để test performance (optional):

```bash
# Cài locust
pip install locust

# Tạo locustfile.py với requests, chạy:
locust -f locustfile.py --host=http://127.0.0.1:8000
```

---

## XI. TROUBLESHOOTING

### 11.1 Port Already in Use

```bash
# Error: Address already in use
# Solution 1: Dùng port khác
uvicorn api.main:app --port 8001

# Solution 2: Kill process trên port 8000
# Windows PowerShell:
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process
```

### 11.2 CSV Files Not Found

```
FileNotFoundError: Cannot find prediction source files.
```

**Nguyên nhân:** Paths không đúng (relative vs. absolute).

**Fix:**

```bash
# Run từ workspace root
cd d:\Download\252\ml\source\weather-forecast
uvicorn api.main:app
```

### 11.3 Type Error (Pydantic)

```
TypeError: Unable to evaluate type annotation 'date'.
```

**Nguyên nhân:** Python 3.9 compatibility issue với Pydantic v2.

**Fix:** (Đã fixed) Import `datetime.date` as `Date`:

```python
from datetime import date as Date

class PredictionRecord(BaseModel):
    date: Date  # ← Tránh naming conflict
```

### 11.4 502 Bad Gateway (Nginx/Reverse Proxy)

```
502 Bad Gateway
```

**Nguyên nhân:** Worker crashed, không có process running.

**Fix:**

```bash
# Restart server
uvicorn api.main:app --workers 4 --access-log
```

---

## XII. DEPLOYMENT (Optional)

### 12.1 Production Checklist

- [ ] Disable `--reload`
- [ ] Set `workers > 1` (multiprocessing)
- [ ] Enable `--access-log` (logging)
- [ ] Use reverse proxy (Nginx)
- [ ] SSL/TLS (HTTPS)
- [ ] Environment variables (`.env`)

### 12.2 Docker (Optional)

```dockerfile
# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build & Run
docker build -t weather-api .
docker run -p 8000:8000 weather-api
```

---

## XIII. QUICK START

### 13.1 5-Minute Setup

```bash
# 1. Kích hoạt venv
.\venv\Scripts\Activate.ps1

# 2. Cài dependencies (nếu chưa cài)
pip install -r requirements.txt

# 3. Chạy server
uvicorn api.main:app --port 8000 --reload

# 4. Test endpoint (mở browser hoặc curl)
# Health check
curl http://127.0.0.1:8000/health

# Get prediction
curl "http://127.0.0.1:8000/predictions/by-date-province?date=2021-05-24&province_code=13"

# 5. Xem Swagger docs
# Mở browser: http://127.0.0.1:8000/docs
```

### 13.2 Test Python Client

```python
import requests

BASE_URL = "http://127.0.0.1:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Get prediction
response = requests.get(
    f"{BASE_URL}/predictions/by-date-province",
    params={"date": "2021-05-24", "province_code": 13}
)
print(response.json())

# Get paginated list
response = requests.get(
    f"{BASE_URL}/predictions",
    params={"page": 1, "limit": 20}
)
print(response.json())
```

---

## XIV. COMMONLY USED COMMANDS

```bash
# Start development server (with reload)
uvicorn api.main:app --reload

# Start production server (no reload, 4 workers)
uvicorn api.main:app --workers 4

# Custom host & port
uvicorn api.main:app --host 127.0.0.1 --port 9000

# With logging
uvicorn api.main:app --access-log --log-level debug

# Run from python
python api/main.py
```

---

## XV. API USAGE EXAMPLES

### 15.1 Frontend Integration (JavaScript)

```javascript
// Fetch prediction via API
fetch(
  "http://127.0.0.1:8000/predictions/by-date-province?date=2021-05-24&province_code=13",
)
  .then((response) => response.json())
  .then((data) => {
    if (data.success) {
      console.log(`Dự báo: ${data.data.predicted_rain_mm} mm`);
      console.log(`Thực tế: ${data.data.actual_rain_mm} mm`);
      const error = Math.abs(
        data.data.predicted_rain_mm - data.data.actual_rain_mm,
      );
      console.log(`Sai số: ${error.toFixed(2)} mm`);
    }
  });
```

### 15.2 Batch Processing (Python)

```python
import requests
import pandas as pd

BASE_URL = "http://127.0.0.1:8000"

# Get all predictions (paginated)
all_records = []
page = 1
while True:
    response = requests.get(
        f"{BASE_URL}/predictions",
        params={"page": page, "limit": 100}
    )
    data = response.json()
    if not data['data']['items']:
        break
    all_records.extend(data['data']['items'])
    page += 1

# Convert to pandas
df = pd.DataFrame(all_records)
df['error_mm'] = abs(df['predicted_rain_mm'] - df['actual_rain_mm'])
print(df[['date', 'province_name', 'error_mm']].head(10))
```

---

## XVI. Support & Troubleshooting Links

- FastAPI Docs: https://fastapi.tiangolo.com
- Uvicorn Docs: https://www.uvicorn.org
- Pydantic Docs: https://docs.pydantic.dev
- HTTP Status Codes: https://http.cat

---

**Happy forecasting! 🌧️**
