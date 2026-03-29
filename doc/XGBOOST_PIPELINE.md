# XGBoost Pipeline

## 1) Mục tiêu
Tài liệu này mô tả chi tiết pipeline XGBoost được dùng để train, evaluate, lưu artifact và inference dữ liệu mới cho bài toán dự báo thời tiết dạng regression.

## 2) Tổng quan kiến trúc pipeline
File thực thi chính: `weather_xgboost_pipeline.py`.

Pipeline được thiết kế theo hướng production-ready với các thành phần tách hàm rõ ràng:
- `load_data`: đọc train/test CSV.
- `preprocess_data`: xử lý datetime, missing, canh cột feature train-test, tạo artifact tiền xử lý.
- `split_data`: chia tập train/validation theo thứ tự thời gian (không shuffle).
- `build_model`: khởi tạo `XGBRegressor` theo config.
- `train_model`: fit model trên train và theo dõi với validation.
- `predict`: sinh dự đoán cho validation/test/new data.
- `evaluate_model`: tính MAE, MSE, RMSE, R2.
- `save_model` / `load_model`: lưu và đọc model joblib.
- `run_training_pipeline`: orchestration toàn bộ train pipeline.
- `run_inference` (và alias `predict_new_samples`): dự đoán mẫu mới bằng model đã lưu.

## 3) Luồng xử lý chi tiết

### 3.1 Input và cấu hình
Tất cả cấu hình được gom trong `PipelineConfig`:
- `train_data_path`, `test_data_path`
- `target_column`
- `validation_size`, `random_seed`
- `output_base_dir`
- `model_params` cho XGBoost

Mặc định đang dùng dữ liệu scaled trong:
- `dataset/test_train_scaled/weather_train_2009_2020_scaled.csv`
- `dataset/test_train_scaled/weather_test_2021_scaled.csv`

### 3.2 Tiền xử lý
Trong `preprocess_data`, pipeline thực hiện:
- Kiểm tra tồn tại target ở cả train và test.
- Tự động phát hiện cột datetime (đặc biệt cột có chữ `date`).
- Chuyển datetime sang ordinal feature (`*_ordinal`) an toàn.
- Tách `X`/`y`, one-hot nếu có categorical còn sót.
- Canh cột train/test bằng `reindex` để đảm bảo cùng schema.
- Xử lý giá trị vô cùng thành NaN.
- Điền missing của feature bằng median train; missing còn lại -> 0.0.
- Điền missing của target bằng median train target.

Đồng thời tạo `PreprocessArtifacts` gồm:
- tên target
- danh sách cột datetime đã xử lý
- danh sách feature sau cùng
- median values để tái sử dụng lúc inference

### 3.3 Chia tập train/validation
`split_data` chia theo thứ tự thời gian:
- train = phần đầu
- validation = phần cuối theo `validation_size`

Thiết kế này phù hợp dữ liệu time-based và tránh leakage do shuffle.

### 3.4 Train và đánh giá
`build_model` dùng `XGBRegressor` với `objective=reg:squarederror`.

`train_model`:
- fit trên train
- theo dõi train/validation qua `eval_set`

`evaluate_model` tính và trả về:
- MAE
- MSE
- RMSE
- R2

### 3.5 Export artifact
Mỗi lần chạy sẽ tạo runtime folder theo timestamp:
- `output/<timestamp>/model`
- `output/<timestamp>/plots`
- `output/<timestamp>/reports`

Nội dung export:
- Model: `xgb_weather_model.joblib`
- Preprocess artifact: `preprocessing_artifacts.joblib`
- Plot:
  - `actual_vs_predicted_test.png`
  - `residual_distribution_test.png`
  - `feature_importance.png`
- Report:
  - `validation_predictions.csv`
  - `test_predictions.csv`
  - `metrics.csv`
  - `results.xlsx`
  - `metrics.json`
- Cấu hình runtime:
  - `config_used.json`

### 3.6 Inference cho dữ liệu mới
`run_inference` thực hiện:
- load model + preprocess artifacts
- đọc file CSV mẫu mới
- áp dụng preprocessing y hệt train-time (datetime, schema alignment, fillna)
- dự đoán và ghi kết quả ra:
  - `output/<timestamp>/inference/inference_predictions.csv`
  - `output/<timestamp>/inference/inference_predictions.xlsx`

## 4) Cách chạy (bắt buộc trong môi trường ảo .venv)

### 4.1 Cài dependency
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4.2 Train
```powershell
.\.venv\Scripts\python.exe .\weather_xgboost_pipeline.py
```

### 4.3 Inference
```powershell
.\.venv\Scripts\python.exe .\weather_xgboost_pipeline.py --mode infer --new-data-path <duong_dan_csv_moi> --model-path <duong_dan_model_joblib> --artifacts-path <duong_dan_artifacts_joblib>
```

## 5 Lưu ý vận hành
- Target mặc định: `max_temp` (có thể đổi bằng `--target`).
- Dữ liệu train/test hiện tại đang là dữ liệu scaled.
- Nếu muốn giữ tính thời gian chặt chẽ hơn nữa, có thể thêm backtesting theo block thời gian ở bản tiếp theo.
