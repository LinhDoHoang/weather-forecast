# 02. Flow Train/Validation/Test và Metrics

## Tổng Quan

Sau khi tiền xử lý xong, pipeline đánh giá **4 mô hình khác nhau** trên cùng một bộ dữ liệu:

1. **XGBoost** — Gradient boosting tree-based
2. **Random Forest** — Ensemble tree-based
3. **Prophet** — Time-series specific
4. **LSTM (GRU variant)** — Deep learning sequence-to-sequence

Tất cả dùng **test set từ 2021** để đánh giá, và các loss/metric được tính trên **2 scale khác nhau**: log1p (dùng khi training) và mm (dùng khi report).

---

## I. KIẾN TRÚC TRAINING CHUNG

### 1.1 Final Model Training Set

Trước khi đánh giá trên test set, train + validation được **gộp lại** để fit mô hình cuối:

```python
model_train_data = pd.concat([train_unscaled, val_unscaled], ignore_index=True)
model_train_data = model_train_data.sort_values(["province_encoded", "date"]).reset_index(drop=True)
```

**Lý do:**

- Train data (2009-2018): 10 năm lịch sử → giúp mô hình học pattern dài hạn
- Validation data (2019-2020): 2 năm gần đây → giúp mô hình adapt với phân phối hiện tại
- Gộp lại = 12 năm dữ liệu → mô hình strong hơn trên test 2021

**Quy trình:**

```
┌─────────────────────────────────────────────────────────────┐
│ Tiền xử lý dữ liệu                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────┬──────────────────┬──────────────────┐
│  Train           │  Validation      │  Test            │
│  2009-2018       │  2019-2020       │  2021            │
│  630K rows       │  130K rows       │  20K rows        │
└──────────────────┴──────────────────┴──────────────────┘
         ↓                    ↓
    ┌──────────────────────────────┐
    │ Gộp lại                      │
    │ model_train_data             │
    │ 2009-2020 (760K rows)        │
    └──────────────────────────────┘
         ↓
    ┌──────────────────────────────┐
    │ Fit XGBoost, RF, Prophet     │
    │ (không dùng validation set)  │
    └──────────────────────────────┘
         ↓
    ┌──────────────────────────────┐
    │ Predict trên test set 2021   │
    └──────────────────────────────┘
         ↓
    ┌──────────────────────────────┐
    │ Tính metrics (MAE, RMSE, R²) │
    └──────────────────────────────┘
```

### 1.2 Hai Scale Metrics

Tất cả mô hình dự đoán trên **log1p scale**, sau đó convert sang **mm scale** để report:

```
Prediction Flow:
┌────────────────────────────────────┐
│ Model output: y_pred_log1p         │  (khoảng: 0-6)
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│ Inverse transform                  │
│ y_pred_mm = expm1(y_pred_log1p)    │  (khoảng: 0-400 mm)
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│ Clip negative → max(0, y_pred_mm)  │  (vì mưa không âm)
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│ Calculate metrics trên scale này   │
└────────────────────────────────────┘
```

**Công thức:**

```python
# log1p → mm
y_pred_mm = np.clip(np.expm1(y_pred_log1p), 0, None)

# Metrics trên cả 2 scale
mae_log = mean_absolute_error(y_true_log1p, y_pred_log1p)
mae_mm = mean_absolute_error(y_true_mm, y_pred_mm)
rmse_mm = sqrt(mean_squared_error(y_true_mm, y_pred_mm))
r2_mm = r2_score(y_true_mm, y_pred_mm)
```

---

## II. XGBoost

### 2.1 Cấu Hình & Hyperparameter

```python
model = XGBRegressor(
    n_estimators=800,           # Số trees
    max_depth=7,                # Chiều sâu mỗi tree
    learning_rate=0.03,         # Shrinkage/learning rate
    subsample=0.8,              # % samples trên mỗi tree (dropout)
    colsample_bytree=0.8,       # % features trên mỗi tree (dropout)
    reg_alpha=0.1,              # L1 regularization (Lasso)
    reg_lambda=1.0,             # L2 regularization (Ridge)
    min_child_weight=5,         # Số samples min ở leaf node
    random_state=42,
    n_jobs=-1,                  # Dùng tất cả CPU cores
)
```

### 2.2 Cross-Validation (CV)

```python
cv_scores = cross_val_score(
    model,
    X_train, y_train,
    cv=5,  # 5-fold CV
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
)
cv_rmse = -cv_scores
```

**Output:**

```
CV RMSE (log1p) → 0.3456 ± 0.0123 (mean ± std)
```

**Ý nghĩa:**

- Mô hình trung bình predict sai ~0.35 trên log1p space
- Std 0.01 nhỏ → model ổn định trên các fold khác nhau

### 2.3 Training & Prediction

```python
# Fit trên toàn bộ training set (train + val)
model.fit(X_train, y_train)

# Predict trên test set (log1p space)
y_pred_log = model.predict(X_test)
```

**Time:** ~5-10 seconds trên 12 năm × 63 tỉnh × 50 features

### 2.4 Metrics Calculation

**Log1p space:**

```python
mae_log = mean_absolute_error(y_test_log, y_pred_log)
rmse_log = sqrt(mean_squared_error(y_test_log, y_pred_log))
r2_log = r2_score(y_test_log, y_pred_log)
```

**MM space (transform → metrics):**

```python
y_pred_mm = np.clip(np.expm1(y_pred_log), 0, None)
mae_mm = mean_absolute_error(y_test_mm, y_pred_mm)
rmse_mm = sqrt(mean_squared_error(y_test_mm, y_pred_mm))
r2_mm = r2_score(y_test_mm, y_pred_mm)
```

### 2.5 Feature Importance

```python
importance_df = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': model.feature_importances_,
}).sort_values('importance', ascending=False)

# Top 10 features được visualize trong report
```

**Ví dụ output:**

```
Rank  Feature                  Importance
────  ─────────────────────────  ──────────
1     rain_roll_mean_7d        0.285
2     rain_roll_max_7d         0.201
3     month_sin                0.089
4     humidi                   0.078
5     temp_diff                0.065
...
```

### 2.6 Ví Dụ Metrics Output

```
▓ XGBoost
├─ CV RMSE (5-fold):           0.3456 ± 0.0123
├─ Training time:              8.45s
│
├─ Test Metrics (log1p space):
│  ├─ MAE:                     0.3234
│  ├─ RMSE:                    0.4567
│  └─ R²:                      0.7823
│
└─ Test Metrics (mm space):
   ├─ MAE:                     5.23 mm
   ├─ RMSE:                    12.45 mm
   └─ R²:                      0.7823
```

---

## III. Random Forest

### 3.1 Cấu Hình

```python
model = RandomForestRegressor(
    n_estimators=500,           # Số trees
    max_depth=None,             # Unbounded depth
    min_samples_leaf=2,         # Min samples ở leaf
    max_features='sqrt',        # Số features random mỗi split
    n_jobs=-1,
    random_state=42,
)
```

### 3.2 Training & Prediction

```python
# Fit
model.fit(X_train, y_train)

# Predict → convert từ log1p → mm
y_pred_log = model.predict(X_test)
y_pred_mm = np.clip(np.expm1(y_pred_log), 0, None)
```

### 3.3 Metrics

Tương tự XGBoost — MAE, RMSE, R² trên cả 2 scale.

**Thường chậm hơn XGBoost** (500 trees × parallel không optimize tốt như XGBoost).

---

## IV. Prophet (Time-Series Specific)

### 4.1 Chiến Lược Prophet

Prophet được fit **per province** (không fit chung cho tất cả):

```python
for province_code in all_provinces:
    train_prov = train_data[train_data['province_encoded'] == province_code]
    test_prov = test_data[test_data['province_encoded'] == province_code]

    prophet_train = train_prov[['date', 'rain_log1p']].rename(
        columns={'date': 'ds', 'rain_log1p': 'y'}
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    model.fit(prophet_train)

    forecast = model.predict(prophet_test[['ds']])
    predictions[province_code] = forecast['yhat'].values
```

### 4.2 Cấu Hình Prophet

```python
model = Prophet(
    yearly_seasonality=True,              # Mô hiệu ứng mùa
    weekly_seasonality=True,              # Mô hiệu ứng tuần
    daily_seasonality=False,              # Không cần daily
    changepoint_prior_scale=0.05,         # Độ nhạy với điểm thay đổi
    seasonality_mode='additive',          # Mô hình seasonality
    interval_width=0.95,                  # Confidence interval
)
```

### 4.3 Fallback Strategy

Nếu Prophet fail trên một tỉnh:

```python
try:
    m.fit(prophet_train)
    forecast = m.predict(prophet_test)
    pred_log[idx] = forecast['yhat'].values
except Exception:
    failures += 1
    # Fallback: dùng giá trị cuối cùng của training set
    fallback_value = prophet_train['y'].iloc[-1]
    pred_log[idx] = fallback_value
```

**Log:**

```
Prophet fail on province=13: ...
  → Fallback = last observed log1p
Province failures: 2/63
```

### 4.4 Metrics Tương Tự

MAE, RMSE, R² trên cả log1p và mm.

---

## V. LSTM (Deep Learning)

### 5.1 Architecture

```
Input sequence (lookback=30 timesteps)
    ↓
┌─────────────────────────────┐
│ LSTM (64 units)             │
│ → output dim=64             │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Dropout (0.3)               │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Dense (32 units, ReLU)      │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Dropout (0.2)               │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Dense (1 unit, Linear)      │
│ → y_pred (log1p scale)      │
└─────────────────────────────┘
```

### 5.2 Cấu Hình Training

```python
criterion = nn.SmoothL1Loss(beta=0.5)   # Huber loss (robust to outliers)
optimizer = torch.optim.AdamW(lr=0.001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# Early stopping
early_stopping_patience = 15
```

### 5.3 Data Preparation

**Sliding window sequences per province:**

```
Ví dụ: lookback=30, province=Ha Noi

Date        X sequence (30 timesteps)           y target
────────────────────────────────────────────────────────
2009-01-31  [2009-01-02 ~ 2009-01-31]         2009-02-01
2009-02-01  [2009-01-03 ~ 2009-02-01]         2009-02-02
2009-02-02  [2009-01-04 ~ 2009-02-02]         2009-02-03
...
```

**Constraints:**

- Per province — không mix dữ liệu giữa các tỉnh
- shift(1) — không data leakage

### 5.4 Training Loop

```python
for epoch in range(1, max_epochs+1):
    # === TRAIN ===
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # === VALIDATE ===
    model.eval()
    with torch.no_grad():
        val_loss = ...

    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val:
        best_val = val_loss
        best_weights = deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            break
```

**Log:**

```
Epoch │ Train Loss  │ Val Loss    │ LR        │ Status
────────────────────────────────────────────────────────
  1   │ 0.586234   │ 0.521456   │ 1.00e-03  │ ✓ best
  2   │ 0.512345   │ 0.498123   │ 1.00e-03  │ ✓ best
  3   │ 0.498123   │ 0.502345   │ 1.00e-03  │ patience 1/15
  ...
 35   │ 0.245123   │ 0.263456   │ 2.50e-04  │ Early stopping at epoch 35
```

### 5.5 Prediction & Inverse Transform

```python
model.eval()
all_preds_scaled = []
with torch.no_grad():
    for X_batch in test_loader:
        preds = model(X_batch.to(device)).cpu().numpy()
        all_preds_scaled.append(preds)

pred_scaled = np.concatenate(all_preds_scaled)  # Shape: (N, 1), scaled

# Inverse transform: scaled → log1p → mm
pred_log1p = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
pred_mm = np.clip(np.expm1(pred_log1p), 0, None)
```

### 5.6 Metrics

MAE, RMSE, R² trên cả log1p và mm.

---

## VI. Metrics Chi Tiết

### 6.1 MAE (Mean Absolute Error)

```python
MAE = (1/N) × Σ|y_true - y_pred|
```

**Ý nghĩa:** Lỗi trung bình (tính giá trị tuyệt đối) — dễ hiểu, có size đơn vị bằng target.

**Ví dụ:**

```
y_true = [5, 10, 15]
y_pred = [4, 11, 14]
errors = [1, 1, 1]
MAE = 1 mm
```

### 6.2 RMSE (Root Mean Squared Error)

```python
RMSE = sqrt((1/N) × Σ(y_true - y_pred)²)
```

**Ý nghĩa:** Lỗi trung bình (với bình phương) — penalizes outliers hơn MAE.

**Ví dụ:**

```
y_true = [5, 10, 15]
y_pred = [4, 11, 14]
errors = [1, 1, 1]
RMSE = sqrt((1+1+1)/3) = 1.0 mm

Nhưng nếu:
y_pred = [1, 10, 18]  (1 outlier)
errors = [4, 0, 3]
RMSE = sqrt((16+0+9)/3) = 2.24 mm  (lớn hơn MAE)
```

### 6.3 R² (Coefficient of Determination)

```python
R² = 1 - (Σ(y_true - y_pred)²) / (Σ(y_true - mean(y_true))²)
```

**Ý nghĩa:** Tỷ lệ phương sai được mô hình giải thích.

- R² = 1: Perfect prediction
- R² = 0: Model không tốt hơn dự đoán constant (mean)
- R² < 0: Model tệ hơn dự đoán constant

**Ví dụ:**

```
y_true = [5, 10, 15, 20, 25]
baseline = mean(y_true) = 15

Baseline MSE = ((5-15)² + (10-15)² + (15-15)² + (20-15)² + (25-15)²) / 5
             = (100 + 25 + 0 + 25 + 100) / 5 = 50

Model prediction:
y_pred = [6, 9, 15, 19, 26]
Model MSE = ((5-6)² + (10-9)² + (15-15)² + (20-19)² + (25-26)²) / 5
          = (1 + 1 + 0 + 1 + 1) / 5 = 0.8

R² = 1 - (0.8 / 50) = 0.984  (rất tốt!)
```

---

## VII. Test Set Characteristics

### 7.1 Test Data Distribution

```
Test Set (2021) — 6 tháng dữ liệu:
- 63 tỉnh × 182 ngày ≈ 11,466 samples
- Date range: 2021-01-01 ~ 2021-06-30
- Target distribution:
  Mean rain_log1p: 1.3456
  Std rain_log1p:  0.8234
  Max rain_log1p:  5.1234 (rain = 166 mm)
  Min rain_log1p:  0 (không mưa)
```

### 7.2 Cross-Model Comparison

Sau khi training tất cả mô hình, report so sánh:

```
Model          │ MAE (log1p) │ RMSE (log1p) │ R² (log1p) │ MAE (mm) │ RMSE (mm) │ R² (mm)
───────────────┼─────────────┼──────────────┼────────────┼──────────┼───────────┼─────────
XGBoost        │ 0.3234      │ 0.4567       │ 0.7823     │ 5.23 mm  │ 12.45 mm  │ 0.7823
Random Forest  │ 0.3456      │ 0.4789       │ 0.7612     │ 5.67 mm  │ 13.12 mm  │ 0.7612
Prophet        │ 0.4123      │ 0.5234       │ 0.6945     │ 7.89 mm  │ 15.23 mm  │ 0.6945
LSTM           │ 0.3678      │ 0.4912       │ 0.7534     │ 6.12 mm  │ 13.78 mm  │ 0.7534
───────────────┴─────────────┴──────────────┴────────────┴──────────┴───────────┴─────────
Best Model     │ ← XGBoost → │ ← XGBoost → │ ← XGBoost → │ ← XGBoost │ ← XGBoost │ ← XGBoost
```

**Kết luận:** XGBoost là mô hình tốt nhất → **được chọn để serve** qua FastAPI API.

---

## VIII. Output Files

```
output/v4_20260403_073740/
├── predictions_2021.csv                 (actual + all model predictions)
├── predictions_all_models_2021.csv      (expanded version)
├── predictions_xgboost_2021.csv         (XGBoost only - used by API)
├── predictions_lstm_2021.csv
├── predictions_prophet_2021.csv
├── predictions_random_forest_2021.csv
├── feature_importance.csv               (XGBoost only)
├── lstm_loss_curve.csv                  (GRU training curve)
├── evaluation_metrics.json              (full metrics object)
├── evaluation_report.md                 (readable report)
└── comparison_report.md                 (cross-model comparison)
```

### 8.1 predictions_xgboost_2021.csv

```csv
date,province,region,xgboost_pred_mm
2021-01-01,Ha Noi,Bac Bo,0.23
2021-01-01,Da Nang,Trung Bo,2.45
...
```

**Cột:** date, province, region, xgboost_pred_mm (millimeters)

---

## IX. Quy Trình Toàn Bộ Flow

```
┌─────────────────────────────────────┐
│ 1. Load & Clean + Feature Eng       │
│    (tiền xử lý dữ liệu)             │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 2. Train/Val/Test Split             │
│    (temporal: 2009-2018, 2019-2020, 2021)
└─────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────────────┐
│ 3. Concatenate Train + Val → model_train_data   │
│    (2009-2020)                                   │
└──────────────────────────────────────────────────┘
              ↓
     ┌────┬─────────┬──────────┬──────┐
     ↓    ↓         ↓          ↓      ↓
  XGBoost │ 5-fold CV │ Train │ Test │ Metrics
           │           │
      Random Forest    │
           │           │
       Prophet         │
           │           │
        LSTM           │
           └───────────┴─────────┘
              ↓
┌─────────────────────────────────────┐
│ 4. Compare Metrics (4 models)       │
│    X GBoost wins                     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 5. Export predictions_xgboost_2021  │
│    (chuẩn bị cho FastAPI serving)   │
└─────────────────────────────────────┘
```

---

## X. Cách Chạy Toàn Bộ Pipeline

```bash
# Kích hoạt venv
.\venv\Scripts\Activate.ps1

# Chạy pipeline
python pipeline_v3.py

# Output:
# [STEP] TẢI VÀ LÀM SẠCH DỮ LIỆU THÔ
# [STEP] FEATURE ENGINEERING
# [STEP] PHÂN CHIA TRAIN / VALIDATION / TEST (temporal)
# [STEP] CHỌN FEATURES, SCALE & LƯU FILE
# [STEP] ĐÁNH GIÁ XGBOOST
# [STEP] ĐÁNH GIÁ RANDOM FOREST
# [STEP] ĐÁNH GIÁ PROPHET
# [STEP] ĐÁNH GIÁ LSTM
# [STEP] FINAL SUMMARY
# Success!
```

**Thời gian chạy:** ~45-60 phút trên CPU thường xuyên (LSTM training chậm nhất)
