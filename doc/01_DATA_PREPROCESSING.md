# 01. Tiền Xử Lý Dữ Liệu (Data Preprocessing)

## Tổng Quan

Pipeline tiền xử lý dữ liệu Vietnam Weather bao gồm 4 bước chính:

1. **Load & Clean** — Tải dữ liệu thô, chuẩn hóa tên, nội suy ngày thiếu, loại bỏ trùng lặp
2. **Feature Engineering** — Tạo các đặc trưng (features) từ dữ liệu gốc
3. **Temporal Split** — Phân chia dữ liệu thành train/validation/test theo thời gian
4. **Scale & Save** — Chuẩn hóa dữ liệu và lưu các file xử lý xong

---

## I. LOAD & CLEAN (Bước 1)

### 1.1 Tải Dữ Liệu Gốc

```python
Input:  dataset/init/weather.csv
Output: DataFrame với cột [province, date, max_temp, min_temp, wind, rain, humidi, cloud, pressure, wind_d]
```

**Công việc chi tiết:**

| Công việc              | Chi Tiết                                                                     |
| ---------------------- | ---------------------------------------------------------------------------- |
| **Đọc CSV**            | Dùng pandas.read_csv(), kiểm tra shape và column names                       |
| **Lưu thông tin**      | In ra số tỉnh, khoảng date min-max, tổng số hàng                             |
| **Chuẩn hóa tên tỉnh** | Áp dụng mapping từ config (ví dụ: "Hanoi" → "Ha Noi") để tránh duplicate tên |

**Ví dụ tên tỉnh:**

```
Trước: "Hanoi", "hanoi", "Ha Noi" → Sau: "Ha Noi"
```

### 1.2 Xử Lý Các Hàng Trùng Lặp

**Tình huống:** Nếu cùng một (province, date) xuất hiện nhiều lần:

- Dùng aggregation (gộp lại)
- Các cột số (max_temp, min_temp, v.v.) → lấy **trung bình**
- Cột `wind_d` (hướng gió) → lấy **first occurrence**

**Ví dụ:**

```
province  date        max_temp  rain
Ha Noi    2009-01-01  25        5
Ha Noi    2009-01-01  26        3
↓
Ha Noi    2009-01-01  25.5      4  (trung bình của 25,26 và 5,3)
```

Số hàng trùng lặp bị loại được in ra trong log:

```
Aggregate 152 duplicate (province, date) rows
```

### 1.3 Nội Suy Ngày Thiếu (Interpolation)

**Vấn đề:** Các tỉnh có thể thiếu dữ liệu một số ngày (do lỗi thu thập, bảo trì).

**Giải pháp — Per Province:**

1. **Tạo range ngày đầy đủ** từ ngày tối thiểu đến ngày tối đa của tỉnh
2. **Nội suy các ngày thiếu** cho các cột số:
   - Dùng `interpolate(method='time')` — linear interpolation theo thời gian
3. **Forward-fill + Backward-fill** hướng gió (`wind_d`):
   - Nếu giá trị thiếu, dùng giá trị trước đó (forward-fill)
   - Nếu vẫn thiếu, dùng giá trị sau đó (backward-fill)

**Ví dụ:**

```
Ha Noi:
2009-01-01  max_temp=25
2009-01-02  max_temp=NaN (thiếu) ← nội suy
2009-01-03  max_temp=23

Sau nội suy:
2009-01-02  max_temp=24 (giá trị trung bình giữa 25 và 23)
```

**Log output:**

```
Nội suy 5,432 ngày thiếu trên 63 tỉnh
```

### 1.4 Làm Tròn Số

Tất cả cột số (max_temp, min_temp, wind, humidi, cloud, pressure) được làm tròn đến số nguyên:

```python
round_cols = ["max_temp", "min_temp", "wind", "humidi", "cloud", "pressure"]
df[round_cols] = df[round_cols].round().astype(int)
```

### 1.5 Validate Dữ Liệu

Các kiểm tra bắt buộc:

| Kiểm Tra     | Nội Dung                           | Hành Động Nếu Lỗi            |
| ------------ | ---------------------------------- | ---------------------------- |
| **No Nulls** | Không có NaN/null values sau xử lý | Exit với lỗi nếu còn null    |
| **Rain ≥ 0** | Tất cả giá trị rain phải ≥ 0       | Exit với lỗi nếu có rain < 0 |

**Ví dụ log:**

```
✓ Không có null values
✓ Rain ≥ 0
```

**Shape quá trình:**

```
Load raw:       80,856 rows
After clean:    80,856 rows (nếu không có thiếu/trùng)
```

---

## II. FEATURE ENGINEERING (Bước 2)

### 2.1 Map Tỉnh → Vùng → Code

Từ file `dataset/province_region_code_mapping.csv`, create 3 cột mới:

- `region` — tên vùng (Bac Bo, Trung Bo, Tay Nguyen, Nam Bo)
- `province_encoded` — mã số tỉnh (0-62)
- `region_encoded` — mã số vùng (0-3)

**Ví dụ:**

```
province        → region        → province_encoded  region_encoded
Ha Noi          → Bac Bo        → 1                 0
Ho Chi Minh     → Nam Bo        → 13                3
```

Nếu có tỉnh không map được → **Exit với lỗi** và liệt kê tỉnh lỗi.

### 2.2 Nhiệt Độ Chênh Lệch (Temperature Delta)

```python
temp_diff = max_temp - min_temp
```

**Ý nghĩa:** Biểu thị độ dao động nhiệt độ trong ngày — có thể liên quan đến điều kiện mây che.

### 2.3 Cyclical Time Encoding (Sin/Cos)

**Tháng (1-12):**

```python
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
```

**Ngày trong năm (1-365):**

```python
day_sin = sin(2π × day_of_year / 365)
day_cos = cos(2π × day_of_year / 365)
```

**Tại sao dùng sin/cos?**

- Các mô hình ML coi "tháng 12" và "tháng 1" gần nhau trong năm (mặc dù khác nhau x 11)
- Sin/cos transform biến điều này thành **khoảng cách Euclidean nhỏ** trong không gian 2D
- Xử lý cyclic nature của thời gian

**Ví dụ:**

```
Tháng 1:   month_sin ≈ 0.5,   month_cos ≈ 0.87
Tháng 12:  month_sin ≈ -0.5,  month_cos ≈ 0.87
→ Gần nhau trong không gian 2D ✓
```

### 2.4 Hướng Gió Encoding

**Compass directions → degrees → sin/cos:**

| Hướng | Độ   | sin   | cos   |
| ----- | ---- | ----- | ----- |
| N     | 0°   | 0.00  | 1.00  |
| NE    | 45°  | 0.71  | 0.71  |
| E     | 90°  | 1.00  | 0.00  |
| S     | 180° | 0.00  | -1.00 |
| W     | 270° | -1.00 | 0.00  |

**Công thức:**

```python
wind_deg = map_compass_to_degrees(wind_d)  # N→0, NE→45, ..., NNW→337.5
wind_dir_sin = sin(radians(wind_deg))
wind_dir_cos = cos(radians(wind_deg))
```

### 2.5 Rainy Season Flag

Mỗi vùng có mùa mưa khác nhau (domain knowledge):

| Vùng                    | Tháng Mưa      |
| ----------------------- | -------------- |
| Bac Bo (Bắc Bộ)         | 5-10 (May-Oct) |
| Trung Bo (Trung Bộ)     | 9-12 (Sep-Dec) |
| Tay Nguyen (Tây Nguyên) | 5-10 (May-Oct) |
| Nam Bo (Nam Bộ)         | 5-11 (May-Nov) |

**Feature:**

```python
is_rainy_season = 1 if (region, month) in rainy_months else 0
```

### 2.6 Humidity × Cloud Interaction

```python
humid_cloud = (humidi × cloud) / 100
```

**Ý nghĩa:** Độ ẩm cao + mây nhiều = điều kiện tiền thuận cho mưa.

### 2.7 Lag Features (Lịch Sử Mưa)

**Công thức cơ bản:**

```python
rain_roll_mean_7d = rain.shift(1).rolling(7, min_periods=1).mean()
```

**Chi tiết:**

- **shift(1)** — để tránh data leakage (chỉ dùng dữ liệu ngày trước, không dùng dữ liệu hôm nay)
- **rolling(w)** — tính trung bình trên cửa sổ `w` ngày
- **Per province** — mỗi tỉnh tính riêng để tránh trộn dữ liệu giữa các tỉnh

**Các lag window: 7, 14, 30 ngày**

| Feature             | Công Thức                   | Ý Nghĩa              |
| ------------------- | --------------------------- | -------------------- |
| `rain_roll_mean_7d` | Trung bình mưa 7 ngày trước | Xu hướng mưa gần đây |
| `rain_roll_max_7d`  | Max mưa trong 7 ngày trước  | Mưa lớn nhất gần đây |
| `rain_roll_std_7d`  | Std dev mưa 7 ngày trước    | Tính ổn định của mưa |

Tương tự cho 14 ngày và 30 ngày.

### 2.8 Target Column Transformation

```python
rain_log1p = log(1 + rain)  # aka np.log1p(rain)
```

**Tại sao?**

- Dữ liệu mưa là **long-tailed** (hầu hết ngày mưa ít, nhưng có ngày mưa rất nhiều)
- log1p transform giảm skewness, giúp mô hình học tốt hơn
- `log1p` instead of `log` để xử lý rain=0 (log(0) undefined)

**Ví dụ:**

```
rain = 0     → log1p = 0
rain = 5     → log1p ≈ 1.79
rain = 100   → log1p ≈ 4.61
```

### 2.9 Feature List Cuối Cùng

Tổng cộng khoảng **50+ features**:

- Thời gian: month, day, year, month_sin/cos, day_sin/cos
- Địa lý: temp_diff, wind_dir_sin/cos, is_rainy_season, humid_cloud
- Lịch sử: rain*roll_mean*_, rain*roll_max*_, rain*roll_std*\* (cho 7d, 14d, 30d)
- Metadata: province_encoded, region_encoded
- Target: rain_log1p (cột y)

---

## III. TEMPORAL SPLIT (Bước 3)

### 3.1 Chiến Lược Phân Chia

**Time-based split — No data leakage:**

```
2009-01-01                           2018-12-31   2019-01-01           2020-12-31   2021-01-01   2021-06-30
|◄───────────────────────────────────►|◄───────────────────────────────►|◄───────────────────────►|
        Training Set (2009-2018)     Validation Set (2019-2020)    Test Set (2021)
        ~10 năm dữ liệu              2 năm dữ liệu                   ~6 tháng dữ liệu
        630,000 rows                 ~130,000 rows                  ~20,000 rows
```

### 3.2 Lý Do Dùng 3-Fold

| Set            | Năm       | Mục Đích                                  | Ghi Chú                                    |
| -------------- | --------- | ----------------------------------------- | ------------------------------------------ |
| **Train**      | 2009-2018 | Huấn luyện mô hình                        | 10 năm dữ liệu lịch sử                     |
| **Validation** | 2019-2020 | Điều chỉnh hyperparameter, early stopping | 2 năm gần đây hơn, giảm phân phối thay đổi |
| **Test**       | 2021      | Đánh giá cuối trên dữ liệu chưa từng thấy | Hoàn toàn mới, mô phỏng dự đoán thực tế    |

**Data leakage check:**

```python
assert train_df['date'].max() < val_df['date'].min()
assert val_df['date'].max() < test_df['date'].min()
```

→ Không có cross-over, an toàn!

### 3.3 Validate & Log

**Log output:**

```
Train shape:          (630000, 60)
Validation shape:     (130000, 60)
Test shape:           (20000, 60)

Train date range:     2009-01-01 → 2018-12-31
Validation date range: 2019-01-01 → 2020-12-31
Test date range:      2021-01-01 → 2021-06-30

Train rain_log1p mean:       1.2345
Validation rain_log1p mean:  1.1234
Test rain_log1p mean:        1.3456

✓ No data leakage
```

---

## IV. SCALE & SAVE (Bước 4)

### 4.1 Chiến Lược Scaling

| Model             | Yêu Cầu Scaling | Chiến Lược                |
| ----------------- | --------------- | ------------------------- |
| **XGBoost**       | ❌ Không        | Dùng dữ liệu **unscaled** |
| **Random Forest** | ❌ Không        | Dùng dữ liệu **unscaled** |
| **Prophet**       | N/A             | Dùng dữ liệu **unscaled** |
| **LSTM**          | ✅ Có           | Dùng dữ liệu **scaled**   |

**Lý do:**

- Tree-based models (XGBoost, RF) không sensitive với khoảng giá trị → không cần scale
- Neural networks (LSTM) rất sensitive → scale để training ổn định

### 4.2 RobustScaler (Dùng cho LSTM)

```python
scaler = RobustScaler()
scaler.fit(train_set)  # ← Chỉ fit trên train set!
scaled_train = scaler.transform(train_set)
scaled_val = scaler.transform(val_set)
scaled_test = scaler.transform(test_set)
```

**Công thức RobustScaler:**

```
scaled_x = (x - Q2) / (Q3 - Q1)
Q1 = 25th percentile
Q2 = 50th percentile (median)
Q3 = 75th percentile
```

**Tại sao RobustScaler?**

- Resilient to outliers (mưa bất thường không ảnh hưởng)
- So với StandardScaler (dùng mean/std), RobustScaler dùng percentile

### 4.3 Features Cần Scale vs. Không Scale

**Không scale** (đã là giá trị normalized):

- cyclical features: month_sin/cos, day_sin/cos, wind_dir_sin/cos
- binary: is_rainy_season
- encoded: province_encoded, region_encoded

**Scale:** Tất cả các features khác (temperature, humidity, etc.)

### 4.4 File Output

```
dataset/v3/
├── weather_train_2009_2018.csv              (unscaled, cho XGBoost/RF)
├── weather_train_2009_2018_scaled.csv       (scaled, cho LSTM)
├── weather_val_2019_2020.csv                (unscaled)
├── weather_val_2019_2020_scaled.csv         (scaled)
├── weather_test_2021.csv                    (unscaled - final test)
├── weather_test_2021_scaled.csv             (scaled)
└── scalers.pkl                              (serialize feature_scaler & target_scaler)
```

### 4.5 Target Scaling

Target column `rain_log1p` cũng được scale riêng:

```python
target_scaler = RobustScaler()
target_scaler.fit(train[['rain_log1p']])
scaled_target_train = target_scaler.transform(train[['rain_log1p']])
```

**Inverse transform khi dự đoán:**

```python
pred_log1p_scaled = model.predict(X_test)
pred_log1p = target_scaler.inverse_transform(pred_log1p_scaled.reshape(-1, 1))
pred_mm = np.expm1(pred_log1p)  # log1p → mm
```

---

## V. Tóm Tắt Chi Tiết

| Bước           | Input          | Output                    | Tổng Thời Gian |
| -------------- | -------------- | ------------------------- | -------------- |
| Load & Clean   | weather.csv    | cleaned_df                | ~2s            |
| Feature Eng    | cleaned_df     | featured_df               | ~5s            |
| Temporal Split | featured_df    | train/val/test            | ~1s            |
| Scale & Save   | train/val/test | 6 CSV files + scalers.pkl | ~2s            |
| **TOTAL**      | Raw CSV        | Ready for modeling        | **~10s**       |

---

## VI. Cách Chạy

```bash
# Kích hoạt venv
.\venv\Scripts\Activate.ps1

# Chạy pipeline (bao gồm tiền xử lý + training)
python pipeline_v3.py
```

**Output:**

```
[STEP] TẢI VÀ LÀM SẠCH DỮ LIỆU THÔ
  [STAT] Loaded → shape: (80856, 10)
  [INFO] Chuẩn hóa tên tỉnh: 64 → 63 unique
  ...
[STEP] FEATURE ENGINEERING
  [INFO] + temp_diff
  [INFO] + month_sin/cos, day_sin/cos (cyclical)
  [INFO] + wind_dir_sin/cos (compass)
  ...
[STEP] PHÂN CHIA TRAIN / VALIDATION / TEST (temporal)
  [STAT] Train shape: (630000, 60)
  [STAT] Validation shape: (130000, 60)
  [STAT] Test shape: (20000, 60)
  [INFO] ✓ No data leakage
[STEP] CHỌN FEATURES, SCALE & LƯU FILE
  [INFO] Saved unscaled → dataset/v3/weather_train_2009_2018.csv
  ...
Success!
```
