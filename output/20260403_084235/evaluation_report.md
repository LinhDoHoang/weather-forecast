# Rainfall Prediction — Evaluation Report

**Run:** 2026-04-03 08:49:42  
**Train period:** 2009–2018  
**Validation period:** 2019–2020  
**Final fit period (for model training):** 2009–2020  
**Test period:** 2021  
**Features:** 44  
**Pipeline time:** 426.4s  

---

## 1. Overall Metrics

| Metric | XGBoost | GRU/LSTM | Winner |
|:-------|--------:|---------:|:------:|
| R² (log1p) | 0.8111 | 0.4466 | XGBoost |
| RMSE (log1p) | 0.4276 | 0.7318 | XGBoost |
| MAE (log1p) | 0.2518 | 0.5003 | XGBoost |
| R² (mm) | 0.4766 | 0.1315 | XGBoost |
| RMSE (mm) | 5.9305 | 7.6393 | XGBoost |
| MAE (mm) | 1.9379 | 2.6974 | XGBoost |
| Train time (s) | 6.9 | 250.1 | — |
| CV RMSE (log1p) | 0.4863 ± 0.0189 | — | — |
| Epochs run | — | 5 | — |

> **Best model:** XGBoost &nbsp;|&nbsp; **Rating:** XUẤT SẮC

---

## 2. Feature Importance (XGBoost — Top 15)

| Rank | Feature | Importance |
|-----:|:--------|----------:|
| 1 | `wet_streak_days` | 0.2852 |
| 2 | `wet_to_dry` | 0.2178 |
| 3 | `consecutive_dry_days` | 0.1833 |
| 4 | `rain_yesterday` | 0.0769 |
| 5 | `heavy_rain_yesterday` | 0.0574 |
| 6 | `dry_to_wet` | 0.0202 |
| 7 | `rain_ewm_7d` | 0.0180 |
| 8 | `humidi` | 0.0119 |
| 9 | `is_rainy_season` | 0.0113 |
| 10 | `rain_ewm_14d` | 0.0110 |
| 11 | `min_temp` | 0.0069 |
| 12 | `humid_cloud` | 0.0068 |
| 13 | `temp_diff` | 0.0067 |
| 14 | `rain_roll_mean_30d` | 0.0055 |
| 15 | `region_encoded` | 0.0055 |

---

## 3. Features Used

| # | Feature |
|--:|:--------|
| 1 | `max_temp` |
| 2 | `min_temp` |
| 3 | `wind` |
| 4 | `humidi` |
| 5 | `cloud` |
| 6 | `pressure` |
| 7 | `temp_diff` |
| 8 | `month_sin` |
| 9 | `month_cos` |
| 10 | `day_sin` |
| 11 | `day_cos` |
| 12 | `wind_dir_sin` |
| 13 | `wind_dir_cos` |
| 14 | `is_rainy_season` |
| 15 | `humid_cloud` |
| 16 | `rain_yesterday` |
| 17 | `rain_roll_mean_7d` |
| 18 | `rain_roll_max_7d` |
| 19 | `rain_roll_std_7d` |
| 20 | `rain_roll_mean_14d` |
| 21 | `rain_roll_max_14d` |
| 22 | `rain_roll_std_14d` |
| 23 | `rain_roll_mean_30d` |
| 24 | `rain_roll_max_30d` |
| 25 | `rain_roll_std_30d` |
| 26 | `pressure_diff` |
| 27 | `consecutive_dry_days` |
| 28 | `temp_humid_interaction` |
| 29 | `region_encoded` |
| 30 | `province_encoded` |
| 31 | `rain_lag_2d` |
| 32 | `rain_lag_3d` |
| 33 | `rain_lag_5d` |
| 34 | `rain_roll_mean_3d` |
| 35 | `rain_roll_max_3d` |
| 36 | `rain_roll_std_3d` |
| 37 | `rain_ewm_7d` |
| 38 | `rain_ewm_14d` |
| 39 | `heavy_rain_yesterday` |
| 40 | `dry_to_wet` |
| 41 | `wet_to_dry` |
| 42 | `wet_streak_days` |
| 43 | `pressure_roll_mean_3d` |
| 44 | `wind_roll_mean_3d` |

---

## 4. Preprocessing Notes

- **Target:** `rain` (mm) → `log1p` transformed (reduces skewness)
- **Scaling:** RobustScaler for GRU input features (resistant to outliers)
- **Split:** strict temporal — train/validation/test, no data leakage
- **Cyclical encoding:** `sin/cos` for month, day-of-year, wind direction
- **Lag features:** rain rolled mean/max/std at 7d, 14d, 30d per province
- **Domain features:** `is_rainy_season`, `consecutive_dry_days`, `humid_cloud`, `pressure_diff`, `temp_humid_interaction`
