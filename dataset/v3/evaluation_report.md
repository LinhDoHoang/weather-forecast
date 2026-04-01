# Rainfall Prediction — Evaluation Report

**Run:** 2026-04-01 21:29:51  
**Train period:** 2009–2020  
**Test period:** 2021 (Jan–Jun)  
**Features:** 30  
**Pipeline time:** 1314.2s  

---

## 1. Overall Metrics

| Metric | XGBoost | GRU/LSTM | Winner |
|:-------|--------:|---------:|:------:|
| R² (log1p) | 0.6843 | 0.4394 | XGBoost |
| RMSE (log1p) | 0.5527 | 0.7366 | XGBoost |
| MAE (log1p) | 0.3615 | 0.5114 | XGBoost |
| R² (mm) | 0.4536 | 0.1331 | XGBoost |
| RMSE (mm) | 6.0594 | 7.6327 | XGBoost |
| MAE (mm) | 2.1643 | 2.7060 | XGBoost |
| Train time (s) | 9.5 | 1258.0 | — |
| CV RMSE (log1p) | 0.6127 ± 0.0188 | — | — |
| Epochs run | — | 21 | — |

> **Best model:** XGBoost &nbsp;|&nbsp; **Rating:** TỐT

---

## 2. Feature Importance (XGBoost — Top 15)

| Rank | Feature | Importance |
|-----:|:--------|----------:|
| 1 | `consecutive_dry_days` | 0.4205 |
| 2 | `rain_yesterday` | 0.2173 |
| 3 | `is_rainy_season` | 0.0623 |
| 4 | `rain_roll_mean_7d` | 0.0410 |
| 5 | `humidi` | 0.0347 |
| 6 | `min_temp` | 0.0252 |
| 7 | `humid_cloud` | 0.0224 |
| 8 | `rain_roll_mean_14d` | 0.0133 |
| 9 | `temp_diff` | 0.0120 |
| 10 | `rain_roll_mean_30d` | 0.0115 |
| 11 | `day_cos` | 0.0115 |
| 12 | `region_encoded` | 0.0113 |
| 13 | `pressure` | 0.0104 |
| 14 | `month_cos` | 0.0095 |
| 15 | `pressure_diff` | 0.0088 |

---

## 3. Evaluation by Region

| Region | N | XGB MAE | XGB RMSE | XGB R² | GRU MAE | GRU RMSE | GRU R² |
|:-------|--:|--------:|---------:|-------:|--------:|---------:|-------:|
| Bac Bo | 1859 | 2.96 | 7.63 | 0.3626 | 3.51 | 9.17 | 0.0801 |
| Nam Bo | 2535 | 1.64 | 4.10 | 0.4487 | 1.96 | 5.12 | 0.1422 |
| Tay Nguyen | 507 | 2.45 | 6.73 | 0.4873 | 3.23 | 7.96 | 0.2832 |
| Trung Bo | 1690 | 2.00 | 6.34 | 0.5361 | 2.78 | 8.74 | 0.1194 |

---

## 4. Evaluation by Month

| Month | N | XGB MAE | XGB RMSE | XGB R² | GRU MAE | GRU RMSE | GRU R² |
|------:|--:|--------:|---------:|-------:|--------:|---------:|-------:|
| Jan | 1209 | 0.94 | 3.00 | 0.6551 | 1.40 | 4.07 | 0.3652 |
| Feb | 1092 | 0.68 | 3.71 | 0.2988 | 0.93 | 4.28 | 0.0660 |
| Mar | 1209 | 0.63 | 1.82 | 0.6711 | 0.86 | 2.70 | 0.2716 |
| Apr | 1170 | 2.83 | 6.60 | 0.3170 | 3.59 | 8.06 | -0.0182 |
| May | 1209 | 4.36 | 8.51 | 0.2894 | 4.95 | 10.10 | 0.0007 |
| Jun | 702 | 4.33 | 10.23 | 0.4865 | 5.56 | 13.91 | 0.0506 |

---

## 5. Features Used

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

---

## 6. Preprocessing Notes

- **Target:** `rain` (mm) → `log1p` transformed (reduces skewness)
- **Scaling:** RobustScaler for GRU input features (resistant to outliers)
- **Split:** strict temporal — no data leakage across train/test
- **Cyclical encoding:** `sin/cos` for month, day-of-year, wind direction
- **Lag features:** rain rolled mean/max/std at 7d, 14d, 30d per province
- **Domain features:** `is_rainy_season`, `consecutive_dry_days`, `humid_cloud`, `pressure_diff`, `temp_humid_interaction`
