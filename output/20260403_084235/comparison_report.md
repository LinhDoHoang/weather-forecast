# Model Comparison

Models: XGBoost, Random Forest, Prophet, LSTM/GRU.
Accuracy follows the current repository style: Accuracy (%) = R2 * 100.

Best by R2 (mm): **XGBoost** (0.4766)

| Model | Config | R2 (log1p) | RMSE (log1p) | MAE (log1p) | Accuracy log1p (%) | R2 (mm) | RMSE (mm) | MAE (mm) | Accuracy mm (%) | Train time (s) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| XGBoost | unscaled | 0.8111 | 0.4276 | 0.2518 | 81.11 | 0.4766 | 5.9305 | 1.9379 | 47.66 | 6.88 |
| RandomForest | unscaled | 0.7947 | 0.4458 | 0.2628 | 79.47 | 0.3866 | 6.4205 | 2.0528 | 38.66 | 31.67 |
| LSTM/GRU | scaled (RobustScaler) | 0.4466 | 0.7318 | 0.5003 | 44.66 | 0.1315 | 7.6393 | 2.6974 | 13.15 | 250.11 |
| Prophet | unscaled (per-province time series) | 0.2672 | 0.8421 | 0.6398 | 26.72 | 0.0689 | 7.9099 | 3.0392 | 6.89 | 22.05 |