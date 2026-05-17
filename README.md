# 📈 CO2 Emissions — Time Series Forecasting

> Comparing ARIMA, SARIMA, LSTM, and Transformer models on CO2 emission data. A practical guide to choosing the right forecasting model.

![Python](https://img.shields.io/badge/Python-3.x-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange) ![Statsmodels](https://img.shields.io/badge/Statsmodels-ARIMA-green) ![Kaggle](https://img.shields.io/badge/Dataset-CO2-lightgrey)

---

## What this project covers

End-to-end time series forecasting pipeline — from EDA and preprocessing to classical models (ARIMA/SARIMA) and modern deep learning models (LSTM/Transformer), with a focus on proper parameter tuning and model selection.

---

## Key Concepts Learned

### 🔄 Rolling Cross-Validation

Standard k-fold cross-validation **cannot** be used for time series — randomly shuffling data causes the model to "see the future" during training (data leakage). The solution is **rolling cross-validation**: the training window expands forward in time, and the model is only ever tested on data it has never seen.

<img width="1322" height="428" alt="Rolling cross-validation" src="https://github.com/user-attachments/assets/35d54142-6276-4e3c-89b6-769cf492fd8c" />

---

### 📊 Additive vs Multiplicative — Decomposition

Decomposing the time series into trend, seasonality, and residuals reveals whether the seasonal fluctuations stay constant over time (additive) or grow with the trend (multiplicative). Parallel seasonal components → additive model → no Box-Cox transformation needed.

<img width="1115" height="444" alt="Seasonal decomposition" src="https://github.com/user-attachments/assets/d29b46d0-57a1-491e-b1f5-e140802d98da" />

---

### 🎛️ Parameter Tuning — ACF & PACF

- **ACF** (Autocorrelation Function) → choose `q` (Moving Average order): the lag where the function drops into the confidence band
- **PACF** (Partial Autocorrelation Function) → choose `p` (AutoRegressive order): same logic

> Both dropped to 0 at lag 3 → **ARIMA(3,1,3)**

Key lesson: too many parameters → **overfitting**. A simpler model that generalizes beats a complex model that memorizes.

---

### 📉 ARIMA — Good for Short-Term Only

ARIMA is a linear model. As the forecast horizon grows, it fails to capture the non-linear trend in the data. Short steps (10–30): decent. Long steps (90+): poor.

<img width="1290" height="444" alt="ARIMA forecasting steps" src="https://github.com/user-attachments/assets/6b46f253-3185-4045-a330-3faeafd0fb9c" />

---

## Model Comparison

| Model | Short-term | Long-term | Notes |
|---|---|---|---|
| ARIMA(3,1,3) | ⚠️ Medium | ❌ Poor | No seasonality, linear only |
| SARIMA(3,1,3)(1,1,1)₁₂ | ✅ R²=0.90 | ⚠️ R²=0.82 | Best for seasonal short-term |
| LSTM | ⚠️ Decent | ✅ Good | Captures long-term dependencies |
| Transformer | ❌ Poor | ❌ Poor | Needs far more than ~500 data points |

---

## How to Choose a Model

<img width="866" height="473" alt="Model selection framework" src="https://github.com/user-attachments/assets/f23b25e8-a2bb-4797-b6d7-ba1f21999e2c" />

```
Large dataset?        → Transformer
Need long-term?       → LSTM
Data is seasonal?     → SARIMA
Otherwise            → ARIMA
```

---

## Stack

```bash
pip install pandas seaborn matplotlib statsmodels tensorflow scikit-learn
```

---

## 📄 Full Report

[![Report PDF](https://img.shields.io/badge/Report-PDF-red?logo=adobeacrobatreader)](https://github.com/Oussama-Ouarezki/Timeseries_co2/blob/main/Projet_data_mining-10.pdf)

---

## Conclusion

> This project taught three core time series lessons: **(1)** always use rolling cross-validation — never standard k-fold; **(2)** ACF/PACF are powerful tools for parameter selection, but more parameters ≠ better model; **(3)** no single model wins across all horizons — understanding your data size and forecast horizon is what drives the right choice.
