# 📈 Google Stock Price Forecasting

**🚀 [Live App → Open Streamlit](http://t-i-m-e-series-forecasting.streamlit.app/)**
> ⚠️ *App may be sleeping — click the link and wait 1–2 minutes for it to wake up (Streamlit free tier).*

Built a signal-generation system on 10 years of OHLCV data using SARIMAX + Random Forest + XGBoost ensemble.
- Test R² = 0.849 | SARIMAX RMSE = 22.87
- Granger causality for exogenous feature selection (High_lag6, Low_lag9, Open_lag7)
- TimeSeriesSplit CV | Rolling volatility + momentum features
