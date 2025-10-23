# Sales and Demand Forecasting Using Time Series Analysis

## Project Overview  
Independent publishers face significant risk from over- or under-stocking without reliable demand forecasts, particularly for seasonal titles. This project develops and evaluates time series forecasting models to deliver actionable demand insights that can guide stocking strategies.  

The objective was to benchmark classical statistical methods (**SARIMA**) against machine learning (**XGBoost**), deep learning (**LSTM**), and hybrid approaches (**SARIMA + LSTM**, sequential and parallel). Models were evaluated using **MAE**, **MAPE**, and forecast prediction intervals to quantify uncertainty.  

## Data Preparation  

### Data Overview  
- Dataset: **Nielsen BookScan**  
- Initial filtering: 61 ISBNs with sales beyond **2024-07-01**  
- Selected titles for deep-dive analysis:  
  - *The Alchemist (TA)*  
  - *The Very Hungry Caterpillar (TVHC)*  
- Time period: **2012–2024** (628 weekly observations per title)  
- Forecast horizon: **last 32 weeks (8 months)**  

Both series display strong annual seasonality, with holiday peaks and a COVID-19 dip in 2020. Weekly data served as the main modelling input, with monthly series derived for comparison.  

### Preprocessing  
- Dates converted to `datetime` index with regularised weekly frequency  
- Monthly totals derived via resampling  
- Seasonal-trend decomposition identified **multiplicative seasonality** → applied Box-Cox transformation (+1 shift for zeros)  
- Stationarity checks (ACF, PACF) → weekly series approximately stationary, monthly required differencing (d=1)  

## Modelling Approach  

### Models Evaluated  
1. **SARIMA** – baseline classical model  
2. **XGBoost** – recursive pipeline with detrending and deseasonalisation  
3. **LSTM** – sliding window sequences, tuned via KerasTuner  
4. **Hybrid Sequential** – SARIMA residuals modelled with LSTM  
5. **Hybrid Parallel** – weighted average of SARIMA and LSTM predictions  

### Validation Strategy  
- Expanding-window cross-validation  
- Hyperparameter tuning via grid search (XGBoost) and KerasTuner (LSTM)  
- Evaluation on holdout set (last 32 weeks) using **MAE** and **MAPE**  

## Results  

### The Alchemist (Weekly Sales)  
- **Best Model:** Hybrid Sequential (SARIMA + LSTM)  
- **Performance:** MAE = 134.29, MAPE = 19.8%  
- Outperformed SARIMA baseline (MAPE = 21.8%)  

| Model              | MAE    | MAPE (%) |
|---------------------|--------|----------|
| Hybrid Sequential   | 134.29 | 19.8     |
| Hybrid Parallel     | 137.93 | 21.5     |
| SARIMA             | 141.64 | 21.8     |
| XGBoost            | 157.43 | 22.7     |
| LSTM               | 185.16 | 28.5     |

### The Very Hungry Caterpillar (Weekly Sales)  
- **Best Model:** Hybrid Parallel (SARIMA + LSTM, 23% LSTM weight)  
- **Performance:** MAE = 504.46, MAPE = 22.0%  
- Improved over SARIMA baseline (MAPE = 23.3%)  

| Model              | MAE    | MAPE (%) |
|---------------------|--------|----------|
| Hybrid Parallel     | 504.46 | 22.0     |
| SARIMA             | 523.72 | 23.3     |
| Hybrid Sequential   | 524.11 | 23.3     |
| XGBoost            | 562.55 | 24.4     |
| LSTM               | 653.21 | 28.4     |


### Monthly Forecasting  
- Models trained directly on monthly data consistently **underperformed** compared to weekly-trained models aggregated to monthly.  
- Example (The Alchemist, SARIMA):  
  - Weekly (aggregated): MAE = 468.70, MAPE = 17.1%  
  - Monthly (direct): MAE = 897.47, MAPE = 44.0%  
- Conclusion: **Use weekly models as the forecasting baseline** and aggregate to monthly if required.  

## Conclusions & Recommendations  
- **Hybrid models** consistently delivered the strongest performance:  
  - *The Alchemist*: Hybrid Sequential SARIMA + LSTM  
  - *The Very Hungry Caterpillar*: Hybrid Parallel SARIMA + LSTM  
- **Weekly data** is superior to direct monthly modelling.  
- Standalone ML/DL models (XGBoost, LSTM) captured level/trend but struggled with seasonal and event-driven spikes.  
- SARIMA handled seasonality well, and hybridisation allowed recovery of both seasonal structure and nonlinear dynamics.  

### Future Work  
- Incorporate **calendar features** (holidays, events) as exogenous regressors  
- Explore **transformer-based sequence models** for long-range seasonality  
- Develop **automated pipelines** for real-time publisher forecasting  

## Tech Stack  
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, scikit-learn, statsmodels, XGBoost, TensorFlow/Keras, Matplotlib, Seaborn  
- **Techniques:** Time Series Forecasting, ARIMA/SARIMA, Gradient Boosting, Deep Learning (LSTM), Hybrid Models, Cross-Validation, Hyperparameter Tuning  

## Repository Structure 
```
├── book_inventory_demand_notebook.ipynb # Jupyter notebook with full analysis
├── book_inventory_demand_report.pdf # Stakeholder report 
├── README.md # Project documentation
```
