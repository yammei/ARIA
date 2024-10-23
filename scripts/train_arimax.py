from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from model_inference import generate_forecast
import statsmodels.api as sm
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os

# Load data
def load_data(filename):
    stock_data = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
    business_days = pd.date_range(start=stock_data.index.min(), end=stock_data.index.max(), freq='B')
    return stock_data, business_days

# Load tickers from correlation CSV
def load_competitor_tickers(csv_file):
    try:
        tickers_df = pd.read_csv(csv_file)
        tickers = tickers_df['Ticker'].tolist()
        # Ensure PANW is excluded from competitors (since it is the target)
        tickers = [ticker for ticker in tickers if ticker != 'PANW']
        return tickers
    except Exception as e:
        print(f"Error loading tickers from CSV: {e}")
        return []

# Prepare variables
def prepare_variables(stock_data, business_days, competitors):
    panw_data = stock_data[stock_data['Ticker'] == 'PANW']
    endog = panw_data['Close']

    # Exogenous variables: EMA values + competitor close prices
    exog = panw_data[['EMA_50', 'EMA_200']]

    for competitor in competitors:
        competitor_data = stock_data[stock_data['Ticker'] == competitor].reindex(panw_data.index).fillna(0)
        exog[f'{competitor}_Close_lag1'] = competitor_data['Close'].shift(1)

    endog = endog.reindex(business_days).fillna(method='ffill')
    exog = exog.reindex(business_days).fillna(method='ffill')
    exog = exog.fillna(method='bfill')

    return endog, exog

# Normalize exog data
def normalize_exog(exog):
    scaler = MinMaxScaler()
    exog_scaled = pd.DataFrame(scaler.fit_transform(exog), columns=exog.columns, index=exog.index)
    return exog_scaled, scaler

# Train the model
def train_model(endog_train, exog_train, order=(1, 1, 1)):
    model = sm.tsa.SARIMAX(endog_train, exog=exog_train, order=order)
    model_fit = model.fit()
    return model_fit

# Forecast model
def forecast_model(model_fit, endog_test, exog_test):
    forecast = model_fit.forecast(steps=len(endog_test), exog=exog_test)
    return pd.Series(forecast, index=endog_test.index)

# Evaluate model and append metrics to CSV
def evaluate_model(actual, forecast, timeframe, metrics_path="../logs/ARIMAX_metrics.csv"):
    comparison_df = pd.DataFrame({
        'Actual': actual,
        'Forecast': forecast
    })
    comparison_df.dropna(inplace=True)

    if not comparison_df.empty:
        mae = mean_absolute_error(comparison_df['Actual'], comparison_df['Forecast'])
        rmse = np.sqrt(mean_squared_error(comparison_df['Actual'], comparison_df['Forecast']))
        print(f'MAE: {mae}')
        print(f'RMSE: {rmse}')

        # Create a DataFrame for the metrics with the timeframe
        metrics_df = pd.DataFrame({
            'Timeframe (months)': [timeframe],
            'MAE': [mae],
            'RMSE': [rmse]
        })

        # Append to CSV, creating the file if it doesn't exist
        if not os.path.isfile(metrics_path):
            metrics_df.to_csv(metrics_path, index=False)
        else:
            metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)

        print(f"MAE and RMSE for timeframe {timeframe} months saved to {metrics_path}")

    return comparison_df

# Save model and scaler
def save_model(model, scaler, feature_names, model_file="sarimax_model.pkl", scaler_file="scaler.pkl", features_file="features.pkl"):
    file_path_prefix = '../model/'
    if not os.path.exists(file_path_prefix):
        os.makedirs(file_path_prefix)
    joblib.dump(model, os.path.join(file_path_prefix, model_file))
    joblib.dump(scaler, os.path.join(file_path_prefix, scaler_file))
    joblib.dump(feature_names, os.path.join(file_path_prefix, features_file))
    print(f"Model saved as {model_file}, scaler saved as {scaler_file}, and feature names saved as {features_file}")

# Main workflow
def main():
    forecasting = True
    timeframes = [3]
    for timeframe in timeframes:
        # Load tickers from correlation CSV
        correlation_csv_path = '../logs/CORRELATION_top_comovement.csv'
        competitors = load_competitor_tickers(correlation_csv_path)

        # Preprocess training data
        training_data = f"../training_data/ARIMAX_{timeframe}mo_training_data.csv"          # Specify training data file path
        stock_data, business_days = load_data(training_data)                                # Load training data
        endog, exog = prepare_variables(stock_data, business_days, competitors)             # Seperate endogenous and exogenous variables
        exog_scaled, scaler = normalize_exog(exog)                                          # Normalize training data values
        split_index = int(len(endog) * 0.8)                                                 # Ratio for training and validation split
        endog_train, endog_test = endog[:split_index], endog[split_index:]                  # Split training and validation (endogenous)
        exog_train, exog_test = exog_scaled[:split_index], exog_scaled[split_index:]        # Split training and validation (exogenous)

        # Train and save model
        model_file_path = 'sARIMAX_Short_Term_PANW_Forecast.pkl'                    # Specify model fine path
        model_fit = train_model(endog_train, exog_train, order=(1, 1, 1))           # Train model
        feature_names = exog_scaled.columns.tolist()                                # Save feature names
        save_model(model_fit, scaler, feature_names, model_file=model_file_path)    # Save model
        print("Feature names used during training:", feature_names)

        # Evaluate model across different training datasets
        if forecasting:
            forecast_results_self = forecast_model(model_fit, endog_test, exog_test)
            _ = evaluate_model(endog_test, forecast_results_self, timeframe, metrics_path='../logs/ARIMAX_metrics')

if __name__ == "__main__":
    main()
