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
        # print(f"Error loading tickers from CSV: {e}")
        return []

# Prepare variables
def prepare_variables(stock_data, business_days, competitors):
    panw_data = stock_data[stock_data['Ticker'] == 'PANW']
    endog = panw_data['Close']

    # Exogenous variables: EMA values + competitor close prices
    exog = panw_data[['EMA_12', 'EMA_26', 'Volume']]

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
def evaluate_model(actual, forecast, i_timeframes, j_timeframes, p, q, metrics_path="../logs/ARIMAX_metrics.csv"):
    # Create DataFrame to compare actual vs forecast
    comparison_df = pd.DataFrame({
        'Actual': actual,
        'Forecast': forecast
    })
    comparison_df.dropna(inplace=True)

    if not comparison_df.empty:
        # Calculate metrics
        mae = mean_absolute_error(comparison_df['Actual'], comparison_df['Forecast'])
        rmse = np.sqrt(mean_squared_error(comparison_df['Actual'], comparison_df['Forecast']))
        print(f'MAE: {mae}')
        print(f'RMSE: {rmse}')

        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            '(p, d, q)': [(p, 1, q)],
            'Training Timeframe': [i_timeframes],
            'Comovement Timeframe': [j_timeframes],
            'MAE': [mae],
            'RMSE': [rmse]
        })

        # Append to CSV, creating the file if it doesn't exist
        if not os.path.isfile(metrics_path):
            metrics_df.to_csv(metrics_path, index=False)
        else:
            metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)

        # Select the last 10 days for actual vs forecast comparison
        recent_comparison_df = comparison_df.tail(10)
        print("Actual vs Forecast (Last 10 Days):\n", recent_comparison_df)

        # Save the last 10 days to CSV for further analysis
        recent_comparison_csv_path = f'../logs/ARIMAX_actual_vs_forecast_last_10days_{i_timeframes}mo.csv'
        recent_comparison_df.to_csv(recent_comparison_csv_path)
        print(f"Actual vs Forecast for last 10 days saved to {recent_comparison_csv_path}")

    return comparison_df


# Save model and scaler
def save_model(model, scaler, feature_names, model_file="sarimax_model.pkl", scaler_file="scaler.pkl", features_file="features.pkl"):
    file_path_prefix = '../model/'
    if not os.path.exists(file_path_prefix):
        os.makedirs(file_path_prefix)
    joblib.dump(model, os.path.join(file_path_prefix, model_file))
    joblib.dump(scaler, os.path.join(file_path_prefix, scaler_file))
    joblib.dump(feature_names, os.path.join(file_path_prefix, features_file))


# Main workflow
def main():
    # i_timeframes = [2, 3, 4, 5, 6]
    # j_timeframes = [12, 24, 48, 60, 96, 120]
    # p_range = range(1,6)
    # q_range = range(1,6)
    forecasting = True
    i_timeframes = [4]
    j_timeframes = [120]
    p_range = range(3, 4)
    q_range = range(3, 4)

    for i in i_timeframes:
        for j in j_timeframes:
            for p in p_range:
                for q in q_range:
                    correlation_csv_path = f'../logs/CORRELATION_top_comovement_{j}months.csv'
                    competitors = load_competitor_tickers(correlation_csv_path)

                    # Preprocess training data
                    training_data = f"../training_data/ARIMAX_{i}mo_training_data.csv"  # Specify training data file path
                    stock_data, business_days = load_data(training_data)                 # Load training data
                    endog, exog = prepare_variables(stock_data, business_days, competitors)  # Prepare variables
                    exog_scaled, scaler = normalize_exog(exog)                           # Normalize training data
                    split_index = int(len(endog) * 0.8)                                  # Split data
                    endog_train, endog_test = endog[:split_index], endog[split_index:]   # Training and testing split (endog)
                    exog_train, exog_test = exog_scaled[:split_index], exog_scaled[split_index:]  # Exog split

                    # Train and save model
                    model_file_path = 'sARIMAX_Short_Term_PANW_Forecast.pkl'
                    model_fit = train_model(endog_train, exog_train, order=(p, 1, q))  # Train model
                    feature_names = exog_scaled.columns.tolist()                      # Save feature names
                    save_model(model_fit, scaler, feature_names, model_file=model_file_path)

                    # Evaluate the model
                    if forecasting:
                        forecast_results_self = forecast_model(model_fit, endog_test, exog_test)
                        _ = evaluate_model(endog_test, forecast_results_self, i, j, p, q, metrics_path='../logs/ARIMAX_metrics')

                        # Generate a 3-day forecast
                        forecast_df = generate_forecast(model_fit, exog_scaled, stock_data['Close'], days=5, scaler=scaler)
                        forecast_df.to_csv(f'../logs/ARIMAX_3day_forecast_{i}mo.csv')

if __name__ == "__main__":
    main()