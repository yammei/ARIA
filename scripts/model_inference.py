import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

# Function to load the saved model and scaler
def load_model(model_filename="../model/sarimax_model.pkl", scaler_filename="../model/scaler.pkl", features_filename="../model/features.pkl"):
    model_fit = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    feature_names = joblib.load(features_filename)  # Load saved feature names
    return model_fit, scaler, feature_names

# Function to load new data for inference
def load_recent_data(filename, exog_vars, competitors):
    stock_data = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
    exog = stock_data[exog_vars]

    for competitor in competitors:
        competitor_data = stock_data[stock_data['Ticker'] == competitor]
        exog[f'{competitor}_Close_lag1'] = competitor_data['Close'].shift(1)

    return stock_data, exog

def load_competitor_tickers(csv_file):
    try:
        tickers_df = pd.read_csv(csv_file)
        tickers = tickers_df['Ticker'].tolist()
        return tickers
    except Exception as e:
        print(f"Error loading tickers from CSV: {e}")
        return []

# Normalize exogenous variables using the saved scaler
def normalize_exog_for_inference(exog, scaler, feature_names):
    exog = exog.reindex(columns=feature_names)

    # Fill any remaining NaNs with zeros to avoid errors
    exog.fillna(0, inplace=True)

    exog_scaled = pd.DataFrame(scaler.transform(exog), columns=exog.columns, index=exog.index)
    
    # Debugging step: Print the first few rows of exog before and after scaling
    print("Exogenous variables before scaling:\n", exog.head())
    print("Exogenous variables after scaling:\n", exog_scaled.head())
    
    return exog_scaled

# Denormalize the forecast back to original scale
def denormalize_forecast(forecast, original_data, scaler):
    forecast = forecast.values.reshape(-1, 1)
    original_data_reshaped = original_data.values.reshape(-1, 1)

    # Fit scaler to original data's Close column
    scaler.fit(original_data_reshaped)

    # Debugging step: Print before denormalization
    print("Forecast values before denormalization:\n", forecast)

    # Denormalize forecasted values
    denormalized_forecast = scaler.inverse_transform(forecast).flatten()

    # Debugging step: Print after denormalization
    print("Denormalized forecast values:\n", denormalized_forecast)

    return denormalized_forecast

def generate_forecast(model_fit, exog_scaled, endog, days=1, scaler=None):
    # Get today's date instead of using the last date from the data
    today = pd.Timestamp.today()
    print(f"TODAY: {today}")

    # Use the last exogenous values to extend the forecast for `days` ahead
    last_exog = exog_scaled.iloc[-1]
    future_exog = pd.DataFrame([last_exog] * days, columns=exog_scaled.columns)

    # Create the future business dates starting from today
    future_dates = pd.date_range(today, periods=days, freq='B')
    future_exog.index = future_dates

    # Use the model's forecast function, but make sure the prediction starts from today
    forecast_values = model_fit.get_forecast(steps=days, exog=future_exog).predicted_mean

    # Debugging step: Print forecast values before denormalization
    # print("Raw forecast values:\n", forecast_values)

    # Return the forecast as a DataFrame
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': forecast_values}).set_index('Date')
    print(f"{forecast_df}")
    return forecast_df

# Main function for inference
def main():
    # Load the saved model, scaler, and feature names
    model_file = '../model/sARIMAX_Short_Term_PANW_Forecast.pkl'
    scaler_file = '../model/scaler.pkl'
    features_file = '../model/features.pkl'

    model_fit, scaler, feature_names = load_model(model_file, scaler_file, features_file)

    # Load recent data for inference
    exog_vars = ['EMA_12', 'EMA_26', 'Volume']
    competitors = load_competitor_tickers(csv_file='../logs/CORRELATION_top_comovement_120months.csv')
    stock_data, exog = load_recent_data('../training_data/ARIMAX_4mo_training_data.csv', exog_vars, competitors)

    # Normalize exog data
    exog_scaled = normalize_exog_for_inference(exog, scaler, feature_names)

    # Generate predictions and denormalize
    forecast_df = generate_forecast(model_fit, exog_scaled, stock_data['Close'], days=3, scaler=scaler)

    # Save forecast to CSV
    forecast_df.to_csv('../logs/ARIMAX_forecast_results.csv')
    print("Predictions saved to ARIMAX_forecast_results.csv")

if __name__ == "__main__":
    main()
