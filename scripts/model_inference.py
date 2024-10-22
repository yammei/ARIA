import pandas as pd
import joblib

# Load the saved model, scaler, and feature names
def load_model(model_filename="../model/sARIMAX_Short_Term_PANW_Forecast.pkl", scaler_filename="../model/scaler.pkl", features_filename="../model/features.pkl"):
    model_fit = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    feature_names = joblib.load(features_filename)
    print(f"Feature names loaded from the model: {feature_names}")
    return model_fit, scaler, feature_names

# Function to load the most recent stock data (for inference)
def load_recent_data(filename, exog_vars, competitors):
    stock_data = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
    exog = stock_data[exog_vars].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Add competitor stock prices (shifted by 1 day) as exogenous variables
    for competitor in competitors:
        competitor_data = stock_data[stock_data['Ticker'] == competitor]
        exog.loc[:, f'{competitor}_Close_lag1'] = competitor_data['Close'].shift(1)  # Explicitly use .loc

    return stock_data, exog

def normalize_exog_for_inference(exog, scaler, expected_columns):
    # Ensure the columns match the expected columns from the training
    exog = exog.reindex(columns=expected_columns, fill_value=0)

    # Normalize the entire exog data (including competitor close prices)
    exog_scaled = pd.DataFrame(scaler.transform(exog), columns=exog.columns, index=exog.index)

    print("Expected columns:", expected_columns)
    print("Actual columns in exog after normalization:", exog_scaled.columns)

    return exog_scaled



# Function to generate forecasts for the future using the loaded model
def generate_forecast(model_fit, exog_scaled, endog, days=3):
    # Get the last row of exogenous data to predict the future
    last_exog_values = exog_scaled.iloc[-1]

    # Create a DataFrame for the next few days (e.g., 3 days)
    future_exog = pd.DataFrame([last_exog_values] * days, columns=exog_scaled.columns)

    # Generate future dates
    last_date = endog.index[-1]
    future_dates = pd.date_range(last_date, periods=days + 1, freq='B')[1:]
    future_exog.index = future_dates

    # Forecast for the next 'days' days
    forecast_values = model_fit.forecast(steps=days, exog=future_exog)

    # Create a DataFrame for the forecast results
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': forecast_values
    }).set_index('Date')

    return forecast_df

# Main function for performing inference
def main():
    # Load the saved model, scaler, and feature names
    model_fit, scaler, feature_names = load_model()

    # List of exogenous variables used during training
    exog_vars = ['EMA_12', 'EMA_26']
    competitors = ["CRWD", "CHKP", "ZS", "CSCO", "IBM", "CIBR", "BUG"]

    # Load the most recent stock data for inference
    stock_data, exog = load_recent_data('../training_data/ARIMAX_training_data_v1.csv', exog_vars, competitors)


    # Normalize exog data using the saved scaler
    exog_scaled = normalize_exog_for_inference(exog, scaler, feature_names)

    # Generate the future forecast
    forecast_df = generate_forecast(model_fit, exog_scaled, stock_data['Close'], days=3)

    # Print the forecast
    print("\nForecast for the next 3 days with labeled columns:")
    print(forecast_df)

    # Save forecast results to CSV
    forecast_df.to_csv('../logs/ARIMAX_forecast_future.csv')

if __name__ == "__main__":
    main()
