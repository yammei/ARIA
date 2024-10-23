import yfinance as yf
import pandas as pd

# Read tickers from CSV
def load_tickers_from_csv(csv_file):
    try:
        tickers_df = pd.read_csv(csv_file)
        tickers = tickers_df['Ticker'].tolist()
        if 'PANW' not in tickers:
            tickers.insert(0, 'PANW')
        return tickers
    except Exception as e:
        print(f"Error loading tickers from CSV: {e}")
        return []

# Retrieve stock data and EMAs
def get_stock_data(ticker, start_date, end_date):
    try:
        print(f"Retrieving data for {ticker} from {start_date} to {end_date}...")
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if stock_data.empty:
            print(f"Warning: No data found for {ticker}")
            return pd.DataFrame()

        # Calculate Exponential Moving Averages (12-day, 26-day, 50-day, and 200-day)
        stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
        stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
        stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()
        stock_data['EMA_200'] = stock_data['Close'].ewm(span=200, adjust=False).mean()
        stock_data['Ticker'] = ticker  # Add the Ticker column

        # Filter for only the required columns
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'EMA_12', 'EMA_26', 'EMA_50', 'EMA_200', 'Ticker']]

        return stock_data

    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        return pd.DataFrame()

# Main function to retrieve and save stock data
def main():
    timeframes = [3]  # Example: Retrieve data from the last 3 months
    for timeframe in timeframes:
        csv_file = '../logs/CORRELATION_top_comovement.csv'
        tickers = load_tickers_from_csv(csv_file)
        print(f"Tickers loaded: {tickers}")

        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.today() - pd.DateOffset(months=timeframe)).strftime('%Y-%m-%d')

        all_stock_data = pd.DataFrame()

        for ticker in tickers:
            stock_data = get_stock_data(ticker, start_date, end_date)
            if not stock_data.empty:
                all_stock_data = pd.concat([all_stock_data, stock_data])

        # Reorder columns to ensure Date is the first column
        all_stock_data.reset_index(inplace=True)  # Reset index to ensure 'Date' is a column
        columns_order = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'EMA_12', 'EMA_26', 'EMA_50', 'EMA_200', 'Ticker']
        all_stock_data = all_stock_data[columns_order]

        print(f"DATA:\n{all_stock_data}")

        # Save the data to a CSV file for further analysis
        all_stock_data.to_csv(f"../training_data/ARIMAX_{timeframe}mo_training_data.csv", index=False)
        print("Data retrieval complete. Stock data saved.")

if __name__ == "__main__":
    main()
