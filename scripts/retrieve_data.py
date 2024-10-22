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

        stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
        stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
        stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()
        stock_data['EMA_200'] = stock_data['Close'].ewm(span=200, adjust=False).mean()
        stock_data['Volume'] = stock_data['Volume']

        return stock_data

    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        return pd.DataFrame()

# Main function to retrieve and save stock data
def main():
    for timeframe in range(1, 12):
        csv_file = '../logs/CORRELATION_top_comovement.csv'
        tickers = load_tickers_from_csv(csv_file)
        print(f"Tickers loaded: {tickers}")

        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.today() - pd.DateOffset(months=timeframe)).strftime('%Y-%m-%d')

        all_stock_data = pd.DataFrame()

        for ticker in tickers:
            stock_data = get_stock_data(ticker, start_date, end_date)
            if not stock_data.empty:
                stock_data['Ticker'] = ticker
                all_stock_data = pd.concat([all_stock_data, stock_data])

        all_stock_data.to_csv(f"../training_data/ARIMAX_{timeframe}mo_training_data.csv")
        print("Data retrieval complete. Stock data saved.")

if __name__ == "__main__":
    main()
