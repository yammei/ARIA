import yfinance as yf
import pandas as pd

# Predefined list of cybersecurity and software companies
cybersec_software_tickers = [
    "PANW", "CRWD", "ZS", "FTNT", "CHKP", "CSCO", "AVGO", "IBM", "MSFT",  # Cybersecurity & Enterprise Software
    "SNPS", "CDNS", "ADBE", "NOW", "ORCL",                                # Major Software Companies
    "CIBR", "BUG"                                                         # Cybersecurity ETFs
]

# Function to download stock data
def download_stock_data(tickers, months_back):
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.today() - pd.DateOffset(months=months_back)).strftime('%Y-%m-%d')

    print(f"Downloading stock data from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    return data, start_date, end_date

# Main function
def main(months_back):
    for i in range(months_back, months_back + 1):
        # Ensure PANW and cybersecurity/software tickers are included
        tickers = cybersec_software_tickers.copy()

        # Download stock data
        data, start_date, end_date = download_stock_data(tickers, months_back)

        # Calculate correlation matrix
        correlation_matrix = data.corr()

        # Analyze correlation with PANW
        panw_correlation = correlation_matrix['PANW'].drop('PANW')
        top_companies = panw_correlation.sort_values(ascending=False).head(10)

        # Save to CSV
        output_file = f'../logs/CORRELATION_top_comovement_{i}months.csv'
        top_companies.to_csv(output_file)

        # Print results
        print(top_companies)
        print(f"Saved to '{output_file}' (Data from {start_date} to {end_date})")

# Specify how many months back you want to look
if __name__ == "__main__":
    months_back = 12*8
    main(months_back)
