import yfinance as yf
import pandas as pd

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    sp500_df = table[0]
    return sp500_df['Symbol'].to_list()

tickers = get_sp500_tickers()
if 'PANW' not in tickers:
    tickers.append('PANW')
data = yf.download(tickers, start="2023-01-01", end="2023-12-31")['Adj Close']

correlation_matrix = data.corr()

panw_correlation = correlation_matrix['PANW'].drop('PANW')
top_companies = panw_correlation.sort_values(ascending=False).head(25)
top_companies.to_csv('../logs/CORRELATION_top_comovement.csv')

print(top_companies)
print("Saved to 'CORRELATION_top_comovement.csv'")
