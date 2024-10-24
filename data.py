import requests
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO

class NSE():
    def __init__(self, timeout=10):
        self.base_url = 'https://www.nseindia.com'
        self.session = requests.sessions.Session()
        self.session.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36 Edg/97.0.1072.55",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "accept-language": "en-US,en;q=0.9"
        }
        self.timeout = timeout
        self.session.get(self.base_url, timeout=timeout)

    def getHistoricalData(self, symbol, series, from_date, to_date):
        try:
            all_data = []
            start_date = pd.Timestamp(from_date.strftime('%Y-%m-%d'))
            to_date_str = to_date.strftime('%Y-%m-%d')
            to_date = pd.Timestamp(to_date_str)

            while start_date <= to_date:
                end_date = start_date + pd.DateOffset(years=1)
                if end_date > to_date:
                    end_date = to_date

                url = "/api/historical/cm/equity?symbol={0}&series=[%22{1}%22]&from={2}&to={3}&csv=true".format(
                    symbol.replace('&', '%26'), series, start_date.strftime('%d-%m-%Y'), end_date.strftime('%d-%m-%Y'))
                r = self.session.get(self.base_url + url, timeout=self.timeout)
                df = pd.read_csv(BytesIO(r.content), sep=',', thousands=',')
                df = df.rename(columns={'Date ': 'date', 'OPEN ': 'open', 'HIGH ': 'high', 'LOW ': 'low', 'close ': 'close'})
                df.date = pd.to_datetime(df.date).dt.strftime('%Y-%m-%d')
                df['symbol'] = symbol  # Add the stock symbol as a column
                all_data.insert(0, df)  

                start_date = end_date + pd.DateOffset(days=1)

            if all_data:
                return pd.concat(all_data, ignore_index=True).iloc[::-1]
            else:
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    # Function 1: Save past 10 years' data to a single CSV for all stocks
    def save_past_10_years_data(self, stock_list):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 10)  # Past 10 years

        all_stock_data = pd.DataFrame()  # Initialize an empty DataFrame to store all stock data

        for stock in stock_list:
            print(f"Fetching data for {stock}...")
            df = self.getHistoricalData(stock, 'EQ', start_date, end_date)
            if df is not None:
                all_stock_data = pd.concat([all_stock_data, df], ignore_index=True)  # Append each stock's data
            else:
                print(f"Failed to fetch data for {stock}")

        if not all_stock_data.empty:
            file_name = 'all_stocks_historical_data.csv'
            all_stock_data.to_csv(file_name, index=False)
            print(f"All stock data saved to {file_name}")
        else:
            print("No data fetched for any stock.")

    # Function 2: Return past 7 days' data without saving
    def get_past_7_days_data(self, stock):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=146)  # Past 7 days
        df = self.getHistoricalData(stock, 'EQ', start_date, end_date)
        if df is not None:
            return df
        else:
            print(f"Failed to fetch data for {stock}")
            return None

if __name__ == '__main__':
    nse = NSE()
    
    # Example usage
    stock_list = ['RELIANCE', 'TCS', 'HDFCBANK', 'BRITANNIA', 'ICICIBANK', 'SBIN', 'INFY',
                     'HINDUNILVR', 'ITC', 'LT', 'BAJFINANCE', 'ADANIENT', 'MARUTI', 'NTPC',
                     'AXISBANK', 'HCLTECH', 'TATAMOTORS', 'M&M', 'ULTRACEMCO', 'TITAN', 'ASIANPAINT',
                     'BAJAJ-AUTO', 'WIPRO', 'JSWSTEEL', 'NESTLEIND']  # Modify the stock symbols here

    # Function 1: Save past 10 years' data into a single CSV file for all stocks
    # nse.save_past_10_years_data(stock_list)

    # Function 2: Get past 7 days' data for a single stock without saving
    stock_data = nse.get_past_7_days_data('RELIANCE')
