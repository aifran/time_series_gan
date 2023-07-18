import os
import pandas as pd
import yfinance as yf
import datetime

from config import DATA_DIR


class HistoricalStockDataLoader:
    def __init__(self, ticker:str, start_date, end_date = None, demo=True):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

        self._data = None

    def load(self):
        if self._data is None:
            file_name = os.path.join(DATA_DIR, f"{self.ticker}.csv")
            try:
                self._data = pd.read_csv(file_name)
            except FileNotFoundError:
                self._data = yf.download(self.ticker, self.start_date, self.end_date).reset_index()
                self._data.to_csv(file_name, index = False)
        return self._data

if __name__ == "__main__":
    # Set the start and end date
    start_date = '2020-01-01'
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Set the ticker
    ticker = 'AMZN'

    loader = HistoricalStockDataLoader(ticker, start_date, end_date)
    data = loader.load()
    print("Done")
