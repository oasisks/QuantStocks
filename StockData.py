import yfinance as yf
import csv
import pandas as pd
from os import listdir
from os.path import isfile, join


def data(ticker: str = "AAPL", period: str = "30d", interval: str = "15m"):
    df = yf.download(tickers=ticker, period=period, interval=interval)
    df.drop("Adj Close", axis=1, inplace=True)
    return df


class Universe:
    """
    A universe is a class that contains all of the stock data that would be used for backtesting
    """
    def __init__(self, exchanges: list = ("NYSE", "NASDAQ"), period="60d", interval="15m"):
        # by default, the stocks within this universe will be from the NYSE and NASDAQ exchanges
        self.exchanges = exchanges
        self.period = period
        self.interval = interval

        # we will also be generating all of the dataframes for each ticker
        self.exchange_dfs = self.__generate_dataframes()

    def __generate_dataframes(self) -> dict:
        """

        :return: returns a list of dataframes of stock data gathered from yfinance
        """
        c_exchanges = [exchange.strip(".csv") for exchange in listdir("Exchanges") if isfile(join("Exchanges", exchange))]

        print(c_exchanges)
        # we will gather all of the ticker names from these exchanges (separated by exchanges for filtering later)
        exchange_dfs = {}
        for exchange in self.exchanges:
            dfs = []

            if exchange in c_exchanges:
                directory = f"Exchanges/{exchange}"
                exchange_files = [file.strip(".pkl") for file in listdir(directory)]

                file = open(f"{directory}.csv", "r", encoding="utf-8")
                csv_reader = csv.DictReader(file)

                # if the list is empty
                if not exchange_files:
                    # we need to generate all of the dataframes and save it into that directory
                    for row in csv_reader:
                        ticker = row["Symbol"]
                        if "/" in ticker:
                            continue
                        df = data(ticker, self.period, self.interval)
                        # save these dataframes for future uses
                        df.to_pickle(f"{directory}/{ticker}.pkl")
                        dfs.append(df)
                # if there are already dataframes that exist within the folders, we will just import those
                else:
                    for row in csv_reader:
                        ticker = row["Symbol"]
                        if "/" in ticker:
                            continue
                        # if the ticker already exists within the folder we just add them
                        if ticker in exchange_files:
                            df = pd.read_pickle(f"{directory}/{ticker}.pkl")
                            dfs.append(df)
                        # if it doesn't then we need to create them and save them
                        else:
                            df = data(ticker, self.period, self.interval)
                            df.to_pickle(f"{directory}/{ticker}.pkl")
                            dfs.append(df)
                exchange_dfs[exchange] = dfs
        return exchange_dfs


if __name__ == '__main__':
    universe = Universe()


