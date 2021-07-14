import pandas as pd
import numpy as np
from ta.utils import dropna
from StockData import data
import Indicators as indicators
from backtesting import Backtest, Strategy
from backtesting.test import SMA
from backtesting.lib import crossover



df = data("SNDL")
df = dropna(df)


class GoldenCross(Strategy):
    def init(self):
        # this is the place to create all of the indicators that needs to be used
        # the self.I takes in an indicator function that returns a pandas series
        # however, the self.I function returns a pandas ndarray
        price = pd.Series(self.data.Close)  # of type series
        self.shortEMA = self.I(indicators.ema, price, 20)
        self.shortSMA = self.I(SMA, price, 50)  # this is a np.ndarray
        self.longSMA = self.I(SMA, price, 200)  # this is a np.ndarray

    def next(self):
        if crossover(self.shortSMA, self.longSMA):
            self.buy()
        elif crossover(self.longSMA, self.shortSMA):
            self.sell()


bt = Backtest(df, GoldenCross, commission=0, exclusive_orders=True)
stats = bt.run()
bt.plot()
print(stats)