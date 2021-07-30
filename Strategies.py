import pandas as pd
import numpy as np
from ta.utils import dropna
import Indicators as indicators
from backtesting import Backtest, Strategy
from backtesting.test import SMA
from backtesting.lib import crossover

# df = data("MRNA")
# df = dropna(df)


class VolumeIndicatorOBV(Strategy):
    """

    """
    def init(self):
        # all of the closing prices
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        price = pd.Series(self.data.Close)
        time = pd.Series(self.data.index)
        volume = pd.Series(self.data.Volume)

        first_derivative_obv = indicators.first_derivative_obv(price, volume, time)
        # EMAs for checking bullish/bearish sentiments
        self.shortTermEMA = self.I(indicators.ema, price, 9)
        self.longTermEMA = self.I(indicators.ema, price, 15)

        # EMAs of OBV
        self.shortTermDerivativeOBV = self.I(indicators.ema, first_derivative_obv, 50)
        self.longTermDerivativeOBV = self.I(indicators.ema, first_derivative_obv, 200)

        # VWAP
        self.vwap = self.I(indicators.vwap, high, low, price, volume)

        # OBV
        self.obv = self.I(indicators.obv, price, volume)

    def next(self):
        # we need to check if the short term OBV also crossed above the long term OBV
        if crossover(self.shortTermDerivativeOBV, self.longTermDerivativeOBV):
            # we buy
            self.buy()
        # we also want to sell when the profit/loss is 150 or 100 respectively
        elif self.position.pl >= 30 or self.position.pl <= -10:
            # print(self.position.pl)
            self.position.close()
        # a sell signal if the vwap crosses above the closing price
        elif crossover(self.vwap, self.data.Close):
            self.position.close()


        # print(self.data.Close.s[-1])


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

#
# bt = Backtest(df, VolumeIndicatorOBV, commission=0, exclusive_orders=True, cash=10000)
# stats = bt.run()
# bt.plot()
# print(stats)