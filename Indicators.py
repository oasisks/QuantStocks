import pandas as pd
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator


def ema(close: pd.Series, periods=14, fillna=False):
    return EMAIndicator(close, periods, fillna).ema_indicator()


def atr(close: pd.Series, low: pd.Series, high: pd.Series, periods=14, fillna=False):
    return AverageTrueRange(close=close, high=high, low=low, window=periods, fillna=fillna)


def bollingerBands(close: pd.Series, periods=20, std=2, fillna=False):
    bands = BollingerBands(close, periods, std, fillna)

    d_bands = {"HighBands": bands.bollinger_hband(), "LowBands": bands.bollinger_lband(),
               "MiddleBands": bands.bollinger_mavg(), "HighBands_I": bands.bollinger_hband_indicator(),
               "LowBands_I": bands.bollinger_lband_indicator()}

    return d_bands


def rsi(close: pd.Series, periods=14, fillna=False):
    return RSIIndicator(close, periods, fillna).rsi()


def obv(close: pd.Series, volume: pd.Series, fillna=False):
    return OnBalanceVolumeIndicator.on_balance_volume()

