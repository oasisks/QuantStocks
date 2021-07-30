import pandas as pd
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


def ema(series: pd.Series, periods=14, fillna=False):
    return EMAIndicator(series, periods, fillna).ema_indicator()


def atr(close: pd.Series, low: pd.Series, high: pd.Series, periods=14, fillna=False):
    return AverageTrueRange(close=close, high=high, low=low, window=periods, fillna=fillna)


def bollinger_bands(close: pd.Series, periods=20, std=2, fillna=False):
    bands = BollingerBands(close, periods, std, fillna)

    d_bands = {"HighBands": bands.bollinger_hband(), "LowBands": bands.bollinger_lband(),
               "MiddleBands": bands.bollinger_mavg(), "HighBands_I": bands.bollinger_hband_indicator(),
               "LowBands_I": bands.bollinger_lband_indicator()}

    return d_bands


def rsi(close: pd.Series, periods=14, fillna=False):
    return RSIIndicator(close, periods, fillna).rsi()


def obv(close: pd.Series, volume: pd.Series, fillna=False):
    return OnBalanceVolumeIndicator(close, volume, fillna).on_balance_volume()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period=14, fillna=False):
    return VolumeWeightedAveragePrice(high=high, low=low,
                                      close=close, volume=volume,
                                      window=period, fillna=fillna).volume_weighted_average_price()


def first_derivative_obv(close: pd.Series, volume: pd.Series, time: pd.Series, fillna=False):
    c_obv = obv(close, volume)
    first_derivatives = []

    for index in range(len(c_obv)):
        if index == 0:
            first_derivatives.append(0)
            continue
        current_obv = c_obv[index]
        current_time = time[index]

        previous_obv = c_obv[index - 1]
        previous_time = time[index - 1]

        delta_obv = current_obv - previous_obv
        delta_time = (current_time - previous_time).total_seconds() * 1000  # delta time will be in milliseconds

        derivative = delta_obv / delta_time

        first_derivatives.append(derivative)

    ser = pd.Series(first_derivatives, copy=False)

    return ser
