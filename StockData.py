import os
import yfinance as yf
import csv
import pandas as pd
import math
import requests
import json
from backtesting import Backtest
from Strategies import VolumeIndicatorOBV
from ta.utils import dropna
from os import listdir
from os.path import isfile, join


def data(ticker: str = "AAPL", period: str = "30d", interval: str = "15m"):
    df = yf.download(tickers=ticker, period=period, interval=interval)
    df.drop("Adj Close", axis=1, inplace=True)
    return df


class Universe:
    """
    A universe is a class that contains all of the stock data that would be used for backtesting
    URL: https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&download=true
    URL: https://api.nasdaq.com/api/quote/ABCM/chart?assetclass=stocks&fromdate=2021-07-01&todate=2021-07-28
    """

    def __init__(self, strategies: list, exchanges: list = ("NYSE", "NASDAQ", "AMEX"), period="60d", interval="15m",
                 min_volume=100000, set_industry="", set_sector="Miscellaneous",
                 set_country="United States", min_market_cap=1000000):
        """
        :param strategies: a list of strategies (the strategies must inherent the strategy class)
        :param exchanges: can be NYSE, NASDAQ, or AMEX
        :param period: 60d is the maximum
        :param interval: 5m, 10m, 15m, 30m, 45m, 1h, 1d, and so on
        :param min_volume: the minimum volume (float)
        :param set_industry: the industry the ticker is in (please look below for a list)
        :param set_sector: the sector the ticker is in (please look below for a list)
        :param set_country: the country the screener is interested in (please look below for a list)
        :param min_market_cap: the minimum market cap
        """
        # by default, the stocks within this universe will be from the NYSE and NASDAQ exchanges
        self.strategies = strategies
        self.exchanges = exchanges
        self.period = period
        self.interval = interval

        # screener parameters
        self._min_volume = min_volume
        self._set_industry = set_industry
        self._set_sector = set_sector
        self._set_country = set_country
        self._min_market_cap = min_market_cap

        # we will also be generating all of the dataframes for each ticker
        self.screener = self.__generate_screener()
        self.exchange_dfs = self.__generate_dataframes()
        self.win_rate = []

    def __generate_screener(self) -> pd.DataFrame:
        """

        :param min_volume: the minimum volume the screener will look for
        :param set_industry: the industry that we are interested in screening
        :param set_sector: the sector that we are interested in screening
        :param set_country: the country that the stock was founded in
        :return: a dict of tickers that meets all of the criteria
        TODO: when screening for stocks, incorporate short interest as part of the parameters
        URL: https://www.marketwatch.com/tools/screener/short-interest
        """

        # requests
        url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&download=true"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/92.0.4515.107 Safari/537.36'}
        request = requests.get(url, headers=headers)
        ticker_data = json.loads(request.text)["data"]["rows"]

        # dataframe manipulation
        df = pd.DataFrame(ticker_data)

        volumes = pd.to_numeric(df["volume"], errors="coerce")
        market_caps = pd.to_numeric(df["marketCap"], errors="coerce")

        df.drop(columns=["volume", "marketCap"], axis=1, inplace=True)

        df["volume"] = volumes
        df["marketCap"] = market_caps

        target_ticker = df[(df["volume"] >= self._min_volume) & (df["industry"] == self._set_industry)
                           & (df["marketCap"] >= self._min_market_cap) & (df["country"] == self._set_country)
                           & (df["industry"] == self._set_industry)]
        return target_ticker.reset_index(drop=True)

    def __generate_dataframes(self) -> pd.DataFrame:
        """
        The dataframes are separated into folders base off intervals. Each folder (interval) will contain dataframe
        files. The interval is base off the input when Universe is initialized.

        :return: returns a list of dataframes of stock data gathered from yfinance. If screener is not None, then
        the method will only incorporate tickers within the screener as part of the universe.
        """
        c_exchanges = [exchange.strip(".csv") for exchange in listdir("Exchanges") if
                       isfile(join("Exchanges", exchange))]

        print(c_exchanges)
        # we will gather all of the ticker names from these exchanges (separated by exchanges for filtering later)
        exchange_dfs = {}
        for exchange in self.exchanges:
            dfs = []

            if exchange in c_exchanges:
                directory = f"Exchanges/{exchange}/{self.interval}"
                if not os.path.isdir(directory):
                    os.mkdir(directory)

                exchange_files = [file.strip(".pkl") for file in listdir(directory)]

                file = open(f"Exchanges/{exchange}.csv", "r", encoding="utf-8")
                csv_reader = csv.DictReader(file)

                # if the list is empty
                if not exchange_files:
                    # we need to generate all of the dataframes and save it into that directory
                    for row in csv_reader:
                        ticker = row["Symbol"]

                        # there was some problems with using the in keyword to determine whether the ticker exists in
                        # the screener. Thus, we will be using this instead
                        boolean_findings = self.screener.symbol.str.contains(ticker)
                        total_occurrence = boolean_findings.sum()
                        # if the ticker does not exist within the screener
                        if total_occurrence == 0:
                            continue

                        if "/" in ticker:
                            continue
                        df = data(ticker, self.period, self.interval)
                        print(f"Download {ticker}")
                        # save these dataframes for future uses
                        df.to_pickle(f"{directory}/{ticker}.pkl")
                        dfs.append({ticker: df})
                # if there are already dataframes that exist within the folders, we will just import those
                else:
                    for row in csv_reader:
                        ticker = row["Symbol"]
                        # there was some problems with using the in keyword to determine whether the ticker exists in
                        # the screener. Thus, we will be using this instead
                        boolean_findings = self.screener.symbol.str.contains(ticker)
                        total_occurrence = boolean_findings.sum()
                        # if the ticker does not exist within the screener
                        if total_occurrence == 0:
                            continue

                        if "/" in ticker:
                            continue
                        # if the ticker already exists within the folder we just add them
                        if ticker in exchange_files:
                            df = pd.read_pickle(f"{directory}/{ticker}.pkl")
                            dfs.append({ticker: df})
                        # if it doesn't then we need to create them and save them
                        else:
                            df = data(ticker, self.period, self.interval)
                            print(f"Download {ticker}")
                            df.to_pickle(f"{directory}/{ticker}.pkl")
                            dfs.append({ticker: df})
                exchange_dfs[exchange] = dfs
        return exchange_dfs

    def back_test(self):
        """
        When this function is called, the program will go through the entire universe of stocks and back test each stock
        using the Backtesting.py module.x
        :return:
        """
        index = 0
        for exchange, dfs in self.exchange_dfs.items():
            for df in dfs:
                ticker = list(df.keys())[0]
                # print(f"The current ticker is: {ticker}")
                df = df[ticker]
                df = dropna(df)
                if df.empty:
                    continue
                # print(df)
                for strategy in self.strategies:
                    bt = Backtest(df, strategy, commission=0, exclusive_orders=True, cash=100000)
                    stats = bt.run()  # returns a pd.Series
                    # bt.plot()
                    # print(f"Win rate: {stats['Win Rate [%]']}")
                    # print(f"Finished {ticker}")
                    win_rate = stats["Win Rate [%]"]
                    if math.isnan(win_rate) or win_rate == 0.0:
                        continue
                    self.win_rate.append(win_rate)
                    print(win_rate)
                index += 1
                if index == 100:
                    break
            break

        self.win_rate = sum(self.win_rate) / len(self.win_rate)

    @staticmethod
    def list_of_countries():
        return """
        1. United States
        3. China
        4. Canada
        5. Switzerland
        6. United Kingdom
        7. Hong Kong
        8. Bermuda
        9. Ireland
        10. Netherlands
        11. Germany
        12. Brazil
        13. Luxembourg
        14. Chile
        15. Israel
        16. Sweden
        17. Mexico
        18. Taiwan
        19. Argentina
        20. Cayman Islands
        21. Singapore
        22. Denmark
        23. Australia
        24. South Africa
        25. France
        26. India
        27. Peru
        28. Spain
        29. Panama
        30. Belgium
        31. Japan
        32. Colombia
        33. Greece
        34. Cyprus
        35. Jersey
        36. Uruguay
        37. Guernsey
        38. Macau
        39. Italy
        40. Norway
        41. Costa Rica
        42. Puerto Rico
        43. Kazakhstan
        44. Monaco
        45. Malta
        46. South Korea
        47. Turkey
        48. Portugal
        49. Jordan
        50. Russia
        51. New Zealand
        52. Finland
        53. Isle of Man
        54. Curacao
        55. Bahamas
        56. Philippines
        57. Indonesia
        58. United Arab Emirates"""

    @staticmethod
    def list_of_sector():
        return """
        1. Capital Goods
        2. Basic Industries
        3. Finance
        4. Miscellaneous
        5. Consumer Services
        6. Transportation
        7. Technology
        8. Consumer Durables
        9. Health Care
        10. Consumer Non-Durables
        11. Energy
        12. Public Utilities"""

    @staticmethod
    def list_of_industry():
        return """
        1. Electrical Products
        2. Metal Fabrications
        3. Business Services
        4. Service to the Health Industry
        5. Real Estate Investment Trusts
        6. 
        7. Air Freight/Delivery Services
        8. Investment Managers
        9. Life Insurance
        10. Diversified Commercial Services
        11. Semiconductors
        12. Industrial Machinery/Components
        13. Other Specialty Stores
        14. Computer Manufacturing
        15. Precious Metals
        16. Transportation Services
        17. Other Pharmaceuticals
        18. Major Banks
        19. Biotechnology: Biological Products (No Diagnostic Substances)
        20. Major Pharmaceuticals
        21. Beverages (Production/Distribution)
        22. Biotechnology: In Vitro & In Vivo Diagnostic Substances
        23. Medical/Dental Instruments
        24. Investment Bankers/Brokers/Service
        25. Engineering & Construction
        26. Advertising
        27. Recreational Products/Toys
        28. Property-Casualty Insurers
        29. Aluminum
        30. Hospital/Nursing Management
        31. Food Chains
        32. EDP Services
        33. Military/Government/Technical
        34. Specialty Chemicals
        35. Multi-Sector Companies
        36. Home Furnishings
        37. Finance/Investors Services
        38. Computer Software: Prepackaged Software
        39. Industrial Specialties
        40. Farming/Seeds/Milling
        41. Auto Parts:O.E.M.
        42. Diversified Financial Services
        43. Telecommunications Equipment
        44. Medical/Nursing Services
        45. Oilfield Services/Equipment
        46. Power Generation
        47. Building Products
        48. Building operators
        49. Biotechnology: Laboratory Analytical Instruments
        50. Clothing/Shoe/Accessory Stores
        51. Electric Utilities: Central
        52. Services-Misc. Amusement & Recreation
        53. Trusts Except Educational Religious and Charitable
        54. Real Estate
        55. Biotechnology: Electromedical & Electrotherapeutic Apparatus
        56. Accident &Health Insurance
        57. Other Consumer Services
        58. Managed Health Care
        59. Finance Companies
        60. Water Supply
        61. Medical Specialities
        62. Finance: Consumer Services
        63. Textiles
        64. Professional Services
        65. Aerospace
        66. Radio And Television Broadcasting And Communications Equipment
        67. Specialty Insurers
        68. Major Chemicals
        69. Ophthalmic Goods
        70. Computer Software: Programming Data Processing
        71. Computer peripheral equipment
        72. Oil & Gas Production
        73. Natural Gas Distribution
        74. Commercial Banks
        75. Movies/Entertainment
        76. Containers/Packaging
        77. Broadcasting
        78. Assisted Living Services
        79. Coal Mining
        80. Office Equipment/Supplies/Services
        81. Retail: Computer Software & Peripheral Equipment
        82. Catalog/Specialty Distribution
        83. Trucking Freight/Courier Services
        84. Restaurants
        85. Diversified Manufacture
        86. Auto Manufacturing
        87. Electronic Components
        88. Marine Transportation
        89. Wholesale Distributors
        90. Construction/Ag Equipment/Trucks
        91. Steel/Iron Ore
        92. Oil/Gas Transmission
        93. Agricultural Chemicals
        94. Biotechnology: Commercial Physical & Biological Resarch
        95. Plastic Products
        96. Environmental Services
        97. Savings Institutions
        98. Paints/Coatings
        99. Consumer Electronics/Video Chains
        100. Precision Instruments
        101. Packaged Foods
        102. Department/Specialty Retail Stores
        103. RETAIL: Building Materials
        104. Computer Communications Equipment
        105. Integrated oil Companies
        106. Meat/Poultry/Fish
        107. Specialty Foods
        108. Food Distributors
        109. Apparel
        110. Misc Health and Biotechnology Services
        111. Hotels/Resorts
        112. Homebuilding
        113. Shoe Manufacturing
        114. Rental/Leasing Companies
        115. Banks
        116. Television Services
        117. Pollution Control Equipment
        118. Fluid Controls
        119. Package Goods/Cosmetics
        120. Medical Electronics
        121. Oil Refining/Marketing
        122. Paper
        123. Diversified Electronic Products
        124. Railroads
        125. Building Materials
        126. Publishing
        127. Automotive Aftermarket
        128. Consumer Specialties
        129. Newspapers/Magazines
        130. Other Metals and Minerals
        131. Miscellaneous manufacturing industries
        132. Motor Vehicles
        133. Internet and Information Services
        134. Electronics Distribution
        135. Mining & Quarrying of Nonmetallic Minerals (No Fuels)
        136. Forest Products
        137. Consumer Electronics/Appliances
        138. Ordnance And Accessories
        139. Other Transportation
        140. Miscellaneous
        141. EDP Peripherals
        142. Books
        143. Tobacco
        144. Tools/Hardware"""


if __name__ == '__main__':
    universe = Universe([VolumeIndicatorOBV], interval="5m")
    print(universe.screener)
    print(universe.exchange_dfs)
    # universe.back_test()
    # print(universe.win_rate)
