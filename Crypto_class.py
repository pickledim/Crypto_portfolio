#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 11:11:23 2021

@author: dimitrisglenis
"""

import pandas as pd
import numpy as np
import re
import time
from cryptocmd import CmcScraper
import pickle
import os
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from datetime import datetime
import clean_dir as cd


def __datetime(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d')


def remove_coins(to_remove, selected_coins):
    """
    Removes coins from a list

    Parameters
    ----------
    to_remove : TYPE list
        The coins that ou want to remove
    selected_coins : TYPE list
        the list of which you want to remove the coins

    Returns
    -------
    selected_coins : TYPE list
        DESCRIPTION.

    """

    for coin in to_remove:
        try:
            selected_coins.remove(coin)
        except ValueError:
            pass

    return selected_coins


def scrap_coin(coin, coin_csv_name):
    """
    Scraps the prices of the coins from CoinMarketCap

    Parameters
    ----------
    coin : TYPE string
        DESCRIPTION.
    coin_csv_name : TYPE string
        DESCRIPTION.

    Returns
    -------
    df : TYPE dataframe
        the historic data of the coin

    """
    print(f'\n ===== {coin} ===== \n')
    # initialise scraper without time interval
    scraper = CmcScraper(coin)

    # export the data as csv file, you can also pass optional `name` parameter
    scraper.export("csv", name=coin_csv_name)

    # Pandas dataFrame for the same data
    df = scraper.get_dataframe()

    return df


def regularize(weights, portfolio):
    # =============================================================================
    # Regularize
    # =============================================================================

    unreg = 0
    for weight, perc in weights.items():
        if perc > 0:
            unreg += perc
    reg = 1 / unreg
    for coin, value in portfolio.items():
        portfolio[coin] = value * reg

    print(f'Invest {np.array(list(portfolio.values())).sum()}')

    return portfolio


def portfolio_optimization(df, selected_coins, _budget, _mu_method, _cov_method, _obj_function, _drop=False):
    mu_mapping = {
        'mean': expected_returns.mean_historical_return,
        'exp': expected_returns.ema_historical_return,
        'capm': expected_returns.capm_return
    }

    cov_mapping = {
        'sample': risk_models.sample_cov,
        'exp': risk_models.exp_cov
    }

    df = df[selected_coins]
    # keep the coins that are in the market for at least 2 months
    for i in df.columns:
        pos1 = df[i].last_valid_index()
        pos2 = df[i].first_valid_index()
        if pos1 is not None and pos2 is not None:
            start = __datetime(pos1)
            end = __datetime(pos2)
            delta = end - start
            if delta.days < 60:
                df.drop(i, axis=1, inplace=True)
        else:
            df.drop(i, axis=1, inplace=True)

    if _drop:
        df.dropna(inplace=True)

    df = df.sort_index(ascending=True)

    # Calculate expected returns and sample covariance
    mu = mu_mapping[_mu_method](df, compounding=True, frequency=365)
    mu.replace([np.inf, -np.inf], np.nan, inplace=True)
    mu.dropna(inplace=True)
    # print(mu)
    ind_list = mu.index
    df = df[ind_list]
    S = cov_mapping[_cov_method](df, frequency=365)

    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))

    obj_f_mapping = {
        'quadratic': ef.max_quadratic_utility,
        'sharpe': ef.max_sharpe,
        'min_volat': ef.min_volatility
    }

    weights = obj_f_mapping[_obj_function]()

    clean_weights = ef.clean_weights()

    ef.portfolio_performance(verbose=True)

    port = {}
    for coin, perc in weights.items():
        if perc > 0:
            port[coin] = perc * _budget

    return port, mu, clean_weights


def get_p_l(portfolio, in_price_coins, final_price_coins):
    """
    Returns the profit-loss of the selected coins

    :param portfolio: The optimised portfolio dict
    :param in_price_coins: the initial prices of the coins series
    :param final_price_coins: the final prices of the coins series
    :return:
    portfolio in a df
    the p_l
    """

    p_l = 0
    coins_list = []
    amount_list = []
    n_coins_list = []

    for coin, amount in portfolio.items():
        in_price = in_price_coins[coin]
        fin_price = final_price_coins[coin]
        n_coins_bought = amount / in_price
        coins_list.append(coin)
        amount_list.append(amount)
        n_coins_list.append(n_coins_bought)
        p_l += n_coins_bought * (fin_price - in_price)

    print(f'\n Profit Loss: {p_l}e\n')

    portfolio = pd.DataFrame({'Coin': coins_list, 'Amount': amount_list, 'n_coins': n_coins_list})
    portfolio.sort_values(by=['Amount'], ascending=False, inplace=True)

    return portfolio, p_l


class Cryptos:

    def __init__(self, choice, _budget, _n_coins, _hodl):

        if choice:
            # loads the top100 coins
            with open(r"top_100.pickle", "rb") as input_file:
                self.coins = list(pickle.load(input_file))
            self.csv_name = 'Top_100_cryptos'
        else:
            # loads all the crypto coins
            with open(r"top_1000.pickle", "rb") as input_file:
                self.coins = list(pickle.load(input_file))
            self.csv_name = 'All_cryptos'

        self.budget = _budget
        self.hodl = _hodl
        self._n_coins = _n_coins
        self.p_l = int()
        self.selected_coins_of_past = []
        self.selected_coins = []
        self.market_cap = []
        self.date = pd.DataFrame()
        self.df_prices = pd.DataFrame()
        self.df_market_cap = pd.DataFrame()
        self.df_prices_past = pd.DataFrame()
        self.portfolio_from_past = pd.DataFrame()
        self.portfolio = pd.DataFrame()

    def regex_coins(self, file):

        file1 = open(f"{file}", "r")
        lines = file1.readlines()
        file1.close()

        cryptos = []

        for line in lines:
            if re.match(r"[A-Z]+[A-Z]+[A-Z]*[\s]+", line):
                cryptos.append(line.lstrip().rstrip())
        cryptos2 = []
        for crypto in cryptos:
            if ' ' not in crypto:
                cryptos2.append(crypto)

        self.coins = list(set(cryptos2))

    def get_prices_df(self):

        self.coins = self.coins[:self._n_coins]  # cheating -->stupid patch for coinmarket rate limit

        cryptos_list = []

        # scrap each coin of the coins list
        for i, coin in enumerate(self.coins):
            print(i)
            if i % 25 == 0 and i != 0:
                print('sleeping for 1 min...')
                time.sleep(60)
            df = scrap_coin(coin, f'{coin}_all_time')  # scraps the selected coins
            cryptos_list.append(df['Close'])  # keep the closing value column
            self.market_cap.append(df['Market Cap'])  # append the historic data of MC for each coin
            if coin == 'BTC':
                self.date = df[
                    'Date']  # assign the BTC date column as the date column of the prices df (BTC is the oldest)

        # =====================================================================
        # Post Pros
        # =====================================================================
        self.df_prices = pd.concat(cryptos_list, axis=1)  # create the prices df
        self.df_prices.columns = self.coins  # add the coin names
        self.df_prices['Date'] = self.date  # add the dates
        self.df_prices = self.df_prices.set_index(self.df_prices['Date'].values)  # set the date column as index
        self.df_prices.drop(columns=['Date'], axis=1, inplace=True)  # drop the Date column

        df = self.df_prices.copy()

        # make nan the ico of each coin check AAVE
        for i in df.columns:
            j = 0
            while j < 2:
                pos = df[i].last_valid_index()
                df.at[pos, i] = np.nan
                # df[i].loc[pos] = np.nan
                j += 1

        self.df_prices = df.copy()

        cwd = os.getcwd()
        cd.move_files(cwd, os.path.join(cwd, 'scrapped_data'))

        self.df_prices.to_csv(f'{self.csv_name}.csv')

    def get_market_cap_df(self):
        """
        creates the df of all the coins market cap
        :return:
        """
        self.df_market_cap = pd.concat(self.market_cap, axis=1)
        self.df_market_cap.columns = self.coins
        self.df_market_cap['Date'] = self.date
        self.df_market_cap = self.df_market_cap.set_index(self.df_market_cap['Date'].values)
        self.df_market_cap.drop(columns=['Date'], axis=1, inplace=True)
        self.df_market_cap.to_csv(f'{self.csv_name}_market_cap.csv')

    def validate_from_past(self, _n_coins, _n_days, _mu_method, _cov_method, _obj_function, _drop, _scrap=False):

        if not _scrap:
            self.df_prices = pd.read_csv(f'{self.csv_name}.csv', index_col=0)

        df_mc = pd.read_csv(f'{self.csv_name}_market_cap.csv', index_col=0)

        df_mc = df_mc.iloc[_n_days:, :]  # see the market cap of the coins from the specific day
        df_365 = df_mc.iloc[0, :].T  # take the market cap of only that day
        df_365.dropna(inplace=True)  #
        coins = df_365.nlargest(_n_coins, )  # keep the n largest coins in terms of market cap
        self.selected_coins_of_past = list(coins.index)  # store the names in a list

        stable_coins = ['USDT', 'USDC', 'CUSDC', 'BUSD', 'UST', 'PAX', 'DAI', 'CDAI',
                        'CETH', 'BTG', 'ETC', 'HUSD', 'TUSD', 'USDN', 'CUSDT']

        # remove the stable coins, the shitty coins and the ones that you cannot buy
        self.selected_coins_of_past = remove_coins(stable_coins, self.selected_coins_of_past)

        df_coins = self.df_prices.columns
        df_coins = df_coins.values.tolist()
        coins = []
        for coin in self.selected_coins_of_past:
            if coin in df_coins:
                coins.append(coin)
        coins = self.selected_coins_of_past

        self.df_prices = self.df_prices[coins]
        if self.hodl:
            # if you hodl check the prices of today
            prices_now = self.df_prices.iloc[0, :]
        else:
            # if you trade check the prices after one year of the investment
            prices_now = self.df_prices.iloc[_n_days - 365, :]

        # take the prices up to n_days
        self.df_prices_past = self.df_prices.iloc[_n_days:, :]
        # take the prices of the n_days
        prices_past = self.df_prices.iloc[_n_days, :]

        # portfolio optimization
        portfolio, mu, weights = portfolio_optimization(self.df_prices_past, coins, self.budget,
                                                        _mu_method, _cov_method, _obj_function, _drop)
        # regularize the portfolio
        self.portfolio_from_past = regularize(weights, portfolio)
        # get profit_loss
        self.portfolio_from_past, self.p_l = get_p_l(self.portfolio_from_past, prices_past, prices_now)

    def optimize_portfolio(self, _n_coins, _mu_method, _cov_method, _obj_function, _drop=False, _scrap=False):

        if not _scrap:
            self.df_prices = pd.read_csv(f'{self.csv_name}.csv', index_col=0)
            self.coins = self.df_prices.columns

        df_mc = pd.read_csv(f'{self.csv_name}_market_cap.csv', index_col=0)

        # take the n coins of the largest market cap
        df_365 = df_mc.iloc[0, :].T

        coins = df_365.nlargest(_n_coins, )
        df_365.dropna(inplace=True)
        self.selected_coins = list(coins.index)

        stable_coins = ['USDT', 'USDC', 'CUSDC', 'BUSD', 'UST', 'PAX', 'DAI', 'CDAI',
                        'CETH', 'BTG', 'ETC', 'HUSD', 'TUSD', 'USDN', 'CUSDT']

        # remove the stable coins, the shitty coins and the ones that you cannot buy
        self.selected_coins = remove_coins(stable_coins, self.selected_coins)

        df_coins = self.df_prices.columns
        df_coins = df_coins.values.tolist()
        coins = []
        for coin in self.selected_coins:
            if coin in df_coins:
                coins.append(coin)
        self.selected_coins = coins
        portfolio, mu, weights = portfolio_optimization(self.df_prices, self.selected_coins, self.budget,
                                                        _mu_method, _cov_method, _obj_function, _drop)
        portfolio = regularize(weights, portfolio)
        coins_list = []
        amount_list = []
        n_coins_list = []
        for coin, amount in portfolio.items():
            price = self.df_prices[coin].iloc[1]
            n_coins_bought = amount / price
            coins_list.append(coin)
            amount_list.append(amount)
            n_coins_list.append(n_coins_bought)

        self.portfolio = pd.DataFrame({'Coin': coins_list, 'Amount': amount_list, 'n_coins': n_coins_list})
        self.portfolio.sort_values(by=['Amount'], ascending=False, inplace=True)


if __name__ == "__main__":
    top_100 = True
    n_coins = 20
    n_days = 90
    mu_method = 'mean'
    cov_method = 'exp'
    obj_function = 'quadratic'
    drop = False
    budget = 100
    hodl = True
    scrap = False
    crypto_class_20c = Cryptos(top_100, budget, n_coins, hodl)

    crypto_class_20c.get_prices_df()
    crypto_class_20c.get_market_cap_df()
    crypto_class_20c.validate_from_past(n_coins, n_days, mu_method, cov_method, obj_function, drop, scrap)
    crypto_class_20c.optimize_portfolio(n_coins, mu_method, cov_method, obj_function, drop, scrap)
    print(f'\nn_coins={n_coins}\n')
    print(crypto_class_20c.portfolio)

    n_coins = 10
    crypto_class_10c = Cryptos(top_100, budget, n_coins, hodl)
    crypto_class_10c.optimize_portfolio(n_coins, mu_method, cov_method, obj_function, drop, scrap)
    print(f'\nn_coins={n_coins}\n')
    print(crypto_class_10c.portfolio)
