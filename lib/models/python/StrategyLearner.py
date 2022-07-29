import datetime as dt
import pandas as pd
import util as ut
import indicators
import QLearner as ql


class StrategyLearner(object):
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.net_position = 0
        self.learner = None

    def discretize(self, data, steps):
        values = data.to_numpy().flatten()
        return pd.cut(values, steps, labels=False)

    def get_trading_indicators(self, prices, symbol):
        steps = 10
        exp_average_df = indicators.exponential_moving_average(prices, symbol)
        momentum_df = indicators.momentum(prices, symbol)
        death_cross_df = indicators.death_cross(prices, symbol)
        stochastic = indicators.stochastic_oscillator(prices, symbol)

        exp_avg = self.discretize(exp_average_df, steps)
        momentum = self.discretize(momentum_df, steps)
        death_cross = self.discretize(death_cross_df, 3)
        stochastic = self.discretize(stochastic, steps)

        # return [exp_avg, momentum, death_cross]  # works for first test
        return [exp_avg, momentum, stochastic, death_cross]  # works for last test, epoch 200

    def get_trading_state(self, indicators, index):
        state = ""
        for indicator in indicators:
            value = indicator[index]
            state = state + str(value)
        return int(state)

    def add_evidence(
            self,
            symbol="IBM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 1, 1),
            sv=10000,
    ):
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        ######## run non-dyna test ########
        self.learner = ql.QLearner(
            num_states=10000,
            num_actions=2,
            alpha=0.2,
            gamma=0.9,
            rar=0.98,
            radr=0.999,
            dyna=0,
            verbose=False,
        )
        indicator_values = self.get_trading_indicators(prices, symbol)

        count = 0
        epoch = 50
        while count < epoch:
            state = self.get_trading_state(indicator_values, 0)
            action = self.learner.querysetstate(state)

            current_position = self.get_position(action)
            previous_position = 0
            discrete_index = 0

            for index, row in prices.iterrows():
                r = self.get_reward(index, symbol, prices, action, current_position,
                                    previous_position)

                previous_position = current_position

                # Next state
                next_state = self.get_trading_state(indicator_values, discrete_index)
                action = self.learner.query(next_state, r)
                current_position = self.get_position(action)
                discrete_index = discrete_index + 1

            count = count + 1

        if self.verbose:
            print(prices)

        volume_all = ut.get_data(
            syms, dates, colname="Volume"
        )
        volume = volume_all[syms]  # only portfolio symbols
        volume_SPY = volume_all["SPY"]  # only SPY, for comparison later
        if self.verbose:
            print(volume)

    def get_position(self, action):
        position = 0

        # Cover and Buy
        if action == 1:
            position = 1000
        # Short
        elif action == 0:
            position = -1000
        return position

    def get_reward(self, date, symbol, prices, action, current_position, previous_position):
        index = date - dt.timedelta(1)
        today_price = prices.loc[date, symbol]
        yesterday_price = 1

        if index in prices.index:
            yesterday_price = prices.loc[date - dt.timedelta(1), symbol]

        reward = 0
        if self.impact > 0:
            today_price = today_price - (today_price * self.impact)
            yesterday_price = yesterday_price - (yesterday_price * (self.impact * 2))

        if current_position == previous_position:
            reward = 0
        elif action == 1:
            if today_price > yesterday_price:
                reward = 1
            else:
                reward = -1
        elif action == 0:
            if yesterday_price > today_price:
                reward = 1
            else:
                reward = -1
        return reward

    def testPolicy(
            self,
            symbol="IBM",
            sd=dt.datetime(2009, 1, 1),
            ed=dt.datetime(2010, 1, 1),
            sv=10000,
    ):
        self.net_position = 0
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        prices = prices_all[[symbol]]  # only portfolio symbols
        trades_df = pd.DataFrame(0.0, index=dates, columns=["Trades"])
        # trades_SPY = prices_all["SPY"]

        # Indicators
        discrete_indicators = self.get_trading_indicators(prices, symbol)

        discrete_index = 0
        for index, row in prices.iterrows():
            state = self.get_trading_state(discrete_indicators, discrete_index)
            action = self.learner.querysetstate(state)
            discrete_index = discrete_index + 1

            # Do Nothing
            if action == 0:
                self.sell_and_short(index, trades_df)
            # Buy
            if action == 1:
                self.cover_and_buy(index, trades_df)

        return trades_df

    def sell_and_short(self, date, trades):
        if self.net_position == 0:
            trades.at[date, "Trades"] = -1000.0
        if self.net_position > 0:
            # SELL AND SHORT
            trades.at[date, "Trades"] = -2000
        self.net_position = -1000

    def cover_and_buy(self, date, trades):
        if self.net_position == 0:
            trades.at[date, "Trades"] = 1000.0
        if self.net_position < 0:
            # COVER AND BUY
            trades.at[date, "Trades"] = 2000
        self.net_position = 1000


if __name__ == "__main__":
    print()