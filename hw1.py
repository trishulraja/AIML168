# Username: trishulraja

import pandas_datareader as pdr
import datetime

aapl = pdr.get_data_yahoo('AAPL', start = datetime.datetime(2006, 10, 1),
end = datetime.datetime(2012, 1, 1))

aapl.head()
aapl.tail()
aapl.describe()

aapl.index
aapl.columns
ts = aapl['Close'][-10:]

type(ts)

import pandas as pd


print(aapl.loc[pd.Timestamp('2006-11-01'):pd.Timestamp('2006-12-31')].head())

print(aapl.loc['2007'].head())

print(aapl.iloc[22:43])

print(aapl.iloc[[22,43], [2,3]])

sample = aapl

print('sample')

monthly_aapl = aapl.resample('M').mean()
print(monthly_aapl)

aapl.asfreq("M", method = 'bfill')

# aapl['diff'] = aapl.Open - aapl.Close

import matplotlib.pyplot as plt

aapl['Close'].plot(grid=True)
plt.show()

import numpy as np

daily_close = aapl[['Adj Close']]

daily_pct_change = daily_close.pct_change()

daily_pct_change.fillna(0, inplace = True)
print(daily_pct_change)

daily_log_returns = np.log(daily_close.pct_change()+1)
print(daily_log_returns)

# daily_pct_change = daily_close / daily_close.shift(1) -1
# print(daily_pct_change)

daily_pct_change.hist(bins=50)
plt.show()

print(daily_pct_change.describe())

cum_daily_return = (1+daily_pct_change).cumprod()
print(cum_daily_return)

cum_daily_return.plot(figsize=(12,8))
plt.show()

cum_monthly_return = cum_daily_return.resample("M").mean()
print(cum_monthly_return)

def get(tickers, startdate, enddate):
    def data(ticker):
        return(pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
    datas = map(data,tickers)
    return(pd.concat(datas, keys=tickers, names = ['Ticker', 'Date']))

tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG']

all_data = get(tickers, datetime.datetime(2006, 10, 1), datetime.datetime(2012, 1, 1))

daily_close_px = all_data[['Adj Close']].reset_index().pivot('Data', 'Ticker', 'Adj Close')

daily_pct_change = daily_close_px.pct_change()

daily_pct_change.hist(bins=50, sharex = True, figsize =(12,8))
plt.show()

pd.scatter_matrix(daily_pct_change, diagonal = 'kde', alpha = 0.1, figsize = (12.12))
plt.show()

adj_close_px = aapl['Adj Close']

moving_avg = adj_close_px.rolling(window=40).mean()
print(moving_avg[-10:])

aapl['42'] = adj_close_px.rolling(window=40).mean()
aapl['252'] = adj_close_px.rolling(window=252).mean()
aapl[['Adj Close', '42', '252']].plot()
plt.show()

#Volatility

min_periods = 75
vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods)
vol.plot(figsize=(10,8))
plt.show()

#OLS

import statsmodels.api as sm
from pandas.core import datetools

all_adj_close = all_data[['Adj Close']]

all_returns = np.log(all_adj_close / all_adj_close.shift(1))

aapl_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'AAPL']
aapl_returns.index = aapl_returns.index.droplevel('Ticker')

msft_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'MSFT']
msft_returns.index = msft_returns.index.droplevel('Ticker')

return_data = pd.concat([aapl_returns, msft_returns], axis=1)[1:]
return_data.columns = ['AAPL', 'MSFT']

X = sm.add_constant(return_data['AAPL'])
model = sm.OLS(return_data['MSFT'],X).fit()
print(model.summary())

plt.plot(return_data['AAPL'], return_data['MSFT'], 'r.')

ax = plt.axis()

x = np.linspace(ax[0], ax[1] + 0.01)

plt.plot(x, model.params[0] + model.params[1] * x, 'b', lw=2)

plt.grid(True)
plt.axis('tight')
plt.xlabel('Apple Returns')
plt.ylabel('Microsoft returns')
plt.show()

return_data['MSFT'].rolling(window=252).corr(return_data['AAPL']).plot()
plt.show()

# Initialize the short and long windows
short_window = 40
long_window = 100

# Initialize the `signals` DataFrame with the `signal` column
signals = pd.DataFrame(index=aapl.index)
signals['signal'] = 0.0

# Create short simple moving average over the short window
signals['short_mavg'] = aapl['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

# Create long simple moving average over the long window
signals['long_mavg'] = aapl['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

# Create signals
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                            > signals['long_mavg'][short_window:], 1.0, 0.0)   

# Generate trading orders
signals['positions'] = signals['signal'].diff()

# Print `signals`
print(signals)

# Initialize the plot figure
fig = plt.figure()

# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111,  ylabel='Price in $')

# Plot the closing price
aapl['Close'].plot(ax=ax1, color='r', lw=2.)

# Plot the short and long moving averages
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

# Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index, 
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='m')
         
# Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index, 
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='k')
plt.show()

## Backtesting

# Set the initial capital
initial_capital= float(100000.0)

# Create a DataFrame `positions`
positions = pd.DataFrame(index=signals.index).fillna(0.0)

# Buy a 100 shares
positions['AAPL'] = 100*signals['signal']   
  
# Initialize the portfolio with value owned   
portfolio = positions.multiply(aapl['Adj Close'], axis=0)

# Store the difference in shares owned 
pos_diff = positions.diff()

# Add `holdings` to portfolio
portfolio['holdings'] = (positions.multiply(aapl['Adj Close'], axis=0)).sum(axis=1)

# Add `cash` to portfolio
portfolio['cash'] = initial_capital - (pos_diff.multiply(aapl['Adj Close'], axis=0)).sum(axis=1).cumsum()   

# Add `total` to portfolio
portfolio['total'] = portfolio['cash'] + portfolio['holdings']

# Add `returns` to portfolio
portfolio['returns'] = portfolio['total'].pct_change()

# Print the first lines of `portfolio`
print(portfolio.head())

# Create a figure
fig = plt.figure()

ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')

# Plot the equity curve in dollars
portfolio['total'].plot(ax=ax1, lw=2.)

ax1.plot(portfolio.loc[signals.positions == 1.0].index, 
         portfolio.total[signals.positions == 1.0],
         '^', markersize=10, color='m')
ax1.plot(portfolio.loc[signals.positions == -1.0].index, 
         portfolio.total[signals.positions == -1.0],
         'v', markersize=10, color='k')
plt.show()

## Strategy Evaluation

# Isolate the returns of your strategy
returns = portfolio['returns']

# annualized Sharpe ratio
sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())

# Print the Sharpe ratio
print(sharpe_ratio)

# Define a trailing 252 trading day window
window = 252

# Calculate the max drawdown in the past window days for each day 
rolling_max = aapl['Adj Close'].rolling(window, min_periods=1).max()
daily_drawdown = aapl['Adj Close']/rolling_max - 1.0

# Calculate the minimum (negative) daily drawdown
max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()

# Plot the results
daily_drawdown.plot()
max_daily_drawdown.plot()
plt.show()

## CAGR
days = (aapl.index[-1] - aapl.index[0]).days
cagr = ((((aapl['Adj Close'][-1]) / aapl['Adj Close'][1])) ** (365.0/days)) - 1
print(cagr)