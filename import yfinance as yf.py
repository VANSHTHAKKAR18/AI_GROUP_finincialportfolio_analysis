import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define Indian stock tickers (NSE symbols are typically suffixed with '.NS')
tickers = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'ITC.NS']

# Define the time period for analysis (last 20 years)
start_date = '2003-12-01'
end_date = '2023-12-01'

# Download historical stock data
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Check if all tickers have data
print(data.info())

# Calculate daily returns
daily_returns = data.pct_change().dropna()

# Calculate annualized mean returns and covariance matrix
annual_returns = daily_returns.mean() * 252  # 252 trading days in a year
cov_matrix = daily_returns.cov() * 252

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(daily_returns.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Daily Returns (Indian Stocks)')
plt.show()

# Equal weight portfolio for simplicity
weights = np.ones(len(tickers)) / len(tickers)

# Portfolio returns and risk
portfolio_return = np.dot(weights, annual_returns)
portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

print(f"Expected Annual Return: {portfolio_return:.2%}")
print(f"Portfolio Risk (Standard Deviation): {portfolio_risk:.2%}")

# Monte Carlo Simulation for Risk-Return Trade-off
num_simulations = 10000
results = np.zeros((3, num_simulations))

for i in range(num_simulations):
    # Random weights
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    
    # Expected return and risk
    port_return = np.dot(weights, annual_returns)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Sharpe Ratio (Assuming risk-free rate is 0 for simplicity)
    sharpe_ratio = port_return / port_risk
    
    results[0, i] = port_return
    results[1, i] = port_risk
    results[2, i] = sharpe_ratio

# Plotting the simulated portfolios
plt.figure(figsize=(12, 8))
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Monte Carlo Simulation of Portfolio Risk-Return (Indian Stocks)')
plt.show()