import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define Indian stock tickers (NSE symbols are typically suffixed with '.NS')
tickers = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'ITC.NS']

# Define the time period for analysis (last 1 year as per your request)
start_date = '2023-12-01'
end_date = '2024-12-01'

# Download historical stock data
print("Downloading data...")
data = yf.download(tickers, start=start_date, end=end_date)

# Check the structure of the data
print("Columns in the data:")
print(data.columns)

# Print first few rows to inspect the data
print("\nSample data:")
print(data.head())

# If 'Adj Close' is available, use it. Otherwise, use 'Close'.
if 'Adj Close' in data.columns.levels[0]:
    data = data['Adj Close']
else:
    data = data['Close']  # fallback to 'Close' if 'Adj Close' is not available

# Drop rows with missing data
data.dropna(inplace=True)

# Plot individual stock graphs (Date-wise for each stock)
for ticker in tickers:
    plt.figure(figsize=(12, 6))
    plt.plot(data[ticker], label=ticker, color='blue')
    plt.title(f'{ticker} - Adjusted Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot combined graph for all stocks
plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(data[ticker], label=ticker)
plt.title('Adjusted Closing Prices for All Stocks')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate daily returns
daily_returns = data.pct_change().dropna()

# Calculate annualized returns using compounded return method
# Compound daily returns and then annualize
annual_returns = (1 + daily_returns).prod()**(252 / len(daily_returns)) - 1

# Convert annual returns to percentage
annual_returns_percentage = annual_returns * 100  # Multiply by 100 to convert to percentage

# Calculate covariance matrix for risk calculation
cov_matrix = daily_returns.cov() * 252

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(daily_returns.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Daily Returns')
plt.show()

# Equal weight portfolio for simplicity
weights = np.ones(len(tickers)) / len(tickers)

# Portfolio expected return and risk
portfolio_return = np.dot(weights, annual_returns)
portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Bar plot of annualized returns in percentage
plt.figure(figsize=(10, 6))
annual_returns_percentage.sort_values().plot(kind='bar', color='skyblue')
plt.title('Annualized Returns for Individual Stocks')
plt.ylabel('Return (%)')
plt.xlabel('Stock')
plt.grid(True)
plt.show()

# Pie chart of equal weights
plt.figure(figsize=(8, 8))
plt.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Portfolio Allocation (Equal Weights)')
plt.show()

# Monte Carlo Simulation for Portfolio Optimization
num_simulations = 10000
results = np.zeros((3, num_simulations))

for i in range(num_simulations):
    # Generate random weights
    random_weights = np.random.random(len(tickers))
    random_weights /= np.sum(random_weights)
    
    # Calculate portfolio return and risk
    port_return = np.dot(random_weights, annual_returns)
    port_risk = np.sqrt(np.dot(random_weights.T, np.dot(cov_matrix, random_weights)))
    
    # Sharpe Ratio (Assuming risk-free rate = 0)
    sharpe_ratio = port_return / port_risk
    
    results[0, i] = port_return
    results[1, i] = port_risk
    results[2, i] = sharpe_ratio

# Scatter plot of Monte Carlo simulation results
plt.figure(figsize=(12, 8))
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.7)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Monte Carlo Simulation: Risk vs. Return')
plt.grid(True)
plt.show()

# Display the annualized returns as percentage
print("\nAnnualized Returns (Percentage):")
print(annual_returns_percentage)

# Portfolio expected return and risk in percentage
print(f"\nExpected Annual Return of Portfolio: {portfolio_return * 100:.2f}%")
print(f"Portfolio Risk (Standard Deviation): {portfolio_risk * 100:.2f}%")
