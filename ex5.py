import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define Indian stock tickers (NSE symbols are typically suffixed with '.NS')
tickers = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'ITC.NS']

# Define the time period for analysis (last 1 year)
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

# --- 1. Plot Historical Adjusted Closing Prices for each stock ---
plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(data[ticker], label=ticker)
plt.title('Historical Adjusted Closing Prices (All Stocks)')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.show()

# --- 2. Calculate daily returns ---
daily_returns = data.pct_change(fill_method=None).dropna()

# --- 3. Calculate annualized mean returns and covariance matrix ---
annual_returns = daily_returns.mean() * 252  # 252 trading days in a year
cov_matrix = daily_returns.cov() * 252

# --- 4. Plot correlation matrix ---
plt.figure(figsize=(10, 8))
sns.heatmap(daily_returns.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Daily Returns')
plt.show()

# --- 5. Bar plot of annualized returns ---
plt.figure(figsize=(10, 6))
annual_returns.sort_values().plot(kind='bar', color='skyblue')
plt.title('Annualized Returns for Individual Stocks')
plt.ylabel('Return (%)')
plt.xlabel('Stock')
plt.grid(True)
plt.show()

# --- 6. Monte Carlo Simulation for Portfolio Optimization ---
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

# --- 7. Investment and Profit/Loss Calculation ---
# Ask the user how much money they want to invest in each stock individually
investment_per_stock = []
for ticker in tickers:
    investment = float(input(f"How much money would you like to invest in {ticker} (in INR): "))
    investment_per_stock.append(investment)

# Calculate the initial price of each stock
initial_prices = data.iloc[0]

# Calculate the final value of each stock in the portfolio based on the user investment
final_prices = data.iloc[-1]
final_values = (final_prices / initial_prices) * np.array(investment_per_stock)

# Calculate total value of portfolio after the investment period
total_value = final_values.sum()

# Calculate the profit or loss from the portfolio
profit_or_loss = total_value - sum(investment_per_stock)
print(f"Total portfolio value: ₹{total_value:.2f}")
print(f"Total profit/loss: ₹{profit_or_loss:.2f}")

# --- 8. Pie Chart of Custom Portfolio Allocation ---
plt.figure(figsize=(8, 8))
plt.pie(investment_per_stock, labels=tickers, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title(f'Portfolio Allocation (Based on User Input)')
plt.show()

# --- 9. Pie Chart of Total Investment vs Profit/Loss ---
total_investment = sum(investment_per_stock)
total_gain_loss = profit_or_loss

# Pie chart for total investment and profit/loss
labels = ['Total Investment', 'Total Gain/Loss']
sizes = [total_investment, total_gain_loss]
colors = ['lightcoral', 'lightgreen']
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=[f'{label} (₹{size:.2f})' for label, size in zip(labels, sizes)],
        autopct='%1.1f%%', startangle=90, colors=colors)
plt.title(f'Investment vs Profit/Loss (₹{profit_or_loss:.2f})')
plt.show()

# --- 10. Pie Chart of Return Amount for Each Stock ---
return_amounts = final_values - np.array(investment_per_stock)  # Return for each stock

# Pie chart for return amounts
plt.figure(figsize=(8, 8))
plt.pie(return_amounts, labels=[f'{ticker} (₹{amount:.2f})' for ticker, amount in zip(tickers, return_amounts)],
        autopct='%1.1f%%', startangle=90, colors=sns.color_palette('muted'))
plt.title('Return Amount for Each Stock in Portfolio')
plt.show()
