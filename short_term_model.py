import yfinance as yf
import pandas as pd
import seaborn as sns


# Note: In Testing Phase
"""
Short Term Prediction Model
This module is designed to handle short-term stock predictions using historical data.
For now, example dataset is AAPL
"""

# Download 1 year of AAPL historical data
aapl_data = yf.download('AAPL', period='1y', auto_adjust=True)

# Display the first few rows
print(aapl_data.head())

# Calculate and display the correlation matrix
correlation_matrix = aapl_data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

import matplotlib.pyplot as plt

# Plot closing price over time
plt.figure(figsize=(12, 6))
plt.plot(aapl_data.index, aapl_data['Close'], label='AAPL Close Price')
plt.title('AAPL Closing Price - Last 1 Year')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Plot correlation matrix as a heatmap

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('AAPL Feature Correlation Matrix')
plt.show()