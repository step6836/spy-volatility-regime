# Data

## SPY OHLCV Data (2019-2024)

**Source:** Yahoo Finance  
**Symbol:** SPY (SPDR S&P 500 ETF Trust)  
**Period:** January 2, 2019 - December 31, 2024  
**Frequency:** Daily (trading days only)  
**Records:** ~1,510 trading days

### File

`spy_clean.csv` - Cleaned SPY price data

### Columns

| Column | Type | Description |
|--------|------|-------------|
| Date | datetime | Trading date (YYYY-MM-DD) |
| Open | float | Opening price |
| High | float | Highest intraday price |
| Low | float | Lowest intraday price |
| Close | float | Closing price |
| Volume | int | Trading volume |

### Data Cleaning

The dataset has been cleaned with the following steps:

1. **Type Conversion:** All OHLCV columns converted to numeric types, handling any non-numeric values
2. **Date Parsing:** Dates standardized to datetime format and sorted chronologically
3. **Missing Values:** Rows with missing/null values removed
4. **Non-Trading Days:** Weekends and market holidays naturally excluded (only actual trading days present)
5. **Duplicates:** Removed 

No outlier removal was performed - all extreme values represent actual market conditions (e.g., COVID-19 crash in March 2020).

### Data Coverage

- **Pre-COVID:** Jan 2019 - Feb 2020 (normal volatility)
- **COVID Crash:** Mar 2020 (extreme volatility spike)
- **Recovery:** Apr 2020 - Dec 2021 (elevated volatility)
- **Recent:** 2022-2024 (mixed regime periods)

This 6-year window captures multiple volatility regimes including tail events, making it suitable for regime classification training.

### Usage

```python
import pandas as pd

df = pd.read_csv('spy_clean.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
```

### Notes

- Data sourced from Yahoo Finance for educational/research purposes
- For production use, consider extending to 10+ years for better tail event coverage
- Check Yahoo Finance Terms of Service for commercial usage restrictions