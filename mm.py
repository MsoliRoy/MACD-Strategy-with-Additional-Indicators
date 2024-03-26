import pandas as pd
import numpy as np

class MMStrategy:
    """
    A class representing a financial trading strategy.

    Attributes:
        fast_length (int): Length of the fast exponential moving average (EMA) for MACD calculation.
        slow_length (int): Length of the slow EMA for MACD calculation.
        signal_length (int): Length of the signal line for MACD calculation.
        ma_period (int): Period for calculating the moving average (MA).
        rsi_period (int): Period for calculating the Relative Strength Index (RSI).
    """
    def __init__(self, fast_length=12, slow_length=26, signal_length=9, ma_period=200, rsi_period=14):
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.signal_length = signal_length
        self.ma_period = ma_period
        self.rsi_period = rsi_period

    def calculate_macd(self, close_data):
        """Calculates MACD and related indicators."""
        ema_fast = close_data.ewm(span=self.fast_length, min_periods=0, adjust=False).mean()
        ema_slow = close_data.ewm(span=self.slow_length, min_periods=0, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal_length, min_periods=0, adjust=False).mean()
        delta = macd - signal_line
        return macd, signal_line, delta

    def calculate_rsi(self, close_data):
        """Calculates Relative Strength Index (RSI)."""
        delta = close_data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period, min_periods=0).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_period, min_periods=0).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_obv(self, close_data, volume_data):
        """Calculates On-Balance Volume (OBV)."""
        obv_values = np.where(close_data.diff() > 0, volume_data.diff(), np.where(close_data.diff() < 0, -volume_data.diff(), 0))
        obv = pd.Series(np.cumsum(obv_values), index=close_data.index)
        return obv

    def generate_signals(self, close_data, volume_data):
        """Generates buy/sell signals based on the strategy."""
        # Calculate MACD
        macd, signal_line, delta = self.calculate_macd(close_data)
        
        # Calculate RSI
        rsi = self.calculate_rsi(close_data)
        
        # Calculate Moving Average
        ma = close_data.rolling(window=self.ma_period, min_periods=0).mean()
        
        # Calculate OBV
        obv = self.calculate_obv(close_data, volume_data)
        
        # Generate signals
        long_condition = (delta > 0) & (close_data > ma) & (rsi < 70) & (obv > obv.rolling(window=20, min_periods=0).mean())
        short_condition = (delta < 0) & (close_data < ma) & (rsi > 30) & (obv < obv.rolling(window=20, min_periods=0).mean())
        
        long_entry = np.where(long_condition, 1, 0)
        short_entry = np.where(short_condition, -1, 0)
        
        return pd.DataFrame({'close': close_data, 'long_entry': long_entry, 'short_entry': short_entry})


# Example usage
# Load data (replace this with your actual data)
close_data = pd.Series([100, 102, 105, 103, 101])
volume_data = pd.Series([1000, 1200, 1100, 900, 1000])

# Create an instance of FinancialStrategy
strategy = FinancialStrategy()

# Generate signals
signals = strategy.generate_signals(close_data, volume_data)

print(signals)
