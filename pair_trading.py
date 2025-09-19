"""
The script implements a Gaussian Copula-based pairs trading strategy that:
  - Downloads stock data via yfinance
  - Fits marginal distributions and copula parameters
  - Generates trading signals based on mispricing
  - Backtests the strategy with transaction costs
  - Provides performance metrics and visualization

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')


class GaussianCopulaPairsTrading:
    def __init__(self, lookback_window=252):
        """
        Gaussian Copula-based Pairs Trading Strategy

        Parameters:
        - lookback_window: Number of days for parameter estimation
        """
        self.lookback_window = lookback_window
        self.marginal_params_x = None
        self.marginal_params_y = None
        self.copula_corr = None

    def fit_marginal_distribution(self, returns, distribution='t'):
        """Fit marginal distribution to returns"""
        if distribution == 't':
            # Fit Student's t-distribution
            params = stats.t.fit(returns)
            return {'dist': 't', 'params': params}
        elif distribution == 'norm':
            # Fit normal distribution
            params = stats.norm.fit(returns)
            return {'dist': 'norm', 'params': params}
        else:
            raise ValueError("Supported distributions: 't', 'norm'")

    def empirical_cdf(self, data):
        """Calculate empirical CDF (rank-based transformation)"""
        n = len(data)
        ranks = stats.rankdata(data)
        return ranks / (n + 1)

    def fit_copula(self, returns_x, returns_y):
        """
        Fit Gaussian copula to the pair of return series
        """
        # Transform to uniform variables using empirical CDF
        u = self.empirical_cdf(returns_x)
        v = self.empirical_cdf(returns_y)

        # Transform to standard normal variables
        z_u = stats.norm.ppf(u)
        z_v = stats.norm.ppf(v)

        # Estimate correlation parameter
        correlation = np.corrcoef(z_u, z_v)[0, 1]

        return correlation

    def update_parameters(self, returns_x, returns_y):
        """Update model parameters using rolling window"""
        # Fit marginal distributions
        self.marginal_params_x = self.fit_marginal_distribution(returns_x, 't')
        self.marginal_params_y = self.fit_marginal_distribution(returns_y, 't')

        # Fit copula
        self.copula_corr = self.fit_copula(returns_x, returns_y)

    def calculate_conditional_quantile(self, u_x, quantile=0.5):
        """
        Calculate conditional quantile of Y given X using Gaussian copula

        Parameters:
        - u_x: CDF value of X
        - quantile: Desired conditional quantile (0.5 for median)
        """
        if self.copula_corr is None:
            raise ValueError("Model not fitted yet")

        # Transform to standard normal
        z_x = stats.norm.ppf(u_x)

        # Calculate conditional mean and variance
        rho = self.copula_corr
        cond_mean = rho * z_x
        cond_var = 1 - rho ** 2

        # Calculate conditional quantile
        z_y_quantile = cond_mean + np.sqrt(cond_var) * stats.norm.ppf(quantile)
        u_y_quantile = stats.norm.cdf(z_y_quantile)

        return u_y_quantile

    def generate_trading_signals(self, prices_x, prices_y, entry_threshold=0.05, exit_threshold=0.01):
        """
        Generate trading signals based on copula model

        Parameters:
        - entry_threshold: Threshold for entering trades (probability deviation)
        - exit_threshold: Threshold for exiting trades
        """
        returns_x = prices_x.pct_change().dropna()
        returns_y = prices_y.pct_change().dropna()

        signals = pd.DataFrame(index=prices_x.index[1:])
        signals['signal'] = 0
        signals['spread'] = np.nan

        for i in range(self.lookback_window, len(returns_x)):
            # Get rolling window data
            window_returns_x = returns_x.iloc[i - self.lookback_window:i]
            window_returns_y = returns_y.iloc[i - self.lookback_window:i]

            # Update model parameters
            self.update_parameters(window_returns_x, window_returns_y)

            # Current return values
            current_return_x = returns_x.iloc[i]
            current_return_y = returns_y.iloc[i]

            # Calculate empirical CDF values
            u_x = (window_returns_x <= current_return_x).mean()
            u_y = (window_returns_y <= current_return_y).mean()

            # Calculate expected conditional quantile
            expected_u_y = self.calculate_conditional_quantile(u_x, 0.5)

            # Calculate mispricing signal
            mispricing = u_y - expected_u_y
            signals.loc[signals.index[i], 'spread'] = mispricing

            # Generate trading signals
            if abs(mispricing) > entry_threshold:
                if mispricing > 0:  # Y is overvalued relative to X
                    signals.loc[signals.index[i], 'signal'] = -1  # Short Y, Long X
                else:  # Y is undervalued relative to X
                    signals.loc[signals.index[i], 'signal'] = 1  # Long Y, Short X
            elif abs(mispricing) < exit_threshold:
                signals.loc[signals.index[i], 'signal'] = 0  # Exit position

        return signals

    def backtest_strategy(self, prices_x, prices_y, signals, transaction_cost=0.001):
        """
        Backtest the pairs trading strategy
        """
        returns_x = prices_x.pct_change()
        returns_y = prices_y.pct_change()

        # Calculate strategy returns
        strategy_returns = []
        position = 0

        for i, (idx, row) in enumerate(signals.iterrows()):
            if i == 0:
                strategy_returns.append(0)
                position = row['signal']
                continue

            # Previous position
            prev_position = position
            current_signal = row['signal']

            # Calculate return based on position
            if prev_position == 1:  # Long Y, Short X
                trade_return = returns_y.loc[idx] - returns_x.loc[idx]
            elif prev_position == -1:  # Short Y, Long X
                trade_return = returns_x.loc[idx] - returns_y.loc[idx]
            else:
                trade_return = 0

            # Account for transaction costs when position changes
            if current_signal != prev_position and current_signal != 0:
                trade_return -= transaction_cost

            strategy_returns.append(trade_return)
            position = current_signal

        signals['strategy_returns'] = strategy_returns
        signals['cumulative_returns'] = (1 + pd.Series(strategy_returns, index=signals.index)).cumprod()

        return signals


# Configuration - Change ticker pairs here
TICKER_PAIRS = {
    'AAPL_TSLA': ['AAPL', 'TSLA'],
    'AAPL_MSFT': ['AAPL', 'MSFT'], 
    'SPY_QQQ': ['SPY', 'QQQ'],
    'GLD_SLV': ['GLD', 'SLV']}

# Select which pair to test (change this to test different pairs)
SELECTED_PAIR = 'GLD_SLV'

# Example usage and demonstration
def demonstrate_copula_pairs_trading():
    """Demonstrate the Gaussian Copula pairs trading strategy"""

    # Download sample data using configured tickers
    tickers = TICKER_PAIRS[SELECTED_PAIR]
    print(f"Downloading sample data for {tickers[0]}/{tickers[1]}...")
    raw_data = yf.download(tickers, start='2020-01-01', end='2024-01-01')
    data = raw_data['Close']  # Use 'Close' instead of 'Adj Close'

    # Initialize strategy
    strategy = GaussianCopulaPairsTrading(lookback_window=60)

    # Generate signals
    print("Generating trading signals...")
    signals = strategy.generate_trading_signals(
        data[tickers[0]],
        data[tickers[1]],
        entry_threshold=0.1,
        exit_threshold=0.02)

    # Backtest strategy
    print("Backtesting strategy...")
    results = strategy.backtest_strategy(data[tickers[0]], data[tickers[1]], signals)

    # Calculate performance metrics
    total_return = results['cumulative_returns'].iloc[-1] - 1
    annual_return = (results['cumulative_returns'].iloc[-1] ** (252 / len(results))) - 1
    volatility = results['strategy_returns'].std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    max_drawdown = (results['cumulative_returns'] / results['cumulative_returns'].cummax() - 1).min()

    print(f"\n=== Strategy Performance ===")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annual_return:.2%}")
    print(f"Volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot normalized prices
    normalized_stock1 = data[tickers[0]] / data[tickers[0]].iloc[0]
    normalized_stock2 = data[tickers[1]] / data[tickers[1]].iloc[0]

    axes[0].plot(normalized_stock1.index, normalized_stock1, label=f'{tickers[0]} (normalized)', alpha=0.7)
    axes[0].plot(normalized_stock2.index, normalized_stock2, label=f'{tickers[1]} (normalized)', alpha=0.7)
    axes[0].set_title('Normalized Stock Prices')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot mispricing signal
    axes[1].plot(results.index, results['spread'], label='Mispricing Signal', color='purple', alpha=0.7)
    axes[1].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Entry Threshold')
    axes[1].axhline(y=-0.1, color='r', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0.02, color='g', linestyle='--', alpha=0.5, label='Exit Threshold')
    axes[1].axhline(y=-0.02, color='g', linestyle='--', alpha=0.5)
    axes[1].set_title('Copula-based Mispricing Signal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot strategy performance
    axes[2].plot(results.index, results['cumulative_returns'], label='Strategy Returns', color='darkblue', linewidth=2)
    axes[2].set_title('Cumulative Strategy Returns')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pair_trading_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'pair_trading_results.png'")
    plt.show()

    return results, strategy


def test_pair(pair_name):
    """
    Test a specific pair by name
    
    Args:
        pair_name: Key from TICKER_PAIRS dictionary
    """
    global SELECTED_PAIR
    SELECTED_PAIR = pair_name
    print(f"\n=== Testing {pair_name} ===")
    return demonstrate_copula_pairs_trading()

# Run demonstration
if __name__ == "__main__":
    results, strategy = demonstrate_copula_pairs_trading()
    
    # Uncomment below to test multiple pairs
    # for pair_name in TICKER_PAIRS.keys():
    #     test_pair(pair_name)