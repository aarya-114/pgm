"""
Hidden Markov Model implementation for stock market regime detection.
This module contains the core functionality for training HMMs and analyzing stock data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.stats import norm
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import yfinance as yf

# Set plotting style
plt.style.use('ggplot')
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = [14, 8]

def get_stock_data(ticker='SPY', period='5y'):
    """
    Downloads stock data for the specified ticker and period.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    period : str
        Time period (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with stock data including calculated log returns
    """
    # Get data
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    
    # Calculate log returns
    data['LogReturn'] = np.log(data['Close']).diff().dropna()
    data = data.dropna()
    
    print(f"Downloaded {len(data)} days of data for {ticker}")
    return data

def train_hmm(returns, n_states=3, n_iter=1000):
    """
    Train HMM on the returns data with specified number of hidden states.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of log returns
    n_states : int
        Number of hidden states in the model
    n_iter : int
        Number of iterations for model training
        
    Returns:
    --------
    tuple
        (states_df, state_params, transition_matrix, model)
        - states_df: DataFrame with original returns and assigned states
        - state_params: DataFrame with parameters for each state
        - transition_matrix: State transition probability matrix
        - model: Trained GaussianHMM model
    """
    # Prepare the data
    X = returns.values.reshape(-1, 1)
    
    # Initialize and train the HMM
    model = GaussianHMM(n_components=n_states, 
                      covariance_type="full", 
                      n_iter=n_iter,
                      random_state=42)
    
    model.fit(X)
    
    # Get the hidden states
    hidden_states = model.predict(X)
    
    # Get state characteristics
    state_means = model.means_.flatten()
    state_vars = np.sqrt(np.array([np.diag(model.covars_[i])[0] for i in range(n_states)]))
    
    # Sort states by mean returns
    sort_idx = np.argsort(state_means)
    state_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sort_idx)}
    
    # Remap the hidden states
    remapped_states = np.array([state_mapping[state] for state in hidden_states])
    
    # Create a DataFrame with the results
    states_df = pd.DataFrame({
        'Return': returns.values,
        'State': remapped_states
    }, index=returns.index)
    
    # Calculate transition matrix
    transition_matrix = model.transmat_
    remapped_transition = np.zeros_like(transition_matrix)
    for i in range(n_states):
        for j in range(n_states):
            remapped_transition[state_mapping[i], state_mapping[j]] = transition_matrix[i, j]
    
    # Calculate average duration of states
    state_durations = []
    for state in range(n_states):
        # Probability of staying in the same state
        p_stay = remapped_transition[state, state]
        # Expected duration = 1 / (1 - p_stay)
        expected_duration = 1 / (1 - p_stay) if p_stay < 1 else float('inf')
        state_durations.append(expected_duration)
    
    # Organize state parameters
    state_params = pd.DataFrame({
        'Mean': [state_means[sort_idx[i]] for i in range(n_states)],
        'Std Dev': [state_vars[sort_idx[i]] for i in range(n_states)],
        'Annualized Return': [state_means[sort_idx[i]] * 252 * 100 for i in range(n_states)],
        'Annualized Volatility': [state_vars[sort_idx[i]] * np.sqrt(252) * 100 for i in range(n_states)],
        'Avg Duration (days)': state_durations
    })
    
    return states_df, state_params, remapped_transition, model

def plot_returns_with_states(data, states_df, state_params, title='Stock Returns with HMM States'):
    """
    Plot the returns colored by hidden state.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Original stock data
    states_df : pandas.DataFrame
        DataFrame with returns and state assignments
    state_params : pandas.DataFrame
        State parameters
    title : str
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    n_states = len(state_params)
    colors = ['red', 'gray', 'green'] if n_states == 3 else plt.cm.viridis(np.linspace(0, 1, n_states))
    
    # Plot returns colored by state
    for i in range(n_states):
        mask = states_df['State'] == i
        ax.scatter(
            states_df.index[mask], 
            states_df['Return'][mask],
            color=colors[i],
            label=f"State {i} (Ann. Return: {state_params.loc[i, 'Annualized Return']:.1f}%, Vol: {state_params.loc[i, 'Annualized Volatility']:.1f}%)",
            alpha=0.6,
            s=30
        )
    
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Log Return', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    return fig

def plot_state_distributions(state_params, returns_range=None):
    """
    Plot the probability density functions of each state.
    
    Parameters:
    -----------
    state_params : pandas.DataFrame
        State parameters including mean and standard deviation
    returns_range : array-like, optional
        Range of returns for plotting
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    n_states = len(state_params)
    colors = ['red', 'gray', 'green'] if n_states == 3 else plt.cm.viridis(np.linspace(0, 1, n_states))
    
    if returns_range is None:
        # Create a reasonable range based on the state parameters
        max_std = state_params['Std Dev'].max()
        max_mean_abs = state_params['Mean'].abs().max()
        margin = 3 * max_std
        min_return = min(state_params['Mean'].min() - margin, -0.03)
        max_return = max(state_params['Mean'].max() + margin, 0.03)
        returns_range = np.linspace(min_return, max_return, 1000)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i in range(n_states):
        mean = state_params.loc[i, 'Mean']
        std = state_params.loc[i, 'Std Dev']
        
        pdf = norm.pdf(returns_range, mean, std)
        
        ax.plot(
            returns_range, 
            pdf, 
            color=colors[i], 
            lw=2,
            label=f"State {i} (μ={mean:.4f}, σ={std:.4f})"
        )
        
        # Fill under the curve
        ax.fill_between(returns_range, pdf, alpha=0.3, color=colors[i])
    
    ax.set_title('Return Distributions by Market State', fontsize=16)
    ax.set_xlabel('Daily Log Return', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    return fig

def plot_state_timeline(states_df, state_params):
    """
    Plot the states over time as a colorized timeline.
    
    Parameters:
    -----------
    states_df : pandas.DataFrame
        DataFrame with returns and state assignments
    state_params : pandas.DataFrame
        State parameters
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    n_states = len(state_params)
    colors = ['red', 'gray', 'green'] if n_states == 3 else plt.cm.viridis(np.linspace(0, 1, n_states))
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Prepare data for area plot
    state_data = pd.DataFrame(index=states_df.index)
    
    for i in range(n_states):
        state_data[f'State {i}'] = (states_df['State'] == i).astype(int)
    
    # Plot stacked area (only one state will be active at a time)
    ax.stackplot(
        state_data.index, 
        [state_data[f'State {i}'] for i in range(n_states)],
        colors=colors,
        labels=[f"State {i}" for i in range(n_states)],
        alpha=0.7
    )
    
    ax.set_title('Market Regime Timeline', fontsize=16)
    ax.set_ylabel('Active State', fontsize=14)
    ax.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    return fig

def plot_transition_matrix(transition_matrix):
    """
    Visualize the transition matrix as a heatmap.
    
    Parameters:
    -----------
    transition_matrix : numpy.ndarray
        State transition probability matrix
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    n_states = transition_matrix.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        transition_matrix, 
        annot=True, 
        cmap='Blues', 
        fmt='.2f',
        xticklabels=[f'State {i}' for i in range(n_states)],
        yticklabels=[f'State {i}' for i in range(n_states)]
    )
    
    ax.set_title('State Transition Probabilities', fontsize=16)
    ax.set_xlabel('To State', fontsize=14)
    ax.set_ylabel('From State', fontsize=14)
    plt.tight_layout()
    return fig

def compute_performance_by_state(data, states_df):
    """
    Compute performance metrics for each state.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Original stock data
    states_df : pandas.DataFrame
        DataFrame with returns and state assignments
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with performance metrics for each state
    """
    # Merge the state information with original price data
    combined = pd.concat([data['Close'], states_df['State']], axis=1).dropna()
    
    # Calculate daily returns (non-log)
    combined['Return'] = combined['Close'].pct_change().dropna()
    combined = combined.dropna()
    
    # Get unique states
    states = sorted(combined['State'].unique())
    
    # Calculate performance metrics for each state
    results = []
    for state in states:
        state_data = combined[combined['State'] == state]
        
        # Skip if too few data points
        if len(state_data) < 5:
            continue
        
        # Calculate metrics
        total_days = len(state_data)
        mean_return = state_data['Return'].mean()
        annualized_return = (1 + mean_return) ** 252 - 1
        volatility = state_data['Return'].std() * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        cum_returns = (1 + state_data['Return']).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max - 1).min()
        
        # Calculate percent of positive days
        positive_days = (state_data['Return'] > 0).sum() / total_days
        
        results.append({
            'State': state,
            'Days': total_days,
            'Mean Daily Return': mean_return,
            'Annualized Return': annualized_return * 100,  # as percentage
            'Annualized Volatility': volatility * 100,  # as percentage
            'Sharpe Ratio': sharpe,
            'Max Drawdown': drawdown * 100,  # as percentage
            'Positive Days %': positive_days * 100  # as percentage
        })
    
    return pd.DataFrame(results).set_index('State')

def plot_regime_price_action(data, states_df, window_size=90):
    """
    Plot the price action with shaded regions for each state.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Original stock data
    states_df : pandas.DataFrame
        DataFrame with returns and state assignments
    window_size : int, optional
        Window size for moving average
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    n_states = states_df['State'].nunique()
    colors = ['red', 'gray', 'green'] if n_states == 3 else plt.cm.viridis(np.linspace(0, 1, n_states))
    
    # Create a time series of the states
    state_ts = states_df['State']
    
    # Create a combined dataframe
    combined = pd.concat([data['Close'], state_ts], axis=1).dropna()
    
    # Calculate moving average if window_size is provided
    if window_size:
        combined['MA'] = combined['Close'].rolling(window=window_size).mean()
    
    # Identify contiguous regions of the same state
    regions = []
    current_state = None
    start_idx = None
    
    for i, (date, row) in enumerate(combined.iterrows()):
        if row['State'] != current_state:
            if current_state is not None:
                regions.append({
                    'start': start_idx,
                    'end': date,
                    'state': current_state
                })
            current_state = row['State']
            start_idx = date
    
    # Add the last region
    if current_state is not None and start_idx is not None:
        regions.append({
            'start': start_idx,
            'end': combined.index[-1],
            'state': current_state
        })
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot price
    ax.plot(combined.index, combined['Close'], color='black', linewidth=1.5, label='Price')
    
    # Plot moving average if available
    if window_size and 'MA' in combined.columns:
        ax.plot(combined.index, combined['MA'], color='blue', linewidth=1.5, linestyle='--', 
                label=f'{window_size}-Day MA')
    
    # Shade regions by state
    y_min, y_max = ax.get_ylim()
    for region in regions:
        ax.axvspan(region['start'], region['end'], 
                  color=colors[int(region['state'])], 
                  alpha=0.2)
    
    # Restore y limits
    ax.set_ylim(y_min, y_max)
    
    # Add legend for states
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], alpha=0.2, label=f'State {i}')
                      for i in range(n_states)]
    
    # Combine with line legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_elements, labels + [f'State {i}' for i in range(n_states)],
             loc='upper left', fontsize=12)
    
    ax.set_title('Price Action with Market Regimes', fontsize=16)
    ax.set_ylabel('Price', fontsize=14)
    plt.tight_layout()
    return fig

def predict_next_state_probabilities(model, current_state):
    """
    Predict the probabilities of moving to each state in the next time step.
    
    Parameters:
    -----------
    model : hmmlearn.hmm.GaussianHMM
        Trained HMM model
    current_state : int
        Current state index
        
    Returns:
    --------
    numpy.ndarray
        Array of probabilities for transitioning to each state
    """
    # Get the transition matrix
    transition_matrix = model.transmat_
    
    # The probability of transitioning to each state given the current state
    next_state_probs = transition_matrix[current_state]
    
    return next_state_probs

def run_backtesting(data, states_df, state_params, initial_capital=10000):
    """
    Backtest a simple strategy based on market states.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Original stock data
    states_df : pandas.DataFrame
        DataFrame with returns and state assignments
    state_params : pandas.DataFrame
        State parameters
    initial_capital : float, optional
        Initial capital for backtesting
        
    Returns:
    --------
    tuple
        (backtest_data, performance)
        - backtest_data: DataFrame with backtest results
        - performance: Series with performance metrics
    """
    # Merge the price data with state information
    backtest_data = pd.concat([
        data['Close'],
        states_df['State']
    ], axis=1).dropna()
    
    # Calculate daily returns (percentage)
    backtest_data['Return'] = backtest_data['Close'].pct_change()
    backtest_data = backtest_data.dropna()
    
    # Initialize strategy columns
    backtest_data['Strategy'] = 0.0  # Strategy daily returns
    backtest_data['BuyHold'] = backtest_data['Return']  # Buy and hold returns
    backtest_data['Exposure'] = 0.0  # Exposure level
    
    # Number of states
    n_states = len(state_params)
    
    # Define exposure for each state (bullish: 1, neutral: 0.5, bearish: 0)
    state_exposures = {}
    for i in range(n_states):
        if i == 0:  # Bearish (lowest returns)
            state_exposures[i] = 0.0
        elif i == n_states - 1:  # Bullish (highest returns)
            state_exposures[i] = 1.0
        else:  # Neutral state(s)
            state_exposures[i] = 0.5
    
    # Apply the strategy
    for i in range(1, len(backtest_data)):
        yesterday = backtest_data.index[i-1]
        today = backtest_data.index[i]
        
        # State yesterday
        state = backtest_data.loc[yesterday, 'State']
        
        # Set exposure based on yesterday's state
        backtest_data.loc[today, 'Exposure'] = state_exposures[state]
        
        # Calculate strategy return
        backtest_data.loc[today, 'Strategy'] = backtest_data.loc[today, 'Return'] * backtest_data.loc[today, 'Exposure']
    
    # Calculate cumulative returns
    backtest_data['Strategy_Cum'] = (1 + backtest_data['Strategy']).cumprod() * initial_capital
    backtest_data['BuyHold_Cum'] = (1 + backtest_data['BuyHold']).cumprod() * initial_capital
    
    # Calculate performance metrics
    strategy_return = backtest_data['Strategy'].mean() * 252
    strategy_vol = backtest_data['Strategy'].std() * np.sqrt(252)
    strategy_sharpe = strategy_return / strategy_vol if strategy_vol > 0 else 0
    
    bh_return = backtest_data['BuyHold'].mean() * 252
    bh_vol = backtest_data['BuyHold'].std() * np.sqrt(252)
    bh_sharpe = bh_return / bh_vol if bh_vol > 0 else 0
    
    # Calculate drawdowns
    strategy_dd = (backtest_data['Strategy_Cum'] / backtest_data['Strategy_Cum'].cummax() - 1).min()
    bh_dd = (backtest_data['BuyHold_Cum'] / backtest_data['BuyHold_Cum'].cummax() - 1).min()
    
    # Prepare performance summary
    performance = {
        'Strategy Ann. Return': strategy_return * 100,  # as percentage
        'Buy & Hold Ann. Return': bh_return * 100,  # as percentage
        'Strategy Ann. Volatility': strategy_vol * 100,  # as percentage
        'Buy & Hold Ann. Volatility': bh_vol * 100,  # as percentage
        'Strategy Sharpe': strategy_sharpe,
        'Buy & Hold Sharpe': bh_sharpe,
        'Strategy Max Drawdown': strategy_dd * 100,  # as percentage
        'Buy & Hold Max Drawdown': bh_dd * 100,  # as percentage
        'Strategy Final Value': backtest_data['Strategy_Cum'].iloc[-1],
        'Buy & Hold Final Value': backtest_data['BuyHold_Cum'].iloc[-1]
    }
    
    return backtest_data, pd.Series(performance)



def plot_backtest_results(backtest_data, performance):
    """
    Plot the backtest results.
    
    Parameters:
    -----------
    backtest_data : pandas.DataFrame
        DataFrame with backtest results
    performance : pandas.Series
        Series with performance metrics
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot cumulative returns
    ax1.plot(backtest_data['Strategy_Cum'], color='blue', linewidth=2, label='HMM Strategy')
    ax1.plot(backtest_data['BuyHold_Cum'], color='gray', linewidth=2, label='Buy & Hold')
    
    ax1.set_title('Backtest Results: HMM Regime-Based Strategy vs Buy & Hold', fontsize=16)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot exposure
    ax2.fill_between(backtest_data.index, backtest_data['Exposure'], color='green', alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Market Exposure', fontsize=14)
    ax2.set_xlabel('Date', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add text with performance metrics
    performance_text = "\n".join([
        f"Strategy Ann. Return: {performance['Strategy Ann. Return']:.2f}%",
        f"Buy & Hold Ann. Return: {performance['Buy & Hold Ann. Return']:.2f}%",
        f"Strategy Sharpe: {performance['Strategy Sharpe']:.2f}",
        f"Buy & Hold Sharpe: {performance['Buy & Hold Sharpe']:.2f}",
        f"Strategy Max DD: {performance['Strategy Max Drawdown']:.2f}%",
        f"Buy & Hold Max DD: {performance['Buy & Hold Max Drawdown']:.2f}%"
    ])
    
    # Position the text box in the upper right of the first subplot
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax1.text(0.02, 0.98, performance_text, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig

def analyze_stock_with_hmm(ticker='SPY', period='5y', n_states=3, n_iter=1000):
    """
    Perform a complete HMM analysis of the given stock.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    period : str
        Time period for historical data
    n_states : int
        Number of hidden states in the model
    n_iter : int
        Number of iterations for model training
        
    Returns:
    --------
    dict
        Dictionary with analysis results including:
        - data: Original stock data
        - states_df: DataFrame with state assignments
        - state_params: State parameters
        - transition_matrix: State transition probabilities
        - model: Trained HMM model
        - performance_by_state: Performance metrics by state
        - backtest_data: Backtest results
        - performance: Performance metrics
        - plots: Dictionary of plot figures
    """
    # Get data
    print(f"Analyzing {ticker} with a {n_states}-state HMM model...")
    data = get_stock_data(ticker, period)
    
    # Train HMM
    states_df, state_params, transition_matrix, model = train_hmm(data['LogReturn'], n_states, n_iter)
    
    # Print state parameters
    print("\nMarket Regime Characteristics:")
    print(state_params)
    
    # Print transition matrix
    print("\nState Transition Probabilities:")
    print(pd.DataFrame(
        transition_matrix, 
        index=[f'From State {i}' for i in range(n_states)],
        columns=[f'To State {i}' for i in range(n_states)]
    ))
    
    # Calculate and print performance by state
    performance_by_state = compute_performance_by_state(data, states_df)
    print("\nPerformance Metrics by State:")
    print(performance_by_state)
    
    # Predict current state and transition probabilities
    current_state = states_df['State'].iloc[-1]
    next_state_probs = predict_next_state_probabilities(model, current_state)
    
    print(f"\nCurrent Market State: {current_state}")
    print("Probabilities for Next Day's State:")
    for i, prob in enumerate(next_state_probs):
        print(f"  State {i}: {prob:.2%}")
    
    # Run backtesting
    backtest_data, performance = run_backtesting(data, states_df, state_params)
    
    print("\nBacktest Performance:")
    print(performance)
    
    # Generate plots
    plots = {}
    plots['returns_with_states'] = plot_returns_with_states(data, states_df, state_params, 
                                                           f"{ticker} Returns with HMM States")
    plots['state_distributions'] = plot_state_distributions(state_params)
    plots['state_timeline'] = plot_state_timeline(states_df, state_params)
    plots['transition_matrix'] = plot_transition_matrix(transition_matrix)
    plots['regime_price_action'] = plot_regime_price_action(data, states_df)
    plots['backtest_results'] = plot_backtest_results(backtest_data, performance)
    
    return {
        'data': data,
        'states_df': states_df,
        'state_params': state_params,
        'transition_matrix': transition_matrix,
        'model': model,
        'performance_by_state': performance_by_state,
        'backtest_data': backtest_data,
        'performance': performance,
        'plots': plots
    }

# Additional functions for the web application

def get_state_description(state_idx, state_params):
    """
    Generate a human-readable description of a market state.
    
    Parameters:
    -----------
    state_idx : int
        State index
    state_params : pandas.DataFrame
        State parameters
        
    Returns:
    --------
    str
        Human-readable description of the state
    """
    n_states = len(state_params)
    
    # Get state characteristics
    ann_return = state_params.loc[state_idx, 'Annualized Return']
    ann_vol = state_params.loc[state_idx, 'Annualized Volatility']
    avg_duration = state_params.loc[state_idx, 'Avg Duration (days)']
    
    # Determine state type based on parameters
    if state_idx == 0:  # Lowest return state
        state_type = "Bearish"
        description = f"This is a bearish market regime characterized by negative returns (annualized {ann_return:.2f}%) "
        description += f"and relatively high volatility ({ann_vol:.2f}%). "
        description += f"On average, this state lasts for {avg_duration:.1f} trading days when it occurs."
        
    elif state_idx == n_states - 1:  # Highest return state
        state_type = "Bullish"
        description = f"This is a bullish market regime characterized by strong positive returns (annualized {ann_return:.2f}%) "
        
        if ann_vol > 15:
            description += f"with high volatility ({ann_vol:.2f}%). "
        else:
            description += f"with moderate volatility ({ann_vol:.2f}%). "
            
        description += f"On average, this state lasts for {avg_duration:.1f} trading days when it occurs."
        
    else:  # Intermediate states
        if ann_return > 0:
            state_type = "Neutral-Positive"
            description = f"This is a neutral-positive market regime with modest returns (annualized {ann_return:.2f}%) "
        else:
            state_type = "Neutral-Negative"
            description = f"This is a neutral-negative market regime with slight negative returns (annualized {ann_return:.2f}%) "
            
        if ann_vol < 10:
            description += f"and low volatility ({ann_vol:.2f}%). "
        else:
            description += f"and moderate volatility ({ann_vol:.2f}%). "
            
        description += f"On average, this state lasts for {avg_duration:.1f} trading days when it occurs."
    
    return {
        'state_type': state_type,
        'description': description
    }

def get_market_insight(current_state, next_state_probs, state_params, performance_by_state):
    """
    Generate market insights based on the current state and transition probabilities.
    
    Parameters:
    -----------
    current_state : int
        Current state index
    next_state_probs : array-like
        Probabilities of transitioning to each state
    state_params : pandas.DataFrame
        State parameters
    performance_by_state : pandas.DataFrame
        Performance metrics by state
        
    Returns:
    --------
    dict
        Dictionary with market insights
    """
    n_states = len(state_params)
    
    # Get current state description
    current_state_info = get_state_description(current_state, state_params)
    
    # Find most likely next state
    most_likely_next = np.argmax(next_state_probs)
    probability = next_state_probs[most_likely_next]
    
    # Determine if a state change is likely
    if most_likely_next != current_state and probability > 0.3:
        state_change = "likely"
    elif most_likely_next != current_state and probability > 0.2:
        state_change = "possible"
    else:
        state_change = "unlikely"
    
    # Generate a market outlook
    if current_state == 0:  # Bearish
        if state_change == "likely" and most_likely_next > current_state:
            outlook = "The market appears to be at the end of a bearish phase, with signs of improvement ahead."
        elif state_change == "possible" and most_likely_next > current_state:
            outlook = "While still in a bearish regime, there's some probability of a shift to a more favorable market condition."
        else:
            outlook = "The market remains in a bearish state with limited signs of immediate improvement."
    
    elif current_state == n_states - 1:  # Bullish
        if state_change == "likely" and most_likely_next < current_state:
            outlook = "The bullish market phase may be nearing its end, with increased probability of a market regime shift."
        elif state_change == "possible" and most_likely_next < current_state:
            outlook = "While still bullish, there are some early warning signs of a potential market regime change."
        else:
            outlook = "The market remains in a strong bullish state with good probability of continuation."
    
    else:  # Neutral states
        if state_change == "likely" and most_likely_next > current_state:
            outlook = "The market shows promising signs of transitioning to a more positive regime."
        elif state_change == "likely" and most_likely_next < current_state:
            outlook = "Caution is warranted as the market shows increased probability of moving to a less favorable regime."
        else:
            outlook = "The market is likely to remain in its current neutral state in the near term."
    
    # Add information about current returns and volatility
    curr_state_return = performance_by_state.loc[current_state, 'Annualized Return']
    curr_state_vol = performance_by_state.loc[current_state, 'Annualized Volatility']
    
    current_dynamics = f"In the current regime, the market has historically shown an annualized return of {curr_state_return:.2f}% "
    current_dynamics += f"with {curr_state_vol:.2f}% volatility."
    
    return {
        'current_state_type': current_state_info['state_type'],
        'current_state_description': current_state_info['description'],
        'most_likely_next_state': most_likely_next,
        'transition_probability': probability,
        'state_change': state_change,
        'outlook': outlook,
        'current_dynamics': current_dynamics
    }

def get_trading_recommendation(current_state, next_state_probs, state_params, n_states=3):
    """
    Generate a trading recommendation based on the current state and transition probabilities.
    
    Parameters:
    -----------
    current_state : int
        Current state index
    next_state_probs : array-like
        Probabilities of transitioning to each state
    state_params : pandas.DataFrame
        State parameters
    n_states : int
        Number of states in the model
        
    Returns:
    --------
    dict
        Dictionary with trading recommendation
    """
    # Basic strategy: 
    # - State 0 (bearish): No exposure (cash)
    # - Intermediate states: 50% exposure
    # - Highest state (bullish): Full exposure
    
    # Current recommended exposure
    if current_state == 0:
        current_exposure = 0.0
        exposure_text = "0% (Cash)"
    elif current_state == n_states - 1:
        current_exposure = 1.0
        exposure_text = "100% (Fully Invested)"
    else:
        current_exposure = 0.5
        exposure_text = "50% (Partially Invested)"
    
    # Expected state change
    most_likely_next = np.argmax(next_state_probs)
    probability = next_state_probs[most_likely_next]
    
    # Recommendation
    if current_state == 0:  # Bearish
        if most_likely_next > current_state and probability > 0.3:
            recommendation = "Consider adding some market exposure as probability of market improvement is increasing."
            action = "INCREASE EXPOSURE"
        else:
            recommendation = "Maintain minimal market exposure as bearish conditions are likely to persist."
            action = "MAINTAIN CASH"
            
    elif current_state == n_states - 1:  # Bullish
        if most_likely_next < current_state and probability > 0.3:
            recommendation = "Consider reducing exposure as the bullish phase may be showing signs of ending."
            action = "REDUCE EXPOSURE"
        else:
            recommendation = "Maintain full market exposure as bullish conditions are likely to persist."
            action = "STAY INVESTED"
            
    else:  # Neutral
        if most_likely_next > current_state and probability > 0.3:
            recommendation = "Consider increasing market exposure as conditions appear to be improving."
            action = "INCREASE EXPOSURE"
        elif most_likely_next < current_state and probability > 0.3:
            recommendation = "Consider reducing market exposure as conditions may be deteriorating."
            action = "REDUCE EXPOSURE"
        else:
            recommendation = "Maintain balanced market exposure as neutral conditions are likely to persist."
            action = "MAINTAIN BALANCE"
    
    # Add risk warning
    risk_warning = "This is a systematic recommendation based solely on the HMM model and historical patterns. "
    risk_warning += "Always consider your personal risk tolerance, investment goals, and other market factors "
    risk_warning += "before making investment decisions."
    
    return {
        'current_exposure': current_exposure,
        'exposure_text': exposure_text,
        'recommendation': recommendation,
        'action': action,
        'risk_warning': risk_warning
    }