from flask import Flask, render_template, request, jsonify, redirect, url_for
import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import base64
from datetime import datetime, timedelta
import yfinance as yf
import json
from hmm_model import train_hmm, get_stock_data, plot_returns_with_states
from hmm_model import plot_state_distributions, plot_state_timeline, plot_transition_matrix
from hmm_model import plot_regime_price_action, run_backtesting, plot_backtest_results
from hmm_model import compute_performance_by_state, predict_next_state_probabilities

app = Flask(__name__)

# Set global plotting style
plt.style.use('ggplot')
sns.set_style('darkgrid')

# Cache for analysis results
analysis_cache = {}

@app.route('/')
def index():
    """Main page with form for analysis"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle form submission and run analysis"""
    ticker = request.form.get('ticker', 'SPY').upper()
    period = request.form.get('period', '5y')
    n_states = int(request.form.get('n_states', 3))
    
    # Generate a unique cache key
    cache_key = f"{ticker}_{period}_{n_states}_{datetime.now().strftime('%Y%m%d')}"
    
    try:
        # Get data
        data = get_stock_data(ticker, period)
        
        # Get current market price and daily change
        current_data = yf.Ticker(ticker).history(period='1d')
        current_price = float(current_data['Close'].iloc[-1])
        previous_close = float(current_data['Open'].iloc[-1])
        price_change = ((current_price - previous_close) / previous_close) * 100
        
        # Train HMM
        states_df, state_params, transition_matrix, model = train_hmm(data['LogReturn'], n_states)
        
        # Calculate performance by state
        performance_by_state = compute_performance_by_state(data, states_df)
        
        # Current state and prediction
        current_state = states_df['State'].iloc[-1]
        next_state_probs = predict_next_state_probabilities(model, current_state)
        
        # Run backtesting
        backtest_data, performance = run_backtesting(data, states_df, state_params)
        
        # Generate plots and encode them as base64
        plot_images = {}
        
        # Returns with states plot
        fig_returns = plot_returns_with_states(data, states_df, state_params, f"{ticker} Returns with HMM States")
        plot_images['returns'] = fig_to_base64(fig_returns)
        plt.close(fig_returns)
        
        # State distributions
        fig_dist = plot_state_distributions(state_params)
        plot_images['distributions'] = fig_to_base64(fig_dist)
        plt.close(fig_dist)
        
        # State timeline
        fig_timeline = plot_state_timeline(states_df, state_params)
        plot_images['timeline'] = fig_to_base64(fig_timeline)
        plt.close(fig_timeline)
        
        # Transition matrix
        fig_trans = plot_transition_matrix(transition_matrix)
        plot_images['transitions'] = fig_to_base64(fig_trans)
        plt.close(fig_trans)
        
        # Price action with regimes
        fig_price = plot_regime_price_action(data, states_df)
        plot_images['price_action'] = fig_to_base64(fig_price)
        plt.close(fig_price)
        
        # Backtest results
        fig_backtest = plot_backtest_results(backtest_data, performance)
        plot_images['backtest'] = fig_to_base64(fig_backtest)
        plt.close(fig_backtest)
        
        # Convert DataFrame to JSON-serializable format
        state_params_json = state_params.to_dict(orient='records')
        performance_by_state_json = performance_by_state.reset_index().to_dict(orient='records')
        performance_json = performance.to_dict()
        
        # Store results in cache
        analysis_cache[cache_key] = {
            'ticker': ticker,
            'period': period,
            'n_states': n_states,
            'state_params': state_params_json,
            'performance_by_state': performance_by_state_json,
            'current_state': int(current_state),
            'next_state_probs': next_state_probs.tolist(),
            'performance': performance_json,
            'plot_images': plot_images,
            'current_price': current_price,
            'price_change': price_change,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return redirect(url_for('results', key=cache_key))
        
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/results/<key>')
def results(key):
    """Display analysis results"""
    if key not in analysis_cache:
        return redirect(url_for('index'))
    
    results = analysis_cache[key]
    return render_template('results.html', results=results)

@app.route('/api/tickers')
def get_tickers():
    """API endpoint to get stock ticker suggestions"""
    query = request.args.get('q', '').upper()
    
    # This is a simplified example - in a real app, you'd query a proper database
    # of stock tickers or use an API
    common_tickers = {
        'SPY': 'SPDR S&P 500 ETF Trust',
        'QQQ': 'Invesco QQQ Trust (NASDAQ-100 Index)',
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc.',
        'AMZN': 'Amazon.com Inc.',
        'TSLA': 'Tesla Inc.',
        'META': 'Meta Platforms Inc.',
        'NVDA': 'NVIDIA Corporation',
        'BRK.B': 'Berkshire Hathaway Inc.',
        'JPM': 'JPMorgan Chase & Co.',
        'JNJ': 'Johnson & Johnson',
        'V': 'Visa Inc.',
        'PG': 'Procter & Gamble Co.',
        'XOM': 'Exxon Mobil Corporation',
        'DIS': 'The Walt Disney Company',
        'HD': 'Home Depot Inc.',
        'BAC': 'Bank of America Corporation',
        'INTC': 'Intel Corporation',
        'VZ': 'Verizon Communications Inc.'
    }
    
    if query:
        # Filter tickers that start with the query
        results = {k: v for k, v in common_tickers.items() if k.startswith(query)}
    else:
        # Return all common tickers if no query
        results = common_tickers
    
    return jsonify([{'symbol': k, 'name': v} for k, v in results.items()])

def fig_to_base64(fig):
    """Convert a matplotlib figure to base64 for embedding in HTML"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)