import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from flask import Flask, request, render_template, jsonify
from scipy.optimize import basinhopping

np.random.seed(42) 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot():
    try:
        tickers = request.form.get('tickers', '').split(',')
        start_date = request.form.get('start_date', '')
        end_date = request.form.get('end_date', '')

        print(f"📥 Received Tickers: {tickers}")
        print(f"📥 Start Date: {start_date}")
        print(f"📥 End Date: {end_date}")

        if not tickers or tickers == ['']:
            return jsonify({'error': 'No tickers provided'}), 400

        if not start_date or not end_date:
            return jsonify({'error': 'Start and end dates must be provided'}), 400

        data = get_data_for_tickers(tickers, start_date, end_date)

        if data['adj_close_df'].empty:
            return jsonify({'error': 'No stock data found. Check your tickers and date range.'}), 400

        optimal_portfolio = optimize_portfolio(data['adj_close_df'])

        
        if np.isnan(optimal_portfolio['expected_return']) or np.isnan(optimal_portfolio['volatility']) or np.isnan(optimal_portfolio['sharpe_ratio']):
            return jsonify({'error': 'Portfolio optimization failed due to NaN values'}), 500

        response = {
            'tickers': tickers,
            'dates': data['dates'],
            'allocations': optimal_portfolio['weights'],
            'expected_returns': optimal_portfolio['expected_return'],
            'risk': optimal_portfolio['volatility'],
            'sharpe_ratio': optimal_portfolio['sharpe_ratio']
        }

        print(f"✅ Sending Response: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"❌ ERROR: {e}")  
        return jsonify({'error': str(e)}), 500


def get_data_for_tickers(tickers, start, end):
    adj_close_df = pd.DataFrame()

    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, auto_adjust=False)

        if data.empty:
            print(f"⚠️ No data found for {ticker}. Skipping...")
            continue

        adj_close_df[ticker] = data['Adj Close']


    adj_close_df.dropna(how='all', inplace=True)

    if adj_close_df.empty:
        raise ValueError("❌ ERROR: No stock data available. Check tickers and date range.")

    return {
        'adj_close_df': adj_close_df,
        'dates': adj_close_df.index.strftime('%Y-%m-%d').tolist()
    }


def optimize_portfolio(adj_close_df):
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    log_returns = log_returns.dropna()

    if log_returns.empty:
        raise ValueError("❌ ERROR: Log returns are empty. Check your data.")

    cov_matrix = log_returns.cov() * 252
    risk_free_rate = 0.0454

    def standard_deviation(weights, cov_matrix):
        return np.sqrt(weights.T @ cov_matrix @ weights)

    def expected_return(weights, log_returns):
        return np.sum(log_returns.mean() * weights) * 252

    def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

    def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

    num_assets = len(adj_close_df.columns)
    if num_assets == 0:
        raise ValueError("❌ ERROR: No assets found in adj_close_df.")

    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(0, 1) for _ in range(num_assets)]
    initial_weights = np.array([1 / num_assets] * num_assets)

    optimize_results = basinhopping(
    neg_sharpe_ratio, initial_weights,
    minimizer_kwargs={'args': (log_returns, cov_matrix, risk_free_rate), 'method': 'SLSQP', 'constraints': constraints, 'bounds': bounds},
    niter=10  
)
    
    return {
        'weights': list(optimize_results.x),
        'expected_return': expected_return(optimize_results.x, log_returns),
        'volatility': standard_deviation(optimize_results.x, cov_matrix),
        'sharpe_ratio': sharpe_ratio(optimize_results.x, log_returns, cov_matrix, risk_free_rate)
    }

if __name__ == '__main__':
    app.run(debug=True)
