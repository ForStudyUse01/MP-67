import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import requests
import time
from datetime import datetime, timedelta

def sort_by_relevance(results, query):
    """Sort results by relevance to search query"""
    def get_relevance_score(result):
        name = result['name'].lower()
        symbol = result['symbol'].lower()
        query_lower = query.lower()
        
        # Highest priority: exact match at start
        if name.startswith(query_lower) or symbol.startswith(query_lower):
            return (4, len(name))  # Shorter names first
        # High priority: contains exact match
        if query_lower in name or query_lower in symbol:
            return (3, len(name))
        # Medium priority: partial word match
        if any(word.startswith(query_lower) for word in name.split()):
            return (2, len(name))
        # Low priority: other matches
        return (1, len(name))
    
    return sorted(results, key=get_relevance_score, reverse=True)

# Function to safely get stock data with retries
def get_stock_data_safe(symbol, period, max_retries=3):
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if not data.empty:
                return data
            time.sleep(1)  # Wait before retry
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                st.error(f"Error fetching data: {str(e)}")
                return None
            time.sleep(1)  # Wait before retry
    return None

# Function to search for stocks using Yahoo Finance API
def search_stocks(query):
    if not query:
        return []
    
    try:
        base_url = "https://query1.finance.yahoo.com/v1/finance/search"
        params = {
            'q': query,
            'quotesCount': 10,
            'newsCount': 0,
            'enableFuzzyQuery': True,
            'quotesQueryId': 'tss_match_phrase_query'
        }
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
            
        data = response.json()
        
        if 'quotes' in data:
            results = []
            for quote in data['quotes']:
                if 'symbol' in quote:
                    name = quote.get('longname', quote.get('shortname', 'N/A'))
                    exchange = quote.get('exchange', 'N/A')
                    symbol = quote['symbol']
                    
                    # Filter out non-stock items
                    if not any(x in symbol for x in ['-USD', 'CURRENCY', 'INDEX']):
                        results.append({
                            'symbol': symbol,
                            'name': name,
                            'exchange': exchange
                        })
            return results
    except Exception as e:
        st.error(f"Search error: {str(e)}")
    return []

# Function to get USD to INR exchange rate
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_usd_to_inr_rate():
    try:
        data = get_stock_data_safe("INR=X", period="1d")
        if data is not None and not data.empty:
            return data['Close'].iloc[-1]
    except:
        pass
    return 83.5  # Default value if unable to fetch

# Function to calculate moving averages
def calculate_moving_averages(data):
    """Calculate Simple and Exponential Moving Averages"""
    # Calculate 20-day and 50-day SMAs
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    
    # Calculate 20-day and 50-day EMAs
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
    
    return data

# Function to make predictions
def make_predictions(symbol, period='1y'):
    try:
        # Get stock data
        data = get_stock_data_safe(symbol, period)
        if data is None:
            st.error("Unable to fetch stock data for predictions")
            return None
        
        # Calculate moving averages
        data = calculate_moving_averages(data)
        
        # Generate predictions for next 5 days
        last_close = data['Close'].iloc[-1]
        last_sma20 = data['SMA20'].iloc[-1]
        last_sma50 = data['SMA50'].iloc[-1]
        last_ema20 = data['EMA20'].iloc[-1]
        last_ema50 = data['EMA50'].iloc[-1]
        
        # Calculate weights based on recent performance
        weights = {
            'Close': 0.4,
            'SMA20': 0.15,
            'SMA50': 0.15,
            'EMA20': 0.15,
            'EMA50': 0.15
        }
        
        # Calculate predictions for next 5 days
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=5, freq='B')
        predictions_list = []
        current_price = last_close
        
        for _ in range(5):
            prediction = (current_price * weights['Close'] + 
                        last_sma20 * weights['SMA20'] +
                        last_sma50 * weights['SMA50'] +
                        last_ema20 * weights['EMA20'] +
                        last_ema50 * weights['EMA50'])
            predictions_list.append(prediction)
            current_price = prediction  # Use previous prediction for next day
        
        # Create future predictions DataFrame
        future_predictions = pd.DataFrame({
            'Date': future_dates,
            'Predicted': predictions_list
        })
        
        # Calculate trend direction based on 5-day prediction
        trend = 'Upward' if predictions_list[-1] > last_close else 'Downward'
        change_percent = ((predictions_list[-1] - last_close) / last_close) * 100
        
        # Create prediction results
        results = {
            'Current Price': round(last_close, 2),
            'Predicted Prices': [round(p, 2) for p in predictions_list],
            'Predicted Dates': future_dates,
            'Predicted Trend': trend,
            'Predicted Change %': round(change_percent, 2)
        }
        
        # Visualization of moving averages with dark theme
        fig = go.Figure()
        
        # Add closing price
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['Close'],
            name='Close Price',
            line=dict(color='#00BFFF')  # Deep sky blue
        ))
        
        # Add future predictions
        fig.add_trace(go.Scatter(
            x=future_predictions['Date'],
            y=future_predictions['Predicted'],
            name='Predicted Price',
            line=dict(color='#00BFFF', dash='dot')  # Matching blue with dot pattern
        ))
        
        # Add SMA20
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['SMA20'],
            name='20-day SMA',
            line=dict(color='#FFA500')  # Orange
        ))
        
        # Add SMA50
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['SMA50'],
            name='50-day SMA',
            line=dict(color='#FF4500', dash='dash')  # Orange red
        ))
        
        # Add EMA20
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['EMA20'],
            name='20-day EMA',
            line=dict(color='#32CD32')  # Lime green
        ))
        
        # Add EMA50
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['EMA50'],
            name='50-day EMA',
            line=dict(color='#9370DB', dash='dash')  # Medium purple
        ))
        
        # Update layout with dark theme
        fig.update_layout(
            title=f'Stock Price and Moving Averages for {symbol}',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            paper_bgcolor='#1E1E1E',  # Dark background
            plot_bgcolor='#1E1E1E',   # Dark plot area
            font=dict(
                color='#FFFFFF'  # White text
            ),
            xaxis=dict(
                gridcolor='#333333',  # Darker grid lines
                zerolinecolor='#333333'
            ),
            yaxis=dict(
                gridcolor='#333333',  # Darker grid lines
                zerolinecolor='#333333'
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(30, 30, 30, 0.8)",  # Dark legend background
                bordercolor="rgba(255, 255, 255, 0.3)",  # Light border
                borderwidth=1,
                font=dict(color='#FFFFFF')  # White legend text
            )
        )
        
        # Add crossover signals
        signal_text = []
        for i in range(1, len(data)):
            # Check for SMA crossovers
            if (data['SMA20'].iloc[i-1] <= data['SMA50'].iloc[i-1] and 
                data['SMA20'].iloc[i] > data['SMA50'].iloc[i]):
                signal_text.append(f"Bullish SMA Crossover at {data.index[i].strftime('%Y-%m-%d')}")
            elif (data['SMA20'].iloc[i-1] >= data['SMA50'].iloc[i-1] and 
                  data['SMA20'].iloc[i] < data['SMA50'].iloc[i]):
                signal_text.append(f"Bearish SMA Crossover at {data.index[i].strftime('%Y-%m-%d')}")
        
        if signal_text:
            results['Signals'] = '<br>'.join(signal_text)
        
        return results, fig
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# Check if a stock is already in INR (Indian exchanges)
def is_indian_stock(symbol):
    return symbol.endswith(('.NS', '.BO', '.BSE', '.NSE'))

# Initialize session state
if 'stock_symbol' not in st.session_state:
    st.session_state.stock_symbol = None
if 'stock_name' not in st.session_state:
    st.session_state.stock_name = None

# Set up the page with dark theme
st.set_page_config(
    page_title="Stock Market App with Predictions",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': 'Stock Market Prediction App'
    }
)

# Set darker theme
st.markdown("""
<style>
    /* Main app background and text */
    .stApp {
        background-color: #0A0A0A;
        color: #FAFAFA;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #121212;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00BFFF !important;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #00BFFF !important;
        font-weight: bold !important;
    }
    
    /* Metric labels */
    [data-testid="stMetricLabel"] {
        color: #CCCCCC !important;
    }
    
    /* Positive and negative delta colors */
    [data-testid="stMetricDelta"][data-value^="-"] {
        color: #FF4040 !important;
        font-weight: bold !important;
    }
    
    [data-testid="stMetricDelta"]:not([data-value^="-"]) {
        color: #32CD32 !important;
        font-weight: bold !important;
    }
    
    /* Select box styling */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1E1E1E;
        border-color: #333333;
        color: #FAFAFA;
    }
    .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: #00BFFF;
    }
    div[role="listbox"] {
        background-color: #1E1E1E;
        max-height: 300px !important;
        overflow-y: auto !important;
    }
    div[role="option"] {
        color: #FAFAFA;
        padding: 8px 12px;
    }
    div[role="option"]:hover {
        background-color: #333333;
    }
    
    /* Text input styling */
    .stTextInput input {
        background-color: #1E1E1E;
        color: #FAFAFA;
        border-color: #333333;
    }
    .stTextInput input:hover, .stTextInput input:focus {
        border-color: #00BFFF;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #1E1E1E;
        color: #FAFAFA;
        border-color: #333333;
    }
    .stButton button:hover {
        border-color: #00BFFF;
        color: #00BFFF;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1E1E1E;
    }
    .stTabs [data-baseweb="tab"] {
        color: #CCCCCC;
    }
    .stTabs [aria-selected="true"] {
        color: #00BFFF !important;
        border-bottom-color: #00BFFF !important;
    }
    
    /* Dataframe styling */
    .stDataFrame table {
        background-color: #1E1E1E;
    }
    .stDataFrame th {
        background-color: #333333;
        color: #FAFAFA;
    }
    .stDataFrame td {
        color: #FAFAFA;
    }
    
    /* Scrollbar styles */
    ::-webkit-scrollbar {
        width: 8px;
        background: #0A0A0A;
    }
    ::-webkit-scrollbar-track {
        background: #1E1E1E;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: #333333;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00BFFF;
    }
    
    /* Warning and info boxes */
    .stAlert {
        background-color: #1E1E1E;
        border-color: #333333;
    }
</style>
""", unsafe_allow_html=True)

# Create a two-column layout for the header
header_col1, header_col2 = st.columns([2, 1])

with header_col1:
    st.title("📈 Stock Market App with Predictions (INR)")

with header_col2:
    # Initialize session state for dropdown
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ''
    if 'filtered_stocks' not in st.session_state:
        st.session_state.filtered_stocks = []
    if 'stock_history' not in st.session_state:
        st.session_state.stock_history = []

    # Add CSS for forced scrollbar in dropdown
    st.markdown("""
    <style>
    /* Target Streamlit selectbox dropdown menu */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="select"] > div > div > div[role="listbox"],
    div[role="listbox"] {
        max-height: 200px !important;
        overflow-y: auto !important;
    }

    /* Ensure scrollbar is visible in WebKit browsers */
    div[role="listbox"]::-webkit-scrollbar {
        -webkit-appearance: none;
        width: 10px;
    }

    div[role="listbox"]::-webkit-scrollbar-thumb {
        border-radius: 5px;
        background-color: rgba(0,0,0,.5);
        -webkit-box-shadow: 0 0 1px rgba(255,255,255,.5);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Search box
    search_query = st.text_input(
        "🔍 Search Stock",
        value="",  # Don't use session state here to avoid auto-search
        placeholder="Type to filter stocks...",
        key="search_input"
    )
    
    # Only search when query changes and is not empty
    if search_query and len(search_query) >= 1:
        with st.spinner("Searching..."):
            results = search_stocks(search_query)
            results = sort_by_relevance(results, search_query)
            
            if results:
                # Format results for dropdown
                formatted_results = [
                    {
                        'label': f"{result['name']} - {result['symbol']}",
                        'value': result['symbol'],
                        'name': result['name'],
                        'exchange': result['exchange']
                    }
                    for result in results
                    if not any(x in result['symbol'] for x in ['-USD', 'CURRENCY', 'INDEX'])
                ]
                
                # Limit to 10 results to ensure dropdown doesn't get too large
                if len(formatted_results) > 10:
                    formatted_results = formatted_results[:10]
                    st.info(f"Showing top 10 matches for '{search_query}'")
                
                # Update session state
                st.session_state.filtered_stocks = formatted_results
            else:
                st.session_state.filtered_stocks = []
                st.info("No matching stocks found")
    
    # Show dropdown only if we have results
    if st.session_state.filtered_stocks:
        # Create dropdown for stock selection with height attribute
        stock_options = [stock['label'] for stock in st.session_state.filtered_stocks]
        
        # Create dropdown for stock selection with height attribute
        selected_stock = st.selectbox(
            "Select a Stock",
            options=stock_options,
            key="stock_dropdown"
        )
        
        # Handle selection
        if st.button("View Stock Data", key="view_stock_btn"):
            # Find the selected stock info
            for stock in st.session_state.filtered_stocks:
                if stock['label'] == selected_stock:
                    # Set the selected stock
                    st.session_state.stock_symbol = stock['value']
                    st.session_state.stock_name = stock['name']
                    
                    # Update history
                    if stock not in st.session_state.stock_history:
                        st.session_state.stock_history.insert(0, stock)
                        if len(st.session_state.stock_history) > 5:
                            st.session_state.stock_history.pop()
                    
                    # Rerun to refresh the page
                    st.rerun()
    elif search_query:
        st.info("No matching stocks found")

    # Display Recently Viewed Stocks
    if st.session_state.stock_history:
        st.markdown("### Recently Viewed")
        cols = st.columns(1)
        with cols[0]:
            for stock in st.session_state.stock_history:
                if st.button(
                    f"📈 {stock['label']}",
                    key=f"history_{stock['value']}",
                    help=f"Click to view {stock['name']}",
                    use_container_width=True
                ):
                    st.session_state.stock_symbol = stock['value']
                    st.session_state.stock_name = stock['name']
                    st.rerun()

# Sidebar for other inputs
st.sidebar.header("Settings")

# If no stock selected from search, allow manual input
if not st.session_state.stock_symbol:
    manual_input = st.sidebar.text_input("Or enter stock symbol manually (e.g., RELIANCE.NS)", "")
    if manual_input:
        st.session_state.stock_symbol = manual_input
        # For manual input, use the symbol as the name until we get data
        st.session_state.stock_name = manual_input

# Create a dropdown for time period selection with more options
time_periods = {
    "5 Days": "5d", 
    "1 Week": "1wk", 
    "1 Month": "1mo", 
    "3 Months": "3mo", 
    "6 Months": "6mo", 
    "1 Year": "1y", 
    "5 Years": "5y", 
    "Maximum": "max"
}
selected_period = st.sidebar.selectbox("Select Time Period", list(time_periods.keys()))
period = time_periods[selected_period]

# Moving Average information in sidebar
st.sidebar.header("Moving Average Analysis")
st.sidebar.info("""
**Moving Average Indicators:**
- 20-day SMA (Short-term trend)
- 50-day SMA (Medium-term trend)
- 20-day EMA (Short-term momentum)
- 50-day EMA (Medium-term momentum)

When short-term MA crosses above long-term MA, it may indicate an upward trend.
When short-term MA crosses below long-term MA, it may indicate a downward trend.
""")

# Get the USD to INR exchange rate
usd_to_inr_rate = get_usd_to_inr_rate()
st.sidebar.info(f"Current Exchange Rate: 1 USD = ₹{usd_to_inr_rate:.2f}")

# Main app logic
if st.session_state.stock_symbol:
    # Check if stock is from Indian exchange
    is_inr_stock = is_indian_stock(st.session_state.stock_symbol)
    if is_inr_stock:
        st.sidebar.success(f"Note: {st.session_state.stock_symbol} is already quoted in INR")
    
    # Show a loading message
    with st.spinner(f"Loading data for {st.session_state.stock_symbol}..."):
        # Get the stock data
        data = get_stock_data_safe(st.session_state.stock_symbol, period)
        
        # If we got data back
        if data is not None and not data.empty:
            try:
                # Display the stock name and current price
                # Get the ticker info to get the company name
                try:
                    ticker = yf.Ticker(st.session_state.stock_symbol)
                    company_name = ticker.info.get('longName', ticker.info.get('shortName', st.session_state.stock_name))
                    if company_name:
                        st.session_state.stock_name = company_name
                except:
                    # If we can't get the company name, use what we have
                    company_name = st.session_state.stock_name or st.session_state.stock_symbol
                
                # Display header with company name and symbol
                st.header(f"{company_name} ({st.session_state.stock_symbol}) Stock Data")
                
                # Get the most recent closing price
                current_price_original = data['Close'].iloc[-1]
                
                # Convert to INR if not already in INR
                if not is_inr_stock:
                    current_price = current_price_original * usd_to_inr_rate
                else:
                    current_price = current_price_original
                
                # Calculate the price change percentage
                if len(data) > 1:
                    previous_price = data['Close'].iloc[-2]
                    price_change = ((current_price_original - previous_price) / previous_price * 100)
                else:
                    price_change = 0
                
                # Display the price with a color indicator for up/down
                st.metric("Current Price", f"₹{current_price:.2f}", f"{price_change:.2f}%")
                
                # Create tabs for different views
                tab1, tab2 = st.tabs(["Historical Data", "Predictions"])
                
                with tab1:
                    # Create a price chart
                    st.subheader("Stock Price History")
                    
                    # Create a line chart using plotly
                    fig = go.Figure()
                    
                    # Convert price data to INR if not already in INR
                    if not is_inr_stock:
                        y_values = data['Close'] * usd_to_inr_rate
                    else:
                        y_values = data['Close']
                        
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=y_values,
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#0052cc')
                    ))
                    
                    # Customize the chart
                    fig.update_layout(
                        title=f"{st.session_state.stock_symbol} Price History",
                        xaxis_title="Date", 
                        yaxis_title="Price (INR)",
                        height=400,
                        hovermode="x unified"
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show a small table of recent prices
                    st.subheader("Recent Price Data (INR)")
                    
                    # Clone the dataframe to avoid modifying the original
                    display_data = data[['Open', 'High', 'Low', 'Close']].copy()
                    
                    # Convert price data to INR if not already in INR
                    if not is_inr_stock:
                        for col in display_data.columns:
                            display_data[col] = display_data[col] * usd_to_inr_rate
                    
                    # Display only the most recent 5 days, with newest first
                    st.dataframe(display_data.sort_index(ascending=False).head(5))
                
                with tab2:
                    st.subheader("Price Predictions")
                    
                    # Check if we have enough data for prediction
                    if len(data) > 50:  # Need at least 50 days for moving averages
                        try:
                            # Make predictions using moving averages
                            predictions = make_predictions(st.session_state.stock_symbol, period)
                            
                            if predictions:
                                results, fig = predictions
                                
                                # Convert predictions to INR if needed
                                if not is_inr_stock:
                                    results['Current Price'] = results['Current Price'] * usd_to_inr_rate
                                    results['Predicted Prices'] = [p * usd_to_inr_rate for p in results['Predicted Prices']]
                                
                                # Display prediction results
                                st.subheader("Price Predictions")
                                
                                # Create columns for current price and 5-day predictions
                                cols = st.columns(6)
                                
                                # Display current price
                                with cols[0]:
                                    st.metric("Current", f"₹{results['Current Price']:.2f}")
                                
                                # Display 5-day predictions
                                for i in range(5):
                                    with cols[i+1]:
                                        date = results['Predicted Dates'][i].strftime('%Y-%m-%d')
                                        price = results['Predicted Prices'][i]
                                        change = ((price - results['Current Price']) / results['Current Price']) * 100
                                        st.metric(f"Day {i+1}", f"₹{price:.2f}", f"{change:.1f}%")
                                
                                # Display the chart
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display crossover signals if available
                                if 'Signals' in results:
                                    st.subheader("Recent Crossover Signals")
                                    st.markdown(results['Signals'], unsafe_allow_html=True)
                            else:
                                st.warning("Could not generate predictions. Please try a different time period.")
                        except Exception as e:
                            st.error(f"Error in prediction: {str(e)}")
                    else:
                        # More helpful message when there's not enough data
                        st.warning(
                            "Not enough data for predictions. This stock needs at least 50 days of historical data."
                            "\n\nPlease try one of these solutions:"
                            "\n1. Select a longer time period (e.g., 1 Year or 5 Years)"
                            "\n2. Choose a stock with more historical data"
                        )
                        
                        # Show available time periods
                        st.info("Available time periods: " + ", ".join(list(time_periods.keys())))
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
        else:
            # Show error if no data is found
            st.error(f"No data found for {st.session_state.stock_symbol}. Please check the symbol and try again.")
else:
    # Prompt user to enter a stock symbol
    st.info("Please search for a stock or enter a stock symbol to see data.")

# Footer
st.caption("Mini Project MP-67") 
st.caption("Mentor:Rasika Ransing") 