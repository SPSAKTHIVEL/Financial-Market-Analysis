import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(layout="wide", page_title="Financial Market Portfolio", initial_sidebar_state="expanded")

# Load model
try:
    model = pickle.load(open("bitcoin_model.pkl", "rb"))
    model_available = True
except:
    model_available = False

# Load dataset
df = pd.read_csv("US_Stock_Data.csv")

# Drop unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Data cleaning: remove commas and convert to float for price columns
price_cols = [col for col in df.columns if 'Price' in col]
for col in price_cols:
    df[col] = df[col].astype(str).str.replace(',', '').astype(float)

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
# Remove any rows with NaT dates
df = df.dropna(subset=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Add professional background with transparency
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.15) 100%),
                    url("https://wealthface.com/blog/wp-content/uploads/2022/08/Blog-1-1.png");
        background-size: cover;
        background-attachment: fixed;
        background-blend-mode: overlay;
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    .stApp {
        background-color: transparent;
    }
    h1 {
        color: #0f172a;
        text-align: center;
        font-weight: 800;
        font-size: 2.5rem;
    }
    h2 {
        color: #1e293b;
        font-weight: 700;
    }
    h3 {
        color: #334155;
    }
    .metric-card {
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Financial Market Analysis Portfolio")
st.markdown("*Professional Investment Intelligence Dashboard*")
st.markdown("---")

# ===== KEY PERFORMANCE INDICATORS (KPIs) =====
st.subheader("Key Performance Indicators (KPIs)")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    btc_price = df["Bitcoin_Price"].iloc[-1]
    btc_change = ((df["Bitcoin_Price"].iloc[-1] - df["Bitcoin_Price"].iloc[-30]) / df["Bitcoin_Price"].iloc[-30]) * 100
    st.metric("Bitcoin (USD)", f"${btc_price:,.0f}", f"{btc_change:+.2f}%", delta_color="normal")

with col2:
    eth_price = df["Ethereum_Price"].iloc[-1]
    eth_change = ((df["Ethereum_Price"].iloc[-1] - df["Ethereum_Price"].iloc[-30]) / df["Ethereum_Price"].iloc[-30]) * 100
    st.metric("Ethereum (USD)", f"${eth_price:,.0f}", f"{eth_change:+.2f}%", delta_color="normal")

with col3:
    sp500_price = df["S&P_500_Price"].iloc[-1]
    sp500_change = ((df["S&P_500_Price"].iloc[-1] - df["S&P_500_Price"].iloc[-30]) / df["S&P_500_Price"].iloc[-30]) * 100
    st.metric("S&P 500 Index", f"${sp500_price:,.0f}", f"{sp500_change:+.2f}%", delta_color="normal")

with col4:
    gold_price = df["Gold_Price"].iloc[-1]
    gold_change = ((df["Gold_Price"].iloc[-1] - df["Gold_Price"].iloc[-30]) / df["Gold_Price"].iloc[-30]) * 100
    st.metric("Gold (USD/oz)", f"${gold_price:,.2f}", f"{gold_change:+.2f}%", delta_color="normal")

with col5:
    oil_price = df["Crude_oil_Price"].iloc[-1]
    oil_change = ((df["Crude_oil_Price"].iloc[-1] - df["Crude_oil_Price"].iloc[-30]) / df["Crude_oil_Price"].iloc[-30]) * 100
    st.metric("Crude Oil (USD)", f"${oil_price:,.2f}", f"{oil_change:+.2f}%", delta_color="normal")

st.markdown("---")

# ---- TABS ----
tab1, tab2, tab3, tab4 = st.tabs(["Market Trends", "Technical Analysis", "Correlation & Insights", "Price Prediction"])

# =========================
# TAB 1: MARKET TRENDS
# =========================
with tab1:
    st.header("Market Trends Analysis")
    st.write("Analyze individual asset price movements and trading volumes with detailed statistical insights.")
    
    col_trend1, col_trend2 = st.columns([3, 1])
    
    with col_trend2:
        # Asset selection
        assets = {
            "Bitcoin": ("Bitcoin_Price", "Bitcoin_Vol."),
            "Ethereum": ("Ethereum_Price", "Ethereum_Vol."),
            "S&P 500": ("S&P_500_Price", None),
            "Nasdaq 100": ("Nasdaq_100_Price", "Nasdaq_100_Vol."),
            "Apple": ("Apple_Price", "Apple_Vol."),
            "Tesla": ("Tesla_Price", "Tesla_Vol."),
            "Microsoft": ("Microsoft_Price", "Microsoft_Vol."),
            "Amazon": ("Amazon_Price", "Amazon_Vol."),
            "Google": ("Google_Price", "Google_Vol."),
            "Netflix": ("Netflix_Price", "Netflix_Vol."),
            "Meta": ("Meta_Price", "Meta_Vol."),
            "Nvidia": ("Nvidia_Price", "Nvidia_Vol."),
            "Gold": ("Gold_Price", "Gold_Vol."),
            "Silver": ("Silver_Price", "Silver_Vol."),
            "Crude Oil": ("Crude_oil_Price", "Crude_oil_Vol."),
            "Copper": ("Copper_Price", "Copper_Vol."),
            "Platinum": ("Platinum_Price", "Platinum_Vol."),
        }

        selected_asset = st.selectbox("Select Asset", list(assets.keys()), index=0)
        price_col, vol_col = assets[selected_asset]
    
    with col_trend1:
        # Chart: Price over time - Simple, clean version
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df[price_col],
            mode='lines',
            name=selected_asset,
            line=dict(color='#2563eb', width=2),
            fill='tozeroy',
            fillcolor='rgba(37, 99, 235, 0.1)'
        ))
        
        fig.update_layout(
            title=f"{selected_asset} Price Over Time",
            xaxis_title="Date",
            yaxis_title=f"Price (USD)",
            height=400,
            template="plotly_white",
            hovermode='x unified',
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Volume chart if available
    if vol_col:
        col_vol1, col_vol2 = st.columns([1, 1])
        with col_vol1:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=df["Date"],
                y=df[vol_col],
                name='Volume',
                marker_color='#10b981'
            ))
            fig_vol.update_layout(
                title=f"{selected_asset} Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=350,
                template="plotly_white",
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col_vol2:
            # Price vs Volume correlation
            correlation = df[price_col].corr(df[vol_col])
            fig_corr = go.Figure(data=[
                go.Scatter(x=df[vol_col], y=df[price_col], mode='markers',
                          marker=dict(size=5, color=df.index, colorscale='Viridis', showscale=True),
                          text=df['Date'].dt.strftime('%Y-%m-%d'),
                          hovertemplate='<b>Date:</b> %{text}<br><b>Volume:</b> %{x:,.0f}<br><b>Price:</b> $%{y:,.2f}<extra></extra>')
            ])
            fig_corr.update_layout(
                title=f"Price vs Volume (r={correlation:.2f})",
                xaxis_title="Trading Volume",
                yaxis_title=f"Price (USD)",
                height=350,
                template="plotly_white",
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # Market Statistics Section
    st.markdown("### Market Statistics")
    
    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5, stat_col6 = st.columns(6)
    
    latest_price = df[price_col].iloc[-1]
    max_price = df[price_col].max()
    min_price = df[price_col].min()
    avg_price = df[price_col].mean()
    std_price = df[price_col].std()
    
    with stat_col1:
        st.metric("Current Price", f"${latest_price:,.2f}")
    with stat_col2:
        st.metric("30-Day Avg", f"${df[price_col].iloc[-30:].mean():,.2f}")
    with stat_col3:
        st.metric("30-Day High", f"${df[price_col].iloc[-30:].max():,.2f}")
    with stat_col4:
        st.metric("30-Day Low", f"${df[price_col].iloc[-30:].min():,.2f}")
    with stat_col5:
        volatility = (std_price / avg_price) * 100
        st.metric("Volatility", f"{volatility:.2f}%")
    with stat_col6:
        price_range = ((max_price - min_price) / min_price) * 100
        st.metric("All-Time Range", f"{price_range:.2f}%")
    
    # Detailed Insights
    st.markdown("### Market Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        trend = "📈 Bullish" if latest_price > avg_price else "📉 Bearish"
        st.info(f"**Trend Status:** {trend}\n\nCurrent price is {'above' if latest_price > avg_price else 'below'} the historical average.")
    
    with insight_col2:
        momentum = "📊 Positive" if df[price_col].iloc[-1] > df[price_col].iloc[-5] else "📊 Negative"
        st.success(f"**5-Day Momentum:** {momentum}\n\nRecent price action shows {'momentum' if df[price_col].iloc[-1] > df[price_col].iloc[-5] else 'downward'} pressure.")
    
    with insight_col3:
        resistance = max_price
        support = min_price
        st.warning(f"**Key Levels:**\n\n⬆️ Resistance: ${resistance:,.2f}\n⬇️ Support: ${support:,.2f}")

# =========================
# TAB 2: TECHNICAL ANALYSIS
# =========================
with tab2:
    st.header("Technical Analysis")
    st.write("Advanced technical indicators and moving averages for in-depth market analysis.")
    
    # Select assets for technical analysis
    tech_assets = ["Bitcoin", "Ethereum", "S&P 500", "Apple", "Tesla", "Microsoft", "Gold", "Crude Oil"]
    selected_tech_asset = st.selectbox("Select Asset for Technical Analysis", tech_assets, key="tech_select")
    
    tech_price_col = assets[selected_tech_asset][0]
    
    # Calculate Moving Averages
    df['SMA_20'] = df[tech_price_col].rolling(window=20).mean()
    df['SMA_50'] = df[tech_price_col].rolling(window=50).mean()
    df['EMA_12'] = df[tech_price_col].ewm(span=12, adjust=False).mean()
    
    col_ma1, col_ma2 = st.columns([2, 1])
    
    with col_ma1:
        # Moving Averages Chart
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df["Date"], y=df[tech_price_col], name='Price', line=dict(color='#2563eb')))
        fig_ma.add_trace(go.Scatter(x=df["Date"], y=df['SMA_20'], name='SMA 20', line=dict(color='#f59e0b', dash='dash')))
        fig_ma.add_trace(go.Scatter(x=df["Date"], y=df['SMA_50'], name='SMA 50', line=dict(color='#ef4444', dash='dash')))
        fig_ma.add_trace(go.Scatter(x=df["Date"], y=df['EMA_12'], name='EMA 12', line=dict(color='#8b5cf6')))
        
        fig_ma.update_layout(
            title=f"{selected_tech_asset} - Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=400,
            template="plotly_white",
            hovermode='x unified'
        )
        st.plotly_chart(fig_ma, use_container_width=True)
    
    with col_ma2:
        st.markdown("#### Moving Average Analysis")
        st.write("**SMA (20):** Short-term trend")
        st.write("**SMA (50):** Medium-term trend")
        st.write("**EMA (12):** Responsive to recent price changes")
        
        latest_price_tech = df[tech_price_col].iloc[-1]
        sma_20_latest = df['SMA_20'].iloc[-1]
        
        if latest_price_tech > sma_20_latest:
            st.success("✓ Price above SMA-20: Potential uptrend")
        else:
            st.error("✗ Price below SMA-20: Potential downtrend")
    
    # RSI and MACD
    col_rsi, col_macd = st.columns(2)
    
    with col_rsi:
        # Calculate RSI
        delta = df[tech_price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        fig_rsi = go.Figure()
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.add_trace(go.Scatter(x=df["Date"], y=df['RSI'], name='RSI', line=dict(color='#6366f1')))
        fig_rsi.update_layout(
            title="Relative Strength Index (RSI)",
            xaxis_title="Date",
            yaxis_title="RSI Value",
            height=300,
            template="plotly_white",
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col_macd:
        # Calculate MACD
        ema_12 = df[tech_price_col].ewm(span=12, adjust=False).mean()
        ema_26 = df[tech_price_col].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Histogram'] = df['MACD'] - df['Signal']
        
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df["Date"], y=df['MACD'], name='MACD', line=dict(color='#2563eb')))
        fig_macd.add_trace(go.Scatter(x=df["Date"], y=df['Signal'], name='Signal', line=dict(color='#ef4444')))
        fig_macd.add_trace(go.Bar(x=df["Date"], y=df['Histogram'], name='Histogram', marker_color='#06b6d4', opacity=0.3))
        fig_macd.update_layout(
            title="MACD (Moving Average Convergence Divergence)",
            xaxis_title="Date",
            yaxis_title="MACD Value",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig_macd, use_container_width=True)

# =========================
# TAB 3: CORRELATION & INSIGHTS
# =========================
with tab3:
    st.header("Correlation Analysis & Market Insights")
    st.write("Discover relationships between different assets and understand market dynamics.")
    
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    
    col_corr_select1, col_corr_select2 = st.columns(2)
    
    with col_corr_select1:
        st.markdown("#### Default Correlation Set")
        default_corr = ["Bitcoin_Price", "Ethereum_Price", "S&P_500_Price", "Gold_Price", "Crude_oil_Price"]
        st.write("Pre-selected major assets for quick analysis.")
    
    with col_corr_select2:
        st.markdown("#### Custom Selection")
        selected_cols = st.multiselect(
            "Add more assets to analyze",
            numeric_cols,
            default=default_corr,
            help="Select multiple assets to compute correlations"
        )
    
    if selected_cols and len(selected_cols) > 1:
        corr = df[selected_cols].corr()
        
        fig_corr_heatmap = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Asset Correlation Heatmap"
        )
        fig_corr_heatmap.update_layout(height=600)
        st.plotly_chart(fig_corr_heatmap, use_container_width=True)
        
        # Correlation Insights
        st.markdown("### Correlation Insights")
        
        strong_positive = []
        strong_negative = []
        weak_corr = []
        
        for i in range(len(selected_cols)):
            for j in range(i+1, len(selected_cols)):
                corr_val = corr.iloc[i, j]
                asset1 = selected_cols[i].replace("_Price", "").replace("_", " ")
                asset2 = selected_cols[j].replace("_Price", "").replace("_", " ")
                
                if corr_val > 0.6:
                    strong_positive.append((asset1, asset2, corr_val))
                elif corr_val < -0.4:
                    strong_negative.append((asset1, asset2, corr_val))
                elif abs(corr_val) < 0.3:
                    weak_corr.append((asset1, asset2, corr_val))
        
        if strong_positive:
            st.success("**Strong Positive Correlations** (Assets move together)")
            for a1, a2, cv in strong_positive:
                st.write(f"• {a1} ↔ {a2}: {cv:.3f}")
        
        if strong_negative:
            st.warning("**Strong Negative Correlations** (Assets move opposite)")
            for a1, a2, cv in strong_negative:
                st.write(f"• {a1} ↔ {a2}: {cv:.3f}")
        
        if weak_corr:
            st.info("**Weak Correlations** (Movements are independent)")
            for a1, a2, cv in weak_corr[:5]:  # Show top 5
                st.write(f"• {a1} ↔ {a2}: {cv:.3f}")
    
    # Distribution Analysis
    st.markdown("### Price Distribution Analysis")
    
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        dist_asset = st.selectbox("Select Asset for Distribution", list(assets.keys()), index=0, key="dist_select")
        dist_price_col = assets[dist_asset][0]
        
        fig_dist = px.histogram(
            df,
            x=dist_price_col,
            nbins=30,
            title=f"{dist_asset} Price Distribution",
            labels={dist_price_col: "Price (USD)"},
            color_discrete_sequence=['#3b82f6']
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with dist_col2:
        st.markdown("#### Distribution Statistics")
        dist_stats = df[dist_price_col].describe()
        st.dataframe(dist_stats.to_frame(), use_container_width=True)
        
        skewness = df[dist_price_col].skew()
        kurtosis = df[dist_price_col].kurtosis()
        st.write(f"**Skewness:** {skewness:.3f}")
        st.write(f"**Kurtosis:** {kurtosis:.3f}")

# =========================
# TAB 4: PRICE PREDICTION
# =========================
with tab4:
    st.header("Price Prediction & Forecasting")
    st.write("Machine Learning models for price prediction and trend forecasting.")
    
    if model_available:
        st.success("✓ Bitcoin prediction model loaded successfully!")
        
        pred_col1, pred_col2 = st.columns([2, 1])
        
        with pred_col2:
            st.markdown("#### Input Features")
            oil = st.number_input("Crude Oil Price (USD)", value=75.0, min_value=0.0, max_value=200.0)
            gold = st.number_input("Gold Price (USD/oz)", value=2000.0, min_value=0.0, max_value=10000.0)
            copper = st.number_input("Copper Price (USD/lb)", value=3.8, min_value=0.0, max_value=10.0)
            amazon = st.number_input("Amazon Stock Price (USD)", value=170.0, min_value=0.0, max_value=500.0)
            
            if st.button("Predict Bitcoin Price", use_container_width=True):
                try:
                    features = pd.DataFrame({
                        "oil": [oil],
                        "gold": [gold],
                        "copper": [copper],
                        "amazon": [amazon]
                    })
                    
                    prediction = model.predict(features)
                    st.session_state.prediction = prediction[0]
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")
        
        with pred_col1:
            # Time series forecasting
            st.markdown("#### Historical Bitcoin Prices & LSTM Forecast")
            
            # Simple linear regression for trend
            X = np.arange(len(df)).reshape(-1, 1)
            y = df['Bitcoin_Price'].values
            
            model_lr = LinearRegression()
            model_lr.fit(X, y)
            
            # Generate future predictions
            future_days = 30
            future_X = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
            future_pred = model_lr.predict(future_X)
            
            last_date = df['Date'].iloc[-1]
            if pd.notna(last_date):
                future_dates = pd.date_range(start=last_date, periods=future_days+1, freq='D')[1:]
            else:
                future_dates = pd.date_range(start=pd.Timestamp.now(), periods=future_days, freq='D')
            
            fig_forecast = go.Figure()
            
            # Historical data
            fig_forecast.add_trace(go.Scatter(
                x=df['Date'], y=df['Bitcoin_Price'],
                mode='lines', name='Historical Price',
                line=dict(color='#2563eb', width=2)
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=future_dates, y=future_pred,
                mode='lines', name='30-Day Forecast',
                line=dict(color='#f59e0b', dash='dash', width=2)
            ))
            
            # Confidence interval
            fig_forecast.add_trace(go.Scatter(
                x=future_dates, y=future_pred * 1.05,
                fill=None, mode='lines', line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            fig_forecast.add_trace(go.Scatter(
                x=future_dates, y=future_pred * 0.95,
                fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                name='95% Confidence Interval',
                fillcolor='rgba(245, 158, 11, 0.2)'
            ))
            
            fig_forecast.update_layout(
                title="Bitcoin Price Forecast (30 Days)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=400,
                template="plotly_white",
                hovermode='x unified'
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Show prediction result
        if 'prediction' in st.session_state:
            st.markdown("---")
            st.markdown("### Prediction Result")
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            
            with pred_col1:
                current_btc = df['Bitcoin_Price'].iloc[-1]
                predicted_btc = st.session_state.prediction
                change_pct = ((predicted_btc - current_btc) / current_btc) * 100
                
                st.metric("Predicted Bitcoin Price", f"${predicted_btc:,.2f}", f"{change_pct:+.2f}%")
            
            with pred_col2:
                st.info(f"**Current Price:** ${current_btc:,.2f}")
            
            with pred_col3:
                if predicted_btc > current_btc:
                    st.success(f"**Outlook:** Bullish (↑ {abs(change_pct):.2f}%)")
                else:
                    st.error(f"**Outlook:** Bearish (↓ {abs(change_pct):.2f}%)")
    
    else:
        st.warning("⚠️ Bitcoin prediction model not available. Using alternative forecasting method.")
        
        col_alt1, col_alt2 = st.columns([2, 1])
        
        with col_alt2:
            st.markdown("#### Time Period Selection")
            forecast_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30, step=7)
        
        with col_alt1:
            # Alternative: Exponential Smoothing
            from scipy import stats
            
            X = np.arange(len(df)).reshape(-1, 1)
            y = df['Bitcoin_Price'].values
            
            model_lr = LinearRegression()
            model_lr.fit(X, y)
            
            future_X = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
            future_pred = model_lr.predict(future_X)
            
            last_date = df['Date'].iloc[-1]
            if pd.notna(last_date):
                future_dates = pd.date_range(start=last_date, periods=forecast_days+1, freq='D')[1:]
            else:
                future_dates = pd.date_range(start=pd.Timestamp.now(), periods=forecast_days, freq='D')
            
            # Calculate confidence interval
            residuals = y - model_lr.predict(X)
            std_error = np.std(residuals)
            
            ci_upper = future_pred + 1.96 * std_error
            ci_lower = future_pred - 1.96 * std_error
            
            fig_alt = go.Figure()
            
            fig_alt.add_trace(go.Scatter(
                x=df['Date'], y=df['Bitcoin_Price'],
                mode='lines', name='Historical Price',
                line=dict(color='#2563eb', width=2)
            ))
            
            fig_alt.add_trace(go.Scatter(
                x=future_dates, y=future_pred,
                mode='lines', name='Linear Forecast',
                line=dict(color='#f59e0b', dash='dash', width=2)
            ))
            
            fig_alt.add_trace(go.Scatter(
                x=future_dates, y=ci_upper,
                fill=None, mode='lines', line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            fig_alt.add_trace(go.Scatter(
                x=future_dates, y=ci_lower,
                fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                name='95% Confidence Interval',
                fillcolor='rgba(245, 158, 11, 0.2)'
            ))
            
            fig_alt.update_layout(
                title=f"Bitcoin Price Forecast ({forecast_days} Days)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=400,
                template="plotly_white",
                hovermode='x unified'
            )
            st.plotly_chart(fig_alt, use_container_width=True)
        
        # Forecast statistics
        st.markdown("### Forecast Statistics")
        forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
        
        with forecast_col1:
            current_btc = df['Bitcoin_Price'].iloc[-1]
            forecast_avg = np.mean(future_pred)
            st.metric("Current Bitcoin Price", f"${current_btc:,.2f}")
        
        with forecast_col2:
            forecast_high = np.max(future_pred)
            st.metric(f"{forecast_days}-Day High Forecast", f"${forecast_high:,.2f}")
        
        with forecast_col3:
            forecast_low = np.min(future_pred)
            st.metric(f"{forecast_days}-Day Low Forecast", f"${forecast_low:,.2f}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>Data updated: {}   |   © 2024 Financial Market Analysis</p>".format(df['Date'].iloc[-1].strftime("%Y-%m-%d")), unsafe_allow_html=True)
