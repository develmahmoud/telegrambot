import os
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
import telegram # Added for Telegram notifications
import flask from Flask
from threading import Thread


app = Flask(__name__)

# --- API Keys & Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EODHD_API_KEY = os.getenv("EODHD_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") # Added placeholder
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")     # Added placeholder

# --- Indicator Functions (Obfuscated Names) ---
def compute_market_energy(data):
    data["Price_Change"] = data["Close"].pct_change()
    data["EVOL"] = data["Volume"] * (data["Price_Change"] ** 2)
    return data

def adjust_time_factor(data, lookback=5):
    data["Price_Velocity"] = data["Close"].diff()
    velocity = data["Price_Velocity"].rolling(window=lookback).mean()
    c = 1
    time_numeric = np.arange(len(data))
    denominator = np.sqrt(1 - (velocity**2 / c**2))
    denominator = np.where(denominator > 0, denominator, np.nan)
    data["ATDI"] = np.where(velocity.notna() & pd.notna(denominator), time_numeric / denominator, np.nan)
    return data

def calculate_price_force(data):
    data["Price_Velocity"] = data["Close"].diff()
    data["RMI"] = data["Volume"] * data["Price_Velocity"]
    return data

def derive_smoothed_average(data, period=5):
    if "Close" not in data.columns or len(data) < period:
        return pd.Series(index=data.index, dtype=float)
    return data["Close"].rolling(window=period, min_periods=max(1, period // 2)).mean()

def assess_market_flux(data):
    data["Price_Change"] = data["Close"].diff()
    time_numeric = np.arange(1, len(data) + 1)
    data["NVI"] = data["Volume"] * (data["Price_Change"] / time_numeric)
    return data

def measure_price_impetus(data):
    data["Price_Movement"] = data["Close"].diff()
    data["NMI"] = data["Price_Movement"].shift(1) * data["Volume"]
    return data

def evaluate_counter_force(data):
    data["Price_Change"] = data["Close"].diff()
    data["NRI"] = - (data["Price_Change"].shift(1) * data["Volume"])
    return data

def track_price_trajectory(data):
    data["Price_Change"] = data["Close"].diff()
    time_numeric = np.arange(len(data))
    volume_safe = data["Volume"].replace(0, np.nan)
    data["GUMI"] = (data["Price_Change"] * time_numeric) / volume_safe
    return data

def determine_friction_level(data):
    data["Price_Change"] = data["Close"].diff()
    data["GRI"] = data["Volume"] / data["Price_Change"].replace(0, np.nan)
    return data

def model_price_descent(data):
    data["Price_Change"] = data["Close"].diff()
    time_numeric = np.arange(1, len(data) + 1)
    data["GFI"] = - (data["Price_Change"] / time_numeric)
    return data

def quantify_field_variance(data):
    data["Price_Change"] = data["Close"].pct_change()
    time_numeric = np.arange(1, len(data) + 1)
    data["QVI"] = data["Volume"] * (data["Price_Change"] ** 2 / time_numeric)
    return data

def gauge_thermal_flow(data, lookback=5):
    data["Price_Change"] = data["Close"].diff()
    temperature_factor = data["Volume"].rolling(window=lookback).mean() / 1000
    data["TMI"] = data["Volume"] * (data["Price_Change"] / temperature_factor.replace(0, np.nan))
    return data

def correlate_data_streams(data):
    data["Price_Change_Diff"] = data["Close"].diff().fillna(0)
    data["Volume_Filled"] = data["Volume"].fillna(0)
    if len(data["Price_Change_Diff"]) >= 2 and len(data["Volume_Filled"]) >= 2 and not data["Price_Change_Diff"].isnull().all() and not data["Volume_Filled"].isnull().all():
        if data["Price_Change_Diff"].var() != 0 and data["Volume_Filled"].var() != 0:
            correlation_matrix = np.corrcoef(data["Price_Change_Diff"], data["Volume_Filled"])
            data["CCI"] = correlation_matrix[0,1]
        else:
            data["CCI"] = 0
    else:
        data["CCI"] = np.nan
    return data

def analyze_field_strength(data):
    data["Price_Change"] = data["Close"].diff()
    time_numeric = np.arange(1, len(data) + 1)
    data["MVI"] = (data["Price_Change"] * data["Volume"]) / (time_numeric)
    return data

def derive_wave_pattern(data):
    data["Price_Change"] = data["Close"].diff()
    wave_frequency = data["Volume"].rolling(window=5).mean() / 100
    volume_safe = data["Volume"].replace(0, np.nan)
    data["SWI"] = (data["Price_Change"] ** 2 / volume_safe) * wave_frequency
    return data

def calculate_flow_pressure(data):
    data["Price_Change"] = data["Close"].diff()
    pressure_ratio = data["Volume"].rolling(window=5).mean() / 1000
    data["BPI"] = (data["Volume"] / data["Price_Change"].replace(0, np.nan)) * pressure_ratio
    return data

def estimate_uncertainty_index(data):
    data["Price_Volatility"] = data["Close"].pct_change().abs()
    data["Volume_Uncertainty"] = data["Volume"].pct_change().abs()
    time_numeric = np.arange(1, len(data) + 1)
    data["HUI"] = (data["Price_Volatility"] * data["Volume_Uncertainty"]) / time_numeric
    return data

def identify_recursive_patterns(data, lookback=5):
    def calculate_correlation(x):
        if pd.Series(x).notna().sum() < 2 or pd.Series(x).var() == 0:
            return np.nan
        return np.corrcoef(x, np.arange(len(x)))[0,1]
    fractal_pattern = data["Close"].rolling(window=lookback).apply(calculate_correlation, raw=True)
    data["Fractal_Pattern"] = fractal_pattern
    data["Volume_Scale"] = data["Volume"].rolling(window=lookback).mean() / 1000
    data["FAI"] = data["Fractal_Pattern"] * data["Volume_Scale"]
    return data

def evaluate_system_dynamics(data, lookback=5):
    data["Price_Volatility"] = data["Close"].pct_change().abs()
    data["Volume_Chaos"] = data["Volume"].pct_change().abs().replace(0, np.nan)
    time_numeric = np.arange(1, len(data) + 1) / 1000
    data["Time_Scale"] = time_numeric
    data["CTI"] = (data["Price_Volatility"] ** 2 / data["Volume_Chaos"]) * data["Time_Scale"]
    return data

def calculate_adx_indicator(data_hlc, period=14):
    if not all(col in data_hlc.columns for col in ["High", "Low", "Close"]) or len(data_hlc) < period + 1:
        return pd.Series(index=data_hlc.index, dtype=float)
    df = data_hlc.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR_calc"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["DM_plus_calc"] = np.where((df["High"] - df["High"].shift(1)) > (df["Low"].shift(1) - df["Low"]), np.maximum(df["High"] - df["High"].shift(1), 0), 0)
    df["DM_minus_calc"] = np.where((df["Low"].shift(1) - df["Low"]) > (df["High"] - df["High"].shift(1)), np.maximum(df["Low"].shift(1) - df["Low"], 0), 0)
    tr_smoothed = df["TR_calc"].ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    dm_plus_smoothed = df["DM_plus_calc"].ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    dm_minus_smoothed = df["DM_minus_calc"].ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    di_plus = (dm_plus_smoothed / tr_smoothed.replace(0, np.nan)) * 100
    di_minus = (dm_minus_smoothed / tr_smoothed.replace(0, np.nan)) * 100
    dx = (abs(di_plus - di_minus) / abs(di_plus + di_minus).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return adx

# --- Machine Learning & External APIs ---
def predictive_model_processor(data_input, training_lookback=100, feature_lags=None, test_mode=False, mock_rf_signal=0):
    if test_mode:
        return mock_rf_signal

    if feature_lags is None:
        feature_lags = [1, 3, 5, 10, 15]
    
    df = data_input.copy()
    min_required_data_for_processor = training_lookback + max(feature_lags) + 1 

    if len(df) < min_required_data_for_processor:
        return 0 

    df = compute_market_energy(df)
    df = calculate_price_force(df)
    df = estimate_uncertainty_index(df)
    df = evaluate_system_dynamics(df)

    for lag in feature_lags:
        df[f"Price_Pct_Change_Lag_{lag}"] = df["Close"].pct_change(periods=lag)
        df[f"Volume_Shift_Lag_{lag}"] = df["Volume"].shift(lag)

    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0) 

    feature_columns = ["EVOL", "RMI", "HUI", "CTI"] + \
                      [f"Price_Pct_Change_Lag_{lag}" for lag in feature_lags] + \
                      [f"Volume_Shift_Lag_{lag}" for lag in feature_lags]
    
    df.dropna(subset=feature_columns + ["Target"], inplace=True)

    if len(df) < training_lookback: 
        return 0 

    train_df = df.iloc[-training_lookback:]
    X_train = train_df[feature_columns]
    y_train = train_df["Target"]

    if len(X_train) == 0 or len(y_train) == 0 or len(X_train) != len(y_train):
        return 0 
    
    if len(np.unique(y_train)) < 2:
        return 0

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
    try:
        model.fit(X_train, y_train)
    except ValueError as e:
        return 0 

    latest_features_data = data_input.copy() 
    latest_features_data = compute_market_energy(latest_features_data)
    latest_features_data = calculate_price_force(latest_features_data)
    latest_features_data = estimate_uncertainty_index(latest_features_data)
    latest_features_data = evaluate_system_dynamics(latest_features_data)
    for lag in feature_lags:
        latest_features_data[f"Price_Pct_Change_Lag_{lag}"] = latest_features_data["Close"].pct_change(periods=lag)
        latest_features_data[f"Volume_Shift_Lag_{lag}"] = latest_features_data["Volume"].shift(lag)

    current_features_for_prediction = latest_features_data[feature_columns].iloc[-1:]
    
    if current_features_for_prediction.isnull().any().any():
        return 0 

    predicted_signal_raw = model.predict(current_features_for_prediction)
    
    if predicted_signal_raw[0] == 1:
        return 1 
    else:
        return -1 

    return 0 

def get_bitcoin_sentiment_from_gemini(test_mode=False, mock_response=None):
    if test_mode:
        return mock_response

    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GEMINI_API_KEY:
        return "NEUTRAL"
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-pro")
        prompt = (
            "What is the current public sentiment about Bitcoin in the market, considering recent news and social media discussions? "
            "Please provide a brief assessment of whether the sentiment is very positive, positive, neutral, negative, or very negative. "
            "Your answer should be one of these: VERY_POSITIVE, POSITIVE, NEUTRAL, NEGATIVE, VERY_NEGATIVE."
        )
        response = model.generate_content(prompt)
        if response.text:
            valid_sentiments = ["VERY_POSITIVE", "POSITIVE", "NEUTRAL", "NEGATIVE", "VERY_NEGATIVE"]
            cleaned_response = response.text.strip().upper()
            for sentiment_keyword in valid_sentiments:
                if sentiment_keyword in cleaned_response:
                    return sentiment_keyword
            return "NEUTRAL"
        else:
            return "NEUTRAL"
    except Exception as e:
        return "NEUTRAL"

def get_gold_economic_calendar(api_key=None, days_ahead=3, test_mode=False, mock_calendar=None):
    if test_mode and mock_calendar is not None:
        return mock_calendar
    
    if api_key is None or api_key == "YOUR_EODHD_API_KEY_HERE":
        return pd.DataFrame(columns=["date", "event", "country", "importance", "actual", "forecast", "previous"])
    
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        future_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        url = f"https://eodhistoricaldata.com/api/calendar/economic?api_token={api_key}&fmt=json&from={today}&to={future_date}"
        response = requests.get(url)
        
        if response.status_code != 200:
            return pd.DataFrame(columns=["date", "event", "country", "importance", "actual", "forecast", "previous"])
        
        calendar_data = pd.DataFrame(response.json())
        
        if calendar_data.empty:
            return calendar_data
        
        gold_keywords = ["gold", "precious metal", "xau", "inflation", "federal reserve", 
                        "interest rate", "usd", "dollar", "safe haven", "bullion"]
        
        gold_events = calendar_data[calendar_data.apply(
            lambda row: any(keyword in str(row.get("event", "")).lower() for keyword in gold_keywords), axis=1
        )]
        
        important_countries = ["US", "CN", "IN", "EU"]
        important_events = ["cpi", "nfp", "gdp", "fomc", "interest rate decision", "inflation"]
        
        additional_events = calendar_data[
            (calendar_data["country"].isin(important_countries)) & 
            (calendar_data["importance"] == "HIGH") &
            (calendar_data.apply(
                lambda row: any(event in str(row.get("event", "")).lower() for event in important_events), axis=1
            ))
        ]
        
        gold_related_events = pd.concat([gold_events, additional_events]).drop_duplicates()
        
        if not gold_related_events.empty and "date" in gold_related_events.columns:
            gold_related_events = gold_related_events.sort_values("date")
        
        return gold_related_events
    
    except Exception as e:
        return pd.DataFrame(columns=["date", "event", "country", "importance", "actual", "forecast", "previous"])

def is_safe_to_trade_gold(gold_calendar, hours_threshold=2):
    if gold_calendar.empty:
        return True, "No upcoming gold-related events"
    
    now = datetime.now()
    
    for _, event in gold_calendar.iterrows():
        if "date" not in event or pd.isna(event["date"]):
            continue
            
        try:
            event_time = datetime.strptime(event["date"], "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            try:
                event_time = datetime.strptime(event["date"], "%Y-%m-%d")
            except (ValueError, TypeError):
                continue
        
        hours_until_event = (event_time - now).total_seconds() / 3600
        
        if hours_until_event < hours_threshold and hours_until_event > 0:
            importance = str(event.get("importance", "")).upper()
            if importance == "HIGH":
                return False, f"High importance event \t{event.get('event', 'Unknown')}\t coming in {hours_until_event:.1f} hours"
            elif importance == "MEDIUM" and hours_until_event < 1:
                return False, f"Medium importance event \t{event.get('event', 'Unknown')}\t coming in {hours_until_event:.1f} hours"
    
    return True, "No imminent gold-related events"

def adjust_trade_size_based_on_events(original_size, gold_calendar, rf_signal):
    if gold_calendar.empty:
        return original_size, "No upcoming events to consider"
    
    now = datetime.now()
    adjustment_factor = 1.0
    reason = ""
    
    for _, event in gold_calendar.iterrows():
        if "date" not in event or pd.isna(event["date"]):
            continue
            
        try:
            event_time = datetime.strptime(event["date"], "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            try:
                event_time = datetime.strptime(event["date"], "%Y-%m-%d")
            except (ValueError, TypeError):
                continue
        
        hours_until_event = (event_time - now).total_seconds() / 3600
        
        if hours_until_event < 24 and hours_until_event > 0:
            importance = str(event.get("importance", "")).upper()
            if importance == "HIGH":
                adjustment_factor *= 0.5
                reason += f"Reduced by 50% due to upcoming high importance event \t{event.get('event', 'Unknown')}\t. "
            elif importance == "MEDIUM":
                adjustment_factor *= 0.7
                reason += f"Reduced by 30% due to upcoming medium importance event \t{event.get('event', 'Unknown')}\t. "
        
        event_text = str(event.get("event", "")).lower()
        if rf_signal == 1 and any(kw in event_text for kw in ["growth", "positive", "bullish", "increase"]):
            adjustment_factor *= 1.2
            reason += f"Increased by 20% due to potentially positive event \t{event.get('event', 'Unknown')}\t. "
        elif rf_signal == -1 and any(kw in event_text for kw in ["contraction", "negative", "bearish", "decrease"]):
            adjustment_factor *= 1.2
            reason += f"Increased by 20% due to potentially negative event \t{event.get('event', 'Unknown')}\t. "

    new_size = original_size * adjustment_factor
    return new_size, reason.strip()

# --- Trading Logic Class ---
class AdvancedTradingLogic:
    def __init__(self, symbol, api_source, stop_loss_pct=0.01, take_profit_pct=0.03, adx_threshold=25, eod_api_key=None):
        self.symbol = symbol
        self.api_source = api_source
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.adx_threshold = adx_threshold
        self.eod_api_key = eod_api_key
        self.data_5m = pd.DataFrame()
        self.data_4h = pd.DataFrame()
        self.data_1d = pd.DataFrame()
        self.cached_gold_calendar = pd.DataFrame() # Cache for EOD calendar
        self.last_calendar_fetch_time = 0 # Timestamp of last EOD fetch
        self.calendar_cache_duration_seconds = 24 * 60 * 60 # Cache for 24 hours

    def fetch_data(self):
        # Placeholder for actual data fetching logic (ccxt or yfinance)
        # This should populate self.data_5m, self.data_4h, self.data_1d
        print(f"Fetching data for {self.symbol} from {self.api_source}...")
        # Example: Fetching dummy data
        self.data_5m = self._generate_dummy_data(100)
        self.data_4h = self._generate_dummy_data(100, interval="4h")
        self.data_1d = self._generate_dummy_data(100, interval="1d")
        return True

    def _generate_dummy_data(self, count, interval="5m"):
        # Generates dummy OHLCV data for testing
        freq_map = {"5m": "5min", "4h": "4h", "1d": "D"} # Corrected freq mapping
        pandas_freq = freq_map.get(interval, interval)
        
        end_dt = datetime.now()
        try:
            # Attempt to generate dates ensuring they don't go too far back
            # Calculate a reasonable start date based on count and frequency
            if pandas_freq == "5min":
                start_dt = end_dt - timedelta(minutes=5 * count)
            elif pandas_freq == "4h":
                start_dt = end_dt - timedelta(hours=4 * count)
            elif pandas_freq == "D":
                start_dt = end_dt - timedelta(days=count)
            else:
                # Default fallback if frequency is unknown
                start_dt = end_dt - timedelta(days=count) 
            
            # Ensure start date is not excessively old to prevent OutOfBounds
            min_allowed_date = pd.Timestamp("1970-01-01") 
            if start_dt < min_allowed_date:
                start_dt = min_allowed_date + timedelta(seconds=1) # Adjust slightly forward
                print(f"Warning: Calculated start date for dummy data ({interval}) was before {min_allowed_date}. Adjusting start date.")

            dates = pd.date_range(start=start_dt, end=end_dt, freq=pandas_freq)
            # If generated dates are more than needed, take the most recent ones
            if len(dates) > count:
                dates = dates[-count:]
            # If fewer dates generated (e.g., due to start/end constraints), use what we have
            elif len(dates) < count:
                 print(f"Warning: Could only generate {len(dates)} dummy data points for {interval} instead of {count}.")

        except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
            print(f"Error: OutOfBoundsDatetime encountered for {interval} with count {count}. Cannot generate dummy data.")
            return pd.DataFrame()
        except ValueError as ve:
             print(f"Error generating date range for {interval}: {ve}. Cannot generate dummy data.")
             return pd.DataFrame()

        if dates.empty:
             print(f"Error: Failed to generate valid date range for dummy data ({interval}).")
             return pd.DataFrame()

        data = pd.DataFrame(index=dates)
        data["Open"] = np.random.rand(len(dates)) * 100 + 50000
        data["High"] = data["Open"] + np.random.rand(len(dates)) * 10
        data["Low"] = data["Open"] - np.random.rand(len(dates)) * 10
        data["Close"] = data["Open"] + (np.random.rand(len(dates)) - 0.5) * 15
        data["Volume"] = np.random.rand(len(dates)) * 1000 + 100
        return data

    def _get_cached_gold_calendar(self):
        current_time = time.time()
        if not self.cached_gold_calendar.empty and (current_time - self.last_calendar_fetch_time < self.calendar_cache_duration_seconds):
            print("Using cached EOD economic calendar data.")
            return self.cached_gold_calendar
        else:
            print("Fetching new EOD economic calendar data...")
            try:
                new_calendar = get_gold_economic_calendar(api_key=self.eod_api_key)
                if not new_calendar.empty:
                    self.cached_gold_calendar = new_calendar
                    self.last_calendar_fetch_time = current_time
                    print("EOD calendar cache updated.")
                    return self.cached_gold_calendar
                else:
                    print("Failed to fetch new EOD calendar data or data was empty. Using previous cache if available.")
                    return self.cached_gold_calendar if not self.cached_gold_calendar.empty else pd.DataFrame()
            except Exception as e:
                print(f"Error fetching EOD calendar: {e}. Using previous cache if available.")
                return self.cached_gold_calendar if not self.cached_gold_calendar.empty else pd.DataFrame()

    def process_market_conditions(self):
        if self.data_5m.empty or self.data_4h.empty or self.data_1d.empty:
            print("Insufficient data to process market conditions.")
            return 0, "NO_SIGNAL", "Insufficient data"

        sma_4h = derive_smoothed_average(self.data_4h, period=20)
        sma_1d = derive_smoothed_average(self.data_1d, period=20)
        current_price = self.data_5m["Close"].iloc[-1]
        trend = "NEUTRAL"
        if not sma_4h.empty and not sma_1d.empty:
            if current_price > sma_4h.iloc[-1] and current_price > sma_1d.iloc[-1]:
                trend = "UPTREND"
            elif current_price < sma_4h.iloc[-1] and current_price < sma_1d.iloc[-1]:
                trend = "DOWNTREND"

        adx_5m = calculate_adx_indicator(self.data_5m, period=14)
        is_ranging = False
        if not adx_5m.empty and adx_5m.iloc[-1] < self.adx_threshold:
            is_ranging = True

        if is_ranging:
            return 0, "NO_SIGNAL", f"Ranging market detected (ADX {adx_5m.iloc[-1]:.2f} < {self.adx_threshold})"

        rf_signal = predictive_model_processor(self.data_5m)

        final_signal = 0
        reason = ""

        if "BTC" in self.symbol:
            sentiment = get_bitcoin_sentiment_from_gemini()
            reason = f"RF Signal: {rf_signal}, Trend: {trend}, Sentiment: {sentiment}. "
            if rf_signal == 1 and trend == "UPTREND" and sentiment in ["VERY_POSITIVE", "POSITIVE"]:
                final_signal = 1
                reason += "BUY signal confirmed by trend and positive sentiment."
            elif rf_signal == -1 and trend == "DOWNTREND" and sentiment in ["VERY_NEGATIVE", "NEGATIVE"]:
                final_signal = -1
                reason += "SELL signal confirmed by trend and negative sentiment."
            else:
                reason += "Signal filtered out by trend or sentiment."

        elif "GC=F" in self.symbol or "XAU" in self.symbol:
            gold_calendar = self._get_cached_gold_calendar()
            is_safe, safety_reason = is_safe_to_trade_gold(gold_calendar)
            reason = f"RF Signal: {rf_signal}, Trend: {trend}, Safety: {is_safe} ({safety_reason}). "
            if not is_safe:
                final_signal = 0
                reason += "Trading suspended due to imminent event."
            elif rf_signal == 1 and trend == "UPTREND":
                final_signal = 1
                reason += "BUY signal confirmed by trend."
            elif rf_signal == -1 and trend == "DOWNTREND":
                final_signal = -1
                reason += "SELL signal confirmed by trend."
            else:
                reason += "Signal filtered out by trend."

        else:
            reason = f"RF Signal: {rf_signal}, Trend: {trend}. "
            if rf_signal == 1 and trend == "UPTREND":
                final_signal = 1
                reason += "BUY signal confirmed by trend."
            elif rf_signal == -1 and trend == "DOWNTREND":
                final_signal = -1
                reason += "SELL signal confirmed by trend."
            else:
                reason += "Signal filtered out by trend."

        action = "NO_SIGNAL"
        if final_signal == 1:
            action = "BUY"
        elif final_signal == -1:
            action = "SELL"

        return final_signal, action, reason

    def calculate_position_size(self, account_balance, risk_per_trade=0.01):
        risk_amount = account_balance * risk_per_trade
        if self.data_5m.empty or len(self.data_5m) < 1:
             print("Error: Cannot calculate position size due to insufficient 5m data.")
             return 0, 0
        entry_price = self.data_5m["Close"].iloc[-1]
        stop_loss_price_buy = entry_price * (1 - self.stop_loss_pct)
        stop_loss_price_sell = entry_price * (1 + self.stop_loss_pct)
        
        risk_per_unit_buy = entry_price - stop_loss_price_buy
        risk_per_unit_sell = stop_loss_price_sell - entry_price
        
        position_size_buy = risk_amount / risk_per_unit_buy if risk_per_unit_buy > 0 else 0
        position_size_sell = risk_amount / risk_per_unit_sell if risk_per_unit_sell > 0 else 0
        
        if "GC=F" in self.symbol or "XAU" in self.symbol:
            gold_calendar = self._get_cached_gold_calendar()
            rf_signal_placeholder = 0 # Placeholder - ideally get from process_market_conditions
            position_size_buy, reason_buy = adjust_trade_size_based_on_events(position_size_buy, gold_calendar, 1)
            position_size_sell, reason_sell = adjust_trade_size_based_on_events(position_size_sell, gold_calendar, -1)
            print(f"Gold Size Adjustment (Buy Check): {reason_buy}")
            print(f"Gold Size Adjustment (Sell Check): {reason_sell}")

        return position_size_buy, position_size_sell

    def execute_trade(self, action, position_size):
        if self.data_5m.empty or len(self.data_5m) < 1:
             print("Error: Cannot execute trade due to insufficient 5m data.")
             return False
        entry_price = self.data_5m["Close"].iloc[-1]
        if action == "BUY":
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
            print(f"Executing BUY {position_size:.4f} {self.symbol} @ {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
        elif action == "SELL":
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)
            print(f"Executing SELL {position_size:.4f} {self.symbol} @ {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
        return True

    def send_notification(self, message):
        # Updated Telegram notification logic
        if TELEGRAM_BOT_TOKEN != "YOUR_TELEGRAM_BOT_TOKEN_HERE" and TELEGRAM_CHAT_ID != "YOUR_TELEGRAM_CHAT_ID_HERE":
            try:
                bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
                bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
                print(f"Telegram notification sent: {message}")
            except Exception as e:
                print(f"Failed to send Telegram notification: {e}")
                print(f"NOTIFICATION (Fallback): {message}") # Fallback to console
        else:
            print(f"NOTIFICATION (Telegram not configured): {message}")

# --- Execution Functions ---
def execute_primary_operations(symbol, api_source, eod_api_key):
    logic = AdvancedTradingLogic(symbol, api_source, eod_api_key=eod_api_key)
    if logic.fetch_data():
        signal, action, reason = logic.process_market_conditions()
        print(f"[{datetime.now()}] Symbol: {symbol}, Signal: {signal}, Action: {action}, Reason: {reason}")
        if action != "NO_SIGNAL":
            account_balance = 10000
            size_buy, size_sell = logic.calculate_position_size(account_balance)
            position_size = size_buy if action == "BUY" else size_sell
            if position_size > 0:
                logic.execute_trade(action, position_size)
                # Use the updated send_notification method
                logic.send_notification(f"Trade Signal for {symbol}: {action} {position_size:.4f} units. Reason: {reason}")
            else:
                print("Calculated position size is zero, no trade executed.")
    else:
        print(f"Failed to fetch data for {symbol}.")

def execute_multi_asset_operations():
    assets = [
        {"symbol": "BTC/USDT", "api_source": "ccxt"}, 
        {"symbol": "GC=F", "api_source": "yfinance"} # Gold Futures
    ]
    
    logics = {asset["symbol"]: AdvancedTradingLogic(asset["symbol"], asset["api_source"], eod_api_key=EODHD_API_KEY) for asset in assets}
    
    while True:
        print(f"\n--- Checking Assets @ {datetime.now()} ---")
        for asset in assets:
            symbol = asset["symbol"]
            logic = logics[symbol]
            
            if logic.fetch_data():
                signal, action, reason = logic.process_market_conditions()
                print(f"[{datetime.now()}] Symbol: {symbol}, Signal: {signal}, Action: {action}, Reason: {reason}")
                if action != "NO_SIGNAL":
                    account_balance = 10000 # Dummy balance
                    size_buy, size_sell = logic.calculate_position_size(account_balance)
                    position_size = size_buy if action == "BUY" else size_sell
                    if position_size > 0:
                        logic.execute_trade(action, position_size)
                        # Use the updated send_notification method
                        logic.send_notification(f"Trade Signal for {symbol}: {action} {position_size:.4f} units. Reason: {reason}")
                    else:
                        print(f"Calculated position size is zero for {symbol}, no trade executed.")
            else:
                print(f"Failed to fetch data for {symbol}.")
        
        print("--- Waiting for next cycle --- ")
        time.sleep(300) # Check every 5 minutes

# --- Backtesting Functions (Simplified) ---
def run_simple_backtest(symbol, api_source, eod_api_key, start_date="2023-01-01", end_date="2023-12-31"):
    print(f"\n--- Running Simple Backtest for {symbol} ---")
    logic = AdvancedTradingLogic(symbol, api_source, eod_api_key=eod_api_key)
    
    print("Fetching historical data for backtest...")
    # Use the corrected dummy data generation
    logic.data_5m = logic._generate_dummy_data(2000, interval="5m") 
    logic.data_4h = logic._generate_dummy_data(500, interval="4h")
    logic.data_1d = logic._generate_dummy_data(250, interval="1d")
    print("Historical data fetched (dummy).")

    if logic.data_5m.empty or logic.data_4h.empty or logic.data_1d.empty:
        print("Failed to get sufficient historical data for backtest.")
        return

    trades = []
    balance = 10000
    position = 0 
    entry_price = 0
    position_size = 0 

    print("Simulating trades...")
    
    # --- FIX: Robust start index calculation ---
    min_required_lookback = 100 # Minimum data points needed for indicators/ML
    
    # Find the earliest timestamp present in 4h and 1d data
    min_start_time_4h = logic.data_4h.index[0] if not logic.data_4h.empty else pd.Timestamp.min
    min_start_time_1d = logic.data_1d.index[0] if not logic.data_1d.empty else pd.Timestamp.min
    required_start_time = max(min_start_time_4h, min_start_time_1d)

    # Find the first index in 5m data that is >= required_start_time
    start_index_loc = logic.data_5m.index.searchsorted(required_start_time)
    
    # Ensure start_index allows for minimum lookback and is within bounds
    start_index = max(min_required_lookback, start_index_loc)
    start_index = min(start_index, len(logic.data_5m) - 1)
    # --- END FIX ---
    
    if start_index >= len(logic.data_5m):
        print("Not enough data points to start backtest simulation after aligning timeframes and lookback.")
        return

    for i in range(start_index, len(logic.data_5m)):
        current_data_5m = logic.data_5m.iloc[:i+1]
        current_time_5m = logic.data_5m.index[i]
        
        # --- FIX: Use .loc for slicing based on time to avoid index issues ---
        current_data_4h = logic.data_4h.loc[logic.data_4h.index <= current_time_5m]
        current_data_1d = logic.data_1d.loc[logic.data_1d.index <= current_time_5m]
        # --- END FIX ---
        
        # Check if sliced data is empty before proceeding
        if current_data_4h.empty or current_data_1d.empty:
            # print(f"Skipping step {i} due to empty 4h/1d data after slicing at {current_time_5m}")
            continue
            
        temp_logic = AdvancedTradingLogic(symbol, api_source, eod_api_key=eod_api_key)
        temp_logic.data_5m = current_data_5m
        temp_logic.data_4h = current_data_4h
        temp_logic.data_1d = current_data_1d
        temp_logic.cached_gold_calendar = logic.cached_gold_calendar # Pass cache
        temp_logic.last_calendar_fetch_time = logic.last_calendar_fetch_time # Pass cache time

        signal, action, reason = temp_logic.process_market_conditions()
        
        # Update main logic cache if it was updated in temp_logic
        logic.cached_gold_calendar = temp_logic.cached_gold_calendar
        logic.last_calendar_fetch_time = temp_logic.last_calendar_fetch_time

        current_price = current_data_5m["Close"].iloc[-1]

        # Simplified trade logic for backtest
        if action == "BUY" and position <= 0:
            if position == -1: # Close existing short position
                profit = (entry_price - current_price) * abs(position_size)
                balance += profit
                trades.append({"time": logic.data_5m.index[i], "type": "Close Short", "price": current_price, "profit": profit, "balance": balance})
            position = 1
            entry_price = current_price
            size_buy, _ = temp_logic.calculate_position_size(balance)
            position_size = size_buy if size_buy > 0 else 0
            if position_size > 0:
                trades.append({"time": logic.data_5m.index[i], "type": "Buy", "price": entry_price, "size": position_size, "balance": balance})
            else:
                position = 0 # Reset position if size is zero
        
        elif action == "SELL" and position >= 0:
            if position == 1: # Close existing long position
                profit = (current_price - entry_price) * abs(position_size)
                balance += profit
                trades.append({"time": logic.data_5m.index[i], "type": "Close Long", "price": current_price, "profit": profit, "balance": balance})
            position = -1
            entry_price = current_price
            _, size_sell = temp_logic.calculate_position_size(balance)
            position_size = size_sell if size_sell > 0 else 0
            if position_size > 0:
                trades.append({"time": logic.data_5m.index[i], "type": "Sell", "price": entry_price, "size": position_size, "balance": balance})
            else:
                position = 0 # Reset position if size is zero

    print("Backtest simulation complete.")
    if trades:
        trades_df = pd.DataFrame(trades)
        print("Backtest Results:")
        print(trades_df.tail())
        print(f"Final Balance: {balance:.2f}")
    else:
        print("No trades executed during backtest.")

def run_multi_asset_backtest(start_date="2023-01-01", end_date="2023-12-31"):
     assets = [
        {"symbol": "BTC/USDT", "api_source": "ccxt"}, 
        {"symbol": "GC=F", "api_source": "yfinance"} 
    ]
     for asset in assets:
         run_simple_backtest(asset["symbol"], asset["api_source"], EODHD_API_KEY, start_date, end_date)


@app.route("/")
def home():
    return "Trading bot is running"

def run_web():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

def run_bot():
    run_multi_asset_backtest()


if __name__ == "__main__":
    print("Starting Trading Bot with Dummy Web Server...")
    Thread(target=run_web).start()
    run_bot()
    # Choose one execution mode:
    # 1. Multi-Asset Live Trading:
    # execute_multi_asset_operations()
    
    # 2. Single Asset Live Trading (Example: Bitcoin):
    # execute_primary_operations(symbol="BTC/USDT", api_source="ccxt", eod_api_key=EODHD_API_KEY)
    
    # 3. Single Asset Live Trading (Example: Gold):
    # execute_primary_operations(symbol="GC=F", api_source="yfinance", eod_api_key=EODHD_API_KEY)

    # 4. Multi-Asset Backtesting:
    
    
    # 5. Single Asset Backtesting (Example: Bitcoin):
    # run_simple_backtest(symbol="BTC/USDT", api_source="ccxt", eod_api_key=EODHD_API_KEY)
    
    # 6. Single Asset Backtesting (Example: Gold):
    # run_simple_backtest(symbol="GC=F", api_source="yfinance", eod_api_key=EODHD_API_KEY)

    print("Trading Bot finished or running in background.")

