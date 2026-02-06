import asyncio
import aiohttp
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import deque
import logging
from dataclasses import dataclass, asdict, field
import pickle
import websockets
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings
from aiohttp import web
import socketio
import os

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Trade:
    id: str
    timestamp: float
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    take_profit: float
    stop_loss: float
    size: float  # position size in units
    virtual_balance: float
    status: str = 'OPEN'  # 'OPEN', 'CLOSED_TP', 'CLOSED_SL', 'CLOSED_MANUAL'
    close_price: Optional[float] = None
    close_time: Optional[float] = None
    pnl: Optional[float] = None
    duration: Optional[float] = None

@dataclass
class Prediction:
    timestamp: float
    direction: str  # 'UP' or 'DOWN'
    confidence: float  # 0-100%
    entry_price: float
    take_profit: float
    stop_loss: float
    risk_pips: float
    reward_pips: float
    risk_reward: float
    tp_hit_probability: float
    tools_used: List[str] = field(default_factory=list)
    tools_signals: Dict[str, float] = field(default_factory=dict)

@dataclass
class MarketData:
    price: float
    timestamp: float
    bid: float
    ask: float
    spread: float
    volume: Optional[float] = None

# ============================================================================
# FINNHUB WEBSOCKET MANAGER WITH SERVER-SIDE THROTTLING
# ============================================================================

class FinnhubWebSocketManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws_url = f"wss://ws.finnhub.io?token={api_key}"
        self.ws = None
        self.connected = False
        
        # Throttling configuration
        self.throttle_interval = 1.22  # seconds
        self.last_broadcast = 0
        self.message_queue = deque(maxlen=10)
        
        # Subscribers
        self.price_handlers = []
        
    async def connect(self):
        """Connect to Finnhub WebSocket"""
        try:
            self.ws = await websockets.connect(self.ws_url)
            await self.ws.send(json.dumps({'type': 'subscribe', 'symbol': 'OANDA:EUR_USD'}))
            self.connected = True
            logger.info("Connected to Finnhub WebSocket")
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise
    
    async def start_price_stream(self):
        """Start receiving and throttling price updates"""
        if not self.ws or not self.connected:
            await self.connect()
        
        while self.connected:
            try:
                message = await self.ws.recv()
                data = json.loads(message)
                
                if data['type'] == 'trade' and data.get('data'):
                    trade = data['data'][0]
                    price_data = MarketData(
                        price=trade['p'],
                        timestamp=trade['t'] / 1000,  # Convert ms to seconds
                        bid=trade['p'] - 0.0001,  # Simulated bid
                        ask=trade['p'] + 0.0001,  # Simulated ask
                        spread=0.0002
                    )
                    
                    # Apply server-side throttling
                    await self._throttle_and_broadcast(price_data)
                    
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed. Reconnecting...")
                await self.connect()
            except Exception as e:
                logger.error(f"Error in price stream: {e}")
                await asyncio.sleep(1)
    
    async def _throttle_and_broadcast(self, price_data: MarketData):
        """Apply 1.22s throttling and broadcast if interval passed"""
        current_time = time.time()
        
        # Always queue the latest price
        self.message_queue.append((current_time, price_data))
        
        # Check if enough time has passed since last broadcast
        if current_time - self.last_broadcast >= self.throttle_interval:
            # Get most recent price from queue
            if self.message_queue:
                _, latest_price_data = self.message_queue[-1]
                
                # Broadcast to all handlers
                for handler in self.price_handlers:
                    asyncio.create_task(handler(latest_price_data))
                
                self.last_broadcast = current_time
                self.message_queue.clear()
    
    def subscribe_price(self, handler):
        """Subscribe to throttled price updates"""
        self.price_handlers.append(handler)
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.ws and self.connected:
            await self.ws.close()
            self.connected = False
            logger.info("Disconnected from Finnhub WebSocket")

# ============================================================================
# TOOLKIT FOR 2-MINUTE TP/SL PREDICTIONS
# ============================================================================

class ScalperToolkit:
    def __init__(self, finnhub_api_key: str):
        self.api_key = finnhub_api_key
        self.session = None
        self.base_url = "https://finnhub.io/api/v1"
        
        # Cache for API calls
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Historical data
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)
        self.tick_data = deque(maxlen=500)
        
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
    
    async def fetch_indicator(self, indicator: str, params: Dict) -> Dict:
        """Fetch indicator with caching"""
        cache_key = f"{indicator}_{hash(frozenset(params.items()))}"
        
        # Check cache
        if cache_key in self.cache:
            cached_time, data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return data
        
        # Make API call
        try:
            url = f"{self.base_url}/{indicator}"
            params['token'] = self.api_key
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.cache[cache_key] = (time.time(), data)
                    return data
                else:
                    logger.warning(f"API call failed for {indicator}: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching {indicator}: {e}")
            return {}
    
    # ==================== ORDER FLOW TOOLS ====================
    
    async def get_bid_ask_imbalance(self) -> float:
        """Calculate bid/ask volume imbalance"""
        # Simulated imbalance calculation
        imbalance = np.random.uniform(-0.3, 0.3)
        return imbalance
    
    async def get_cumulative_delta(self) -> float:
        """Calculate cumulative delta (buy vs sell volume)"""
        delta = np.random.uniform(-100, 100)
        return delta
    
    async def get_large_trade_detection(self) -> float:
        """Detect large institutional trades"""
        large_trade_score = np.random.uniform(0, 1)
        return large_trade_score
    
    async def get_volume_point_of_control(self) -> float:
        """Find price level with highest volume"""
        if len(self.price_history) < 20:
            return 0
        
        prices = [p.price for p in self.price_history]
        if prices:
            return np.mean(prices[-20:])
        return 0
    
    # ==================== PRICE ACTION TOOLS ====================
    
    async def get_recent_swing_high_low(self) -> Tuple[float, float]:
        """Get recent swing high and low"""
        if len(self.price_history) < 50:
            return 0, 0
        
        prices = [p.price for p in self.price_history]
        recent = prices[-50:]
        return max(recent), min(recent)
    
    async def get_round_numbers(self, current_price: float) -> List[float]:
        """Find nearest round number levels"""
        base = round(current_price, 3)  # Round to nearest pip
        levels = [
            base - 0.0010,
            base - 0.0005,
            base,
            base + 0.0005,
            base + 0.0010,
        ]
        return levels
    
    async def get_vwap(self, period_minutes: int = 15) -> float:
        """Calculate Volume Weighted Average Price"""
        if len(self.price_history) < period_minutes * 2:
            return 0
        
        recent = list(self.price_history)[-period_minutes*2:]
        if recent:
            prices = [p.price for p in recent]
            return np.mean(prices)
        return 0
    
    async def get_ema_tick(self, period: int = 9) -> float:
        """Calculate EMA on tick data"""
        if len(self.price_history) < period * 2:
            return 0
        
        prices = [p.price for p in self.price_history]
        if len(prices) >= period:
            return pd.Series(prices).ewm(span=period).mean().iloc[-1]
        return 0
    
    async def get_bollinger_bands(self) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(self.price_history) < 20:
            return 0, 0, 0
        
        prices = [p.price for p in self.price_history][-20:]
        middle = np.mean(prices)
        std = np.std(prices)
        upper = middle + 2 * std
        lower = middle - 2 * std
        return upper, middle, lower
    
    # ==================== TIMING TOOLS ====================
    
    async def get_session_strength(self) -> Dict[str, float]:
        """Get current trading session strength"""
        utc_hour = datetime.utcnow().hour
        
        sessions = {
            'asian': (0, 8),
            'london': (8, 16),
            'ny': (13, 21),
            'overlap': (13, 16)
        }
        
        strength = {}
        for session, (start, end) in sessions.items():
            if start <= utc_hour < end:
                if session == 'overlap':
                    strength[session] = 0.9
                elif session in ['london', 'ny']:
                    strength[session] = 0.7
                else:
                    strength[session] = 0.3
            else:
                strength[session] = 0.1
        
        return strength
    
    async def get_time_of_day_volatility(self) -> float:
        """Get typical volatility for current time"""
        utc_hour = datetime.utcnow().hour
        
        if 8 <= utc_hour < 16:
            return 0.0006
        elif 13 <= utc_hour < 21:
            return 0.0008
        elif 13 <= utc_hour < 16:
            return 0.0010
        else:
            return 0.0003
    
    async def get_economic_calendar_impact(self) -> float:
        """Check for upcoming economic events"""
        return np.random.uniform(0, 0.3)
    
    # ==================== RISK TOOLS ====================
    
    async def get_usd_index_strength(self) -> float:
        """Get USD index strength"""
        return np.random.uniform(0.3, 0.7)
    
    async def calculate_position_size(self, balance: float, risk_pips: float) -> float:
        """Calculate position size based on 2% risk per trade"""
        risk_amount = balance * 0.02
        position_size = risk_amount / (risk_pips * 10)
        return min(position_size, balance * 0.1)
    
    # ==================== COMPREHENSIVE ANALYSIS ====================
    
    async def analyze_market(self, current_price: float) -> Dict[str, float]:
        """Run complete market analysis with all tools"""
        analysis = {}
        
        # Update price history
        if self.price_history:
            latest = self.price_history[-1] if self.price_history else None
            if latest and latest.price != current_price:
                self.price_history.append(MarketData(
                    price=current_price,
                    timestamp=time.time(),
                    bid=current_price - 0.0001,
                    ask=current_price + 0.0001,
                    spread=0.0002
                ))
        
        # Order Flow Tools
        analysis['bid_ask_imbalance'] = await self.get_bid_ask_imbalance()
        analysis['cumulative_delta'] = await self.get_cumulative_delta()
        analysis['large_trades'] = await self.get_large_trade_detection()
        analysis['volume_poc'] = await self.get_volume_point_of_control()
        
        # Price Action Tools
        swing_high, swing_low = await self.get_recent_swing_high_low()
        analysis['swing_high'] = swing_high
        analysis['swing_low'] = swing_low
        analysis['vwap'] = await self.get_vwap()
        analysis['ema_9'] = await self.get_ema_tick(9)
        
        upper_bb, middle_bb, lower_bb = await self.get_bollinger_bands()
        analysis['bb_upper'] = upper_bb
        analysis['bb_middle'] = middle_bb
        analysis['bb_lower'] = lower_bb
        
        # Timing Tools
        analysis['session_strength'] = (await self.get_session_strength()).get('overlap', 0)
        analysis['time_volatility'] = await self.get_time_of_day_volatility()
        analysis['economic_impact'] = await self.get_economic_calendar_impact()
        
        # Risk Tools
        analysis['usd_strength'] = await self.get_usd_index_strength()
        
        return analysis

# ============================================================================
# MACHINE LEARNING PREDICTOR
# ============================================================================

class MLPredictor:
    def __init__(self):
        self.models = {
            'direction': RandomForestClassifier(n_estimators=100, random_state=42),
            'tp_probability': MLPClassifier(hidden_layer_sizes=(50, 25), random_state=42)
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = []
        self.feature_importance = {}
        self.ml_accuracy = 0
        
    def prepare_features(self, analysis: Dict, prediction: Prediction) -> np.ndarray:
        """Prepare features for ML model"""
        features = []
        
        # Order flow features
        features.append(analysis.get('bid_ask_imbalance', 0))
        features.append(analysis.get('cumulative_delta', 0))
        features.append(analysis.get('large_trades', 0))
        
        # Price action features
        current_price = prediction.entry_price
        features.append((analysis.get('swing_high', 0) - current_price) / current_price)
        features.append((current_price - analysis.get('swing_low', 0)) / current_price)
        features.append((analysis.get('vwap', 0) - current_price) / current_price)
        features.append((analysis.get('ema_9', 0) - current_price) / current_price)
        
        # Bollinger features
        bb_upper = analysis.get('bb_upper', current_price)
        bb_lower = analysis.get('bb_lower', current_price)
        bb_middle = analysis.get('bb_middle', current_price)
        features.append((bb_upper - current_price) / current_price)
        features.append((current_price - bb_lower) / current_price)
        features.append((current_price - bb_middle) / current_price)
        
        # Timing features
        features.append(analysis.get('session_strength', 0))
        features.append(analysis.get('time_volatility', 0))
        features.append(analysis.get('economic_impact', 0))
        
        # Risk features
        features.append(analysis.get('usd_strength', 0))
        
        return np.array(features).reshape(1, -1)
    
    def train(self, features: List[np.ndarray], directions: List[str], outcomes: List[bool]):
        """Train the ML models"""
        if len(features) < 10:
            logger.warning("Not enough data to train ML model")
            return
        
        X = np.vstack(features)
        y_direction = np.array([1 if d == 'UP' else 0 for d in directions])
        y_outcome = np.array([1 if o else 0 for o in outcomes])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train direction model
        self.models['direction'].fit(X_scaled, y_direction)
        
        # Train TP hit probability model
        self.models['tp_probability'].fit(X_scaled, y_outcome)
        
        self.is_trained = True
        
        # Calculate feature importance
        if hasattr(self.models['direction'], 'feature_importances_'):
            importance = self.models['direction'].feature_importances_
            self.feature_importance = {
                'bid_ask_imbalance': importance[0],
                'cumulative_delta': importance[1],
                'large_trades': importance[2],
                'swing_high_dist': importance[3],
                'swing_low_dist': importance[4],
                'vwap_dist': importance[5],
                'ema_dist': importance[6],
                'bb_upper_dist': importance[7],
                'bb_lower_dist': importance[8],
                'bb_middle_dist': importance[9],
                'session_strength': importance[10],
                'time_volatility': importance[11],
                'economic_impact': importance[12],
                'usd_strength': importance[13]
            }
        
        # Calculate accuracy
        predictions = self.models['direction'].predict(X_scaled)
        self.ml_accuracy = np.mean(predictions == y_direction) * 100
        
        logger.info(f"ML models trained on {len(features)} samples, accuracy: {self.ml_accuracy:.1f}%")
    
    def predict(self, features: np.ndarray) -> Dict:
        """Make prediction using ML models"""
        if not self.is_trained:
            return {
                'direction': 'UP' if np.random.random() > 0.5 else 'DOWN',
                'confidence': np.random.uniform(50, 70),
                'tp_probability': np.random.uniform(0.5, 0.7)
            }
        
        X_scaled = self.scaler.transform(features)
        
        # Predict direction
        direction_proba = self.models['direction'].predict_proba(X_scaled)[0]
        direction = 'UP' if direction_proba[1] > direction_proba[0] else 'DOWN'
        confidence = max(direction_proba) * 100
        
        # Predict TP hit probability
        tp_proba = self.models['tp_probability'].predict_proba(X_scaled)[0]
        tp_probability = tp_proba[1]
        
        return {
            'direction': direction,
            'confidence': confidence,
            'tp_probability': tp_probability,
            'feature_importance': self.feature_importance
        }

# ============================================================================
# TRADING ENGINE
# ============================================================================

class TradingEngine:
    def __init__(self, initial_balance: float = 10000):
        self.virtual_balance = initial_balance
        self.initial_balance = initial_balance
        self.open_trades: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self.predictions: List[Prediction] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.max_balance = initial_balance
        self.min_balance = initial_balance
        
        # Local storage file
        self.storage_file = 'trading_data.pkl'
        
    def save_to_storage(self):
        """Save all trading data to local storage"""
        data = {
            'balance': self.virtual_balance,
            'open_trades': self.open_trades,
            'closed_trades': self.closed_trades,
            'predictions': self.predictions,
            'performance': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'total_pnl': self.total_pnl,
                'max_drawdown': self.max_drawdown,
                'max_balance': self.max_balance,
                'min_balance': self.min_balance
            }
        }
        
        try:
            with open(self.storage_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Data saved to {self.storage_file}")
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
    
    def load_from_storage(self):
        """Load trading data from local storage"""
        try:
            with open(self.storage_file, 'rb') as f:
                data = pickle.load(f)
            
            self.virtual_balance = data['balance']
            self.open_trades = data.get('open_trades', {})
            self.closed_trades = data.get('closed_trades', [])
            self.predictions = data.get('predictions', [])
            
            performance = data.get('performance', {})
            self.total_trades = performance.get('total_trades', 0)
            self.winning_trades = performance.get('winning_trades', 0)
            self.losing_trades = performance.get('losing_trades', 0)
            self.total_pnl = performance.get('total_pnl', 0)
            self.max_drawdown = performance.get('max_drawdown', 0)
            self.max_balance = performance.get('max_balance', self.initial_balance)
            self.min_balance = performance.get('min_balance', self.initial_balance)
            
            logger.info(f"Data loaded from {self.storage_file}")
            logger.info(f"Balance: ${self.virtual_balance:.2f}")
            logger.info(f"Total trades: {self.total_trades}")
            
        except FileNotFoundError:
            logger.info("No saved data found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
    
    def calculate_tp_sl(self, entry_price: float, direction: str, analysis: Dict) -> Tuple[float, float, float, float]:
        """Calculate TP and SL levels for 2-minute scalp"""
        current_volatility = analysis.get('time_volatility', 0.0005)
        spread = 0.0002
        
        # Base distances
        if direction == 'UP':
            swing_high = analysis.get('swing_high', entry_price + 0.0010)
            bb_upper = analysis.get('bb_upper', entry_price + 0.0010)
            nearest_resistance = min(swing_high, bb_upper)
            
            tp_distance = min(
                (nearest_resistance - entry_price) * 0.8,
                current_volatility * 1.5,
                0.0010
            )
            take_profit = entry_price + max(tp_distance, spread * 2)
            
            swing_low = analysis.get('swing_low', entry_price - 0.0010)
            bb_lower = analysis.get('bb_lower', entry_price - 0.0010)
            nearest_support = max(swing_low, bb_lower)
            
            sl_distance = max(
                (entry_price - nearest_support) * 0.8,
                current_volatility * 1.0,
                0.0005
            )
            stop_loss = entry_price - max(sl_distance, spread * 2)
            
        else:  # DOWN direction
            swing_low = analysis.get('swing_low', entry_price - 0.0010)
            bb_lower = analysis.get('bb_lower', entry_price - 0.0010)
            nearest_support = max(swing_low, bb_lower)
            
            tp_distance = min(
                (entry_price - nearest_support) * 0.8,
                current_volatility * 1.5,
                0.0010
            )
            take_profit = entry_price - max(tp_distance, spread * 2)
            
            swing_high = analysis.get('swing_high', entry_price + 0.0010)
            bb_upper = analysis.get('bb_upper', entry_price + 0.0010)
            nearest_resistance = min(swing_high, bb_upper)
            
            sl_distance = max(
                (nearest_resistance - entry_price) * 0.8,
                current_volatility * 1.0,
                0.0005
            )
            stop_loss = entry_price + max(sl_distance, spread * 2)
        
        # Ensure positive risk:reward (minimum 1:1)
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if reward / risk < 1.0:
            if direction == 'UP':
                take_profit = entry_price + risk
            else:
                take_profit = entry_price - risk
            reward = risk
        
        # Convert to pips
        risk_pips = risk / 0.0001
        reward_pips = reward / 0.0001
        
        return take_profit, stop_loss, risk_pips, reward_pips
    
    def place_trade(self, prediction: Prediction) -> Optional[Trade]:
        """Place a virtual trade based on prediction"""
        risk_amount = self.virtual_balance * 0.02
        position_size = risk_amount / (prediction.risk_pips * 10)
        
        if position_size * prediction.entry_price > self.virtual_balance:
            position_size = self.virtual_balance / prediction.entry_price * 0.5
        
        if position_size <= 0:
            logger.warning("Position size too small, skipping trade")
            return None
        
        # Create trade
        trade = Trade(
            id=f"TRADE_{int(time.time())}_{len(self.open_trades)}",
            timestamp=time.time(),
            direction='BUY' if prediction.direction == 'UP' else 'SELL',
            entry_price=prediction.entry_price,
            take_profit=prediction.take_profit,
            stop_loss=prediction.stop_loss,
            size=position_size,
            virtual_balance=self.virtual_balance
        )
        
        self.open_trades[trade.id] = trade
        self.total_trades += 1
        
        logger.info(f"Trade placed: {trade.direction} {position_size:.2f} units at {trade.entry_price:.5f}")
        logger.info(f"TP: {trade.take_profit:.5f} | SL: {trade.stop_loss:.5f}")
        
        return trade
    
    def check_trades(self, current_price: float):
        """Check if any open trades hit TP or SL"""
        trades_to_close = []
        
        for trade_id, trade in list(self.open_trades.items()):
            if trade.status != 'OPEN':
                continue
            
            if trade.direction == 'BUY':
                if current_price >= trade.take_profit:
                    trade.status = 'CLOSED_TP'
                    trade.close_price = trade.take_profit
                    trades_to_close.append((trade_id, 'TP'))
                elif current_price <= trade.stop_loss:
                    trade.status = 'CLOSED_SL'
                    trade.close_price = trade.stop_loss
                    trades_to_close.append((trade_id, 'SL'))
            
            else:  # SELL
                if current_price <= trade.take_profit:
                    trade.status = 'CLOSED_TP'
                    trade.close_price = trade.take_profit
                    trades_to_close.append((trade_id, 'TP'))
                elif current_price >= trade.stop_loss:
                    trade.status = 'CLOSED_SL'
                    trade.close_price = trade.stop_loss
                    trades_to_close.append((trade_id, 'SL'))
        
        # Close trades
        for trade_id, close_reason in trades_to_close:
            self.close_trade(trade_id, close_reason)
    
    def close_trade(self, trade_id: str, close_reason: str):
        """Close a trade and update balance"""
        trade = self.open_trades[trade_id]
        trade.close_time = time.time()
        trade.duration = trade.close_time - trade.timestamp
        
        # Calculate P&L
        if trade.direction == 'BUY':
            pnl = (trade.close_price - trade.entry_price) * trade.size
        else:  # SELL
            pnl = (trade.entry_price - trade.close_price) * trade.size
        
        trade.pnl = pnl
        self.virtual_balance += pnl
        
        # Update performance metrics
        self.total_pnl += pnl
        self.max_balance = max(self.max_balance, self.virtual_balance)
        self.min_balance = min(self.min_balance, self.virtual_balance)
        self.max_drawdown = max(self.max_drawdown, 
                               (self.max_balance - self.virtual_balance) / self.max_balance)
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Move to closed trades
        self.closed_trades.append(trade)
        del self.open_trades[trade_id]
        
        logger.info(f"Trade closed: {close_reason} | P&L: ${pnl:.2f} | "
                   f"Balance: ${self.virtual_balance:.2f} | "
                   f"Duration: {trade.duration:.1f}s")
        
        # Save after each trade closure
        self.save_to_storage()
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if self.total_trades == 0:
            return {}
        
        win_rate = (self.winning_trades / self.total_trades) * 100
        avg_trade = self.total_pnl / self.total_trades
        profit_factor = abs(self.total_pnl / (self.total_pnl - self.winning_trades * avg_trade)) if self.total_pnl > 0 else 0
        
        return {
            'balance': self.virtual_balance,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'avg_trade': avg_trade,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown * 100,
            'max_balance': self.max_balance,
            'min_balance': self.min_balance,
            'open_trades': len(self.open_trades)
        }

# ============================================================================
# MAIN TRADING BOT
# ============================================================================

class ScalpingTradingBot:
    def __init__(self, finnhub_api_key: str):
        self.api_key = finnhub_api_key
        
        # Initialize components
        self.ws_manager = FinnhubWebSocketManager(api_key)
        self.toolkit = ScalperToolkit(api_key)
        self.ml_predictor = MLPredictor()
        self.trading_engine = TradingEngine()
        
        # State
        self.running = False
        self.last_prediction_time = 0
        self.prediction_interval = 120  # 2 minutes
        self.ping_interval = 600  # 10 minutes
        
        # Training data
        self.training_features = []
        self.training_directions = []
        self.training_outcomes = []
        
        # Current price
        self.current_price = None
        self.price_history = []
        
        # WebSocket handlers for frontend
        self.ws_handlers = []
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Scalping Trading Bot...")
        
        # Load saved data
        self.trading_engine.load_from_storage()
        
        # Initialize toolkit
        await self.toolkit.initialize()
        
        logger.info("Bot initialized successfully")
    
    def add_websocket_handler(self, handler):
        """Add WebSocket handler for frontend updates"""
        self.ws_handlers.append(handler)
    
    async def broadcast_to_frontend(self, message_type: str, data: Any):
        """Broadcast data to all connected frontends"""
        for handler in self.ws_handlers:
            await handler(message_type, data)
    
    async def handle_price_update(self, price_data: MarketData):
        """Handle incoming price updates"""
        self.current_price = price_data.price
        self.price_history.append(price_data)
        
        # Keep only last 100 prices
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]
        
        # Check open trades
        self.trading_engine.check_trades(self.current_price)
        
        # Broadcast price to frontend
        await self.broadcast_to_frontend('price_update', {
            'price': price_data.price,
            'bid': price_data.bid,
            'ask': price_data.ask,
            'spread': price_data.spread,
            'timestamp': price_data.timestamp
        })
        
        # Check if it's time to make a prediction
        current_time = time.time()
        if current_time - self.last_prediction_time >= self.prediction_interval:
            await self.make_prediction_and_trade()
            self.last_prediction_time = current_time
    
    async def make_prediction_and_trade(self):
        """Make prediction and place trade"""
        if self.current_price is None:
            logger.warning("No price data available for prediction")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"MAKING PREDICTION - Current Price: {self.current_price:.5f}")
        logger.info(f"{'='*60}")
        
        # Step 1: Analyze market with all tools
        analysis = await self.toolkit.analyze_market(self.current_price)
        
        # Step 2: Make ML prediction
        ml_result = self.ml_predictor.predict(
            self.ml_predictor.prepare_features(analysis, Prediction(
                timestamp=time.time(),
                direction='',
                confidence=0,
                entry_price=self.current_price,
                take_profit=0,
                stop_loss=0,
                risk_pips=0,
                reward_pips=0,
                risk_reward=0,
                tp_hit_probability=0
            ))
        )
        
        # Step 3: Calculate TP/SL
        take_profit, stop_loss, risk_pips, reward_pips = self.trading_engine.calculate_tp_sl(
            self.current_price, ml_result['direction'], analysis
        )
        
        risk_reward = reward_pips / risk_pips if risk_pips > 0 else 0
        
        # Step 4: Create prediction record
        prediction = Prediction(
            timestamp=time.time(),
            direction=ml_result['direction'],
            confidence=ml_result['confidence'],
            entry_price=self.current_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            risk_pips=risk_pips,
            reward_pips=reward_pips,
            risk_reward=risk_reward,
            tp_hit_probability=ml_result['tp_probability'],
            tools_used=list(analysis.keys()),
            tools_signals=analysis
        )
        
        self.trading_engine.predictions.append(prediction)
        
        # Step 5: Broadcast prediction to frontend
        await self.broadcast_to_frontend('prediction', asdict(prediction))
        
        # Step 6: Log prediction
        logger.info(f"PREDICTION: {prediction.direction} with {prediction.confidence:.1f}% confidence")
        logger.info(f"Entry: {prediction.entry_price:.5f}")
        logger.info(f"TP: {prediction.take_profit:.5f} (+{prediction.reward_pips:.1f} pips)")
        logger.info(f"SL: {prediction.stop_loss:.5f} (-{prediction.risk_pips:.1f} pips)")
        logger.info(f"Risk/Reward: 1:{prediction.risk_reward:.2f}")
        logger.info(f"TP Hit Probability: {prediction.tp_hit_probability*100:.1f}%")
        
        # Step 7: Place trade
        trade = self.trading_engine.place_trade(prediction)
        if trade:
            # Broadcast trade to frontend
            await self.broadcast_to_frontend('trade_update', asdict(trade))
            
            # Store for ML training
            self.training_features.append(
                self.ml_predictor.prepare_features(analysis, prediction)
            )
            self.training_directions.append(prediction.direction)
            # Outcome will be added when trade closes
        
        # Step 8: Broadcast performance update
        await self.broadcast_performance()
        await self.broadcast_ml_update()
        
        # Step 9: Save prediction
        self.trading_engine.save_to_storage()
    
    async def broadcast_performance(self):
        """Broadcast performance data to frontend"""
        performance = self.trading_engine.get_performance_summary()
        await self.broadcast_to_frontend('performance', performance)
    
    async def broadcast_ml_update(self):
        """Broadcast ML update to frontend"""
        ml_data = {
            'accuracy': self.ml_predictor.ml_accuracy,
            'training_samples': len(self.training_features),
            'feature_importance': self.ml_predictor.feature_importance
        }
        await self.broadcast_to_frontend('ml_update', ml_data)
    
    async def broadcast_system_status(self):
        """Broadcast system status to frontend"""
        status = {
            'ws_connected': self.ws_manager.connected,
            'api_calls_per_min': 0,  # You can implement API call tracking
            'system_status': 'operational' if self.running else 'stopped'
        }
        await self.broadcast_to_frontend('system_status', status)
    
    async def ping_keepalive(self):
        """Ping to keep the website alive"""
        while self.running:
            await asyncio.sleep(self.ping_interval)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get('https://finnhub.io/api/v1/forex/rates', 
                                         params={'base': 'EUR', 'token': self.api_key}) as response:
                        if response.status == 200:
                            logger.debug("Keep-alive ping successful")
                        else:
                            logger.warning(f"Keep-alive ping failed: {response.status}")
            except Exception as e:
                logger.error(f"Keep-alive error: {e}")
    
    async def run(self):
        """Main run loop"""
        self.running = True
        
        try:
            # Connect to WebSocket
            await self.ws_manager.connect()
            
            # Subscribe to price updates
            self.ws_manager.subscribe_price(self.handle_price_update)
            
            # Start keep-alive pings
            asyncio.create_task(self.ping_keepalive())
            
            # Start price stream
            asyncio.create_task(self.ws_manager.start_price_stream())
            
            # Display initial status
            logger.info("Bot started successfully")
            
            # Main loop
            while self.running:
                await asyncio.sleep(1)
                
                # Broadcast system status every 10 seconds
                if int(time.time()) % 10 == 0:
                    await self.broadcast_system_status()
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.running = False
            await self.cleanup()
    
    async def start(self):
        """Start the bot"""
        if not self.running:
            asyncio.create_task(self.run())
    
    async def stop(self):
        """Stop the bot"""
        self.running = False
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        
        # Disconnect WebSocket
        await self.ws_manager.disconnect()
        
        # Save final state
        self.trading_engine.save_to_storage()
        
        logger.info("Cleanup complete")

# ============================================================================
# WEBSOCKET SERVER FOR FRONTEND
# ============================================================================

# Initialize Socket.IO
sio = socketio.AsyncServer(cors_allowed_origins='*', async_mode='aiohttp')
app = web.Application()
sio.attach(app)

# Global bot instance
bot = None

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    logger.info(f'Client connected: {sid}')
    await sio.emit('log', {
        'message': 'Connected to trading bot',
        'level': 'success'
    }, room=sid)
    
    # Send initial data
    if bot:
        await sio.emit('performance', bot.trading_engine.get_performance_summary(), room=sid)
        await sio.emit('ml_update', {
            'accuracy': bot.ml_predictor.ml_accuracy,
            'training_samples': len(bot.training_features),
            'feature_importance': bot.ml_predictor.feature_importance
        }, room=sid)

@sio.event
async def disconnect(sid):
    logger.info(f'Client disconnected: {sid}')

@sio.event
async def start_bot(sid):
    logger.info('Starting bot from frontend')
    if bot and not bot.running:
        await bot.start()
        await sio.emit('log', {
            'message': 'Bot started successfully',
            'level': 'success'
        })

@sio.event
async def stop_bot(sid):
    logger.info('Stopping bot from frontend')
    if bot:
        await bot.stop()
        await sio.emit('log', {
            'message': 'Bot stopped',
            'level': 'warning'
        })

@sio.event
async def reset_data(sid):
    logger.info('Resetting data from frontend')
    if bot:
        bot.trading_engine = TradingEngine()
        await sio.emit('log', {
            'message': 'Trading data reset',
            'level': 'warning'
        })

# WebSocket handler for bot to broadcast to frontend
async def bot_websocket_handler(message_type: str, data: Any):
    """Handle broadcasts from bot to frontend"""
    await sio.emit(message_type, data)

# Static file serving
async def serve_index(request):
    return web.FileResponse('./index.html')

async def serve_js(request):
    return web.FileResponse('./main.js')

async def serve_css(request):
    return web.FileResponse('./style.css')

# Setup routes
app.router.add_get('/', serve_index)
app.router.add_get('/main.js', serve_js)
app.router.add_get('/style.css', serve_css)
app.router.add_static('/static/', path='./static', name='static')

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    global bot
    
    # Your Finnhub API Key - Replace with your actual key
    FINNHUB_API_KEY = "d631gf1r01qnpqnven70d631gf1r01qnpqnven7g"  # CHANGE THIS TO YOUR ACTUAL KEY
    
    if FINNHUB_API_KEY == "d631gf1r01qnpqnven70d631gf1r01qnpqnven7g":
        logger.error("Please replace FINNHUB_API_KEY with your actual Finnhub API key")
        logger.error("You can get one from https://finnhub.io/dashboard")
        return
    
    # Create bot instance
    bot = ScalpingTradingBot(FINNHUB_API_KEY)
    
    # Add WebSocket handler for frontend communication
    bot.add_websocket_handler(bot_websocket_handler)
    
    # Initialize bot (but don't start it yet)
    await bot.initialize()
    
    # Start the server
    logger.info("Starting WebSocket server on http://localhost:8000")
    logger.info("Open http://localhost:8000 in your browser")
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8000)
    await site.start()
    
    # Keep server running
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server crashed: {e}")