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
import hmac
import hashlib

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Your Finnhub API Key - Get from environment variable or hardcode
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d632adhr01qnpqnvhljgd632adhr01qnpqnvhlk0")  # CHANGE THIS TO YOUR ACTUAL KEY
FINNHUB_WEBHOOK_SECRET = os.getenv("FINNHUB_WEBHOOK_SECRET", "d631gf1r01qnpqnven8g")  # Your webhook secret

# Port for Render
PORT = int(os.getenv("PORT", 8000))

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
# PRICE MANAGER WITH SIMULATED DATA
# ============================================================================

class PriceManager:
    def __init__(self):
        # Throttling configuration
        self.throttle_interval = 1.22  # seconds
        self.last_broadcast = 0
        self.message_queue = deque(maxlen=10)
        
        # Subscribers
        self.price_handlers = []
        
        # Simulated price
        self.base_price = 1.08500
        self.price_history = deque(maxlen=1000)
        
    async def start_price_stream(self):
        """Start simulated price updates with 1.22s throttling"""
        while True:
            try:
                # Generate simulated price movement
                price_change = np.random.normal(0, 0.0001)  # Small random walk
                self.base_price += price_change
                
                # Keep price in reasonable range
                self.base_price = max(1.08000, min(1.09000, self.base_price))
                
                price_data = MarketData(
                    price=self.base_price,
                    timestamp=time.time(),
                    bid=self.base_price - 0.0001,
                    ask=self.base_price + 0.0001,
                    spread=0.0002
                )
                
                # Apply server-side throttling
                await self._throttle_and_broadcast(price_data)
                
                await asyncio.sleep(0.1)  # Check every 100ms
                
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
    
    def add_price_history(self, price_data: MarketData):
        """Add price to history"""
        self.price_history.append(price_data)

# ============================================================================
# SIMPLIFIED TOOLKIT
# ============================================================================

class ScalperToolkit:
    def __init__(self, price_manager: PriceManager):
        self.price_manager = price_manager
        
    async def analyze_market(self, current_price: float) -> Dict[str, float]:
        """Run complete market analysis with simulated tools"""
        analysis = {}
        
        # Get recent prices for calculations
        prices = [p.price for p in self.price_manager.price_history]
        if not prices:
            prices = [current_price]
        
        # Order Flow Tools (simulated)
        analysis['bid_ask_imbalance'] = np.random.uniform(-0.3, 0.3)
        analysis['cumulative_delta'] = np.random.uniform(-100, 100)
        analysis['large_trades'] = np.random.uniform(0, 1)
        analysis['volume_poc'] = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
        
        # Price Action Tools
        if len(prices) >= 50:
            recent = prices[-50:]
            analysis['swing_high'] = max(recent)
            analysis['swing_low'] = min(recent)
        else:
            analysis['swing_high'] = current_price + 0.0010
            analysis['swing_low'] = current_price - 0.0010
        
        analysis['vwap'] = np.mean(prices[-30:]) if len(prices) >= 30 else current_price
        
        # EMA calculation
        if len(prices) >= 18:
            analysis['ema_9'] = pd.Series(prices).ewm(span=9).mean().iloc[-1]
        else:
            analysis['ema_9'] = current_price
        
        # Bollinger Bands
        if len(prices) >= 20:
            recent_prices = prices[-20:]
            middle = np.mean(recent_prices)
            std = np.std(recent_prices)
            analysis['bb_upper'] = middle + 2 * std
            analysis['bb_middle'] = middle
            analysis['bb_lower'] = middle - 2 * std
        else:
            analysis['bb_upper'] = current_price + 0.0010
            analysis['bb_middle'] = current_price
            analysis['bb_lower'] = current_price - 0.0010
        
        # Timing Tools
        utc_hour = datetime.utcnow().hour
        sessions = {
            'asian': (0, 8),
            'london': (8, 16),
            'ny': (13, 21),
            'overlap': (13, 16)
        }
        
        session_strength = 0
        for session, (start, end) in sessions.items():
            if start <= utc_hour < end:
                if session == 'overlap':
                    session_strength = 0.9
                elif session in ['london', 'ny']:
                    session_strength = 0.7
                else:
                    session_strength = 0.3
                break
        
        analysis['session_strength'] = session_strength
        
        # Time-based volatility
        if 8 <= utc_hour < 16:
            analysis['time_volatility'] = 0.0006
        elif 13 <= utc_hour < 21:
            analysis['time_volatility'] = 0.0008
        elif 13 <= utc_hour < 16:
            analysis['time_volatility'] = 0.0010
        else:
            analysis['time_volatility'] = 0.0003
        
        analysis['economic_impact'] = np.random.uniform(0, 0.3)
        
        # Risk Tools
        analysis['usd_strength'] = np.random.uniform(0.3, 0.7)
        
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
    def __init__(self):
        # Initialize components
        self.price_manager = PriceManager()
        self.toolkit = ScalperToolkit(self.price_manager)
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
        
        # WebSocket handlers for frontend
        self.ws_handlers = []
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Scalping Trading Bot...")
        
        # Load saved data
        self.trading_engine.load_from_storage()
        
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
        self.price_manager.add_price_history(price_data)
        
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
            'ws_connected': True,
            'api_calls_per_min': 0,
            'system_status': 'operational' if self.running else 'stopped'
        }
        await self.broadcast_to_frontend('system_status', status)
    
    async def ping_keepalive(self):
        """Ping to keep the website alive"""
        while self.running:
            await asyncio.sleep(self.ping_interval)
            try:
                logger.debug("Keep-alive ping")
            except Exception as e:
                logger.error(f"Keep-alive error: {e}")
    
    async def run(self):
        """Main run loop"""
        self.running = True
        
        try:
            # Subscribe to price updates
            self.price_manager.subscribe_price(self.handle_price_update)
            
            # Start price stream
            asyncio.create_task(self.price_manager.start_price_stream())
            
            # Start keep-alive pings
            asyncio.create_task(self.ping_keepalive())
            
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

# Finnhub Webhook endpoint
async def finnhub_webhook(request):
    """Handle Finnhub webhook events"""
    try:
        # Verify webhook signature
        signature = request.headers.get('X-Finnhub-Secret', '')
        if signature != FINNHUB_WEBHOOK_SECRET:
            logger.warning(f"Invalid webhook signature: {signature}")
            return web.Response(status=401)
        
        # Get webhook data
        data = await request.json()
        logger.info(f"Received Finnhub webhook: {data}")
        
        # Process the webhook data
        if data.get('type') == 'trade' and data.get('data'):
            trade_data = data['data'][0]
            symbol = trade_data.get('s', '')
            
            if symbol == 'OANDA:EUR_USD' or 'EUR/USD' in symbol:
                # Process EUR/USD trade
                price = trade_data.get('p', 1.08500)
                
                if bot and bot.current_price:
                    # Update bot with real price
                    price_data = MarketData(
                        price=price,
                        timestamp=time.time(),
                        bid=price - 0.0001,
                        ask=price + 0.0001,
                        spread=0.0002
                    )
                    
                    # Update bot
                    await bot.handle_price_update(price_data)
        
        # Always return 200 OK to acknowledge receipt
        return web.Response(status=200)
        
    except json.JSONDecodeError:
        logger.error("Invalid JSON in webhook")
        return web.Response(status=400)
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return web.Response(status=500)

# Static file serving
async def serve_index(request):
    return web.FileResponse('./index.html')

async def serve_js(request):
    return web.FileResponse('./main.js')

async def serve_css(request):
    return web.FileResponse('./style.css')

# Health check endpoint for Render
async def health_check(request):
    return web.json_response({"status": "ok", "timestamp": time.time()})

# Setup routes
app.router.add_get('/', serve_index)
app.router.add_get('/main.js', serve_js)
app.router.add_get('/style.css', serve_css)
app.router.add_post('/webhook/finnhub', finnhub_webhook)
app.router.add_get('/health', health_check)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    global bot
    
    # Create bot instance
    bot = ScalpingTradingBot()
    
    # Add WebSocket handler for frontend communication
    bot.add_websocket_handler(bot_websocket_handler)
    
    # Initialize bot
    await bot.initialize()
    
    # Start the server
    logger.info(f"Starting server on port {PORT}")
    logger.info(f"Dashboard available at: http://localhost:{PORT}")
    logger.info(f"Finnhub webhook endpoint: http://localhost:{PORT}/webhook/finnhub")
    logger.info(f"Health check: http://localhost:{PORT}/health")
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
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