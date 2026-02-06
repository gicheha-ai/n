# app.py - Main Flask application with keep-alive and web interface
from flask import Flask, jsonify, render_template, request
from threading import Thread, Lock
import time
import requests
import atexit
import logging
import os
import random
from datetime import datetime, timedelta

app = Flask(__name__)

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedKeepAlive:
    def __init__(self, interval_minutes=9.5, max_retries=3, app_url=None):
        # Using 9.5 minutes for safety margin
        self.interval = interval_minutes * 60
        self.max_retries = max_retries
        self.app_url = app_url or self.get_app_url()
        self.is_running = False
        self.thread = None
        self.lock = Lock()
        self.consecutive_failures = 0
        self.total_pings = 0
        self.successful_pings = 0
        self.failed_pings = 0
        self.start_time = time.time()
        self.last_ping_time = None
        self.last_ping_status = None
        self.ping_history = []
        self.max_history = 100
        
        logger.info(f"🚀 EnhancedKeepAlive initialized")
        logger.info(f"📡 App URL: {self.app_url}")
        logger.info(f"⏰ Ping interval: {interval_minutes} minutes")
        logger.info(f"🔄 Max retries: {max_retries}")
        
    def get_app_url(self):
        """Get app URL with multiple fallback options"""
        url_sources = [
            os.environ.get('RENDER_EXTERNAL_URL'),
            os.environ.get('RENDER_EXTERNAL_HOSTNAME'),
            os.environ.get('APP_URL'),
            os.environ.get('WEB_URL'),
            os.environ.get('URL'),
            f"http://localhost:{os.environ.get('PORT', '5000')}"
        ]
        
        for url in url_sources:
            if url:
                if not url.startswith(('http://', 'https://')):
                    url = f"https://{url}"  # Assume HTTPS for production
                logger.info(f"✅ Using URL from environment: {url}")
                return url.rstrip('/')
        
        logger.warning("⚠️  No URL found in environment, using localhost")
        return f"http://localhost:{os.environ.get('PORT', '5000')}"
    
    def record_ping(self, success, response_time=None, status_code=None, error=None):
        """Record ping attempt in history"""
        ping_record = {
            'timestamp': time.time(),
            'success': success,
            'response_time': response_time,
            'status_code': status_code,
            'error': error
        }
        
        self.ping_history.append(ping_record)
        
        # Keep only last N records
        if len(self.ping_history) > self.max_history:
            self.ping_history = self.ping_history[-self.max_history:]
    
    def ping_with_retry(self):
        """Ping with retry logic and exponential backoff"""
        start_ping = time.time()
        
        for attempt in range(self.max_retries):
            try:
                # Add cache-busting parameter
                cache_buster = random.randint(1000, 9999)
                ping_url = f"{self.app_url}/api/health?_={cache_buster}&attempt={attempt}"
                
                # Add timeout and headers
                headers = {
                    'User-Agent': 'KeepAlive/2.0',
                    'X-KeepAlive-Timestamp': str(time.time())
                }
                
                response = requests.get(
                    ping_url, 
                    headers=headers,
                    timeout=10,
                    verify=False  # For self-signed certs in dev
                )
                
                response_time = time.time() - start_ping
                
                if response.status_code == 200:
                    data = response.json()
                    
                    self.consecutive_failures = 0
                    self.successful_pings += 1
                    self.last_ping_time = time.time()
                    self.last_ping_status = 'success'
                    
                    uptime_seconds = data.get('uptime', 0)
                    uptime_hours = uptime_seconds / 3600
                    
                    logger.info(
                        f"✅ Ping #{self.total_pings} successful | "
                        f"Response: {response_time:.2f}s | "
                        f"Uptime: {uptime_hours:.1f}h"
                    )
                    
                    self.record_ping(
                        success=True,
                        response_time=response_time,
                        status_code=response.status_code
                    )
                    
                    return True
                else:
                    self.record_ping(
                        success=False,
                        response_time=response_time,
                        status_code=response.status_code,
                        error=f"HTTP {response.status_code}"
                    )
                    
            except requests.exceptions.Timeout:
                response_time = time.time() - start_ping
                logger.warning(f"⏰ Timeout on attempt {attempt + 1}/{self.max_retries}")
                self.record_ping(
                    success=False,
                    response_time=response_time,
                    error="Timeout"
                )
            except requests.exceptions.ConnectionError as e:
                response_time = time.time() - start_ping
                logger.warning(f"🔌 Connection error on attempt {attempt + 1}/{self.max_retries}: {e}")
                self.record_ping(
                    success=False,
                    response_time=response_time,
                    error="Connection Error"
                )
            except Exception as e:
                response_time = time.time() - start_ping
                logger.error(f"❌ Error on attempt {attempt + 1}/{self.max_retries}: {str(e)[:100]}")
                self.record_ping(
                    success=False,
                    response_time=response_time,
                    error=str(e)[:100]
                )
            
            # Exponential backoff before retry
            if attempt < self.max_retries - 1:
                backoff = (2 ** attempt) * 2  # 2, 4, 8 seconds
                logger.info(f"⏳ Waiting {backoff}s before retry...")
                time.sleep(backoff)
        
        # All retries failed
        with self.lock:
            self.consecutive_failures += 1
            self.failed_pings += 1
            self.last_ping_time = time.time()
            self.last_ping_status = 'failed'
        
        logger.error(f"🔥 All {self.max_retries} ping attempts failed. "
                    f"Consecutive failures: {self.consecutive_failures}")
        
        # Critical alert after multiple consecutive failures
        if self.consecutive_failures >= 3:
            logger.critical("🚨 CRITICAL: Multiple consecutive ping failures! "
                          "App might go to sleep!")
        
        return False
    
    def keep_alive_loop(self):
        """Main loop with self-healing"""
        logger.info("🔄 Keep-alive loop started")
        
        while self.is_running:
            with self.lock:
                self.total_pings += 1
            
            success = self.ping_with_retry()
            
            # Adaptive sleep based on success/failure
            sleep_time = self.interval
            if not success and self.consecutive_failures > 0:
                # If failing, check more frequently
                sleep_time = min(60, self.interval / 2)
                logger.info(f"⚠️  Using adaptive sleep of {sleep_time}s due to failures")
            
            # Sleep in chunks to allow for quick stop
            chunk_size = 5  # seconds
            chunks = int(sleep_time / chunk_size)
            for _ in range(chunks):
                if not self.is_running:
                    break
                time.sleep(chunk_size)
    
    def start(self):
        """Start the keep-alive thread"""
        with self.lock:
            if not self.is_running:
                self.is_running = True
                self.thread = Thread(
                    target=self.keep_alive_loop,
                    daemon=True,
                    name="KeepAliveThread"
                )
                self.thread.start()
                logger.info("🚀 Keep-alive thread started successfully")
                return True
        return False
    
    def stop(self):
        """Stop the keep-alive thread"""
        with self.lock:
            self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=10)
            logger.info("🛑 Keep-alive thread stopped")
            return True
        return False
    
    def restart(self):
        """Restart the keep-alive thread"""
        self.stop()
        time.sleep(1)
        return self.start()
    
    def get_stats(self):
        """Get statistics about keep-alive performance"""
        with self.lock:
            success_rate = (self.successful_pings / self.total_pings * 100) if self.total_pings > 0 else 0
            uptime_seconds = time.time() - self.start_time
            next_ping_seconds = 0
            
            if self.last_ping_time:
                time_since_last_ping = time.time() - self.last_ping_time
                next_ping_seconds = max(0, self.interval - time_since_last_ping)
            
            return {
                "is_running": self.is_running,
                "total_pings": self.total_pings,
                "successful_pings": self.successful_pings,
                "failed_pings": self.failed_pings,
                "success_rate": round(success_rate, 1),
                "consecutive_failures": self.consecutive_failures,
                "app_url": self.app_url,
                "interval_minutes": round(self.interval / 60, 1),
                "uptime_hours": round(uptime_seconds / 3600, 1),
                "next_ping_in": round(next_ping_seconds, 0),
                "last_ping_time": self.last_ping_time,
                "last_ping_status": self.last_ping_status,
                "ping_history_count": len(self.ping_history)
            }
    
    def get_ping_history(self, limit=50):
        """Get recent ping history"""
        with self.lock:
            recent_history = self.ping_history[-limit:] if self.ping_history else []
            
            formatted_history = []
            for ping in recent_history:
                formatted_history.append({
                    'timestamp': ping['timestamp'],
                    'time_formatted': datetime.fromtimestamp(ping['timestamp']).strftime('%H:%M:%S'),
                    'success': ping['success'],
                    'response_time': round(ping.get('response_time', 0), 3) if ping.get('response_time') else None,
                    'status_code': ping.get('status_code'),
                    'error': ping.get('error')
                })
            
            return formatted_history

# Create enhanced keep-alive instance (9.5 minutes for safety)
keep_alive = EnhancedKeepAlive(
    interval_minutes=9.5,
    max_retries=3
)

# Global start time for the app
app_start_time = time.time()

# API Routes
@app.route('/api/health')
def api_health():
    """Health check endpoint for keep-alive pings"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - app_start_time,
        "version": "2.0.0",
        "service": "render-keep-alive"
    }), 200

@app.route('/api/stats')
def api_stats():
    """Get keep-alive statistics"""
    stats = keep_alive.get_stats()
    return jsonify(stats)

@app.route('/api/history')
def api_history():
    """Get ping history"""
    limit = request.args.get('limit', default=50, type=int)
    history = keep_alive.get_ping_history(limit=limit)
    return jsonify(history)

@app.route('/api/control', methods=['POST'])
def api_control():
    """Control the keep-alive system"""
    data = request.get_json() or {}
    action = data.get('action', '')
    
    response = {"success": False, "message": ""}
    
    if action == 'start':
        if keep_alive.start():
            response.update({"success": True, "message": "Keep-alive started"})
        else:
            response.update({"message": "Keep-alive already running or failed to start"})
    
    elif action == 'stop':
        if keep_alive.stop():
            response.update({"success": True, "message": "Keep-alive stopped"})
        else:
            response.update({"message": "Keep-alive already stopped or failed to stop"})
    
    elif action == 'restart':
        if keep_alive.restart():
            response.update({"success": True, "message": "Keep-alive restarted"})
        else:
            response.update({"message": "Failed to restart keep-alive"})
    
    elif action == 'force_ping':
        # Force an immediate ping
        success = keep_alive.ping_with_retry()
        response.update({
            "success": success,
            "message": "Force ping completed"
        })
    
    else:
        response.update({"message": f"Unknown action: {action}"})
    
    return jsonify(response)

# Web Interface Routes
@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('index.html')

# Auto-start keep-alive when app starts
if __name__ == "__main__":
    # Register cleanup function
    atexit.register(keep_alive.stop)
    
    # Start the keep-alive thread
    keep_alive.start()
    
    # Start Flask app
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 Starting Flask app on port {port}")
    logger.info(f"🔗 App URL: {keep_alive.app_url}")
    logger.info(f"⏰ Keep-alive interval: {keep_alive.interval/60:.1f} minutes")
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # For production with gunicorn
    @app.before_first_request
    def initialize():
        keep_alive.start()
    atexit.register(keep_alive.stop)