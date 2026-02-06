class TradingDashboard {
    constructor() {
        // WebSocket connection
        this.ws = null;
        this.wsConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        // State
        this.currentPrice = 1.08500;
        this.previousPrice = 1.08500;
        this.priceHistory = [];
        this.maxHistory = 100;
        
        // Performance data
        this.virtualBalance = 10000;
        this.totalTrades = 0;
        this.winningTrades = 0;
        this.totalPnl = 0;
        this.tradeHistory = [];
        this.equityCurve = [];
        
        // ML data
        this.mlAccuracy = 0;
        this.trainingSamples = 0;
        this.featureImportance = {};
        
        // System status
        this.startTime = Date.now();
        this.apiCalls = 0;
        this.cycleCount = 0;
        this.activeTrade = null;
        this.prediction = null;
        this.toolSignals = {};
        
        // Countdowns
        this.predictionCountdown = 120;
        this.tradeTimer = 0;
        this.analysisCountdown = 120;
        
        // Charts
        this.priceChart = null;
        this.equityChart = null;
        this.learningChart = null;
        
        // DOM Elements
        this.initializeElements();
        
        // Start
        this.init();
    }
    
    initializeElements() {
        // Price elements
        this.elements = {
            currentPrice: document.getElementById('currentPrice'),
            bidPrice: document.getElementById('bidPrice'),
            askPrice: document.getElementById('askPrice'),
            spread: document.getElementById('spread'),
            priceChange: document.getElementById('priceChange'),
            changeDirection: document.getElementById('changeDirection'),
            changeAmount: document.getElementById('changeAmount'),
            
            // Trade elements
            tradeStatus: document.getElementById('tradeStatus'),
            tradeDirection: document.getElementById('tradeDirection'),
            entryPrice: document.getElementById('entryPrice'),
            takeProfit: document.getElementById('takeProfit'),
            stopLoss: document.getElementById('stopLoss'),
            tradeSize: document.getElementById('tradeSize'),
            progressFill: document.getElementById('progressFill'),
            tradeTimer: document.getElementById('tradeTimer'),
            
            // Prediction elements
            predictionCountdown: document.getElementById('predictionCountdown'),
            predictedDirection: document.getElementById('predictedDirection'),
            confidenceValue: document.getElementById('confidenceValue'),
            confidenceBadge: document.getElementById('confidenceBadge'),
            expectedTP: document.getElementById('expectedTP'),
            expectedSL: document.getElementById('expectedSL'),
            expectedRR: document.getElementById('expectedRR'),
            signalGrid: document.getElementById('signalGrid'),
            
            // Performance elements
            virtualBalance: document.getElementById('virtualBalance'),
            winRate: document.getElementById('winRate'),
            totalTrades: document.getElementById('totalTrades'),
            totalPnl: document.getElementById('totalPnl'),
            profitFactor: document.getElementById('profitFactor'),
            maxDrawdown: document.getElementById('maxDrawdown'),
            avgTrade: document.getElementById('avgTrade'),
            tradeList: document.getElementById('tradeList'),
            
            // ML elements
            mlStatus: document.getElementById('mlStatus'),
            mlAccuracy: document.getElementById('mlAccuracy'),
            trainingSamples: document.getElementById('trainingSamples'),
            topFeature: document.getElementById('topFeature'),
            accuracyProgress: document.getElementById('accuracyProgress'),
            featureList: document.getElementById('featureList'),
            
            // System elements
            statusDot: document.getElementById('statusDot'),
            connectionStatus: document.getElementById('connectionStatus'),
            liveClock: document.getElementById('liveClock'),
            cycleCount: document.getElementById('cycleCount'),
            wsStatus: document.getElementById('wsStatus'),
            updateRate: document.getElementById('updateRate'),
            apiCalls: document.getElementById('apiCalls'),
            uptime: document.getElementById('uptime'),
            systemIcon: document.getElementById('systemIcon'),
            systemStatus: document.getElementById('systemStatus'),
            lastUpdate: document.getElementById('lastUpdate'),
            nextAnalysis: document.getElementById('nextAnalysis'),
            apiStatus: document.getElementById('apiStatus'),
            logContainer: document.getElementById('logContainer'),
            
            // Buttons
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),
            resetBtn: document.getElementById('resetBtn')
        };
    }
    
    async init() {
        this.setupCharts();
        this.setupEventListeners();
        this.startClocks();
        this.connectWebSocket();
        this.startDynamicUpdates();
        
        // Load initial data
        this.loadInitialData();
        
        this.log('Dashboard initialized', 'success');
    }
    
    setupCharts() {
        // Price Chart
        const priceCtx = document.getElementById('priceChart').getContext('2d');
        this.priceChart = new Chart(priceCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'EUR/USD',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 300
                },
                plugins: {
                    legend: { display: false },
                    tooltip: { mode: 'index', intersect: false }
                },
                scales: {
                    x: { display: false },
                    y: {
                        display: true,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#aaa' }
                    }
                }
            }
        });
        
        // Equity Chart
        const equityCtx = document.getElementById('equityChart').getContext('2d');
        this.equityChart = new Chart(equityCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Equity Curve',
                    data: [10000],
                    borderColor: '#4dabf7',
                    backgroundColor: 'rgba(77, 171, 247, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { display: false },
                    y: {
                        display: true,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { 
                            color: '#aaa',
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
        
        // Learning Chart
        const learningCtx = document.getElementById('learningChart').getContext('2d');
        this.learningChart = new Chart(learningCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Accuracy',
                        data: [],
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        borderWidth: 2,
                        fill: true
                    },
                    {
                        label: 'Win Rate',
                        data: [],
                        borderColor: '#4ecdc4',
                        backgroundColor: 'rgba(78, 205, 196, 0.1)',
                        borderWidth: 2,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: '#aaa' }
                    }
                },
                scales: {
                    x: { 
                        display: true,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#aaa' }
                    },
                    y: {
                        display: true,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { 
                            color: '#aaa',
                            callback: function(value) {
                                return value + '%';
                            }
                        },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }
    
    setupEventListeners() {
        // Control buttons
        this.elements.startBtn.addEventListener('click', () => this.startBot());
        this.elements.stopBtn.addEventListener('click', () => this.stopBot());
        this.elements.resetBtn.addEventListener('click', () => this.resetData());
        
        // Sound toggle
        document.addEventListener('click', (e) => {
            if (e.target.closest('.btn-sound')) {
                this.toggleSound();
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === ' ') {
                e.preventDefault();
                if (this.wsConnected) {
                    this.stopBot();
                } else {
                    this.startBot();
                }
            }
            if (e.key === 'r' && e.ctrlKey) {
                e.preventDefault();
                this.resetData();
            }
        });
    }
    
    connectWebSocket() {
        const wsUrl = `ws://${window.location.hostname}:8000/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            this.wsConnected = true;
            this.reconnectAttempts = 0;
            this.updateConnectionStatus('connected');
            this.log('WebSocket connected successfully', 'success');
            this.startBot();
        };
        
        this.ws.onmessage = (event) => {
            this.handleMessage(JSON.parse(event.data));
        };
        
        this.ws.onclose = () => {
            this.wsConnected = false;
            this.updateConnectionStatus('disconnected');
            this.log('WebSocket disconnected', 'warning');
            
            // Attempt to reconnect
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                setTimeout(() => {
                    this.log(`Reconnection attempt ${this.reconnectAttempts}...`, 'info');
                    this.connectWebSocket();
                }, 3000);
            }
        };
        
        this.ws.onerror = (error) => {
            this.log(`WebSocket error: ${error}`, 'error');
            this.updateConnectionStatus('error');
        };
    }
    
    handleMessage(data) {
        this.elements.lastUpdate.textContent = new Date().toLocaleTimeString();
        
        switch (data.type) {
            case 'price_update':
                this.handlePriceUpdate(data.data);
                break;
            case 'trade_update':
                this.handleTradeUpdate(data.data);
                break;
            case 'prediction':
                this.handlePrediction(data.data);
                break;
            case 'performance':
                this.handlePerformance(data.data);
                break;
            case 'ml_update':
                this.handleMLUpdate(data.data);
                break;
            case 'system_status':
                this.handleSystemStatus(data.data);
                break;
            case 'log':
                this.log(data.message, data.level);
                break;
        }
    }
    
    handlePriceUpdate(priceData) {
        // Update price
        this.previousPrice = this.currentPrice;
        this.currentPrice = priceData.price;
        
        // Update UI
        this.elements.currentPrice.textContent = this.currentPrice.toFixed(5);
        this.elements.bidPrice.textContent = (this.currentPrice - 0.0001).toFixed(5);
        this.elements.askPrice.textContent = (this.currentPrice + 0.0001).toFixed(5);
        this.elements.spread.textContent = '2.0';
        
        // Calculate change
        const change = this.currentPrice - this.previousPrice;
        const changePips = Math.abs(change) / 0.0001;
        
        if (change > 0) {
            this.elements.changeDirection.textContent = '↗';
            this.elements.changeDirection.className = 'direction-up';
            this.elements.changeAmount.textContent = `+${changePips.toFixed(1)} pips`;
            this.elements.changeAmount.className = 'change-up';
        } else if (change < 0) {
            this.elements.changeDirection.textContent = '↘';
            this.elements.changeDirection.className = 'direction-down';
            this.elements.changeAmount.textContent = `-${changePips.toFixed(1)} pips`;
            this.elements.changeAmount.className = 'change-down';
        } else {
            this.elements.changeDirection.textContent = '→';
            this.elements.changeDirection.className = 'direction-neutral';
            this.elements.changeAmount.textContent = '0.0 pips';
            this.elements.changeAmount.className = 'change-neutral';
        }
        
        // Update price history
        this.priceHistory.push({
            time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}),
            price: this.currentPrice
        });
        
        if (this.priceHistory.length > this.maxHistory) {
            this.priceHistory.shift();
        }
        
        // Update chart
        this.updatePriceChart();
        
        // Update trade progress if active
        if (this.activeTrade) {
            this.updateTradeProgress();
        }
    }
    
    handleTradeUpdate(tradeData) {
        this.activeTrade = tradeData;
        
        if (tradeData.status === 'OPEN') {
            // New trade opened
            this.elements.tradeStatus.textContent = 'ACTIVE';
            this.elements.tradeStatus.className = 'status-active';
            
            this.elements.tradeDirection.textContent = tradeData.direction;
            this.elements.tradeDirection.className = tradeData.direction.toLowerCase();
            
            this.elements.entryPrice.textContent = tradeData.entry_price.toFixed(5);
            this.elements.takeProfit.textContent = tradeData.take_profit.toFixed(5);
            this.elements.stopLoss.textContent = tradeData.stop_loss.toFixed(5);
            this.elements.tradeSize.textContent = tradeData.size.toFixed(2);
            
            this.tradeTimer = 0;
            this.startTradeTimer();
            
            this.playSound('tradeOpenSound');
            this.log(`New trade opened: ${tradeData.direction} at ${tradeData.entry_price}`, 'success');
            
        } else if (tradeData.status.startsWith('CLOSED_')) {
            // Trade closed
            const profit = tradeData.pnl || 0;
            
            this.elements.tradeStatus.textContent = `CLOSED (${profit >= 0 ? 'WIN' : 'LOSS'})`;
            this.elements.tradeStatus.className = profit >= 0 ? 'status-win' : 'status-loss';
            
            this.virtualBalance += profit;
            this.totalPnl += profit;
            this.totalTrades++;
            
            if (profit > 0) this.winningTrades++;
            
            this.tradeHistory.unshift({
                id: tradeData.id,
                direction: tradeData.direction,
                entry: tradeData.entry_price,
                exit: tradeData.close_price,
                pnl: profit,
                duration: tradeData.duration,
                status: tradeData.status
            });
            
            if (this.tradeHistory.length > 10) {
                this.tradeHistory.pop();
            }
            
            // Update equity curve
            this.equityCurve.push(this.virtualBalance);
            this.updateEquityChart();
            this.updatePerformance();
            this.updateTradeList();
            
            this.playSound('tradeCloseSound');
            this.log(`Trade closed: ${profit >= 0 ? 'WIN' : 'LOSS'} $${Math.abs(profit).toFixed(2)}`, 
                     profit >= 0 ? 'success' : 'error');
            
            // Clear active trade
            setTimeout(() => {
                this.activeTrade = null;
                this.elements.tradeStatus.textContent = 'No Active Trade';
                this.elements.tradeStatus.className = 'status-inactive';
                this.elements.tradeDirection.textContent = '--';
                this.elements.entryPrice.textContent = '--.--';
                this.elements.takeProfit.textContent = '--.--';
                this.elements.stopLoss.textContent = '--.--';
                this.elements.tradeSize.textContent = '--';
                this.elements.progressFill.style.width = '0%';
                clearInterval(this.tradeTimerInterval);
            }, 5000);
        }
    }
    
    handlePrediction(predictionData) {
        this.prediction = predictionData;
        this.analysisCountdown = 120;
        
        this.elements.predictedDirection.textContent = predictionData.direction;
        this.elements.predictedDirection.className = predictionData.direction.toLowerCase();
        
        this.elements.confidenceValue.textContent = `${predictionData.confidence.toFixed(1)}%`;
        this.elements.confidenceBadge.className = `confidence-badge ${
            predictionData.confidence > 70 ? 'high' : 
            predictionData.confidence > 50 ? 'medium' : 'low'
        }`;
        
        this.elements.expectedTP.textContent = predictionData.take_profit.toFixed(5);
        this.elements.expectedSL.textContent = predictionData.stop_loss.toFixed(5);
        this.elements.expectedRR.textContent = predictionData.risk_reward.toFixed(2);
        
        // Update tool signals
        this.updateToolSignals(predictionData.tools_signals || {});
        
        this.log(`New prediction: ${predictionData.direction} with ${predictionData.confidence.toFixed(1)}% confidence`, 'info');
        this.cycleCount++;
        this.elements.cycleCount.textContent = this.cycleCount;
    }
    
    handlePerformance(perfData) {
        this.virtualBalance = perfData.balance || this.virtualBalance;
        this.totalTrades = perfData.total_trades || 0;
        this.winningTrades = perfData.winning_trades || 0;
        this.totalPnl = perfData.total_pnl || 0;
        
        this.updatePerformance();
    }
    
    handleMLUpdate(mlData) {
        this.mlAccuracy = mlData.accuracy || this.mlAccuracy;
        this.trainingSamples = mlData.training_samples || this.trainingSamples;
        this.featureImportance = mlData.feature_importance || {};
        
        this.elements.mlAccuracy.textContent = `${this.mlAccuracy.toFixed(1)}%`;
        this.elements.trainingSamples.textContent = this.trainingSamples;
        this.elements.accuracyProgress.style.width = `${this.mlAccuracy}%`;
        
        // Update top feature
        if (Object.keys(this.featureImportance).length > 0) {
            const topFeature = Object.entries(this.featureImportance)
                .sort((a, b) => b[1] - a[1])[0];
            this.elements.topFeature.textContent = this.formatFeatureName(topFeature[0]);
        }
        
        // Update feature list
        this.updateFeatureList();
        
        // Update learning chart
        this.updateLearningChart();
    }
    
    handleSystemStatus(statusData) {
        this.elements.wsStatus.textContent = statusData.ws_connected ? 'Connected' : 'Disconnected';
        this.elements.wsStatus.className = statusData.ws_connected ? 'status-good' : 'status-bad';
        
        this.elements.apiCalls.textContent = `${statusData.api_calls_per_min || 0}/min`;
        this.elements.systemStatus.textContent = statusData.system_status || 'Operational';
        
        if (statusData.system_status === 'error') {
            this.elements.systemIcon.className = 'fas fa-circle status-error';
        } else if (statusData.system_status === 'warning') {
            this.elements.systemIcon.className = 'fas fa-circle status-warning';
        } else {
            this.elements.systemIcon.className = 'fas fa-circle status-good';
        }
    }
    
    updatePriceChart() {
        const labels = this.priceHistory.map(item => item.time);
        const data = this.priceHistory.map(item => item.price);
        
        this.priceChart.data.labels = labels;
        this.priceChart.data.datasets[0].data = data;
        this.priceChart.update('none');
    }
    
    updateEquityChart() {
        const labels = Array.from({length: this.equityCurve.length}, (_, i) => i + 1);
        
        this.equityChart.data.labels = labels;
        this.equityChart.data.datasets[0].data = this.equityCurve;
        this.equityChart.update('none');
    }
    
    updateLearningChart() {
        // Simulate learning progress
        const labels = Array.from({length: this.trainingSamples}, (_, i) => i + 1);
        const accuracyData = Array.from({length: this.trainingSamples}, 
            (_, i) => 50 + Math.min(50, (i / this.trainingSamples) * 50));
        const winRateData = Array.from({length: this.trainingSamples},
            (_, i) => 50 + Math.min(40, (i / this.trainingSamples) * 40));
        
        this.learningChart.data.labels = labels;
        this.learningChart.data.datasets[0].data = accuracyData;
        this.learningChart.data.datasets[1].data = winRateData;
        this.learningChart.update('none');
    }
    
    updateTradeProgress() {
        if (!this.activeTrade) return;
        
        const trade = this.activeTrade;
        const current = this.currentPrice;
        
        let progress = 0;
        
        if (trade.direction === 'BUY') {
            const range = trade.take_profit - trade.stop_loss;
            const position = current - trade.stop_loss;
            progress = Math.max(0, Math.min(100, (position / range) * 100));
        } else {
            const range = trade.stop_loss - trade.take_profit;
            const position = trade.stop_loss - current;
            progress = Math.max(0, Math.min(100, (position / range) * 100));
        }
        
        this.elements.progressFill.style.width = `${progress}%`;
        
        // Update progress bar color based on profit/loss
        if (progress < 50) {
            this.elements.progressFill.style.background = 'linear-gradient(90deg, #ff6b6b, #ff8787)';
        } else if (progress < 80) {
            this.elements.progressFill.style.background = 'linear-gradient(90deg, #ffa94d, #ffc078)';
        } else {
            this.elements.progressFill.style.background = 'linear-gradient(90deg, #40c057, #69db7c)';
        }
    }
    
    updateToolSignals(signals) {
        this.elements.signalGrid.innerHTML = '';
        
        Object.entries(signals).forEach(([tool, value]) => {
            const signal = document.createElement('div');
            signal.className = 'signal-item';
            
            let signalClass = 'neutral';
            let signalText = 'NEUTRAL';
            
            if (value > 0.2) {
                signalClass = 'bullish';
                signalText = 'BULLISH';
            } else if (value < -0.2) {
                signalClass = 'bearish';
                signalText = 'BEARISH';
            }
            
            signal.innerHTML = `
                <div class="signal-name">${this.formatFeatureName(tool)}</div>
                <div class="signal-value ${signalClass}">${signalText}</div>
                <div class="signal-bar">
                    <div class="bar-fill ${signalClass}" style="width: ${Math.abs(value) * 100}%"></div>
                </div>
            `;
            
            this.elements.signalGrid.appendChild(signal);
        });
    }
    
    updateFeatureList() {
        this.elements.featureList.innerHTML = '';
        
        Object.entries(this.featureImportance)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .forEach(([feature, importance]) => {
                const item = document.createElement('div');
                item.className = 'feature-item';
                
                item.innerHTML = `
                    <div class="feature-name">${this.formatFeatureName(feature)}</div>
                    <div class="feature-bar">
                        <div class="feature-fill" style="width: ${importance * 100}%"></div>
                    </div>
                    <div class="feature-value">${(importance * 100).toFixed(1)}%</div>
                `;
                
                this.elements.featureList.appendChild(item);
            });
    }
    
    updatePerformance() {
        const winRate = this.totalTrades > 0 ? (this.winningTrades / this.totalTrades) * 100 : 0;
        const avgTrade = this.totalTrades > 0 ? this.totalPnl / this.totalTrades : 0;
        
        // Calculate profit factor
        const grossProfit = this.winningTrades * Math.abs(avgTrade);
        const grossLoss = (this.totalTrades - this.winningTrades) * Math.abs(avgTrade);
        const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 99 : 0;
        
        // Calculate max drawdown
        let maxDD = 0;
        let peak = this.equityCurve[0] || this.virtualBalance;
        
        for (const equity of this.equityCurve) {
            if (equity > peak) peak = equity;
            const dd = ((peak - equity) / peak) * 100;
            if (dd > maxDD) maxDD = dd;
        }
        
        this.elements.virtualBalance.textContent = `$${this.virtualBalance.toFixed(2)}`;
        this.elements.winRate.textContent = `${winRate.toFixed(1)}%`;
        this.elements.totalTrades.textContent = this.totalTrades;
        this.elements.totalPnl.textContent = `$${this.totalPnl.toFixed(2)}`;
        this.elements.profitFactor.textContent = profitFactor.toFixed(2);
        this.elements.maxDrawdown.textContent = `${maxDD.toFixed(1)}%`;
        this.elements.avgTrade.textContent = `$${avgTrade.toFixed(2)}`;
    }
    
    updateTradeList() {
        this.elements.tradeList.innerHTML = '';
        
        this.tradeHistory.forEach(trade => {
            const item = document.createElement('div');
            item.className = `trade-history-item ${trade.pnl >= 0 ? 'win' : 'loss'}`;
            
            const directionIcon = trade.direction === 'BUY' ? '↗' : '↘';
            const pnlClass = trade.pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
            const pnlSign = trade.pnl >= 0 ? '+' : '';
            
            item.innerHTML = `
                <div class="trade-summary">
                    <span class="trade-direction">${directionIcon} ${trade.direction}</span>
                    <span class="trade-entry">${trade.entry.toFixed(5)}</span>
                    <span class="trade-arrow">→</span>
                    <span class="trade-exit">${trade.exit?.toFixed(5) || '--.--'}</span>
                </div>
                <div class="trade-details">
                    <span class="trade-pnl ${pnlClass}">${pnlSign}$${Math.abs(trade.pnl).toFixed(2)}</span>
                    <span class="trade-duration">${Math.round(trade.duration || 0)}s</span>
                    <span class="trade-status ${trade.status === 'CLOSED_TP' ? 'status-tp' : 'status-sl'}">
                        ${trade.status === 'CLOSED_TP' ? 'TP' : 'SL'}
                    </span>
                </div>
            `;
            
            this.elements.tradeList.appendChild(item);
        });
    }
    
    updateConnectionStatus(status) {
        const statusMap = {
            connected: { dot: 'status-connected', text: 'Connected', color: '#00ff88' },
            disconnected: { dot: 'status-disconnected', text: 'Disconnected', color: '#ff6b6b' },
            error: { dot: 'status-error', text: 'Error', color: '#ff6b6b' },
            connecting: { dot: 'status-connecting', text: 'Connecting...', color: '#ffa94d' }
        };
        
        const currentStatus = statusMap[status] || statusMap.disconnected;
        
        this.elements.statusDot.className = `status-dot ${currentStatus.dot}`;
        this.elements.connectionStatus.textContent = currentStatus.text;
        this.elements.connectionStatus.style.color = currentStatus.color;
        
        this.elements.wsStatus.textContent = currentStatus.text;
        this.elements.wsStatus.className = status === 'connected' ? 'status-good' : 'status-bad';
    }
    
    startClocks() {
        // Live clock
        setInterval(() => {
            const now = new Date();
            this.elements.liveClock.textContent = now.toLocaleTimeString();
        }, 1000);
        
        // Uptime clock
        setInterval(() => {
            const uptime = Date.now() - this.startTime;
            const hours = Math.floor(uptime / 3600000);
            const minutes = Math.floor((uptime % 3600000) / 60000);
            const seconds = Math.floor((uptime % 60000) / 1000);
            
            this.elements.uptime.textContent = 
                `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
        
        // Prediction countdown
        setInterval(() => {
            if (this.analysisCountdown > 0) {
                this.analysisCountdown--;
                this.elements.predictionCountdown.textContent = `${this.analysisCountdown}s`;
                this.elements.nextAnalysis.textContent = `${this.analysisCountdown}s`;
            } else {
                this.analysisCountdown = 120;
            }
        }, 1000);
    }
    
    startTradeTimer() {
        clearInterval(this.tradeTimerInterval);
        this.tradeTimer = 0;
        
        this.tradeTimerInterval = setInterval(() => {
            this.tradeTimer++;
            const minutes = Math.floor(this.tradeTimer / 60);
            const seconds = this.tradeTimer % 60;
            
            this.elements.tradeTimer.textContent = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }
    
    startDynamicUpdates() {
        // Randomize some values for demo purposes
        setInterval(() => {
            if (Math.random() > 0.7) {
                this.elements.apiCalls.textContent = `${Math.floor(40 + Math.random() * 20)}/min`;
            }
        }, 5000);
        
        // Animate the status dot
        setInterval(() => {
            if (this.wsConnected) {
                this.elements.statusDot.style.boxShadow = 
                    `0 0 10px ${Math.random() > 0.5 ? '#00ff88' : '#00ffaa'}`;
            }
        }, 500);
    }
    
    startBot() {
        if (!this.wsConnected) {
            this.log('Cannot start bot: WebSocket not connected', 'error');
            return;
        }
        
        this.ws.send(JSON.stringify({ type: 'start_bot' }));
        this.log('Bot started', 'success');
        
        this.elements.startBtn.disabled = true;
        this.elements.stopBtn.disabled = false;
    }
    
    stopBot() {
        this.ws.send(JSON.stringify({ type: 'stop_bot' }));
        this.log('Bot stopped', 'warning');
        
        this.elements.startBtn.disabled = false;
        this.elements.stopBtn.disabled = true;
    }
    
    resetData() {
        if (confirm('Are you sure you want to reset all trading data? This cannot be undone.')) {
            this.ws.send(JSON.stringify({ type: 'reset_data' }));
            this.log('Data reset requested', 'warning');
        }
    }
    
    loadInitialData() {
        // Load from localStorage
        const savedData = localStorage.getItem('tradingDashboardData');
        
        if (savedData) {
            try {
                const data = JSON.parse(savedData);
                
                this.virtualBalance = data.balance || this.virtualBalance;
                this.totalTrades = data.totalTrades || 0;
                this.winningTrades = data.winningTrades || 0;
                this.totalPnl = data.totalPnl || 0;
                this.tradeHistory = data.tradeHistory || [];
                this.equityCurve = data.equityCurve || [this.virtualBalance];
                
                this.updatePerformance();
                this.updateTradeList();
                this.updateEquityChart();
                
                this.log('Data loaded from localStorage', 'success');
            } catch (e) {
                this.log('Failed to load saved data', 'error');
            }
        }
        
        // Save periodically
        setInterval(() => this.saveToLocalStorage(), 30000);
    }
    
    saveToLocalStorage() {
        const data = {
            balance: this.virtualBalance,
            totalTrades: this.totalTrades,
            winningTrades: this.winningTrades,
            totalPnl: this.totalPnl,
            tradeHistory: this.tradeHistory,
            equityCurve: this.equityCurve
        };
        
        localStorage.setItem('tradingDashboardData', JSON.stringify(data));
    }
    
    log(message, level = 'info') {
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${level}`;
        
        const timestamp = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
        
        const levelIcon = {
            success: '✓',
            error: '✗',
            warning: '⚠',
            info: 'ℹ'
        }[level] || 'ℹ';
        
        logEntry.innerHTML = `
            <span class="log-timestamp">[${timestamp}]</span>
            <span class="log-level">${levelIcon}</span>
            <span class="log-message">${message}</span>
        `;
        
        this.elements.logContainer.prepend(logEntry);
        
        // Keep only last 20 logs
        const logs = this.elements.logContainer.querySelectorAll('.log-entry');
        if (logs.length > 20) {
            logs[logs.length - 1].remove();
        }
        
        // Auto-scroll
        this.elements.logContainer.scrollTop = 0;
        
        // Play alert sound for important messages
        if (level === 'error' || level === 'success') {
            this.playSound('alertSound');
        }
    }
    
    playSound(soundId) {
        const sound = document.getElementById(soundId);
        if (sound) {
            sound.currentTime = 0;
            sound.play().catch(e => console.log('Audio play failed:', e));
        }
    }
    
    formatFeatureName(feature) {
        return feature
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase())
            .replace(/Bid Ask/, 'Bid/Ask')
            .replace(/Rsi/, 'RSI')
            .replace(/Macd/, 'MACD')
            .replace(/Vwap/, 'VWAP')
            .replace(/Atm/, 'ATM')
            .replace(/Tp/, 'TP')
            .replace(/Sl/, 'SL');
    }
    
    toggleSound() {
        const sounds = document.querySelectorAll('audio');
        const isMuted = sounds[0].muted;
        
        sounds.forEach(sound => {
            sound.muted = !isMuted;
        });
        
        this.log(`Sounds ${isMuted ? 'enabled' : 'muted'}`, 'info');
    }
}

// Initialize dashboard when page loads
window.addEventListener('DOMContentLoaded', () => {
    const dashboard = new TradingDashboard();
    window.dashboard = dashboard; // Make available in console for debugging
});