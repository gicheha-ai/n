// static/main.js - Main JavaScript for the keep-alive dashboard
document.addEventListener('DOMContentLoaded', function() {
    // Initialize variables
    let statsData = {};
    let historyData = [];
    let successChart = null;
    let updateInterval;
    let lastUpdateTime = null;
    
    // DOM Elements
    const alertElement = document.getElementById('alert');
    const statusElement = document.getElementById('status');
    const appUrlElement = document.getElementById('appUrl');
    const nextPingElement = document.getElementById('nextPing');
    const totalPingsElement = document.getElementById('totalPings');
    const successRateElement = document.getElementById('successRate');
    const uptimeElement = document.getElementById('uptime');
    const consecutiveFailuresElement = document.getElementById('consecutiveFailures');
    const pingIntervalElement = document.getElementById('pingInterval');
    const estimatedUsageElement = document.getElementById('estimatedUsage');
    const historyBodyElement = document.getElementById('historyBody');
    const lastUpdatedElement = document.getElementById('lastUpdated');
    
    // Control buttons
    const btnStart = document.getElementById('btnStart');
    const btnStop = document.getElementById('btnStop');
    const btnRestart = document.getElementById('btnRestart');
    const btnForcePing = document.getElementById('btnForcePing');
    
    // Initialize the dashboard
    initDashboard();
    
    // Set up auto-refresh
    startAutoRefresh();
    
    // Set up button event listeners
    setupButtonListeners();
    
    // Initialize the chart
    initChart();
    
    // Functions
    function initDashboard() {
        console.log('Initializing dashboard...');
        fetchData();
    }
    
    function startAutoRefresh() {
        // Fetch data every 10 seconds
        updateInterval = setInterval(fetchData, 10000);
        
        // Update next ping countdown every second
        setInterval(updateCountdown, 1000);
    }
    
    function setupButtonListeners() {
        // Add loading states to buttons
        [btnStart, btnStop, btnRestart, btnForcePing].forEach(btn => {
            btn.addEventListener('click', function() {
                if (!this.disabled) {
                    const originalHTML = this.innerHTML;
                    this.innerHTML = '<span class="loading"></span> Processing...';
                    this.disabled = true;
                    
                    // Restore button after 3 seconds if something goes wrong
                    setTimeout(() => {
                        if (this.disabled) {
                            this.innerHTML = originalHTML;
                            this.disabled = false;
                        }
                    }, 3000);
                }
            });
        });
    }
    
    function initChart() {
        const ctx = document.getElementById('successChart').getContext('2d');
        
        successChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Ping Success',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: '#f9fafb'
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(31, 41, 55, 0.9)',
                        titleColor: '#f9fafb',
                        bodyColor: '#f9fafb'
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(75, 85, 99, 0.3)'
                        },
                        ticks: {
                            color: '#9ca3af'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        max: 1,
                        grid: {
                            color: 'rgba(75, 85, 99, 0.3)'
                        },
                        ticks: {
                            color: '#9ca3af',
                            callback: function(value) {
                                return value === 1 ? 'Success' : value === 0 ? 'Failed' : '';
                            }
                        }
                    }
                }
            }
        });
    }
    
    function updateChart(history) {
        if (!successChart || !history || history.length === 0) return;
        
        // Take last 50 pings for the chart
        const recentHistory = history.slice(-50);
        
        // Update chart data
        successChart.data.labels = recentHistory.map(ping => ping.time_formatted);
        successChart.data.datasets[0].data = recentHistory.map(ping => ping.success ? 1 : 0);
        
        successChart.update();
    }
    
    async function fetchData() {
        try {
            // Fetch stats and history in parallel
            const [statsResponse, historyResponse] = await Promise.all([
                fetch('/api/stats'),
                fetch('/api/history?limit=50')
            ]);
            
            if (!statsResponse.ok || !historyResponse.ok) {
                throw new Error('Failed to fetch data');
            }
            
            statsData = await statsResponse.json();
            historyData = await historyResponse.json();
            
            lastUpdateTime = new Date();
            updateUI();
            
        } catch (error) {
            console.error('Error fetching data:', error);
            showAlert('Error fetching data. Please check console.', 'error');
        }
    }
    
    function updateUI() {
        // Update status
        if (statsData.is_running) {
            statusElement.innerHTML = '<span class="status-badge status-active"><i class="fas fa-play-circle"></i> ACTIVE</span>';
            btnStart.disabled = true;
            btnStop.disabled = false;
        } else {
            statusElement.innerHTML = '<span class="status-badge status-inactive"><i class="fas fa-stop-circle"></i> INACTIVE</span>';
            btnStart.disabled = false;
            btnStop.disabled = true;
        }
        
        // Always enable restart and force ping
        btnRestart.disabled = false;
        btnForcePing.disabled = false;
        
        // Update basic stats
        appUrlElement.textContent = statsData.app_url || 'Not available';
        totalPingsElement.textContent = statsData.total_pings || 0;
        successRateElement.textContent = `${statsData.success_rate || 0}%`;
        uptimeElement.textContent = `${statsData.uptime_hours || 0}h`;
        consecutiveFailuresElement.textContent = statsData.consecutive_failures || 0;
        pingIntervalElement.textContent = `${statsData.interval_minutes || 9.5} minutes`;
        
        // Calculate estimated monthly usage
        const hoursPerDay = 24;
        const daysPerMonth = 30;
        const estimatedMonthlyHours = hoursPerDay * daysPerMonth;
        estimatedUsageElement.textContent = `~${estimatedMonthlyHours} hours`;
        
        // Update history table
        updateHistoryTable();
        
        // Update chart
        updateChart(historyData);
        
        // Update last updated time
        if (lastUpdateTime) {
            const timeStr = lastUpdateTime.toLocaleTimeString();
            lastUpdatedElement.textContent = `Last updated: ${timeStr}`;
        }
    }
    
    function updateCountdown() {
        if (!statsData.next_ping_in) {
            nextPingElement.textContent = '--:--';
            return;
        }
        
        let seconds = Math.floor(statsData.next_ping_in);
        
        // If keep-alive is not running, show "N/A"
        if (!statsData.is_running) {
            nextPingElement.textContent = 'N/A';
            return;
        }
        
        // Update the countdown
        seconds--;
        
        if (seconds < 0) {
            // Reset to interval if negative
            seconds = statsData.interval_minutes * 60;
        }
        
        // Update the next ping time in statsData
        statsData.next_ping_in = seconds;
        
        // Format as MM:SS
        const minutes = Math.floor(seconds / 60);
        const secs = seconds % 60;
        nextPingElement.textContent = `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        
        // Color code based on time remaining
        if (seconds < 60) {
            nextPingElement.style.color = 'var(--warning-color)';
        } else if (seconds < 120) {
            nextPingElement.style.color = 'var(--info-color)';
        } else {
            nextPingElement.style.color = 'var(--success-color)';
        }
    }
    
    function updateHistoryTable() {
        if (!historyData || historyData.length === 0) {
            historyBodyElement.innerHTML = `
                <tr>
                    <td colspan="4" style="text-align: center;">No ping history available</td>
                </tr>
            `;
            return;
        }
        
        // Reverse so newest is first
        const reversedHistory = [...historyData].reverse();
        
        let html = '';
        reversedHistory.forEach(ping => {
            const time = new Date(ping.timestamp * 1000);
            const timeStr = time.toLocaleTimeString();
            
            const statusBadge = ping.success 
                ? '<span class="success-badge"><i class="fas fa-check-circle"></i> Success</span>'
                : '<span class="failed-badge"><i class="fas fa-times-circle"></i> Failed</span>';
            
            const responseTime = ping.response_time 
                ? `${(ping.response_time * 1000).toFixed(0)}ms` 
                : 'N/A';
            
            const details = ping.error 
                ? `<span style="color: var(--danger-color);">${ping.error}</span>`
                : (ping.status_code ? `HTTP ${ping.status_code}` : 'OK');
            
            html += `
                <tr>
                    <td>${timeStr}</td>
                    <td>${statusBadge}</td>
                    <td>${responseTime}</td>
                    <td>${details}</td>
                </tr>
            `;
        });
        
        historyBodyElement.innerHTML = html;
    }
    
    async function controlKeepAlive(action) {
        try {
            const response = await fetch('/api/control', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ action: action })
            });
            
            const result = await response.json();
            
            if (result.success) {
                showAlert(result.message, 'success');
                
                // Refresh data after a short delay
                setTimeout(fetchData, 1000);
                
                // Special handling for force ping
                if (action === 'force_ping') {
                    // Update UI immediately to show ping was sent
                    statsData.total_pings = (statsData.total_pings || 0) + 1;
                    updateUI();
                    
                    // Then refresh data after ping completes
                    setTimeout(fetchData, 2000);
                }
            } else {
                showAlert(result.message || 'Action failed', 'error');
            }
            
        } catch (error) {
            console.error('Error controlling keep-alive:', error);
            showAlert('Error: ' + error.message, 'error');
        } finally {
            // Re-enable buttons
            [btnStart, btnStop, btnRestart, btnForcePing].forEach(btn => {
                btn.disabled = false;
                // Reset button text based on current action
                if (btn.id === 'btnStart') btn.innerHTML = '<i class="fas fa-play"></i> Start';
                if (btn.id === 'btnStop') btn.innerHTML = '<i class="fas fa-stop"></i> Stop';
                if (btn.id === 'btnRestart') btn.innerHTML = '<i class="fas fa-redo"></i> Restart';
                if (btn.id === 'btnForcePing') btn.innerHTML = '<i class="fas fa-bolt"></i> Force Ping';
            });
        }
    }
    
    function showAlert(message, type) {
        alertElement.textContent = message;
        alertElement.className = `alert alert-${type}`;
        alertElement.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            alertElement.style.display = 'none';
        }, 5000);
    }
    
    // Make controlKeepAlive available globally for onclick handlers
    window.controlKeepAlive = controlKeepAlive;
    
    // Export functions for debugging if needed
    window.dashboard = {
        fetchData,
        controlKeepAlive,
        getStats: () => statsData,
        getHistory: () => historyData
    };
});