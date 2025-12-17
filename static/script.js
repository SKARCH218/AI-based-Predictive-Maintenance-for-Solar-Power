document.addEventListener('DOMContentLoaded', function() {
    const busVoltageElem = document.getElementById('busVoltage');
    const shuntVoltageElem = document.getElementById('shuntVoltage');
    const loadVoltageElem = document.getElementById('loadVoltage');
    const current_mAElem = document.getElementById('current_mA');
    const power_mWElem = document.getElementById('power_mW');
    const accumulatedEnergy_mWhElem = document.getElementById('accumulatedEnergy_mWh');
    const lastUpdatedElem = document.getElementById('last_updated');
    const statusBadge = document.getElementById('status_badge');
    const boardSelect = document.getElementById('boardSelect');
    let selectedBoard = '';

    async function loadBoards() {
        try {
            const res = await fetch('/api/boards');
            const boards = await res.json();
            if (!boardSelect) return;
            boardSelect.innerHTML = '';
            const allOpt = document.createElement('option');
            allOpt.value = '';
            allOpt.textContent = '전체';
            boardSelect.appendChild(allOpt);
            (boards || []).forEach(b => {
                const opt = document.createElement('option');
                opt.value = b;
                opt.textContent = b;
                boardSelect.appendChild(opt);
            });
            boardSelect.value = selectedBoard;
        } catch (e) {
            console.error('보드 목록 로딩 실패', e);
        }
    }

    let powerChart;
    let predictionHistoryChart;

    // 차트 초기화 함수
    function initChart(initialData) {
        const ctx = document.getElementById('powerChart').getContext('2d');
        powerChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: initialData.map(d => new Date(d.x).toLocaleTimeString()), // 시간 레이블
                datasets: [{
                    label: '발전량 (mW)',
                    data: initialData.map(d => d.y),
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'category',
                        title: {
                            display: true,
                            text: '시간'
                        },
                        ticks: {
                            color: '#ffffff'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: '발전량 (mW)'
                        },
                        beginAtZero: true,
                        ticks: {
                            color: '#ffffff'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                }
            }
        });
    }

    // 실시간 데이터 업데이트 함수
    async function updateRealtimeData() {
        try {
            const response = await fetch(`/api/data${selectedBoard ? `?board_id=${encodeURIComponent(selectedBoard)}` : ''}`);
            const data = await response.json();

            if (Object.keys(data).length > 0) {
                busVoltageElem.textContent = data.bus_voltage !== undefined ? data.bus_voltage.toFixed(2) : 'N/A';
                shuntVoltageElem.textContent = data.shunt_voltage !== undefined ? data.shunt_voltage.toFixed(2) : 'N/A';
                loadVoltageElem.textContent = data.load_voltage !== undefined ? data.load_voltage.toFixed(2) : 'N/A';
                current_mAElem.textContent = data.current_ma !== undefined ? data.current_ma.toFixed(2) : 'N/A';
                power_mWElem.textContent = data.power_mw !== undefined ? data.power_mw.toFixed(2) : 'N/A';
                accumulatedEnergy_mWhElem.textContent = data.accumulated_energy_mwh !== undefined ? data.accumulated_energy_mwh.toFixed(2) : 'N/A';
                lastUpdatedElem.textContent = new Date(data.timestamp).toLocaleString();
            }
        } catch (error) {
            console.error('Error fetching real-time data:', error);
            lastUpdatedElem.textContent = '데이터 로드 실패';
        }
    }

    // 최신 예측 상태 업데이트
    async function updatePredictionStatus() {
        try {
            const res = await fetch(`/api/prediction/latest${selectedBoard ? `?board_id=${encodeURIComponent(selectedBoard)}` : ''}`);
            const pred = await res.json();
            if (pred && pred.status) {
                const status = String(pred.status).toUpperCase();
                statusBadge.textContent = status;
                statusBadge.classList.remove('normal', 'warning', 'alert');
                if (status === 'NORMAL') statusBadge.classList.add('normal');
                else if (status === 'WARNING') statusBadge.classList.add('warning');
                else if (status === 'ALERT') statusBadge.classList.add('alert');
                // 예측 셀 강조
                const grid = document.getElementById('panelGrid');
                if (grid) {
                    Array.from(grid.children).forEach(c => c.classList.remove('alert'));
                    let cells = [];
                    const raw = pred.cells;
                    if (raw) {
                        if (typeof raw === 'string') {
                            try { cells = JSON.parse(raw); } catch { cells = []; }
                        } else if (Array.isArray(raw)) {
                            cells = raw;
                        } else {
                            cells = [];
                        }
                    }
                    (cells || []).forEach(cell => {
                        const r = Number(cell.row) - 1;
                        const c = Number(cell.col) - 1;
                        if (!Number.isInteger(r) || !Number.isInteger(c)) return;
                        const idx = r * 6 + c;
                        const el = grid.children[idx];
                        if (el) el.classList.add('alert');
                    });
                }
            }
        } catch (e) {
            console.error('Prediction fetch error', e);
        }
    }

    // 축별 최신값 렌더링 + 보드 라벨 업데이트
    async function updateAxisLatest() {
        try {
            const res = await fetch(`/api/axis/latest${selectedBoard ? `?board_id=${encodeURIComponent(selectedBoard)}` : ''}`);
            const rows = await res.json();
            const rowLabels = document.getElementById('rowLabels');
            const colLabels = document.getElementById('colLabels');
            if (!rowLabels || !colLabels) return;
            rowLabels.innerHTML = '';
            colLabels.innerHTML = '';
            const latestByAxis = {};
            // 중복 축 제거: 동일 축명에 대해 id 최댓값 기준 최신값만 유지
            const pickLatest = {};
            (rows || []).forEach(r => {
                const k = (r.axis || '').toLowerCase();
                if (!k) return;
                if (!pickLatest[k] || (r.id ?? 0) > (pickLatest[k].id ?? -1)) {
                    pickLatest[k] = r;
                }
            });
            Object.values(pickLatest).forEach(r => {
                latestByAxis[(r.axis || '').toLowerCase()] = r;
            });

            // 패널 그리드 컬러링 (간단 매핑 예시)
            const grid = document.getElementById('panelGrid');
            if (grid && !grid.dataset.built) {
                // 5x6 = 30 셀 생성
                for (let i = 0; i < 30; i++) {
                    const cell = document.createElement('div');
                    cell.className = 'panel-cell';
                    const label = document.createElement('span');
                    label.className = 'label';
                    label.textContent = `(${Math.floor(i/6)+1}, ${i%6+1})`;
                    cell.appendChild(label);
                    grid.appendChild(cell);
                }
                grid.dataset.built = '1';
            }
            if (grid) {
                const cells = Array.from(grid.children);
                // 매우 단순한 예: x1~x5는 행 강도, y1~y6은 열 강도처럼 합성
                const xKeys = ['x1','x2','x3','x4','x5'];
                const yKeys = ['y1','y2','y3','y4','y5','y6'];
                // 정규화 함수
                const val = k => Number((latestByAxis[k]?.power_mw) ?? 0);
                const maxX = Math.max(...xKeys.map(val), 1);
                const maxY = Math.max(...yKeys.map(val), 1);
                for (let r = 0; r < 5; r++) {
                    for (let c = 0; c < 6; c++) {
                        const idx = r * 6 + c;
                        const vx = val(xKeys[r]) / maxX; // 0~1
                        const vy = val(yKeys[c]) / maxY; // 0~1
                        const v = Math.max(0, Math.min(1, (vx + vy) / 2));
                        const hue = 140 * v; // 0(빨강쪽)~140(초록쪽) 계열
                        const a = 0.12 + 0.25 * v; // 더 낮은 투명도로 완화
                        const cell = cells[idx];
                        if (cell) {
                                        // 임계값에 의해 클래스 지정: v < 0.4 -> normal, 0.4<=v<0.75 -> warning, >=0.75 -> alert
                                        cell.classList.remove('normal', 'warning', 'alert');
                                        if (v >= 0.75) {
                                            cell.classList.add('alert');
                                        } else if (v >= 0.4) {
                                            cell.classList.add('warning');
                                        } else {
                                            cell.classList.add('normal');
                                        }
                        }
                    }
                }

                // 행/열 라벨 갱신
                const fmt = (axis, v) => `${axis}: ${Number(v).toFixed(1)} mW`;
                rowLabels.innerHTML = '';
                xKeys.forEach(k => {
                    const div = document.createElement('div');
                    div.className = 'row-label';
                    div.textContent = fmt(k, val(k));
                    rowLabels.appendChild(div);
                });
                colLabels.innerHTML = '';
                yKeys.forEach(k => {
                    const div = document.createElement('div');
                    div.className = 'col-label';
                    div.textContent = fmt(k, val(k));
                    colLabels.appendChild(div);
                });
            }
        } catch (e) {
            console.error('축 데이터 로드 실패', e);
        }
    }

    // 차트 데이터 업데이트 함수
    async function updateChart() {
        try {
            const response = await fetch(`/api/data${selectedBoard ? `?board_id=${encodeURIComponent(selectedBoard)}` : ''}`);
            const newData = await response.json();

            if (powerChart && newData && newData.timestamp && newData.power_mw !== undefined) {
                const newLabel = new Date(newData.timestamp).toLocaleTimeString();
                const newDataPoint = newData.power_mw;

                powerChart.data.labels.push(newLabel);
                powerChart.data.datasets[0].data.push(newDataPoint);

                // 차트에 표시할 최대 데이터 포인트 수
                const maxDataPoints = 20;
                if (powerChart.data.labels.length > maxDataPoints) {
                    powerChart.data.labels.shift();
                    powerChart.data.datasets[0].data.shift();
                }

                powerChart.update();
            }
        } catch (error) {
            console.error('Error fetching new data for chart:', error);
        }
    }

    // 초기 차트 데이터 로드 함수
    async function initialChartLoad() {
        try {
            const response = await fetch(`/api/history${selectedBoard ? `?board_id=${encodeURIComponent(selectedBoard)}` : ''}`);
            const historyData = await response.json();
            historyData.sort((a, b) => new Date(a.x) - new Date(b.x));
            initChart(historyData);
        } catch (error) {
            console.error('Error fetching history data:', error);
        }
    }

    // 초기 데이터 로드: 보드 목록 먼저 로드
    loadBoards().then(() => initialChartLoad());

    // 1초마다 실시간 데이터 업데이트
    setInterval(updateRealtimeData, 1000);
    // 5초마다 차트 데이터 업데이트
    setInterval(updateChart, 5000);
    // 5초마다 예측 상태 업데이트
    setInterval(updatePredictionStatus, 5000);
    // 5초마다 축별 최신값 업데이트
    setInterval(updateAxisLatest, 5000);

    // 수동 예측 버튼 제거됨

    if (boardSelect) {
        boardSelect.addEventListener('change', () => {
            selectedBoard = boardSelect.value;
            updateRealtimeData();
            initialChartLoad();
            updatePredictionStatus();
            updateAxisLatest();
        });
    }

    // 탭 전환
    const tabLinks = document.querySelectorAll('.tab-link');
    const panes = document.querySelectorAll('.tab-pane');
    tabLinks.forEach(btn => {
        btn.addEventListener('click', () => {
            tabLinks.forEach(b => b.classList.remove('active'));
            panes.forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            const target = document.querySelector(btn.dataset.target);
            if (target) target.classList.add('active');
        });
    });

    // ===== 예지보전 탭 기능 =====
    
    // CNN 예측 결과 업데이트
    async function updateCNNPrediction() {
        try {
            const res = await fetch(`/api/cnn/predict${selectedBoard ? `?board_id=${encodeURIComponent(selectedBoard)}` : ''}`);
            const data = await res.json();
            
            if (data && data.status) {
                // 상태 라벨 업데이트
                const statusLabel = document.getElementById('cnn-status-label');
                const statusIndicator = document.getElementById('cnn-status-indicator');
                const confidenceBadge = document.getElementById('cnn-confidence');
                
                statusLabel.textContent = data.status;
                statusLabel.className = `status-label ${data.status}`;
                confidenceBadge.textContent = `${(data.confidence * 100).toFixed(1)}%`;
                
                // 상태에 따라 인디케이터 색상 변경
                statusIndicator.style.borderColor = getStatusColor(data.status);
                
                // 메타 정보 업데이트
                document.getElementById('cnn-model-version').textContent = data.model_version || '-';
                document.getElementById('cnn-last-update').textContent = new Date(data.timestamp).toLocaleString();
                document.getElementById('cnn-board-id').textContent = data.board_id || '전체';
                
                // 확신도 바 업데이트
                if (data.probabilities) {
                    updateConfidenceBars(data.probabilities);
                }
            }
        } catch (e) {
            console.error('CNN 예측 업데이트 실패:', e);
        }
    }
    
    // 상태별 색상 반환
    function getStatusColor(status) {
        const colors = {
            'NORMAL': '#2ea043',
            // WARNING 색상: 기존 주황(#d29922) -> 노랑(#ffd400)
            'WARNING': '#ffd400',
            'ALERT': '#d73a49',
            'CRITICAL': '#dc2626'
        };
        return colors[status] || '#777';
    }
    
    // 확신도 바 업데이트
    function updateConfidenceBars(probabilities) {
        const classes = ['NORMAL', 'WARNING', 'ALERT', 'CRITICAL'];
        classes.forEach(cls => {
            const prob = probabilities[cls] || 0;
            const percentage = (prob * 100).toFixed(1);
            
            const fillElem = document.getElementById(`conf-${cls.toLowerCase()}`);
            const valueElem = document.getElementById(`conf-${cls.toLowerCase()}-val`);
            
            if (fillElem) fillElem.style.width = `${percentage}%`;
            if (valueElem) valueElem.textContent = `${percentage}%`;
        });
    }
    
    // 패턴 이미지 업데이트
    async function updatePatternImage() {
        try {
            const res = await fetch(`/api/cnn/pattern${selectedBoard ? `?board_id=${encodeURIComponent(selectedBoard)}` : ''}`);
            const data = await res.json();
            
            if (data && data.image) {
                const canvas = document.getElementById('patternCanvas');
                const ctx = canvas.getContext('2d');
                
                // Base64 이미지 로드
                const img = new Image();
                img.onload = function() {
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                img.src = 'data:image/png;base64,' + data.image;
            }
        } catch (e) {
            console.error('패턴 이미지 로드 실패:', e);
        }
    }
    
    // 예측 히스토리 차트 초기화
    function initPredictionHistoryChart() {
        const ctx = document.getElementById('predictionHistoryChart');
        if (!ctx) return;
        
        predictionHistoryChart = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'NORMAL',
                        data: [],
                        borderColor: '#2ea043',
                        backgroundColor: 'rgba(46, 160, 67, 0.1)',
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: 'WARNING',
                        data: [],
                        // WARNING: orange -> yellow
                        borderColor: '#ffd400',
                        backgroundColor: 'rgba(255, 212, 0, 0.12)',
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: 'ALERT',
                        data: [],
                        borderColor: '#d73a49',
                        backgroundColor: 'rgba(215, 58, 73, 0.1)',
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: 'CRITICAL',
                        data: [],
                        borderColor: '#dc2626',
                        backgroundColor: 'rgba(220, 38, 38, 0.1)',
                        fill: true,
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        ticks: { color: '#ffffff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        stacked: true,
                        ticks: { color: '#ffffff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        max: 1
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#ffffff' }
                    }
                }
            }
        });
    }
    
    // 예측 히스토리 차트 업데이트
    async function updatePredictionHistory() {
        try {
            const res = await fetch(`/api/cnn/history${selectedBoard ? `?board_id=${encodeURIComponent(selectedBoard)}` : ''}`);
            const data = await res.json();
            
            if (data && data.length > 0 && predictionHistoryChart) {
                // 최근 20개 데이터
                const recent = data.slice(-20);
                
                predictionHistoryChart.data.labels = recent.map(d => 
                    new Date(d.timestamp).toLocaleTimeString()
                );
                
                // 각 클래스별 확률 데이터
                const classes = ['NORMAL', 'WARNING', 'ALERT', 'CRITICAL'];
                classes.forEach((cls, idx) => {
                    predictionHistoryChart.data.datasets[idx].data = recent.map(d => 
                        d.probabilities ? (d.probabilities[cls] || 0) : 0
                    );
                });
                
                predictionHistoryChart.update();
            }
        } catch (e) {
            console.error('예측 히스토리 로드 실패:', e);
        }
    }
    
    // 모델 성능 지표 업데이트
    async function updateModelMetrics() {
        try {
            const res = await fetch('/api/cnn/model-info');
            const data = await res.json();
            
            if (data) {
                document.getElementById('model-accuracy').textContent = 
                    data.accuracy ? `${data.accuracy.toFixed(1)}%` : '-';
                document.getElementById('model-f1').textContent = 
                    data.f1_score ? data.f1_score.toFixed(3) : '-';
                document.getElementById('model-samples').textContent = 
                    data.prediction_samples || '-';
                document.getElementById('model-training-size').textContent = 
                    data.training_samples || '-';
            }
        } catch (e) {
            console.error('모델 메트릭 로드 실패:', e);
        }
    }
    
    // 교체 날짜 트렌드 차트 초기화
    let replacementTrendChart;
    function initReplacementTrendChart(trendData) {
        const ctx = document.getElementById('replacementTrendChart');
        if (!ctx) return;
        
        if (replacementTrendChart) {
            replacementTrendChart.destroy();
        }
        
        replacementTrendChart = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: trendData.map((_, i) => `${i + 1}`),
                datasets: [{
                    label: '위험도 점수',
                    data: trendData,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: '시간 (최근 → 과거)', color: '#ffffff' },
                        ticks: { color: '#ffffff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        title: { display: true, text: '위험도 점수', color: '#ffffff' },
                        ticks: { color: '#ffffff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        min: 0,
                        max: 100
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#ffffff' }
                    }
                }
            }
        });
    }
    
    // 교체 날짜 예측 업데이트
    async function updateReplacementPrediction() {
        try {
            const res = await fetch(`/api/cnn/replacement-prediction${selectedBoard ? `?board_id=${encodeURIComponent(selectedBoard)}` : ''}`);
            const data = await res.json();
            
            if (data && !data.error) {
                // 교체 날짜
                document.getElementById('replacement-date').textContent = data.replacement_date;
                document.getElementById('replacement-confidence').textContent = `신뢰도: ${data.confidence}`;
                
                // 상세 정보
                document.getElementById('current-health-status').textContent = data.current_status;
                document.getElementById('days-remaining').textContent = `${data.days_remaining}일`;
                document.getElementById('risk-level').textContent = data.risk_level;
                document.getElementById('degradation-rate').textContent = data.degradation_rate;
                
                // 위험도에 따른 스타일링
                const dateElem = document.getElementById('replacement-date');
                dateElem.className = 'replacement-date';
                if (data.days_remaining === 0) {
                    dateElem.classList.add('critical');
                } else if (data.days_remaining <= 7) {
                    dateElem.classList.add('warning');
                } else if (data.days_remaining <= 30) {
                    dateElem.classList.add('alert');
                } else {
                    dateElem.classList.add('normal');
                }
                
                // 트렌드 차트 업데이트
                if (data.trend_data && data.trend_data.length > 0) {
                    initReplacementTrendChart(data.trend_data.reverse());
                }
            } else {
                document.getElementById('replacement-date').textContent = '데이터 부족';
                document.getElementById('replacement-confidence').textContent = '';
            }
        } catch (e) {
            console.error('교체 날짜 예측 로드 실패:', e);
            document.getElementById('replacement-date').textContent = '로드 실패';
        }
    }
    
    // 예지보전 탭 데이터 초기화
    function initPredictionTab() {
        initPredictionHistoryChart();
        updateCNNPrediction();
        updatePatternImage();
        updatePredictionHistory();
        updateModelMetrics();
        updateReplacementPrediction();
    }
    
    // 5초마다 예지보전 데이터 업데이트
    setInterval(() => {
        const predTab = document.getElementById('tab-prediction');
        if (predTab && predTab.classList.contains('active')) {
            updateCNNPrediction();
            updatePredictionHistory();
            updateReplacementPrediction();
        }
    }, 5000);
    
    // 30초마다 패턴 이미지 및 모델 메트릭 업데이트
    setInterval(() => {
        const predTab = document.getElementById('tab-prediction');
        if (predTab && predTab.classList.contains('active')) {
            updatePatternImage();
            updateModelMetrics();
        }
    }, 30000);
    
    // 예지보전 탭 초기 로드
    setTimeout(initPredictionTab, 1000);
});
