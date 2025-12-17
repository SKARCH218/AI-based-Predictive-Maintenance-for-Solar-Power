"""
테스트 데이터 생성기

태양광 패널이 연결되지 않은 상태에서 시뮬레이션 데이터를 생성합니다.
정상, 경고, 이상 상태를 포함한 다양한 패턴의 데이터를 생성하여 DB에 저장합니다.
"""

import sqlite3
import random
import time
import math
from datetime import datetime, timedelta
import numpy as np


class TestDataGenerator:
    """테스트용 태양광 데이터 생성기"""
    
    def __init__(self, db_path='solardata.db'):
        self.db_path = db_path
        self.boards = ['BOARD_001', 'BOARD_002']
        self.axes = {
            'x': ['x1', 'x2', 'x3', 'x4', 'x5'],
            'y': ['y1', 'y2', 'y3', 'y4', 'y5', 'y6']
        }
        
        # 시뮬레이션 상태 (각 축별로 다른 상태 유지)
        self.axis_states = {}
        for board in self.boards:
            self.axis_states[board] = {}
            for axis in self.axes['x'] + self.axes['y']:
                # 80% 정상, 15% 경고, 5% 이상
                rand = random.random()
                if rand < 0.8:
                    state = 'NORMAL'
                elif rand < 0.95:
                    state = 'WARNING'
                else:
                    state = 'ALERT'
                self.axis_states[board][axis] = state
        
        self._init_db()
    
    def _init_db(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # power_data 테이블
        cur.execute('''
            CREATE TABLE IF NOT EXISTS power_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                board_id TEXT,
                axis TEXT,
                bus_voltage REAL,
                shunt_voltage REAL,
                load_voltage REAL,
                current_ma REAL,
                power_mw REAL,
                accumulated_energy_mwh REAL
            )
        ''')
        
        # predictions 테이블
        cur.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                board_id TEXT,
                power_mw REAL,
                status TEXT,
                severity REAL,
                cells TEXT
            )
        ''')
        
        # cnn_predictions 테이블
        cur.execute('''
            CREATE TABLE IF NOT EXISTS cnn_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                board_id TEXT,
                axis TEXT,
                status TEXT,
                confidence REAL,
                probabilities TEXT,
                mean_power REAL,
                std_power REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✓ 데이터베이스 초기화 완료")
    
    def generate_power_data(self, board_id, axis, state='NORMAL'):
        """
        축별 전력 데이터 생성
        
        Args:
            board_id: 보드 ID
            axis: 축 이름 (x1, y2 등)
            state: 상태 (NORMAL, WARNING, ALERT)
        
        Returns:
            전력 데이터 딕셔너리
        """
        # 시간대별 일조량 시뮬레이션 (0-23시)
        hour = datetime.now().hour
        
        # 기본 일조 패턴 (정현파)
        base_intensity = max(0, math.sin((hour - 6) * math.pi / 12))
        
        # 상태별 출력 조정
        if state == 'NORMAL':
            # 정상: 높은 출력
            power_factor = 0.85 + random.uniform(-0.05, 0.05)
            noise = random.gauss(0, 0.02)
        elif state == 'WARNING':
            # 경고: 중간 출력 + 변동성
            power_factor = 0.5 + random.uniform(-0.15, 0.15)
            noise = random.gauss(0, 0.05)
        else:  # ALERT
            # 이상: 낮은 출력 + 큰 변동성
            power_factor = 0.2 + random.uniform(-0.1, 0.1)
            noise = random.gauss(0, 0.08)
        
        # 최종 출력 계산
        max_power = 5000  # mW
        power_mw = max(0, max_power * base_intensity * power_factor + noise * 500)
        
        # 전압/전류 계산
        voltage = 12.0 + random.uniform(-0.5, 0.5)
        current_ma = (power_mw / voltage) if voltage > 0 else 0
        
        return {
            'board_id': board_id,
            'axis': axis,
            'bus_voltage': voltage,
            'shunt_voltage': random.uniform(-0.1, 0.1),
            'load_voltage': voltage + random.uniform(-0.2, 0.2),
            'current_ma': current_ma,
            'power_mw': power_mw,
            'accumulated_energy_mwh': power_mw / 1000  # 간단히 계산
        }
    
    def insert_power_data(self, data):
        """전력 데이터를 DB에 삽입"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute('''
            INSERT INTO power_data (
                board_id, axis, bus_voltage, shunt_voltage, load_voltage,
                current_ma, power_mw, accumulated_energy_mwh
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['board_id'],
            data['axis'],
            data['bus_voltage'],
            data['shunt_voltage'],
            data['load_voltage'],
            data['current_ma'],
            data['power_mw'],
            data['accumulated_energy_mwh']
        ))
        
        conn.commit()
        conn.close()
    
    def generate_prediction_data(self, board_id, power_mw):
        """예측 데이터 생성 (기존 AI 모듈 시뮬레이션)"""
        # 출력에 따른 상태 결정
        if power_mw > 3000:
            status = 'NORMAL'
            severity = random.uniform(0.0, 0.3)
        elif power_mw > 1500:
            status = 'WARNING'
            severity = random.uniform(0.3, 0.7)
        else:
            status = 'ALERT'
            severity = random.uniform(0.7, 1.0)
        
        # 랜덤 셀 선택 (이상 셀)
        cells = []
        if status in ['WARNING', 'ALERT']:
            num_cells = random.randint(1, 3)
            for _ in range(num_cells):
                cells.append({
                    'row': random.randint(1, 5),
                    'col': random.randint(1, 6)
                })
        
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        import json
        cur.execute('''
            INSERT INTO predictions (board_id, power_mw, status, severity, cells)
            VALUES (?, ?, ?, ?, ?)
        ''', (board_id, power_mw, status, severity, json.dumps(cells)))
        
        conn.commit()
        conn.close()
    
    def generate_batch(self):
        """한 번의 배치 데이터 생성 (모든 보드와 축)"""
        for board in self.boards:
            total_power = 0
            for axis in self.axes['x'] + self.axes['y']:
                state = self.axis_states[board][axis]
                data = self.generate_power_data(board, axis, state)
                self.insert_power_data(data)
                total_power += data['power_mw']
            
            # 전체 보드 예측 데이터
            avg_power = total_power / len(self.axes['x'] + self.axes['y'])
            self.generate_prediction_data(board, avg_power)
        
        # 랜덤하게 상태 변경 (5% 확률)
        for board in self.boards:
            for axis in self.axes['x'] + self.axes['y']:
                if random.random() < 0.05:
                    states = ['NORMAL', 'WARNING', 'ALERT']
                    self.axis_states[board][axis] = random.choice(states)
    
    def run_continuous(self, interval=2):
        """연속 데이터 생성 (테스트 서버용)"""
        print(f"✓ 테스트 데이터 생성기 시작 (간격: {interval}초)")
        print(f"  보드: {', '.join(self.boards)}")
        print(f"  축: {len(self.axes['x'] + self.axes['y'])}개")
        print()
        
        try:
            batch_count = 0
            while True:
                self.generate_batch()
                batch_count += 1
                
                if batch_count % 10 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"{batch_count}개 배치 생성 완료")
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n테스트 데이터 생성기 종료")
    
    def generate_initial_history(self, hours=2):
        """초기 히스토리 데이터 생성 (차트용)"""
        print(f"초기 {hours}시간 히스토리 생성 중...")
        
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # 기존 데이터 삭제
        cur.execute("DELETE FROM power_data")
        cur.execute("DELETE FROM predictions")
        cur.execute("DELETE FROM cnn_predictions")
        conn.commit()
        
        # 과거 데이터 생성
        num_points = hours * 60 // 2  # 2분 간격
        start_time = datetime.now() - timedelta(hours=hours)
        
        for i in range(num_points):
            timestamp = start_time + timedelta(minutes=i*2)
            
            for board in self.boards:
                total_power = 0
                for axis in self.axes['x'] + self.axes['y']:
                    state = self.axis_states[board][axis]
                    data = self.generate_power_data(board, axis, state)
                    
                    cur.execute('''
                        INSERT INTO power_data (
                            timestamp, board_id, axis, bus_voltage, shunt_voltage,
                            load_voltage, current_ma, power_mw, accumulated_energy_mwh
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        data['board_id'], data['axis'], data['bus_voltage'],
                        data['shunt_voltage'], data['load_voltage'], data['current_ma'],
                        data['power_mw'], data['accumulated_energy_mwh']
                    ))
                    
                    total_power += data['power_mw']
                
                avg_power = total_power / len(self.axes['x'] + self.axes['y'])
                
                # 예측 데이터
                if avg_power > 3000:
                    status = 'NORMAL'
                    severity = random.uniform(0.0, 0.3)
                elif avg_power > 1500:
                    status = 'WARNING'
                    severity = random.uniform(0.3, 0.7)
                else:
                    status = 'ALERT'
                    severity = random.uniform(0.7, 1.0)
                
                cur.execute('''
                    INSERT INTO predictions (timestamp, board_id, power_mw, status, severity, cells)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (timestamp.strftime('%Y-%m-%d %H:%M:%S'), board, avg_power, status, severity, '[]'))
        
        conn.commit()
        conn.close()
        
        print(f"✓ {num_points}개 히스토리 포인트 생성 완료")


if __name__ == '__main__':
    import sys
    
    generator = TestDataGenerator()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--init':
        # 초기 히스토리만 생성하고 종료
        generator.generate_initial_history(hours=2)
    else:
        # 초기 히스토리 + 연속 생성
        generator.generate_initial_history(hours=2)
        generator.run_continuous(interval=2)
