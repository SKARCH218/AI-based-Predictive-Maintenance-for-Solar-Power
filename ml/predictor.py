"""
실시간 예측 모듈

기능:
1. 실시간 데이터 → 이미지 변환 → CNN 예측
2. 예측 결과 DB 저장
3. 웹 API 통합
"""

import torch
import numpy as np
from typing import Dict, Optional, List
import sqlite3
from datetime import datetime
from pathlib import Path

from ml.image_generator import SolarDataImageGenerator
from ml.cnn_model import ModelManager, SolarPanelCNN, SolarPanelCNNWithAttention


class RealTimePredictor:
    """실시간 태양광 패널 이상 예측기"""
    
    LABEL_MAP = {
        0: 'NORMAL',
        1: 'WARNING',
        2: 'ALERT',
        3: 'CRITICAL'
    }
    
    def __init__(
        self,
        model_version: str = 'latest',
        db_path: str = 'solardata.db',
        device: str = 'cpu',
        confidence_threshold: float = 0.6
    ):
        """
        Args:
            model_version: 사용할 모델 버전
            db_path: 데이터베이스 경로
            device: 'cpu' 또는 'cuda'
            confidence_threshold: 확신도 임계값
        """
        self.db_path = db_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # 모델 로드
        self.model_manager = ModelManager()
        
        if model_version == 'latest':
            versions = self.model_manager.list_versions()
            if not versions:
                raise FileNotFoundError("저장된 모델이 없습니다. 먼저 학습을 진행하세요.")
            model_version = versions[-1]
        
        self.model_version = model_version
        self.model = self.model_manager.load_model(model_version, device=device)
        self.model.eval()
        
        # 모델 메타데이터 로드
        self.model_metadata = self.model_manager.get_metadata(model_version)
        
        # 이미지 생성기
        self.image_generator = SolarDataImageGenerator(db_path=db_path, method='multi')
        
        print(f"✓ 실시간 예측기 준비 완료 (모델: {model_version})")
    
    def predict_current_state(
        self, 
        board_id: Optional[str] = None,
        axis: Optional[str] = None,
        window_size: int = 120
    ) -> Dict:
        """
        현재 상태 예측
        
        Args:
            board_id: 특정 보드 (None이면 전체)
            axis: 특정 축 (None이면 전체)
            window_size: 시계열 윈도우 크기
        
        Returns:
            예측 결과 딕셔너리
        """
        # 최근 데이터로 이미지 생성
        samples = self.image_generator.fetch_timeseries(
            board_id=board_id,
            axis=axis,
            window_size=window_size,
            limit=1
        )
        
        if not samples:
            return {
                'status': 'NO_DATA',
                'message': '충분한 데이터가 없습니다.',
                'board_id': board_id,
                'axis': axis
            }
        
        image, metadata = samples[0]
        
        # 이미지 → Tensor
        if image.ndim == 2:
            image = image[np.newaxis, :, :]
        else:
            image = np.transpose(image, (2, 0, 1))
        
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)
        
        # 예측
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = int(outputs.argmax(dim=1).cpu().numpy()[0])
        
        predicted_label = self.LABEL_MAP[predicted_class]
        confidence = float(probabilities[predicted_class])
        
        # 결과 구성
        result = {
            'status': predicted_label,
            'confidence': confidence,
            'probabilities': {
                self.LABEL_MAP[i]: float(probabilities[i])
                for i in range(len(probabilities))
            },
            'board_id': metadata['board_id'],
            'axis': metadata['axis'],
            'timestamp': metadata['timestamp'],
            'mean_power': metadata['mean_power'],
            'std_power': metadata['std_power'],
            'reliable': confidence >= self.confidence_threshold
        }
        
        return result
    
    def predict_all_axes(self, board_id: Optional[str] = None) -> List[Dict]:
        """
        모든 축에 대해 예측 수행
        
        Args:
            board_id: 특정 보드 (None이면 전체)
        
        Returns:
            축별 예측 결과 리스트
        """
        # X축 (x1~x5)
        x_axes = [f'x{i}' for i in range(1, 6)]
        # Y축 (y1~y6)
        y_axes = [f'y{i}' for i in range(1, 7)]
        
        all_axes = x_axes + y_axes
        results = []
        
        for axis in all_axes:
            try:
                result = self.predict_current_state(
                    board_id=board_id,
                    axis=axis
                )
                results.append(result)
            except Exception as e:
                print(f"축 {axis} 예측 오류: {e}")
                results.append({
                    'status': 'ERROR',
                    'axis': axis,
                    'message': str(e)
                })
        
        return results
    
    def save_prediction_to_db(self, prediction: Dict):
        """
        예측 결과를 cnn_predictions 테이블에 저장
        
        Args:
            prediction: predict_current_state() 결과
        """
        # 데이터 없음 또는 에러 상태는 저장하지 않음
        if prediction.get('status') in ['NO_DATA', 'ERROR']:
            return
        
        # confidence가 없으면 저장하지 않음
        if prediction.get('confidence') is None:
            return
        
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # 테이블 확인/생성
        cur.execute('''
            CREATE TABLE IF NOT EXISTS cnn_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                board_id TEXT,
                axis TEXT,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                prob_normal REAL,
                prob_warning REAL,
                prob_alert REAL,
                prob_critical REAL,
                mean_power REAL,
                std_power REAL,
                reliable INTEGER,
                model_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 삽입
        probs = prediction.get('probabilities', {})
        
        cur.execute('''
            INSERT INTO cnn_predictions (
                board_id, axis, status, confidence,
                prob_normal, prob_warning, prob_alert, prob_critical,
                mean_power, std_power, reliable
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction.get('board_id'),
            prediction.get('axis'),
            prediction.get('status'),
            prediction.get('confidence'),
            probs.get('NORMAL', 0.0),
            probs.get('WARNING', 0.0),
            probs.get('ALERT', 0.0),
            probs.get('CRITICAL', 0.0),
            prediction.get('mean_power'),
            prediction.get('std_power'),
            1 if prediction.get('reliable', False) else 0
        ))
        
        conn.commit()
        conn.close()
    
    def run_periodic_prediction(self, interval_seconds: int = 60):
        """
        주기적으로 예측 실행 (백그라운드 서비스용)
        
        Args:
            interval_seconds: 예측 주기 (초)
        """
        import time
        
        print(f"주기적 예측 시작 (간격: {interval_seconds}초)")
        
        while True:
            try:
                # 모든 축 예측
                results = self.predict_all_axes()
                
                # DB 저장
                for result in results:
                    if result.get('status') not in ['NO_DATA', 'ERROR']:
                        self.save_prediction_to_db(result)
                
                # 요약 출력
                status_counts = {}
                for result in results:
                    status = result.get('status', 'UNKNOWN')
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"예측 완료: {status_counts}")
                
            except Exception as e:
                print(f"예측 오류: {e}")
            
            time.sleep(interval_seconds)


class PredictionCache:
    """예측 결과 캐싱 (성능 최적화)"""
    
    def __init__(self, ttl_seconds: int = 30):
        """
        Args:
            ttl_seconds: 캐시 유효 시간 (초)
        """
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[Dict]:
        """캐시 조회"""
        if key in self.cache:
            result, timestamp = self.cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.ttl:
                return result
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Dict):
        """캐시 저장"""
        self.cache[key] = (value, datetime.now())
    
    def clear(self):
        """캐시 초기화"""
        self.cache.clear()


if __name__ == '__main__':
    print("=== 실시간 예측기 테스트 ===\n")
    
    try:
        # 예측기 생성 (먼저 모델을 학습해야 함)
        predictor = RealTimePredictor(model_version='latest')
        
        # 현재 상태 예측
        result = predictor.predict_current_state()
        
        print(f"예측 결과:")
        print(f"  상태: {result['status']}")
        print(f"  확신도: {result['confidence']:.2%}")
        print(f"  신뢰성: {'높음' if result['reliable'] else '낮음'}")
        print(f"\n확률 분포:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.2%}")
        
        # DB 저장
        predictor.save_prediction_to_db(result)
        print("\n✓ 예측 결과 DB 저장 완료")
        
        # 모든 축 예측
        print("\n=== 전체 축 예측 ===\n")
        all_results = predictor.predict_all_axes()
        
        for res in all_results[:3]:  # 처음 3개만 출력
            if res['status'] != 'NO_DATA':
                print(f"축 {res['axis']}: {res['status']} ({res['confidence']:.2%})")
        
    except FileNotFoundError as e:
        print(f"오류: {e}")
        print("\n먼저 모델을 학습해야 합니다:")
        print("  python -m ml.trainer")
    except Exception as e:
        print(f"오류 발생: {e}")
