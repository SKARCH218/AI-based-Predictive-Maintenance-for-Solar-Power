"""
시계열 데이터를 이미지로 변환하는 모듈

기법:
1. GAF (Gramian Angular Field): 시계열의 각도 관계를 이미지화
2. MTF (Markov Transition Field): 상태 전이 확률을 이미지화
3. Recurrence Plot: 시계열의 재발 패턴을 이미지화
"""

import numpy as np
from typing import Tuple, List, Optional
import sqlite3
from pathlib import Path


class TimeSeriesImageGenerator:
    """시계열 데이터를 2D 이미지로 변환"""
    
    def __init__(self, image_size: int = 64, method: str = 'gaf'):
        """
        Args:
            image_size: 생성할 이미지 크기 (64x64, 128x128 등)
            method: 변환 방법 ('gaf', 'mtf', 'recurrence', 'multi')
        """
        self.image_size = image_size
        self.method = method
        
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """데이터를 [-1, 1] 범위로 정규화"""
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return 2 * (data - min_val) / (max_val - min_val) - 1
    
    def polar_encoding(self, data: np.ndarray) -> np.ndarray:
        """극좌표 인코딩: [-1, 1] → [0, π]"""
        normalized = self.normalize(data)
        return np.arccos(normalized)
    
    def gramian_angular_field(self, data: np.ndarray, method: str = 'summation') -> np.ndarray:
        """
        GAF 변환: 시계열의 각도 관계를 행렬로 표현
        
        Args:
            data: 1D 시계열 데이터
            method: 'summation' (GASF) 또는 'difference' (GADF)
        
        Returns:
            2D GAF 이미지
        """
        # 극좌표 변환
        phi = self.polar_encoding(data)
        n = len(phi)
        
        # GAF 행렬 생성
        if method == 'summation':
            # GASF: cos(φi + φj)
            gaf = np.cos(phi[:, None] + phi[None, :])
        else:
            # GADF: sin(φi - φj)
            gaf = np.sin(phi[:, None] - phi[None, :])
        
        # 리샘플링하여 고정 크기로 변환
        if n != self.image_size:
            gaf = self._resize_image(gaf, self.image_size)
        
        return gaf
    
    def markov_transition_field(self, data: np.ndarray, n_bins: int = 8) -> np.ndarray:
        """
        MTF 변환: 상태 전이 확률을 행렬로 표현
        
        Args:
            data: 1D 시계열 데이터
            n_bins: 양자화 구간 수
        
        Returns:
            2D MTF 이미지
        """
        # 데이터 정규화 및 양자화
        normalized = self.normalize(data)
        bins = np.linspace(-1, 1, n_bins + 1)
        quantized = np.digitize(normalized, bins) - 1
        quantized = np.clip(quantized, 0, n_bins - 1)
        
        # 전이 행렬 계산
        transition_matrix = np.zeros((n_bins, n_bins))
        for i in range(len(quantized) - 1):
            transition_matrix[quantized[i], quantized[i+1]] += 1
        
        # 정규화
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(
            transition_matrix, 
            row_sums, 
            out=np.zeros_like(transition_matrix), 
            where=row_sums != 0
        )
        
        # MTF 생성: 각 시점의 상태 전이 확률 매핑
        mtf = transition_matrix[quantized[:, None], quantized[None, :]]
        
        # 리샘플링
        if len(data) != self.image_size:
            mtf = self._resize_image(mtf, self.image_size)
        
        return mtf
    
    def recurrence_plot(self, data: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Recurrence Plot: 시계열의 재발 패턴 시각화
        
        Args:
            data: 1D 시계열 데이터
            threshold: 유사도 임계값
        
        Returns:
            2D RP 이미지
        """
        normalized = self.normalize(data)
        n = len(normalized)
        
        # 거리 행렬 계산
        distance_matrix = np.abs(normalized[:, None] - normalized[None, :])
        
        # 임계값 적용
        rp = (distance_matrix < threshold).astype(np.float32)
        
        # 리샘플링
        if n != self.image_size:
            rp = self._resize_image(rp, self.image_size)
        
        return rp
    
    def _resize_image(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """이미지 리샘플링 (간단한 보간법)"""
        from scipy.ndimage import zoom
        zoom_factor = target_size / image.shape[0]
        return zoom(image, zoom_factor, order=1)
    
    def generate_multi_channel(self, data: np.ndarray) -> np.ndarray:
        """
        다중 채널 이미지 생성 (RGB 스타일)
        
        Args:
            data: 1D 시계열 데이터
        
        Returns:
            3D 이미지 (height, width, 3)
            - Channel 0: GASF
            - Channel 1: GADF
            - Channel 2: MTF
        """
        gasf = self.gramian_angular_field(data, method='summation')
        gadf = self.gramian_angular_field(data, method='difference')
        mtf = self.markov_transition_field(data)
        
        # 3채널로 스택
        multi_channel = np.stack([gasf, gadf, mtf], axis=-1)
        return multi_channel
    
    def generate_from_timeseries(self, data: np.ndarray) -> np.ndarray:
        """
        시계열 데이터로부터 이미지 생성 (메인 인터페이스)
        
        Args:
            data: 1D 시계열 배열 (예: power_mw 값들)
        
        Returns:
            변환된 이미지 (2D or 3D)
        """
        if self.method == 'gaf':
            return self.gramian_angular_field(data, method='summation')
        elif self.method == 'gadf':
            return self.gramian_angular_field(data, method='difference')
        elif self.method == 'mtf':
            return self.markov_transition_field(data)
        elif self.method == 'recurrence':
            return self.recurrence_plot(data)
        elif self.method == 'multi':
            return self.generate_multi_channel(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")


class SolarDataImageGenerator:
    """태양광 데이터를 이미지로 변환하는 헬퍼 클래스"""
    
    def __init__(self, db_path: str = 'solardata.db', image_size: int = 64, method: str = 'multi'):
        self.db_path = db_path
        self.generator = TimeSeriesImageGenerator(image_size=image_size, method=method)
        
    def fetch_timeseries(
        self, 
        board_id: Optional[str] = None, 
        axis: Optional[str] = None,
        window_size: int = 120,
        limit: int = 1
    ) -> List[Tuple[np.ndarray, dict]]:
        """
        DB에서 시계열 데이터를 가져와 이미지로 변환
        
        Args:
            board_id: 특정 보드 (None이면 전체)
            axis: 특정 축 (None이면 전체)
            window_size: 시계열 윈도우 크기
            limit: 최대 샘플 수
        
        Returns:
            [(이미지, 메타데이터), ...] 리스트
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # 쿼리 구성
        where_clauses = ["power_mw IS NOT NULL"]
        params = []
        
        if board_id:
            where_clauses.append("board_id = ?")
            params.append(board_id)
        
        if axis:
            where_clauses.append("axis = ?")
            params.append(axis)
        
        where_str = " AND ".join(where_clauses)
        
        # 최근 데이터 조회
        query = f"""
            SELECT power_mw, timestamp, board_id, axis, id
            FROM power_data
            WHERE {where_str}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(window_size * limit)
        
        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()
        
        if not rows or len(rows) < window_size:
            return []
        
        # 윈도우 단위로 분할
        results = []
        for i in range(0, min(len(rows), window_size * limit), window_size):
            window = rows[i:i + window_size]
            if len(window) < window_size:
                break
            
            # power_mw 시계열 추출 (시간순 정렬)
            power_values = np.array([row['power_mw'] for row in reversed(window)])
            
            # 이미지 생성
            image = self.generator.generate_from_timeseries(power_values)
            
            # 메타데이터
            metadata = {
                'board_id': window[0]['board_id'],
                'axis': window[0]['axis'],
                'timestamp': window[0]['timestamp'],
                'start_id': window[-1]['id'],
                'end_id': window[0]['id'],
                'mean_power': float(np.mean(power_values)),
                'std_power': float(np.std(power_values))
            }
            
            results.append((image, metadata))
        
        return results
    
    def generate_training_batch(
        self, 
        n_samples: int = 100,
        save_dir: Optional[Path] = None
    ) -> List[Tuple[np.ndarray, str, dict]]:
        """
        학습용 배치 데이터 생성
        
        Args:
            n_samples: 생성할 샘플 수
            save_dir: 저장 디렉토리 (None이면 메모리만)
        
        Returns:
            [(이미지, 라벨, 메타데이터), ...] 리스트
        """
        # predictions 테이블에서 라벨 가져오기
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                p.id, p.timestamp, p.status, p.board_id,
                p.power_mw, p.severity
            FROM predictions p
            ORDER BY p.timestamp DESC
            LIMIT ?
        """, (n_samples,))
        
        predictions = cur.fetchall()
        conn.close()
        
        results = []
        for pred in predictions:
            # 해당 시점 이전 120개 데이터로 이미지 생성
            images = self.fetch_timeseries(
                board_id=pred['board_id'],
                window_size=120,
                limit=1
            )
            
            if images:
                image, metadata = images[0]
                label = pred['status']  # NORMAL, WARNING, ALERT
                
                metadata.update({
                    'prediction_id': pred['id'],
                    'severity': pred['severity'],
                    'actual_power': pred['power_mw']
                })
                
                results.append((image, label, metadata))
                
                # 선택적으로 디스크 저장
                if save_dir:
                    save_dir = Path(save_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    label_dir = save_dir / label
                    label_dir.mkdir(exist_ok=True)
                    
                    filename = f"{pred['id']}_{pred['timestamp'].replace(':', '-')}.npy"
                    np.save(label_dir / filename, image)
        
        return results


if __name__ == '__main__':
    # 테스트 코드
    print("=== 태양광 이미지 생성 테스트 ===")
    
    # 간단한 시계열 테스트
    test_data = np.sin(np.linspace(0, 4 * np.pi, 120)) + np.random.normal(0, 0.1, 120)
    
    gen = TimeSeriesImageGenerator(image_size=64, method='multi')
    image = gen.generate_from_timeseries(test_data)
    
    print(f"생성된 이미지 shape: {image.shape}")
    print(f"이미지 범위: [{image.min():.3f}, {image.max():.3f}]")
    
    # DB 기반 테스트
    try:
        solar_gen = SolarDataImageGenerator(method='multi')
        samples = solar_gen.fetch_timeseries(window_size=120, limit=3)
        print(f"\n생성된 샘플 수: {len(samples)}")
        
        if samples:
            img, meta = samples[0]
            print(f"첫 샘플 shape: {img.shape}")
            print(f"메타데이터: {meta}")
    except Exception as e:
        print(f"DB 테스트 오류 (정상일 수 있음): {e}")
