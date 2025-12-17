"""
지속적 학습 (Continuous Learning) 파이프라인

기능:
1. 주기적으로 새로운 데이터 수집
2. 자동 라벨링 (기존 predictions 활용)
3. 모델 재학습 (Incremental Learning)
4. A/B 테스트 (기존 vs 신규 모델)
5. 성능 모니터링 & 자동 배포
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sqlite3
import json
import time
import schedule

from ml.image_generator import SolarDataImageGenerator
from ml.cnn_model import SolarPanelCNN, SolarPanelCNNWithAttention, ModelManager
from ml.trainer import Trainer, Evaluator, create_dataloaders, SolarPanelDataset


class ContinuousLearningPipeline:
    """지속적 학습 파이프라인"""
    
    def __init__(
        self,
        db_path: str = 'solardata.db',
        models_dir: str = 'models',
        min_new_samples: int = 100,
        retrain_threshold_days: int = 7,
        performance_threshold: float = 0.05  # 5% 성능 향상 시 배포
    ):
        """
        Args:
            db_path: 데이터베이스 경로
            models_dir: 모델 저장 디렉토리
            min_new_samples: 재학습 최소 샘플 수
            retrain_threshold_days: 재학습 주기 (일)
            performance_threshold: 성능 향상 임계값
        """
        self.db_path = db_path
        self.models_dir = Path(models_dir)
        self.min_new_samples = min_new_samples
        self.retrain_threshold_days = retrain_threshold_days
        self.performance_threshold = performance_threshold
        
        self.model_manager = ModelManager(models_dir=models_dir)
        self.image_generator = SolarDataImageGenerator(db_path=db_path, method='multi')
        
        # 학습 로그
        self.training_log_path = self.models_dir / 'training_log.json'
        self.training_log = self._load_training_log()
    
    def _load_training_log(self) -> List[Dict]:
        """학습 로그 로드"""
        if self.training_log_path.exists():
            with open(self.training_log_path, 'r') as f:
                return json.load(f)
        return []
    
    def _save_training_log(self):
        """학습 로그 저장"""
        with open(self.training_log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
    
    def check_new_data(self, last_train_date: Optional[str] = None) -> int:
        """
        마지막 학습 이후 새로운 데이터 수 확인
        
        Args:
            last_train_date: 마지막 학습 날짜 (ISO format)
        
        Returns:
            새로운 샘플 수
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        if last_train_date:
            cur.execute("""
                SELECT COUNT(*) FROM predictions 
                WHERE timestamp > ?
            """, (last_train_date,))
        else:
            cur.execute("SELECT COUNT(*) FROM predictions")
        
        count = cur.fetchone()[0]
        conn.close()
        
        return count
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        재학습 필요 여부 판단
        
        Returns:
            (재학습 필요 여부, 이유)
        """
        if not self.training_log:
            return True, "초기 학습 필요"
        
        last_training = self.training_log[-1]
        last_train_date = last_training['timestamp']
        
        # 새로운 데이터 확인
        new_samples = self.check_new_data(last_train_date)
        
        if new_samples < self.min_new_samples:
            return False, f"새로운 샘플 부족 ({new_samples}/{self.min_new_samples})"
        
        # 마지막 학습 이후 경과 시간
        last_date = datetime.fromisoformat(last_train_date)
        days_since = (datetime.now() - last_date).days
        
        if days_since >= self.retrain_threshold_days:
            return True, f"{days_since}일 경과, 새 샘플 {new_samples}개"
        
        return False, f"재학습 주기 미도달 ({days_since}/{self.retrain_threshold_days}일)"
    
    def prepare_training_data(
        self,
        max_samples: int = 5000,
        use_recent_only: bool = False,
        recent_days: int = 30
    ) -> Tuple:
        """
        학습 데이터 준비
        
        Args:
            max_samples: 최대 샘플 수
            use_recent_only: 최근 데이터만 사용 여부
            recent_days: 최근 N일 데이터
        
        Returns:
            (train_loader, val_loader, test_loader)
        """
        print(f"\n{'='*60}")
        print("학습 데이터 준비 중...")
        print(f"{'='*60}\n")
        
        # 데이터 로더 생성
        loaders = create_dataloaders(
            db_path=self.db_path,
            batch_size=32,
            val_split=0.15,
            test_split=0.15,
            max_samples=max_samples
        )
        
        return loaders
    
    def train_new_model(
        self,
        model_architecture: str = 'attention',
        epochs: int = 50,
        device: str = 'cpu'
    ) -> Tuple[torch.nn.Module, Dict]:
        """
        새 모델 학습
        
        Args:
            model_architecture: 'basic' 또는 'attention'
            epochs: 학습 에폭 수
            device: 'cpu' 또는 'cuda'
        
        Returns:
            (학습된 모델, 성능 메트릭)
        """
        print(f"\n{'='*60}")
        print(f"새 모델 학습 시작")
        print(f"아키텍처: {model_architecture}")
        print(f"디바이스: {device}")
        print(f"{'='*60}\n")
        
        # 데이터 준비
        train_loader, val_loader, test_loader = self.prepare_training_data()
        
        # 모델 생성
        if model_architecture == 'attention':
            model = SolarPanelCNNWithAttention(num_classes=4, in_channels=3)
        else:
            model = SolarPanelCNN(num_classes=4, in_channels=3)
        
        # 트레이너 생성
        trainer = Trainer(
            model, 
            device=device,
            learning_rate=1e-3,
            save_dir='checkpoints_continuous'
        )
        
        # 학습
        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=epochs,
            early_stopping_patience=10,
            save_best=True
        )
        
        # 평가
        evaluator = Evaluator(model, device=device)
        results = evaluator.evaluate(test_loader)
        evaluator.print_report(results)
        
        # 메트릭 추출
        metrics = {
            'accuracy': results['accuracy'],
            'f1_macro': results['classification_report']['macro avg']['f1-score'],
            'f1_weighted': results['classification_report']['weighted avg']['f1-score'],
            'training_history': history
        }
        
        return model, metrics
    
    def compare_models(
        self,
        new_model: torch.nn.Module,
        new_metrics: Dict,
        baseline_version: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        신규 모델과 기존 모델 비교 (A/B 테스트)
        
        Args:
            new_model: 새 모델
            new_metrics: 새 모델 성능
            baseline_version: 비교할 기준 모델 버전
        
        Returns:
            (배포 여부, 이유)
        """
        print(f"\n{'='*60}")
        print("모델 성능 비교 (A/B Test)")
        print(f"{'='*60}\n")
        
        # 기준 모델이 없으면 무조건 배포
        if baseline_version is None:
            if not self.training_log:
                return True, "초기 모델 배포"
            baseline_version = self.training_log[-1]['version']
        
        # 기준 모델 성능 조회
        baseline_metrics = None
        for log in self.training_log:
            if log['version'] == baseline_version:
                baseline_metrics = log['metrics']
                break
        
        if baseline_metrics is None:
            return True, f"기준 모델({baseline_version}) 메트릭 없음"
        
        # 성능 비교
        baseline_acc = baseline_metrics.get('accuracy', 0)
        new_acc = new_metrics.get('accuracy', 0)
        
        improvement = (new_acc - baseline_acc) / max(baseline_acc, 1e-6)
        
        print(f"기준 모델 ({baseline_version}): {baseline_acc:.2f}%")
        print(f"신규 모델: {new_acc:.2f}%")
        print(f"개선율: {improvement*100:.2f}%\n")
        
        if improvement >= self.performance_threshold:
            return True, f"성능 {improvement*100:.2f}% 향상"
        else:
            return False, f"성능 향상 미달 ({improvement*100:.2f}% < {self.performance_threshold*100:.2f}%)"
    
    def deploy_model(
        self,
        model: torch.nn.Module,
        metrics: Dict,
        reason: str
    ) -> str:
        """
        모델 배포 (버전 관리)
        
        Args:
            model: 배포할 모델
            metrics: 성능 메트릭
            reason: 배포 이유
        
        Returns:
            배포된 모델 버전
        """
        # 버전 번호 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v{len(self.training_log)+1}_{timestamp}"
        
        # 모델 저장
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'metrics': metrics,
            'architecture': model.__class__.__name__
        }
        
        self.model_manager.save_model(model, version, metadata)
        
        # 학습 로그 업데이트
        self.training_log.append(metadata)
        self._save_training_log()
        
        print(f"\n{'='*60}")
        print(f"✓ 모델 배포 완료: {version}")
        print(f"  정확도: {metrics['accuracy']:.2f}%")
        print(f"  이유: {reason}")
        print(f"{'='*60}\n")
        
        return version
    
    def run_pipeline(
        self,
        force: bool = False,
        device: str = 'cpu'
    ) -> Optional[str]:
        """
        전체 파이프라인 실행
        
        Args:
            force: 강제 실행 여부
            device: 'cpu' 또는 'cuda'
        
        Returns:
            배포된 모델 버전 (배포 안 하면 None)
        """
        print(f"\n{'='*80}")
        print(f"지속적 학습 파이프라인 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # 재학습 필요 여부 확인
        should_train, reason = self.should_retrain()
        
        if not should_train and not force:
            print(f"✗ 재학습 건너뜀: {reason}\n")
            return None
        
        print(f"✓ 재학습 진행: {reason}\n")
        
        try:
            # 1. 모델 학습
            new_model, metrics = self.train_new_model(
                model_architecture='attention',
                epochs=50,
                device=device
            )
            
            # 2. 성능 비교
            should_deploy, deploy_reason = self.compare_models(new_model, metrics)
            
            if should_deploy or force:
                # 3. 배포
                version = self.deploy_model(new_model, metrics, deploy_reason)
                return version
            else:
                print(f"\n✗ 배포 건너뜀: {deploy_reason}\n")
                return None
        
        except Exception as e:
            print(f"\n✗ 파이프라인 오류: {e}\n")
            import traceback
            traceback.print_exc()
            return None
    
    def schedule_periodic_training(
        self,
        cron_expression: str = "0 2 * * 0",  # 매주 일요일 새벽 2시
        device: str = 'cpu'
    ):
        """
        주기적 학습 스케줄링
        
        Args:
            cron_expression: Cron 표현식 (매주 일요일 새벽 2시)
            device: 'cpu' 또는 'cuda'
        """
        # 간단한 스케줄러 (매주 일요일 02:00)
        schedule.every().sunday.at("02:00").do(
            lambda: self.run_pipeline(device=device)
        )
        
        print(f"✓ 주기적 학습 스케줄 등록: {cron_expression}")
        print(f"  다음 실행: {schedule.next_run()}\n")
        
        # 스케줄 실행 루프
        while True:
            schedule.run_pending()
            time.sleep(60)  # 1분마다 체크


class PerformanceMonitor:
    """모델 성능 모니터링"""
    
    def __init__(self, db_path: str = 'solardata.db'):
        self.db_path = db_path
    
    def get_prediction_accuracy(
        self,
        days: int = 7,
        model_version: Optional[str] = None
    ) -> Dict:
        """
        예측 정확도 계산 (실제 라벨과 비교)
        
        Args:
            days: 최근 N일
            model_version: 특정 모델 버전
        
        Returns:
            정확도 메트릭
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # 예측 결과 조회
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cur.execute("""
            SELECT 
                c.status as predicted,
                p.status as actual,
                c.confidence
            FROM cnn_predictions c
            JOIN predictions p 
                ON c.board_id = p.board_id 
                AND ABS(JULIANDAY(c.timestamp) - JULIANDAY(p.timestamp)) < 0.01
            WHERE c.timestamp > ?
        """, (since_date,))
        
        rows = cur.fetchall()
        conn.close()
        
        if not rows:
            return {'accuracy': 0, 'samples': 0}
        
        # 정확도 계산
        correct = sum(1 for r in rows if r['predicted'] == r['actual'])
        total = len(rows)
        accuracy = 100 * correct / total
        
        # 확신도별 정확도
        high_conf = [r for r in rows if r['confidence'] >= 0.8]
        high_conf_acc = 100 * sum(1 for r in high_conf if r['predicted'] == r['actual']) / max(len(high_conf), 1)
        
        return {
            'accuracy': accuracy,
            'samples': total,
            'high_confidence_accuracy': high_conf_acc,
            'high_confidence_samples': len(high_conf)
        }
    
    def generate_report(self, days: int = 7) -> str:
        """성능 리포트 생성"""
        metrics = self.get_prediction_accuracy(days=days)
        
        report = f"""
{'='*60}
모델 성능 리포트 (최근 {days}일)
{'='*60}

전체 정확도: {metrics['accuracy']:.2f}% ({metrics['samples']} 샘플)
고확신 정확도: {metrics['high_confidence_accuracy']:.2f}% ({metrics['high_confidence_samples']} 샘플)

{'='*60}
        """
        
        return report


if __name__ == '__main__':
    print("=== 지속적 학습 파이프라인 테스트 ===\n")
    
    # 파이프라인 생성
    pipeline = ContinuousLearningPipeline(
        min_new_samples=50,  # 테스트용 낮은 임계값
        retrain_threshold_days=1
    )
    
    # 재학습 필요 여부 확인
    should_train, reason = pipeline.should_retrain()
    print(f"재학습 필요: {should_train} ({reason})")
    
    # 파이프라인 강제 실행 (테스트)
    # version = pipeline.run_pipeline(force=True, device='cpu')
    
    # 성능 모니터링
    monitor = PerformanceMonitor()
    report = monitor.generate_report(days=7)
    print(report)
