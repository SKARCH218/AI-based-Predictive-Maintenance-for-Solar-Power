#!/usr/bin/env python3
"""
CNN 모델 학습 스크립트

사용법:
  python3 train_cnn.py                    # 기본 설정
  python3 train_cnn.py --samples 500      # 샘플 수 지정
  python3 train_cnn.py --epochs 30        # 에폭 수 지정
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

from ml.trainer import Trainer, create_dataloaders
from ml.cnn_model import ModelManager
import torch


def main():
    parser = argparse.ArgumentParser(description='CNN 모델 학습')
    parser.add_argument('--samples', type=int, default=1000, 
                        help='학습 샘플 수 (기본: 1000)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='에폭 수 (기본: 20)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='배치 크기 (기본: 16)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='디바이스 (기본: cpu)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CNN 모델 학습")
    print("=" * 60)
    print(f"샘플 수: {args.samples}")
    print(f"에폭 수: {args.epochs}")
    print(f"배치 크기: {args.batch_size}")
    print(f"디바이스: {args.device}")
    print("=" * 60)
    print()
    
    # 1. 데이터 로더 생성
    print("📊 데이터 준비 중...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=args.batch_size,
            max_samples=args.samples
        )
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        print("\n💡 해결 방법:")
        print("1. 테스트 데이터 생성: python3 test_data_generator.py --init")
        print("2. DB 확인: sqlite3 solardata.db 'SELECT COUNT(*) FROM predictions;'")
        return
    
    # 2. 모델 및 트레이너 생성
    print("\n🤖 모델 초기화 중...")
    trainer = Trainer(
        num_classes=4,
        device=args.device,
        learning_rate=0.001
    )
    
    # 3. 학습
    print("\n🎓 학습 시작...\n")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=5
    )
    
    # 4. 평가
    print("\n📈 모델 평가 중...")
    from ml.trainer import Evaluator
    evaluator = Evaluator(trainer.model, device=args.device)
    results = evaluator.evaluate(test_loader)
    evaluator.print_report(results)
    
    # 5. 모델 저장
    print("\n💾 모델 저장 중...")
    from datetime import datetime
    version = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    model_manager = ModelManager()
    model_manager.save_model(
        model=trainer.model,
        version=version,
        metadata={
            'timestamp': datetime.now().isoformat(),
            'architecture': 'SolarPanelCNN',
            'samples': args.samples,
            'epochs': args.epochs,
            'metrics': {
                'accuracy': float(results['accuracy']),
                'f1_weighted': float(results['classification_report']['weighted avg']['f1-score'])
            }
        }
    )
    
    print(f"\n✅ 학습 완료! 모델 버전: {version}")
    print(f"📁 모델 위치: models/solar_cnn_{version}.pth")
    print("\n💡 서버를 재시작하면 새 모델이 로드됩니다.")


if __name__ == '__main__':
    main()
