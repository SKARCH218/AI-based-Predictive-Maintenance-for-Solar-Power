#!/usr/bin/env python3
"""
CNN 예측을 주기적으로 실행하여 DB에 저장하는 스크립트
"""
import sys
import os
from pathlib import Path

# 현재 스크립트의 디렉토리를 프로젝트 루트로 사용
project_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_dir))

from ml.predictor import RealTimePredictor
import time
from datetime import datetime

print("=" * 60)
print("CNN 예측 자동 저장 스크립트")
print("=" * 60)

predictor = RealTimePredictor(db_path='solardata.db')

print("\n⏰ 30초마다 예측을 실행하고 DB에 저장합니다...")
print("종료하려면 Ctrl+C를 누르세요\n")

try:
    count = 0
    while True:
        count += 1
        
        # 예측 실행
        result = predictor.predict_current_state()
        
        # DB 저장
        predictor.save_prediction_to_db(result)
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # 출력 (confidence가 있을 때만 표시)
        if result.get('confidence') is not None:
            print(f"[{timestamp}] #{count} - {result['status']} ({result['confidence']:.1%}) → DB 저장 완료")
        else:
            print(f"[{timestamp}] #{count} - {result['status']} (데이터 없음)")
        
        time.sleep(30)
        
except KeyboardInterrupt:
    print("\n\n✅ 종료")
