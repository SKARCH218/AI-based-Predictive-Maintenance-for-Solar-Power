#!/usr/bin/env python3
"""
CNN 모듈 로드 테스트
"""
import sys
sys.path.insert(0, '/Users/taehunkim/Documents/태양광')

print("1. 기본 import 테스트...")
try:
    from ml.predictor import RealTimePredictor
    from ml.image_generator import SolarDataImageGenerator
    print("✓ Import 성공")
except Exception as e:
    print(f"✗ Import 실패: {e}")
    sys.exit(1)

print("\n2. 이미지 생성기 테스트...")
try:
    generator = SolarDataImageGenerator(db_path='solardata.db', method='multi')
    print("✓ 이미지 생성기 초기화 성공")
except Exception as e:
    print(f"✗ 이미지 생성기 실패: {e}")
    sys.exit(1)

print("\n3. 시계열 데이터 가져오기 테스트...")
try:
    timeseries = generator.fetch_timeseries(window_size=120, limit=1)
    if timeseries:
        print(f"✓ 시계열 데이터 획득 성공: {len(timeseries)}개")
        img, meta = timeseries[0]
        print(f"  이미지 shape: {img.shape}")
        print(f"  메타: board={meta.get('board_id')}, axis={meta.get('axis')}")
    else:
        print("✗ 시계열 데이터 없음")
except Exception as e:
    print(f"✗ 시계열 데이터 실패: {e}")
    import traceback
    traceback.print_exc()

print("\n4. Predictor 초기화 테스트...")
try:
    predictor = RealTimePredictor(db_path='solardata.db')
    print("✓ Predictor 초기화 성공")
    print(f"  모델 버전: {predictor.model_version}")
    print(f"  메타데이터: {predictor.model_metadata}")
except Exception as e:
    print(f"✗ Predictor 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n5. 예측 테스트...")
try:
    result = predictor.predict_current_state()
    print("✓ 예측 성공")
    print(f"  상태: {result['status']}")
    print(f"  확신도: {result['confidence']:.2%}")
    print(f"  확률: {result['probabilities']}")
except Exception as e:
    print(f"✗ 예측 실패: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ 모든 테스트 완료!")
