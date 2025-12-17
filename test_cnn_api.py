#!/usr/bin/env python3
"""
CNN API 테스트
"""
import requests
import json

BASE_URL = "http://127.0.0.1:5001"

print("=" * 60)
print("CNN API 테스트")
print("=" * 60)

# 1. 예측 테스트
print("\n1️⃣  /api/cnn/predict 테스트...")
try:
    r = requests.get(f"{BASE_URL}/api/cnn/predict", timeout=10)
    print(f"   상태 코드: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   ✅ 성공!")
        print(f"   상태: {data.get('status')}")
        print(f"   확신도: {data.get('confidence'):.2%}")
        print(f"   모델 버전: {data.get('model_version')}")
    else:
        print(f"   ❌ 실패: {r.text}")
except Exception as e:
    print(f"   ❌ 오류: {e}")

# 2. 히스토리 테스트
print("\n2️⃣  /api/cnn/history 테스트...")
try:
    r = requests.get(f"{BASE_URL}/api/cnn/history?limit=3", timeout=10)
    print(f"   상태 코드: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   ✅ 성공! (레코드 {len(data)}개)")
    else:
        print(f"   ❌ 실패: {r.text}")
except Exception as e:
    print(f"   ❌ 오류: {e}")

# 3. 패턴 이미지 테스트
print("\n3️⃣  /api/cnn/pattern 테스트...")
try:
    r = requests.get(f"{BASE_URL}/api/cnn/pattern?window_size=120", timeout=15)
    print(f"   상태 코드: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        if 'image' in data:
            img_len = len(data['image'])
            print(f"   ✅ 성공! (이미지 크기: {img_len} bytes)")
            print(f"   메서드: {data.get('method')}")
        else:
            print(f"   ⚠️  이미지 없음: {data}")
    else:
        print(f"   ❌ 실패: {r.text}")
except Exception as e:
    print(f"   ❌ 오류: {e}")

# 4. 모델 정보 테스트
print("\n4️⃣  /api/cnn/model-info 테스트...")
try:
    r = requests.get(f"{BASE_URL}/api/cnn/model-info", timeout=10)
    print(f"   상태 코드: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   ✅ 성공!")
        print(f"   버전: {data.get('version')}")
        print(f"   정확도: {data.get('accuracy'):.1f}%")
        print(f"   예측 샘플: {data.get('prediction_samples')}")
    else:
        print(f"   ❌ 실패: {r.text}")
except Exception as e:
    print(f"   ❌ 오류: {e}")

print("\n" + "=" * 60)
print("✅ 테스트 완료!")
print("=" * 60)
