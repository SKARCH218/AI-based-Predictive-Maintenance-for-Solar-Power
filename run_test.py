#!/usr/bin/env python3
"""
태양광 예지보전 시스템 - 통합 실행 스크립트

테스트 모드로 전체 시스템을 실행합니다:
1. 테스트 데이터 초기화
2. 백그라운드 데이터 생성기 시작
3. Flask 웹 서버 시작
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.absolute()
os.chdir(PROJECT_DIR)

# PYTHONPATH 설정
os.environ['PYTHONPATH'] = str(PROJECT_DIR)

print("=" * 50)
print("태양광 예지보전 시스템 - 테스트 모드")
print("=" * 50)
print()

# 가상환경 확인
venv_python = PROJECT_DIR / ".venv" / "bin" / "python3"
if not venv_python.exists():
    print("⚠️  가상환경이 없습니다!")
    print("먼저 다음 명령을 실행하세요:")
    print("  python3 -m venv .venv")
    print("  source .venv/bin/activate")
    print("  pip install -r requirements.txt")
    sys.exit(1)

print("✓ 가상환경 확인 완료")

# 1. 초기 데이터 생성
print("\n1️⃣  초기 테스트 데이터 생성 중...")
result = subprocess.run(
    [str(venv_python), "test_data_generator.py", "--init"],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    print(result.stdout)
else:
    print(f"⚠️  데이터 생성 실패: {result.stderr}")

# 2. 백그라운드 데이터 생성기 시작
print("\n2️⃣  백그라운드 데이터 생성기 시작...")
data_gen_process = subprocess.Popen(
    [str(venv_python), "test_data_generator.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
print(f"   PID: {data_gen_process.pid}")

# 잠시 대기
time.sleep(2)

# 3. Flask 서버 시작
print("\n3️⃣  Flask 웹 서버 시작...")
print("   URL: http://127.0.0.1:5001")
print("   URL: http://192.168.1.15:5001 (네트워크)")
print()
print("=" * 50)
print("🎉 서버가 실행 중입니다!")
print("브라우저에서 http://127.0.0.1:5001 접속하세요")
print()
print("종료하려면 Ctrl+C 를 누르세요")
print("=" * 50)
print()

# 종료 핸들러
def signal_handler(sig, frame):
    print("\n\n서버 종료 중...")
    data_gen_process.terminate()
    try:
        data_gen_process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        data_gen_process.kill()
    print("✓ 종료 완료")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Flask 앱 실행 (포그라운드)
try:
    subprocess.run([str(venv_python), "app.py"])
except KeyboardInterrupt:
    signal_handler(None, None)
