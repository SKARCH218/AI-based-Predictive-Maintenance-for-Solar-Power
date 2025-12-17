#!/usr/bin/env python3
"""
태양광 예지보전 시스템 - 통합 실행 스크립트

한 번의 실행으로 모든 서비스를 자동으로 시작합니다:
1. 환경 확인 및 의존성 설치
2. 데이터베이스 초기화
3. 테스트 데이터 생성 (하드웨어 없을 시)
4. 데이터 수집 프로세스 시작
5. AI 예측 루프 시작
6. CNN 예측 저장기 시작
7. Flask 웹 서버 시작

사용법:
    python run.py              # 자동 모드 (하드웨어 감지)
    python run.py --test       # 강제 테스트 모드
    python run.py --no-cnn     # CNN 예측 저장 비활성화
"""

import subprocess
import sys
import time
import os
import signal
import glob
import argparse
from pathlib import Path
from datetime import datetime


class SolarSystemLauncher:
    """태양광 시스템 통합 런처"""
    
    def __init__(self, test_mode=False, enable_cnn_saver=True):
        self.project_dir = Path(__file__).parent.absolute()
        self.test_mode = test_mode
        self.enable_cnn_saver = enable_cnn_saver
        self.processes = {}
        
        # Python 실행 파일
        self.python = sys.executable
        
    def print_header(self, text):
        """섹션 헤더 출력"""
        print(f"\n{'='*70}")
        print(f"  {text}")
        print(f"{'='*70}\n")
    
    def print_step(self, step_num, total, text):
        """단계별 진행 상황 출력"""
        print(f"[{step_num}/{total}] {text}")
    
    def check_environment(self):
        """환경 확인"""
        self.print_step(1, 7, "환경 확인 중...")
        
        # Python 버전 확인
        version = sys.version_info
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("  ✗ Python 3.8 이상이 필요합니다!")
            sys.exit(1)
        
        # 작업 디렉토리 확인
        os.chdir(self.project_dir)
        print(f"  ✓ 작업 디렉토리: {self.project_dir}")
        
        # requirements.txt 확인
        req_file = self.project_dir / "requirements.txt"
        if not req_file.exists():
            print("  ✗ requirements.txt 파일이 없습니다!")
            sys.exit(1)
        print(f"  ✓ requirements.txt 확인")
    
    def install_dependencies(self):
        """의존성 설치"""
        self.print_step(2, 7, "의존성 패키지 설치 중...")
        
        try:
            # pip 업그레이드
            print("  - pip 업그레이드...")
            subprocess.check_call(
                [self.python, "-m", "pip", "install", "--upgrade", "pip"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # requirements 설치
            print("  - 패키지 설치 중... (시간이 걸릴 수 있습니다)")
            subprocess.check_call(
                [self.python, "-m", "pip", "install", "-r", "requirements.txt"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            print("  ✓ 모든 패키지 설치 완료")
            
        except subprocess.CalledProcessError as e:
            print(f"  ✗ 패키지 설치 실패!")
            print(f"  오류: {e.stderr.decode() if e.stderr else str(e)}")
            print("\n  수동 설치를 시도하세요:")
            print(f"    {self.python} -m pip install -r requirements.txt")
            sys.exit(1)
    
    def initialize_database(self):
        """데이터베이스 초기화"""
        self.print_step(3, 7, "데이터베이스 초기화 중...")
        
        import sqlite3
        
        db_path = self.project_dir / "solardata.db"
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # power_data 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS power_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    board_id TEXT,
                    sensor_address TEXT,
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
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT,
                    reason TEXT,
                    power_mw REAL,
                    baseline REAL,
                    threshold REAL,
                    severity REAL,
                    board_id TEXT,
                    cells TEXT
                )
            ''')
            
            # cnn_predictions 테이블
            cursor.execute('''
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
            
            # 데이터 수 확인
            cursor.execute("SELECT COUNT(*) FROM power_data")
            power_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM predictions")
            pred_count = cursor.fetchone()[0]
            
            conn.close()
            
            print(f"  ✓ 데이터베이스 준비 완료")
            print(f"    - power_data: {power_count:,} rows")
            print(f"    - predictions: {pred_count:,} rows")
            
        except Exception as e:
            print(f"  ✗ 데이터베이스 초기화 실패: {e}")
            sys.exit(1)
    
    def detect_hardware(self):
        """하드웨어 감지 (시리얼 포트)"""
        self.print_step(4, 7, "하드웨어 감지 중...")
        
        # macOS/Linux 시리얼 포트 감지
        serial_ports = glob.glob('/dev/tty.*') + glob.glob('/dev/cu.*')
        # Windows COM 포트도 추가 가능
        serial_ports += glob.glob('COM[0-9]*')
        
        if serial_ports:
            print(f"  ✓ 시리얼 포트 감지: {len(serial_ports)}개")
            for port in serial_ports[:3]:  # 처음 3개만 표시
                print(f"    - {port}")
            if len(serial_ports) > 3:
                print(f"    ... 외 {len(serial_ports) - 3}개")
            return False  # 하드웨어 있음 → 테스트 모드 불필요
        else:
            print("  ! 시리얼 포트가 감지되지 않았습니다")
            print("  → 테스트 모드로 전환합니다")
            return True  # 하드웨어 없음 → 테스트 모드 필요
    
    def generate_test_data(self):
        """초기 테스트 데이터 생성"""
        print("  - 초기 테스트 데이터 생성 중...")
        
        try:
            result = subprocess.run(
                [self.python, "test_data_generator.py", "--init"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("  ✓ 초기 데이터 생성 완료")
                # 출력에서 주요 정보만 추출
                for line in result.stdout.split('\n'):
                    if '생성 완료' in line or '✓' in line:
                        print(f"    {line.strip()}")
            else:
                print(f"  ! 데이터 생성 경고: {result.stderr[:100]}")
                
        except subprocess.TimeoutExpired:
            print("  ! 데이터 생성 시간 초과 (계속 진행)")
        except Exception as e:
            print(f"  ! 데이터 생성 실패: {e} (계속 진행)")
    
    def start_process(self, name, command, description):
        """프로세스 시작"""
        print(f"  - {description}...")
        
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=self.project_dir
            )
            
            self.processes[name] = process
            print(f"    ✓ {name} 시작됨 (PID: {process.pid})")
            return process
            
        except Exception as e:
            print(f"    ✗ {name} 시작 실패: {e}")
            return None
    
    def start_services(self):
        """모든 서비스 시작"""
        self.print_step(5, 7, "서비스 시작 중...")
        
        # 1. 데이터 수집 프로세스
        if self.test_mode:
            self.start_process(
                "DATA_GEN",
                [self.python, "test_data_generator.py"],
                "테스트 데이터 생성기 시작"
            )
        else:
            # 환경 변수 설정 (api.py가 TESTMODE 읽음)
            os.environ['TESTMODE'] = '0'
            self.start_process(
                "API",
                [self.python, "api.py"],
                "시리얼 데이터 수집기 시작"
            )
        
        time.sleep(2)
        
        # 2. AI 예측 루프
        self.start_process(
            "AI_PREDICTOR",
            [self.python, "server.py"],
            "AI 예측 루프 시작"
        )
        
        time.sleep(1)
        
        # 3. CNN 예측 저장기 (선택적)
        if self.enable_cnn_saver:
            self.start_process(
                "CNN_SAVER",
                [self.python, "save_predictions.py"],
                "CNN 예측 저장기 시작"
            )
            time.sleep(1)
        
        # 4. Flask 웹 서버 (마지막에 시작)
        self.start_process(
            "WEB_SERVER",
            [self.python, "app.py"],
            "Flask 웹 서버 시작"
        )
    
    def wait_for_server(self, max_wait=15):
        """웹 서버 준비 대기"""
        self.print_step(6, 7, "웹 서버 준비 대기 중...")
        
        import socket
        
        for i in range(max_wait):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', 5001))
                sock.close()
                
                if result == 0:
                    print(f"  ✓ 웹 서버 준비 완료 ({i+1}초)")
                    return True
                    
            except Exception:
                pass
            
            time.sleep(1)
            if i % 3 == 0:
                print(f"  ... 대기 중 ({i+1}/{max_wait}초)")
        
        print("  ! 웹 서버 응답 대기 시간 초과 (계속 진행)")
        return False
    
    def print_status(self):
        """실행 상태 출력"""
        self.print_step(7, 7, "시스템 상태 확인")
        
        print(f"\n  {'프로세스':<20} {'상태':<10} {'PID':<10}")
        print(f"  {'-'*40}")
        
        for name, proc in self.processes.items():
            if proc and proc.poll() is None:
                status = "✓ 실행 중"
                pid = proc.pid
            else:
                status = "✗ 종료됨"
                pid = "-"
            
            print(f"  {name:<20} {status:<10} {pid:<10}")
    
    def print_access_info(self):
        """접속 정보 출력"""
        self.print_header("시스템 준비 완료!")
        
        print("  🌐 웹 인터페이스 접속:")
        print(f"     → http://127.0.0.1:5001")
        print(f"     → http://localhost:5001")
        
        # 네트워크 IP 출력 시도
        try:
            import socket
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            if local_ip and local_ip != '127.0.0.1':
                print(f"     → http://{local_ip}:5001 (네트워크)")
        except:
            pass
        
        print(f"\n  📊 모드: {'테스트 모드 (시뮬레이션)' if self.test_mode else '실제 하드웨어 모드'}")
        print(f"  🧠 CNN 예측: {'활성화' if self.enable_cnn_saver else '비활성화'}")
        
        print(f"\n  ⏹  종료하려면 Ctrl+C 를 누르세요")
        print(f"{'='*70}\n")
    
    def monitor_processes(self):
        """프로세스 모니터링"""
        try:
            while True:
                # 모든 프로세스가 살아있는지 확인
                dead_processes = []
                for name, proc in self.processes.items():
                    if proc and proc.poll() is not None:
                        dead_processes.append(name)
                
                if dead_processes:
                    print(f"\n⚠️  프로세스 종료 감지: {', '.join(dead_processes)}")
                    
                    # 중요 프로세스(WEB_SERVER)가 죽었으면 전체 종료
                    if 'WEB_SERVER' in dead_processes:
                        print("웹 서버가 종료되었습니다. 전체 시스템을 종료합니다.")
                        break
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n\nCtrl+C 감지 - 시스템 종료 중...")
    
    def cleanup(self):
        """프로세스 정리"""
        print("\n모든 프로세스를 종료합니다...\n")
        
        for name, proc in self.processes.items():
            if proc and proc.poll() is None:
                print(f"  - {name} 종료 중...")
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                    print(f"    ✓ {name} 종료됨")
                except subprocess.TimeoutExpired:
                    proc.kill()
                    print(f"    ! {name} 강제 종료됨")
                except Exception as e:
                    print(f"    ! {name} 종료 실패: {e}")
        
        print(f"\n{'='*70}")
        print("  시스템이 안전하게 종료되었습니다")
        print(f"{'='*70}\n")
    
    def run(self):
        """메인 실행 함수"""
        start_time = datetime.now()
        
        self.print_header("태양광 예지보전 시스템 시작")
        print(f"  시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        try:
            # 1. 환경 확인
            self.check_environment()
            
            # 2. 의존성 설치
            self.install_dependencies()
            
            # 3. 데이터베이스 초기화
            self.initialize_database()
            
            # 4. 하드웨어 감지
            if not self.test_mode:
                self.test_mode = self.detect_hardware()
            else:
                self.print_step(4, 7, "테스트 모드 강제 활성화")
            
            # 테스트 모드일 경우 초기 데이터 생성
            if self.test_mode:
                self.generate_test_data()
            
            # 5. 서비스 시작
            self.start_services()
            
            # 6. 웹 서버 준비 대기
            self.wait_for_server()
            
            # 7. 상태 출력
            self.print_status()
            
            # 접속 정보 출력
            self.print_access_info()
            
            # 프로세스 모니터링
            self.monitor_processes()
            
        except KeyboardInterrupt:
            print("\n\nCtrl+C 감지")
        except Exception as e:
            print(f"\n✗ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 정리
            self.cleanup()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"총 실행 시간: {duration:.1f}초\n")


def main():
    """메인 엔트리 포인트"""
    parser = argparse.ArgumentParser(
        description="태양광 예지보전 시스템 통합 실행 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python run.py              # 자동 모드 (하드웨어 자동 감지)
  python run.py --test       # 강제 테스트 모드 (시뮬레이션)
  python run.py --no-cnn     # CNN 예측 저장 비활성화
        """
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='강제 테스트 모드 (하드웨어 없이 시뮬레이션 데이터 사용)'
    )
    
    parser.add_argument(
        '--no-cnn',
        action='store_true',
        help='CNN 예측 저장기 비활성화 (리소스 절약)'
    )
    
    args = parser.parse_args()
    
    # 런처 생성 및 실행
    launcher = SolarSystemLauncher(
        test_mode=args.test,
        enable_cnn_saver=not args.no_cnn
    )
    
    launcher.run()


if __name__ == '__main__':
    main()
