import serial
import time
import json
import sqlite3
import os
import threading
from serial.tools import list_ports

# --- 설정 ---
# 자동 감지할 아두이노 포트 목록(수동 지정 시 이 값을 무시하고 detect_serial_ports 사용)
SERIAL_PORTS = []  # 빈 리스트면 자동 감지 사용
BAUD_RATE = 115200
DB_FILE = 'solardata.db'
# 테스트 모드 설정: True이면 시리얼 대신 파일에서 JSON Lines를 읽습니다.
# 코드에서 직접 testmode = True로 켤 수 있고, 환경변수(TESTMODE=1)로도 제어 가능합니다.
testmode = True
TESTMODE = testmode or (os.getenv('TESTMODE', '0').lower() in ('1', 'true', 'yes'))
TEST_FILE = os.getenv('TEST_FILE', 'testdata.jsonl')

def setup_database():
    """데이터베이스와 테이블을 초기화하고 스키마를 보강합니다."""
    if os.path.exists(DB_FILE):
        print(f"데이터베이스 '{DB_FILE}'가 이미 존재합니다. 계속 진행합니다.")
    else:
        print(f"데이터베이스 '{DB_FILE}'를 생성합니다.")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
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
    # 스키마 보강: 누락 컬럼 추가
    try:
        cursor.execute("PRAGMA table_info(power_data)")
        cols = [row[1] for row in cursor.fetchall()]
        if 'sensor_address' not in cols:
            cursor.execute("ALTER TABLE power_data ADD COLUMN sensor_address TEXT")
            print("스키마 업데이트: sensor_address 컬럼 추가")
        if 'board_id' not in cols:
            cursor.execute("ALTER TABLE power_data ADD COLUMN board_id TEXT")
            print("스키마 업데이트: board_id 컬럼 추가")
        if 'axis' not in cols:
            cursor.execute("ALTER TABLE power_data ADD COLUMN axis TEXT")
            print("스키마 업데이트: axis 컬럼 추가")
    except sqlite3.Error as e:
        print(f"스키마 점검/업데이트 중 오류: {e}")
    conn.commit()
    conn.close()
    print("데이터베이스 설정이 완료되었습니다.")

def reader_thread(port: str):
    """개별 시리얼 포트 리더 스레드."""
    board_id = port  # 기본은 포트명으로 보드 식별
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)
    except serial.SerialException as e:
        print(f"[READER:{port}] 시리얼 오픈 실패: {e}")
        return

    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    print(f"[READER:{port}] 수신 시작")
    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    cursor.execute(
                        '''
                        INSERT INTO power_data (
                            board_id, sensor_address, axis, bus_voltage, shunt_voltage, load_voltage, current_ma, power_mw, accumulated_energy_mwh
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            board_id,
                            data.get('sensorAddress'),
                            data.get('axis'),
                            data.get('busVoltage'),
                            data.get('shuntVoltage'),
                            data.get('loadVoltage'),
                            data.get('current_mA'),
                            data.get('power_mW'),
                            data.get('accumulatedEnergy_mWh'),
                        ),
                    )
                    conn.commit()
                    print(f"[READER:{port}] 저장: {data}")
                except json.JSONDecodeError:
                    print(f"[READER:{port}] JSON 오류: {line}")
                except sqlite3.Error as e:
                    print(f"[READER:{port}] SQLite 오류: {e}")
            else:
                time.sleep(0.05)
    except Exception as e:
        print(f"[READER:{port}] 스레드 오류: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass
        print(f"[READER:{port}] 종료")


def file_reader_thread(file_path: str, board_id: str = 'TEST'):
    """테스트 파일(JSON Lines)을 tail 하며 레코드를 DB에 적재합니다."""
    print(f"[FILE-READER:{board_id}] '{file_path}' 감시 시작")
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    try:
        # 파일이 생길 때까지 대기
        while not os.path.exists(file_path):
            time.sleep(0.2)
        with open(file_path, 'r', encoding='utf-8') as f:
            # 기존 내용은 모두 읽고, 이후 append를 tail
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    f.seek(pos)
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    cursor.execute(
                        '''
                        INSERT INTO power_data (
                            board_id, sensor_address, axis, bus_voltage, shunt_voltage, load_voltage, current_ma, power_mw, accumulated_energy_mwh
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            data.get('board_id') or board_id,
                            data.get('sensorAddress'),
                            data.get('axis'),
                            data.get('busVoltage'),
                            data.get('shuntVoltage'),
                            data.get('loadVoltage'),
                            data.get('current_mA'),
                            data.get('power_mW'),
                            data.get('accumulatedEnergy_mWh'),
                        ),
                    )
                    conn.commit()
                    print(f"[FILE-READER:{board_id}] 저장: {data}")
                except json.JSONDecodeError:
                    print(f"[FILE-READER:{board_id}] JSON 오류: {line}")
                except sqlite3.Error as e:
                    print(f"[FILE-READER:{board_id}] SQLite 오류: {e}")
    except Exception as e:
        print(f"[FILE-READER:{board_id}] 스레드 오류: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass
        print(f"[FILE-READER:{board_id}] 종료")


def main():
    """여러 시리얼 포트에서 데이터를 읽어 DB에 저장."""
    threads = []
    active_ports = []
    if TESTMODE:
        t = threading.Thread(target=file_reader_thread, args=(TEST_FILE, 'TEST'), daemon=True)
        t.start()
        threads.append(t)
        active_ports.append(f"TESTFILE:{TEST_FILE}")
    else:
        ports = SERIAL_PORTS if SERIAL_PORTS else detect_serial_ports()
        for port in ports:
            t = threading.Thread(target=reader_thread, args=(port,), daemon=True)
            t.start()
            threads.append(t)
            active_ports.append(port)
    print(f"활성 소스: {', '.join(active_ports) if active_ports else '없음'}")
    try:
        # 메인 스레드는 유지
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n종료 신호 수신. 리더 스레드 정리 중...")
        # 데몬 스레드라 프로세스 종료 시 자동 종료
        time.sleep(0.5)
    print("수집기 종료")


def detect_serial_ports() -> list[str]:
    """시스템에서 아두이노(또는 유사 장치)로 보이는 포트를 자동 감지합니다."""
    ports = list_ports.comports()
    candidates = []
    for p in ports:
        desc = (p.description or '').lower()
        manf = (getattr(p, 'manufacturer', '') or '').lower()
        hwid = (p.hwid or '').lower()
        if any(k in desc for k in ['arduino', 'ch340', 'usb serial', 'usb-serial']) or \
           any(k in manf for k in ['arduino', 'wch', 'silabs']) or \
           'vid:pid' in hwid:
            candidates.append(p.device)
    # 후보가 없으면 모든 포트를 반환(수동 필터 필요할 수 있음)
    if not candidates:
        candidates = [p.device for p in ports]
    return candidates

if __name__ == "__main__":
    setup_database()
    main()