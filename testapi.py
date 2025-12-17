import os
import json
import time
import random

"""
테스트 제너레이터: JSON Lines 파일(testdata.jsonl)에 센서 측정값을 지속적으로 append합니다.
api.py를 테스트 모드(TESTMODE=1)로 실행하면 이 파일을 tail하여 DB에 적재합니다.

필드 키는 아두이노가 보내는 JSON과 동일하게 맞췄습니다:
- sensorAddress, axis, busVoltage, shuntVoltage, loadVoltage, current_mA, power_mW, accumulatedEnergy_mWh
- board_id (옵션)도 포함 가능
"""

TEST_FILE = os.getenv('TEST_FILE', 'testdata.jsonl')
BOARD_ID = os.getenv('BOARD_ID', 'TEST')

# 환경 구성(다양성 제어)
BASE_X = float(os.getenv('BASE_X', '800'))
BASE_Y = float(os.getenv('BASE_Y', '820'))
JITTER_MW = float(os.getenv('JITTER_MW', '60'))
CYCLE_SEC = float(os.getenv('CYCLE_SEC', '120'))     # 인공 낮/밤 주기(초)
CLOUD_PERIOD_SEC = float(os.getenv('CLOUD_PERIOD_SEC', '45'))
CLOUD_AMPL = float(os.getenv('CLOUD_AMPL', '0.2'))   # 0~0.5 권장
OUTLIER_PROB = float(os.getenv('OUTLIER_PROB', '0.01'))

# 가상의 축/센서 주소 목록 (x1~x5, y1~y6), 센서 주소는 임의 문자열
X_AXES = [f"x{i}" for i in range(1, 6)]
Y_AXES = [f"y{i}" for i in range(1, 7)]
SENSORS = ["0x40", "0x41", "0x44", "0x45"]

# 누적 에너지 상태
accum_energy = {axis: 0.0 for axis in X_AXES + Y_AXES}

# 결함 시나리오(예: x3, y4 약화) - 환경변수로 제어 가능
WEAK_X = os.getenv('WEAK_X', 'x3')
WEAK_Y = os.getenv('WEAK_Y', 'y4')
WEAK_FACTOR = float(os.getenv('WEAK_FACTOR', '0.4'))  # 0~1 배

# 일시적인 구름/음영 이벤트 타이머
_transient_dip_until = 0.0


def stable_factor(key: str, lo: float = 0.9, hi: float = 1.1) -> float:
    """축/센서별로 실행마다 일관된 미세 가중(±10%)을 부여."""
    h = sum(ord(c) for c in key) % 1000
    r = h / 999.0
    return lo + (hi - lo) * r


def diurnal_factor(now: float) -> float:
    """인공 낮/밤 주기. 0~1.2 범위에서 변동(낮에 더 높게)."""
    if CYCLE_SEC <= 0:
        return 1.0
    phase = (now % CYCLE_SEC) / CYCLE_SEC  # 0..1
    # half-sine from 0..pi (0~1), 밤은 0 근처, 낮은 1 근처
    import math
    base = max(0.0, math.sin(math.pi * phase))
    return 0.8 + 0.4 * base  # 0.8~1.2


def cloud_factor(now: float) -> float:
    """저주파 변동 + 간헐적 일시 감소 이벤트."""
    global _transient_dip_until
    import math
    slow = 1.0
    if CLOUD_PERIOD_SEC > 0 and CLOUD_AMPL > 0:
        slow = 1.0 - CLOUD_AMPL * (0.5 + 0.5 * math.sin(2 * math.pi * now / CLOUD_PERIOD_SEC))
        # slow: 1-CLOUD_AMPL..1+CLOUD_AMPL -> 음영 관점으로 1-CLOUD_AMPL..1 범위가 더 자연스러움
        slow = max(0.6, min(1.0, slow))
    # 간헐적 dip 트리거
    if time.time() > _transient_dip_until and random.random() < 0.003:
        _transient_dip_until = now + random.uniform(1.5, 4.0)
    transient = 1.0
    if now < _transient_dip_until:
        transient = random.uniform(0.6, 0.85)
    return max(0.5, min(1.05, slow * transient))


def sensor_factor(addr: str) -> float:
    """센서별 미세 보정(±3%)."""
    return stable_factor(addr, 0.97, 1.03)


def synth_power(axis: str, now: float) -> float:
    base = BASE_X if axis.startswith('x') else BASE_Y
    jitter = random.uniform(-JITTER_MW, JITTER_MW)
    # 축별 일관 가중 + 주기/구름 요인
    axis_gain = stable_factor(axis, 0.9, 1.1)
    env = diurnal_factor(now) * cloud_factor(now)
    val = base * axis_gain * env + jitter
    # 약화 축 적용
    if axis == WEAK_X or axis == WEAK_Y:
        val *= WEAK_FACTOR
    # 드문 이상치(스파이크/딥) 추가(소량)
    if random.random() < OUTLIER_PROB:
        val *= random.uniform(0.75, 1.25)
    return max(0.0, val)


def make_record(axis: str) -> dict:
    now = time.time()
    addr = random.choice(SENSORS)
    power_mw = synth_power(axis, now) * sensor_factor(addr)
    # 전류/전압 상관 유지: P ≈ V*I, 기준 V≈5V
    current_ma = power_mw / 5.0 + random.uniform(-4.5, 4.5)
    # 부하 전압은 약간의 상관(출력 높을수록 소폭 상승), 범위 제한
    load_v = 5.0 + 0.12 * ((power_mw / BASE_Y) - 1.0) + random.uniform(-0.15, 0.15)
    load_v = max(4.6, min(5.4, load_v))
    bus_v = load_v + random.uniform(-0.03, 0.03)
    # 션트 전압은 전류에 비례하는 경향
    shunt_mv = max(0.05, current_ma * 0.001 + random.uniform(-0.03, 0.06))
    acc = accum_energy[axis] + power_mw / 3600.0  # 초당 mW -> mWh 누적
    accum_energy[axis] = acc
    return {
        "board_id": BOARD_ID,
        "sensorAddress": addr,
        "axis": axis,
        "busVoltage": round(bus_v, 3),
        "shuntVoltage": round(shunt_mv, 3),
        "loadVoltage": round(load_v, 3),
        "current_mA": round(current_ma, 3),
        "power_mW": round(power_mw, 3),
        "accumulatedEnergy_mWh": round(acc, 3),
    }


def main():
    print(f"테스트 데이터 생성 시작 -> {TEST_FILE} (BOARD_ID={BOARD_ID})")
    os.makedirs(os.path.dirname(TEST_FILE), exist_ok=True) if os.path.dirname(TEST_FILE) else None
    with open(TEST_FILE, 'a', encoding='utf-8') as f:
        while True:
            # x축 5개 + y축 6개 한 사이클 기록
            for axis in X_AXES + Y_AXES:
                rec = make_record(axis)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                time.sleep(0.05)
            # 사이클 간 간격
            time.sleep(0.3)


if __name__ == '__main__':
    main()
