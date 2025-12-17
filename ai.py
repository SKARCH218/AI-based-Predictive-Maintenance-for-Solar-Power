import sqlite3
import statistics
import time
from typing import Optional, Tuple
import json

DB_FILE = 'solardata.db'


def ensure_tables():
    """Ensure the predictions table exists."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT,              -- NORMAL | WARNING | ALERT
            reason TEXT,              -- human readable reason
            power_mw REAL,            -- latest observed value
            baseline REAL,            -- rolling median/mean baseline
            threshold REAL,           -- dynamic threshold used
            severity REAL             -- 0~1 severity score
        )
        '''
    )
    conn.commit()
    conn.close()


def read_recent_power(n: int = 60, board_id: str | None = None) -> list[float]:
    """Read the most recent N power_mw values from power_data (descending), optionally per board."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    if board_id:
        cur.execute(
            "SELECT power_mw FROM power_data WHERE power_mw IS NOT NULL AND board_id = ? ORDER BY timestamp DESC LIMIT ?",
            (board_id, n),
        )
    else:
        cur.execute(
            "SELECT power_mw FROM power_data WHERE power_mw IS NOT NULL ORDER BY timestamp DESC LIMIT ?",
            (n,),
        )
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows][::-1]  # ascending by time


def compute_baseline(values: list[float]) -> Optional[Tuple[float, float]]:
    """Return (median, mad) as a robust baseline. If insufficient data, return None."""
    if len(values) < 10:
        return None
    med = statistics.median(values)
    abs_dev = [abs(v - med) for v in values]
    mad = statistics.median(abs_dev) if abs_dev else 0.0
    return med, mad


def analyze_once(board_id: str | None = None) -> Optional[dict]:
    """Analyze recent data and insert a prediction if applicable. Returns the inserted record as dict."""
    ensure_tables()
    values = read_recent_power(120, board_id=board_id)
    if not values:
        return None

    latest = values[-1]
    baseline = compute_baseline(values[:-1] or values)
    if baseline is None:
        # Not enough data yet; treat as NORMAL without insert to avoid noise
        return None

    med, mad = baseline
    # Dynamic threshold: med - k * (1.4826 * MAD) ~ robust sigma
    robust_sigma = 1.4826 * (mad if mad > 1e-9 else max(1.0, med * 0.01))
    k_warning = 2.5
    k_alert = 4.0
    warn_thr = med - k_warning * robust_sigma
    alert_thr = med - k_alert * robust_sigma

    if latest <= alert_thr:
        status = 'ALERT'
    elif latest <= warn_thr:
        status = 'WARNING'
    else:
        status = 'NORMAL'

    # severity 0~1, map linearly below warn threshold
    if status == 'NORMAL':
        severity = 0.0
    elif status == 'WARNING':
        severity = min(1.0, max(0.0, (warn_thr - latest) / max(1e-6, warn_thr - alert_thr))) * 0.6
    else:  # ALERT
        severity = 0.6 + min(0.4, (alert_thr - latest) / max(1e-6, med))

    reason = f"latest={latest:.2f}mW, baseline={med:.2f}±{robust_sigma:.2f}, warn<={warn_thr:.2f}, alert<={alert_thr:.2f}"

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    # 확장된 predictions 스키마: board_id, cells 컬럼이 없다면 런타임에 추가
    try:
        cur.execute("PRAGMA table_info(predictions)")
        cols = [row[1] for row in cur.fetchall()]
        if 'board_id' not in cols:
            cur.execute("ALTER TABLE predictions ADD COLUMN board_id TEXT")
            conn.commit()
        if 'cells' not in cols:
            cur.execute("ALTER TABLE predictions ADD COLUMN cells TEXT")
            conn.commit()
    except Exception:
        pass

    # 축 기반 간단 셀 이상 추정(보드별 축값이 충분할 때)
    cells_json = None
    try:
        cells = predict_cells(board_id)
        cells_json = json.dumps(cells) if cells else None
    except Exception:
        cells_json = None

    cur.execute(
        """
        INSERT INTO predictions (status, reason, power_mw, baseline, threshold, severity, board_id, cells)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (status, reason, latest, med, warn_thr, severity, board_id, cells_json)
    )
    conn.commit()
    if board_id:
        cur.execute("SELECT * FROM predictions WHERE board_id = ? ORDER BY id DESC LIMIT 1", (board_id,))
    else:
        cur.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def run_loop(interval_sec: float = 5.0):
    """Run analyze periodically."""
    while True:
        try:
            boards = get_board_ids()
            # 전체(집계) + 보드별 수행
            analyze_once(None)
            for b in boards:
                try:
                    analyze_once(b)
                except Exception as e:
                    print(f"[AI] board {b} analyze error: {e}")
        except Exception as e:
            print(f"[AI] analyze error: {e}")
        time.sleep(interval_sec)


def get_board_ids() -> list[str]:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT board_id FROM power_data WHERE board_id IS NOT NULL")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows


def predict_cells(board_id: Optional[str]) -> list[dict]:
    """보드별 축 최신값을 기반으로 X/Y 축 중 하위 축들을 찾아 교차 지점을 셀로 제안."""
    # 최신 축별 값 수집
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if board_id:
        cur.execute(
            """
            SELECT p.* FROM power_data p
            JOIN (
                SELECT axis, MAX(timestamp) AS mx FROM power_data WHERE board_id = ? AND axis IS NOT NULL GROUP BY axis
            ) t ON p.axis = t.axis AND p.timestamp = t.mx AND p.board_id = ?
            """,
            (board_id, board_id),
        )
    else:
        cur.execute(
            """
            SELECT p.* FROM power_data p
            JOIN (
                SELECT axis, MAX(timestamp) AS mx FROM power_data WHERE axis IS NOT NULL GROUP BY axis
            ) t ON p.axis = t.axis AND p.timestamp = t.mx
            """
        )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return []
    # 그룹 나누기
    x_vals = {}
    y_vals = {}
    for r in rows:
        axis = (r["axis"] or "").lower()
        if axis.startswith("x"):
            x_vals[axis] = r["power_mw"]
        elif axis.startswith("y"):
            y_vals[axis] = r["power_mw"]

    def low_axes(vals: dict[str, float]) -> list[str]:
        if not vals:
            return []
        vs = list(vals.values())
        med = statistics.median(vs)
        mad = statistics.median([abs(v - med) for v in vs]) if len(vs) > 1 else 0.0
        robust_sigma = 1.4826 * (mad if mad > 1e-9 else max(1.0, med * 0.01))
        thr = med - 2.5 * robust_sigma
        return [k for k, v in vals.items() if v <= thr]

    lows_x = low_axes(x_vals)
    lows_y = low_axes(y_vals)
    cells = []
    # 좌표는 축명에서 숫자만 파싱해 r,c 로 저장
    def idx(a: str) -> Optional[int]:
        s = ''.join(ch for ch in a if ch.isdigit())
        return int(s) if s.isdigit() else None
    for ax in lows_x:
        r = idx(ax)
        if r is None:
            continue
        for ay in lows_y:
            c = idx(ay)
            if c is None:
                continue
            cells.append({"x": ax, "y": ay, "row": r, "col": c})
    return cells


if __name__ == '__main__':
    run_loop(5.0)
