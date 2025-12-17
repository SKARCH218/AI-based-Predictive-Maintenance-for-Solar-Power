"""
CNN 예측을 위한 데이터베이스 테이블 생성 스크립트
"""

import sqlite3

DB_FILE = 'solardata.db'


def setup_cnn_tables():
    """CNN 관련 테이블 생성"""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    
    # CNN 예측 결과 테이블
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cnn_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            board_id TEXT,
            axis TEXT,
            status TEXT NOT NULL,
            confidence REAL NOT NULL,
            prob_normal REAL,
            prob_warning REAL,
            prob_alert REAL,
            prob_critical REAL,
            mean_power REAL,
            std_power REAL,
            reliable INTEGER,
            model_version TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 인덱스 생성 (조회 성능 향상)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_cnn_timestamp 
        ON cnn_predictions(timestamp DESC)
    """)
    
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_cnn_board 
        ON cnn_predictions(board_id, timestamp DESC)
    """)
    
    conn.commit()
    conn.close()
    
    print("✓ CNN 예측 테이블 생성 완료")


if __name__ == '__main__':
    print("=== CNN 데이터베이스 설정 ===\n")
    setup_cnn_tables()
    print("\n완료!")
