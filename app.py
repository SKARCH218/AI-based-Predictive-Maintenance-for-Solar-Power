from flask import Flask, render_template, jsonify
from flask import request
import sqlite3
import ai
import os
import json
import base64
from io import BytesIO

app = Flask(__name__)
DB_FILE = 'solardata.db'

# CNN 모델 관련 임포트 (Lazy loading)
CNN_ENABLED = True
CNN_LOADING = True
cnn_predictor = None
image_generator = None

def init_cnn_models():
    """CNN 모델 lazy 초기화 (최초 1회만 실행)"""
    global CNN_ENABLED, CNN_LOADING, cnn_predictor, image_generator
    
    if CNN_ENABLED:
        return True
    
    if CNN_LOADING:
        # 이미 로딩 중이면 대기
        return False
    
    CNN_LOADING = True
    
    try:
        print("CNN 모듈 로딩 중... (약 5-10초 소요)")
        from ml.predictor import RealTimePredictor
        from ml.image_generator import SolarDataImageGenerator
        
        cnn_predictor = RealTimePredictor(db_path=DB_FILE)
        image_generator = SolarDataImageGenerator(db_path=DB_FILE)
        CNN_ENABLED = True
        print("✓ CNN 모듈 로드 완료")
        return True
    except ImportError as e:
        print(f"CNN 모듈을 로드할 수 없습니다: {e}")
        print("기본 기능만 사용 가능합니다.")
        return False
    except FileNotFoundError as e:
        print(f"모델 파일이 없습니다: {e}")
        print("먼저 모델을 학습하세요: python -m ml.trainer")
        return False
    except Exception as e:
        print(f"CNN 초기화 오류: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        CNN_LOADING = False

def query_db(query, args=(), one=False):
    """데이터베이스에 쿼리를 실행하고 결과를 반환합니다."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query, args)
        rv = cur.fetchall()
        conn.close()
        return (rv[0] if rv else None) if one else rv
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None

@app.route('/')
def index():
    """메인 페이지를 렌더링합니다."""
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    """가장 최근의 데이터 포인트 하나를 반환합니다. board_id로 필터 가능."""
    board = request.args.get('board_id')
    if board:
        data = query_db("SELECT * FROM power_data WHERE board_id = ? ORDER BY timestamp DESC LIMIT 1", (board,), one=True)
    else:
        data = query_db("SELECT * FROM power_data ORDER BY timestamp DESC LIMIT 1", one=True)
    if data:
        return jsonify(dict(data))
    return jsonify({})

@app.route('/api/history')
def get_history():
    """최근 데이터 포인트를 반환합니다. board_id로 필터 가능."""
    board = request.args.get('board_id')
    if board:
        history = query_db("SELECT timestamp, power_mw FROM power_data WHERE board_id = ? ORDER BY timestamp DESC LIMIT 20", (board,))
    else:
        history = query_db("SELECT timestamp, power_mw FROM power_data ORDER BY timestamp DESC LIMIT 20")
    if history:
        # JSON 직렬화를 위해 데이터를 [ {x: ..., y: ...} ] 형태로 변환
        formatted_history = [{'x': row['timestamp'], 'y': row['power_mw']} for row in history]
        return jsonify(formatted_history)
    return jsonify([])

@app.route('/api/boards')
def list_boards():
    """사용 가능한 board_id 목록."""
    rows = query_db("SELECT DISTINCT board_id FROM power_data WHERE board_id IS NOT NULL ORDER BY board_id")
    return jsonify([r['board_id'] for r in rows] if rows else [])

@app.route('/api/axis/latest')
def axis_latest():
    """축별 최신 측정값을 반환합니다. board_id로 필터 가능."""
    board = request.args.get('board_id')
    if board:
        rows = query_db(
            """
            SELECT p.* FROM power_data p
            JOIN (
                SELECT axis, MAX(id) AS mxid
                FROM power_data
                WHERE board_id = ? AND axis IS NOT NULL
                GROUP BY axis
            ) t ON p.id = t.mxid
            ORDER BY p.axis
            """,
            (board,),
        )
    else:
        rows = query_db(
            """
            SELECT p.* FROM power_data p
            JOIN (
                SELECT axis, MAX(id) AS mxid
                FROM power_data
                WHERE axis IS NOT NULL
                GROUP BY axis
            ) t ON p.id = t.mxid
            ORDER BY p.axis
            """
        )
    return jsonify([dict(r) for r in rows] if rows else [])
@app.route('/api/prediction/latest')
def get_latest_prediction():
    """최근 예측 결과 1건을 반환합니다."""
    try:
        board = request.args.get('board_id')
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        if board:
            cur.execute("SELECT * FROM predictions WHERE board_id = ? ORDER BY timestamp DESC, id DESC LIMIT 1", (board,))
        else:
            cur.execute("SELECT * FROM predictions ORDER BY timestamp DESC, id DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        return jsonify(dict(row) if row else {})
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return jsonify({})

@app.route('/api/prediction/run-once')
def run_prediction_once():
    """수동으로 한 번 예측을 수행하고 결과를 반환합니다."""
    try:
        board = request.args.get('board_id')
        result = ai.analyze_once(board_id=board)
        return jsonify(result or {})
    except Exception as e:
        return jsonify({"error": str(e)})


# ===== CNN 관련 API 엔드포인트 =====

@app.route('/api/cnn/predict')
def cnn_predict():
    """CNN 모델을 사용한 실시간 예측"""
    if not init_cnn_models():
        return jsonify({"error": "CNN 모델이 로드되지 않았습니다"}), 503
    
    try:
        board = request.args.get('board_id')
        result = cnn_predictor.predict_current_state(board_id=board)
        
        if result:
            return jsonify({
                'status': result['status'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'board_id': result.get('board_id'),
                'timestamp': result.get('timestamp'),
                'model_version': cnn_predictor.model_metadata.get('version', 'unknown')
            })
        else:
            return jsonify({"error": "예측 데이터가 충분하지 않습니다"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/cnn/replacement-prediction')
def cnn_replacement_prediction():
    """확률 기반 교체 날짜 예측"""
    if not init_cnn_models():
        return jsonify({"error": "CNN 모델이 로드되지 않았습니다"}), 503
    
    try:
        from datetime import datetime, timedelta
        import numpy as np
        
        board = request.args.get('board_id')
        
        # 최근 예측 히스토리 가져오기
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        if board:
            cur.execute("""
                SELECT status, confidence, prob_warning, prob_alert, prob_critical, 
                       timestamp, reliable
                FROM cnn_predictions 
                WHERE board_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 30
            """, (board,))
        else:
            cur.execute("""
                SELECT status, confidence, prob_warning, prob_alert, prob_critical, 
                       timestamp, reliable
                FROM cnn_predictions 
                ORDER BY timestamp DESC 
                LIMIT 30
            """)
        
        history = cur.fetchall()
        conn.close()
        
        if not history or len(history) < 3:
            return jsonify({"error": "교체 날짜를 예측하기에 충분한 데이터가 없습니다"}), 404
        
        # 현재 상태 분석
        latest = dict(history[0])
        current_status = latest['status']
        
        # 위험도 점수 계산 (0~100)
        prob_warning = latest.get('prob_warning', 0) or 0
        prob_alert = latest.get('prob_alert', 0) or 0
        prob_critical = latest.get('prob_critical', 0) or 0
        
        risk_score = (prob_warning * 25 + prob_alert * 50 + prob_critical * 100)
        
        # 최근 트렌드 분석 (열화 속도)
        risk_scores = []
        for row in history:
            w = row['prob_warning'] or 0
            a = row['prob_alert'] or 0
            c = row['prob_critical'] or 0
            risk_scores.append(w * 25 + a * 50 + c * 100)
        
        # 열화 속도 계산 (선형 회귀)
        if len(risk_scores) >= 3:
            x = np.arange(len(risk_scores))
            # 간단한 선형 회귀
            degradation_rate = np.polyfit(x, risk_scores, 1)[0]
        else:
            degradation_rate = 0
        
        # 교체 시점 예측
        critical_threshold = 80  # 위험도 80 이상이면 교체 필요
        
        if risk_score >= critical_threshold:
            days_remaining = 0
            replacement_date = datetime.now()
            status_message = "즉시 교체 필요"
            risk_level = "위험"
        elif degradation_rate > 0.1:
            # 현재 위험도에서 임계값까지 도달하는데 걸리는 시간 예측
            days_to_critical = (critical_threshold - risk_score) / (degradation_rate * 0.5)
            days_remaining = max(1, int(days_to_critical))
            replacement_date = datetime.now() + timedelta(days=days_remaining)
            
            if days_remaining <= 7:
                status_message = "긴급 점검 필요"
                risk_level = "높음"
            elif days_remaining <= 30:
                status_message = "주의 관찰 필요"
                risk_level = "중간"
            else:
                status_message = "정상 범위"
                risk_level = "낮음"
        else:
            # 열화가 거의 없거나 개선되는 경우
            days_remaining = 365
            replacement_date = datetime.now() + timedelta(days=365)
            status_message = "양호한 상태"
            risk_level = "낮음"
        
        # 신뢰도 계산
        confidence_level = "높음" if len(history) >= 20 and latest.get('reliable', 0) else "중간"
        
        return jsonify({
            'replacement_date': replacement_date.strftime('%Y년 %m월 %d일'),
            'days_remaining': days_remaining,
            'current_status': current_status,
            'risk_score': round(risk_score, 1),
            'risk_level': risk_level,
            'degradation_rate': f"{degradation_rate:.2f}%/일" if degradation_rate > 0 else "안정",
            'status_message': status_message,
            'confidence': confidence_level,
            'trend_data': risk_scores[:10]  # 최근 10개 트렌드
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/cnn/history')
def cnn_history():
    """CNN 예측 히스토리 조회"""
    try:
        board = request.args.get('board_id')
        limit = int(request.args.get('limit', 50))
        
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        if board:
            cur.execute("""
                SELECT * FROM cnn_predictions 
                WHERE board_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (board, limit))
        else:
            cur.execute("""
                SELECT * FROM cnn_predictions 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
        
        rows = cur.fetchall()
        conn.close()
        
        # probabilities JSON 파싱
        results = []
        for row in rows:
            data = dict(row)
            if data.get('probabilities'):
                try:
                    data['probabilities'] = json.loads(data['probabilities'])
                except:
                    data['probabilities'] = {}
            results.append(data)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/cnn/pattern')
def cnn_pattern():
    """시계열 패턴 이미지 생성 (GAF/MTF)"""
    if not init_cnn_models():
        return jsonify({"error": "이미지 생성기가 로드되지 않았습니다"}), 503
    
    try:
        board = request.args.get('board_id')
        window_size = int(request.args.get('window_size', 64))
        
        # 최근 데이터로 이미지 생성
        timeseries = image_generator.fetch_timeseries(
            board_id=board,
            window_size=window_size,
            limit=1
        )
        
        if not timeseries or len(timeseries) == 0:
            return jsonify({"error": "데이터가 충분하지 않습니다"}), 404
        
        # 이미지 생성
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # GUI 없이 사용
        import matplotlib.pyplot as plt
        
        # fetch_timeseries returns a list of (image, metadata) tuples
        img_tuple = timeseries[0]
        if isinstance(img_tuple, (list, tuple)) and len(img_tuple) >= 1:
            image = img_tuple[0]
            metadata = img_tuple[1] if len(img_tuple) > 1 else {}
        else:
            # backward-compat: if fetch_timeseries returned raw array
            image = img_tuple
            metadata = {}
        
        # 이미지 형태는 (H, W, C) 또는 (C, H, W). 정규화하여 (H, W, 3)으로 맞춥니다.
        if hasattr(image, 'shape'):
            if len(image.shape) == 3 and image.shape[0] == 3:
                # (3, H, W) -> (H, W, 3)
                image = np.transpose(image, (1, 2, 0))
            elif len(image.shape) == 2:
                # 단일 채널 -> 3채널로 복제
                image = np.stack([image, image, image], axis=-1)
        
        # 이미지를 Base64로 인코딩
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis('off')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        # 포함된 메타데이터도 반환
        return jsonify({
            'image': image_base64,
            'method': 'multi-channel GAF/MTF',
            'window_size': window_size,
            'metadata': metadata
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/cnn/model-info')
def cnn_model_info():
    """모델 정보 및 성능 지표"""
    if not init_cnn_models():
        return jsonify({"error": "CNN 모델이 로드되지 않았습니다"}), 503
    
    try:
        # 모델 메타데이터
        metadata = cnn_predictor.model_metadata
        
        # 최근 예측 통계
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) as cnt FROM cnn_predictions")
        prediction_samples = cur.fetchone()[0]
        
        # 학습 데이터 수
        cur.execute("SELECT COUNT(*) as cnt FROM predictions")
        training_samples = cur.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'version': metadata.get('version', 'unknown'),
            'architecture': metadata.get('architecture', 'unknown'),
            'accuracy': metadata.get('metrics', {}).get('accuracy', 0),
            'f1_score': metadata.get('metrics', {}).get('f1_weighted', 0),
            'prediction_samples': prediction_samples,
            'training_samples': training_samples,
            'timestamp': metadata.get('timestamp', '')
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # CNN 모델 사전 로딩 (서버 시작 전)
    print("=" * 60)
    print("태양광 예지보전 시스템 서버 시작")
    print("=" * 60)
    print("\n🔄 CNN 모델 초기화 중...")
    if init_cnn_models():
        print("✅ CNN 기능 활성화\n")
    else:
        print("⚠️  CNN 기능 비활성화 (기본 기능만 사용)\n")
    
    # host='0.0.0.0'으로 설정하여 외부에서도 접속 가능하게 합니다.
    # 테스트 모드: 포트 5001에서 실행
    print("🌐 서버 시작: http://127.0.0.1:5001")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5001, debug=True)
