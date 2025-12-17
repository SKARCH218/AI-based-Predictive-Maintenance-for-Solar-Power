"""
체크포인트(checkpoints/*.pth)를 불러와서 ModelManager 형식으로 저장합니다.
사용법:
  /path/to/venv/bin/python scripts/save_checkpoint_to_model.py
"""
import torch
from pathlib import Path
from datetime import datetime

from ml.cnn_model import SolarPanelCNN, ModelManager

CHECKPOINT_DIR = Path('checkpoints')
MODELS_DIR = Path('models')

# 가장 최근 체크포인트 찾기
ckpt_files = sorted(CHECKPOINT_DIR.glob('*.pth'), key=lambda p: p.stat().st_mtime)
if not ckpt_files:
    print('체크포인트 파일이 없습니다. 먼저 학습을 진행하세요.')
    raise SystemExit(1)

latest = ckpt_files[-1]
print(f'최근 체크포인트: {latest}')

# 체크포인트 로드
ckpt = torch.load(latest, map_location='cpu')
state_dict = ckpt.get('model_state_dict', None)
if state_dict is None:
    print('체크포인트에 model_state_dict가 없습니다.')
    raise SystemExit(1)

# 모델 인스턴스 생성 (기본 구성 사용)
model = SolarPanelCNN(num_classes=4, in_channels=3)
model.load_state_dict(state_dict)

# 모델 저장
manager = ModelManager(models_dir=str(MODELS_DIR))
version = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
metadata = {
    'source_checkpoint': str(latest),
    'converted_at': datetime.now().isoformat()
}
manager.save_model(model, version=version, metadata=metadata)
print('변환 및 저장 완료:', version)
