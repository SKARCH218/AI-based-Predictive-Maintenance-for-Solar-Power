"""
태양광 패널 이상 탐지를 위한 CNN 모델

아키텍처:
1. ResNetLike: 잔차 연결 기반 경량 모델
2. EfficientNetLike: 복합 스케일링 기반 모델
3. CustomCNN: 태양광 특화 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import json
from pathlib import Path


class ResidualBlock(nn.Module):
    """잔차 블록 (ResNet 스타일)"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut 연결
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SolarPanelCNN(nn.Module):
    """
    태양광 패널 이상 탐지 CNN 모델
    
    입력: (batch, 3, 64, 64) - 3채널 GAF 이미지
    출력: (batch, num_classes) - NORMAL, WARNING, ALERT, CRITICAL
    """
    
    def __init__(
        self, 
        num_classes: int = 4,
        in_channels: int = 3,
        base_channels: int = 32,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # 초기 컨볼루션
        self.conv1 = nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # 잔차 블록들
        self.layer1 = self._make_layer(base_channels, base_channels, 2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 8, num_classes)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        """잔차 레이어 생성"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """He 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 입력: (batch, 3, 64, 64)
        x = self.conv1(x)           # → (batch, 32, 32, 32)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)         # → (batch, 32, 16, 16)
        
        x = self.layer1(x)          # → (batch, 32, 16, 16)
        x = self.layer2(x)          # → (batch, 64, 8, 8)
        x = self.layer3(x)          # → (batch, 128, 4, 4)
        x = self.layer4(x)          # → (batch, 256, 2, 2)
        
        x = self.avgpool(x)         # → (batch, 256, 1, 1)
        x = torch.flatten(x, 1)     # → (batch, 256)
        x = self.dropout(x)
        x = self.fc(x)              # → (batch, num_classes)
        
        return x
    
    def predict_proba(self, x):
        """확률 예측"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x):
        """클래스 예측"""
        proba = self.predict_proba(x)
        return torch.argmax(proba, dim=1)


class AttentionBlock(nn.Module):
    """어텐션 메커니즘 블록"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        
    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention


class SolarPanelCNNWithAttention(nn.Module):
    """
    어텐션 메커니즘을 추가한 향상된 모델
    
    이상 패턴의 위치를 강조하여 성능 향상
    """
    
    def __init__(
        self, 
        num_classes: int = 4,
        in_channels: int = 3,
        base_channels: int = 32,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Feature Extractor (SolarPanelCNN과 동일)
        self.conv1 = nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(base_channels, base_channels, 2, stride=1)
        self.attention1 = AttentionBlock(base_channels)
        
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.attention2 = AttentionBlock(base_channels * 2)
        
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.attention3 = AttentionBlock(base_channels * 4)
        
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 8, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.attention1(x)  # 어텐션 적용
        
        x = self.layer2(x)
        x = self.attention2(x)
        
        x = self.layer3(x)
        x = self.attention3(x)
        
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x):
        proba = self.predict_proba(x)
        return torch.argmax(proba, dim=1)


class ModelManager:
    """모델 저장/로드 관리자"""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def save_model(
        self, 
        model: nn.Module, 
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        모델 저장
        
        Args:
            model: PyTorch 모델
            version: 버전 문자열 (예: "v1.0.0")
            metadata: 메타데이터 (정확도, 학습 날짜 등)
        """
        model_path = self.models_dir / f"solar_cnn_{version}.pth"
        meta_path = self.models_dir / f"solar_cnn_{version}_meta.json"
        
        # 모델 저장
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'model_config': {
                'num_classes': model.num_classes,
                'in_channels': model.in_channels,
            }
        }, model_path)
        
        # 메타데이터 저장
        if metadata is None:
            metadata = {}
        
        metadata['version'] = version
        metadata['model_class'] = model.__class__.__name__
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ 모델 저장 완료: {model_path}")
        print(f"✓ 메타데이터: {meta_path}")
    
    def load_model(
        self, 
        version: str,
        model_class: Optional[nn.Module] = None,
        device: str = 'cpu'
    ) -> nn.Module:
        """
        모델 로드
        
        Args:
            version: 버전 문자열
            model_class: 모델 클래스 (None이면 자동 추론)
            device: 'cpu' 또는 'cuda'
        
        Returns:
            로드된 모델
        """
        model_path = self.models_dir / f"solar_cnn_{version}.pth"
        meta_path = self.models_dir / f"solar_cnn_{version}_meta.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일 없음: {model_path}")
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=device)
        
        # 모델 인스턴스 생성
        if model_class is None:
            class_name = checkpoint['model_class']
            if class_name == 'SolarPanelCNN':
                model_class = SolarPanelCNN
            elif class_name == 'SolarPanelCNNWithAttention':
                model_class = SolarPanelCNNWithAttention
            else:
                raise ValueError(f"Unknown model class: {class_name}")
        
        config = checkpoint['model_config']
        model = model_class(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✓ 모델 로드 완료: {model_path}")
        
        # 메타데이터 출력
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            print(f"  메타데이터: {metadata}")
        
        return model
    
    def list_versions(self) -> list:
        """저장된 모델 버전 목록"""
        versions = []
        for path in self.models_dir.glob("solar_cnn_*.pth"):
            version = path.stem.replace("solar_cnn_", "")
            versions.append(version)
        return sorted(versions)
    
    def get_metadata(self, version: str) -> Dict[str, Any]:
        """
        모델 메타데이터 로드
        
        Args:
            version: 버전 문자열
        
        Returns:
            메타데이터 딕셔너리
        """
        meta_path = self.models_dir / f"solar_cnn_{version}_meta.json"
        
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                return json.load(f)
        else:
            # 기본 메타데이터 반환
            return {
                'version': version,
                'timestamp': '',
                'metrics': {}
            }


if __name__ == '__main__':
    print("=== CNN 모델 테스트 ===\n")
    
    # 모델 생성
    model = SolarPanelCNN(num_classes=4, in_channels=3)
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 더미 입력 테스트
    dummy_input = torch.randn(4, 3, 64, 64)  # batch_size=4
    output = model(dummy_input)
    print(f"입력 shape: {dummy_input.shape}")
    print(f"출력 shape: {output.shape}")
    
    proba = model.predict_proba(dummy_input)
    print(f"확률 shape: {proba.shape}")
    print(f"확률 합: {proba.sum(dim=1)}")
    
    # 어텐션 모델 테스트
    print("\n=== 어텐션 모델 테스트 ===\n")
    attention_model = SolarPanelCNNWithAttention(num_classes=4)
    print(f"어텐션 모델 파라미터 수: {sum(p.numel() for p in attention_model.parameters()):,}")
    
    output_att = attention_model(dummy_input)
    print(f"어텐션 모델 출력 shape: {output_att.shape}")
    
    # 모델 저장/로드 테스트
    print("\n=== 모델 저장/로드 테스트 ===\n")
    manager = ModelManager()
    
    # 저장
    manager.save_model(
        model, 
        version="v0.1.0_test",
        metadata={'accuracy': 0.95, 'test': True}
    )
    
    # 로드
    loaded_model = manager.load_model("v0.1.0_test")
    
    # 동일성 검증
    with torch.no_grad():
        out1 = model(dummy_input)
        out2 = loaded_model(dummy_input)
        print(f"저장/로드 후 출력 차이: {torch.abs(out1 - out2).max().item():.6f}")
