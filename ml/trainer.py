"""
CNN 모델 학습 파이프라인

기능:
1. 데이터셋 생성 (이미지 + 라벨)
2. 학습/검증 분할
3. 모델 학습 (조기 종료, 체크포인트)
4. 성능 평가 (정확도, F1, Confusion Matrix)
5. 텐서보드 로깅
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from ml.image_generator import SolarDataImageGenerator
from ml.cnn_model import SolarPanelCNN, SolarPanelCNNWithAttention, ModelManager


class SolarPanelDataset(Dataset):
    """태양광 패널 이미지 데이터셋"""
    
    # 라벨 매핑
    LABEL_MAP = {
        'NORMAL': 0,
        'WARNING': 1,
        'ALERT': 2,
        'CRITICAL': 3  # 추가 클래스 (매우 심각한 고장)
    }
    
    REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
    
    def __init__(self, images: List[np.ndarray], labels: List[str], metadata: List[dict]):
        """
        Args:
            images: 이미지 배열 리스트 (H, W, C)
            labels: 라벨 문자열 리스트
            metadata: 메타데이터 딕셔너리 리스트
        """
        self.images = images
        self.labels = [self.LABEL_MAP.get(label, 0) for label in labels]
        self.metadata = metadata
        
        # 유효성 검사
        assert len(self.images) == len(self.labels) == len(self.metadata)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # NumPy → Torch Tensor
        # (H, W, C) → (C, H, W)
        if image.ndim == 2:
            image = image[np.newaxis, :, :]  # (H, W) → (1, H, W)
        else:
            image = np.transpose(image, (2, 0, 1))  # (H, W, C) → (C, H, W)
        
        image = torch.from_numpy(image).float()
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label, self.metadata[idx]


class Trainer:
    """CNN 모델 학습기"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        save_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # 학습 히스토리
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """1 에폭 학습"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels, _ in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 통계
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        save_best: bool = True
    ):
        """
        모델 학습
        
        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            epochs: 최대 에폭 수
            early_stopping_patience: 조기 종료 인내심
            save_best: 최고 모델 저장 여부
        """
        print(f"{'='*60}")
        print(f"학습 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # 학습
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 검증
            val_loss, val_acc = self.validate(val_loader)
            
            # 스케줄러 업데이트
            self.scheduler.step(val_loss)
            
            # 히스토리 저장
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 출력
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}\n")
            
            # Best 모델 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                if save_best:
                    self.save_checkpoint(f'best_epoch{epoch+1}')
                    print(f"  ✓ Best 모델 저장됨 (val_loss: {val_loss:.4f})\n")
            else:
                patience_counter += 1
            
            # 조기 종료
            if patience_counter >= early_stopping_patience:
                print(f"\n조기 종료: {early_stopping_patience} 에폭 동안 개선 없음")
                break
        
        print(f"\n{'='*60}")
        print(f"학습 완료!")
        print(f"Best Epoch: {self.best_epoch} (val_loss: {self.best_val_loss:.4f})")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_checkpoint(self, name: str):
        """체크포인트 저장"""
        checkpoint_path = self.save_dir / f"{name}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }, checkpoint_path)
    
    def load_checkpoint(self, name: str):
        """체크포인트 로드"""
        checkpoint_path = self.save_dir / f"{name}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        
        print(f"✓ 체크포인트 로드: {checkpoint_path}")
    
    def plot_history(self, save_path: Optional[str] = None):
        """학습 히스토리 플롯"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss History')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy History')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 학습 히스토리 저장: {save_path}")
        else:
            plt.show()


class Evaluator:
    """모델 평가기"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        테스트 데이터 평가
        
        Returns:
            평가 메트릭 딕셔너리
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 정확도
        accuracy = 100 * (all_preds == all_labels).sum() / len(all_labels)
        
        # Classification Report
        class_names = list(SolarPanelDataset.LABEL_MAP.keys())
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        return results
    
    def print_report(self, results: Dict):
        """평가 결과 출력"""
        print(f"\n{'='*60}")
        print(f"모델 평가 결과")
        print(f"{'='*60}\n")
        
        print(f"전체 정확도: {results['accuracy']:.2f}%\n")
        
        report = results['classification_report']
        class_names = list(SolarPanelDataset.LABEL_MAP.keys())
        
        print(f"{'클래스':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 60)
        
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"{class_name:<12} {metrics['precision']:<12.3f} {metrics['recall']:<12.3f} "
                      f"{metrics['f1-score']:<12.3f} {int(metrics['support']):<10}")
        
        print(f"\n{'='*60}\n")
    
    def plot_confusion_matrix(self, results: Dict, save_path: Optional[str] = None):
        """Confusion Matrix 시각화"""
        cm = results['confusion_matrix']
        class_names = list(SolarPanelDataset.LABEL_MAP.keys())
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Confusion Matrix 저장: {save_path}")
        else:
            plt.show()


def create_dataloaders(
    db_path: str = 'solardata.db',
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    window_size: int = 120,
    max_samples: int = 1000,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    데이터 로더 생성
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    print("데이터셋 생성 중...")
    
    # 이미지 생성
    generator = SolarDataImageGenerator(db_path=db_path, method='multi')
    samples = generator.generate_training_batch(n_samples=max_samples)
    
    if not samples:
        raise ValueError("생성된 샘플이 없습니다. DB에 데이터가 충분한지 확인하세요.")
    
    # 언패킹
    images, labels, metadata = zip(*samples)
    images = list(images)
    labels = list(labels)
    metadata = list(metadata)
    
    print(f"생성된 샘플 수: {len(images)}")
    print(f"라벨 분포: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # 데이터셋 생성
    dataset = SolarPanelDataset(images, labels, metadata)
    
    # Train/Val/Test 분할
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}\n")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    print("=== 학습 파이프라인 테스트 ===\n")
    
    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # 데이터 로더 생성
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=16,
            max_samples=100  # 테스트용 적은 샘플
        )
        
        # 모델 생성
        model = SolarPanelCNN(num_classes=4, in_channels=3)
        
        # 트레이너 생성
        trainer = Trainer(model, device=device, learning_rate=1e-3)
        
        # 학습
        history = trainer.fit(
            train_loader, 
            val_loader, 
            epochs=5,  # 테스트용
            early_stopping_patience=3
        )
        
        # 히스토리 플롯
        trainer.plot_history(save_path='training_history.png')
        
        # 평가
        evaluator = Evaluator(model, device=device)
        results = evaluator.evaluate(test_loader)
        evaluator.print_report(results)
        evaluator.plot_confusion_matrix(results, save_path='confusion_matrix.png')
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("DB에 충분한 데이터가 있는지 확인하세요.")
