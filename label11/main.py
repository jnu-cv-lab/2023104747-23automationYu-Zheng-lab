# 修正版 main.py - 修复标签索引问题
import cv2
import mediapipe as mp
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==================== 骨架提取器 ====================
class SkeletonExtractor:
    def __init__(self, target_frames=30):
        self.target_frames = target_frames
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_skeleton_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        skeletons = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                frame_features = []
                for landmark in results.pose_landmarks.landmark:
                    frame_features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                skeletons.append(frame_features)
            else:
                if skeletons:
                    skeletons.append(skeletons[-1])
                else:
                    skeletons.append([0.0] * 132)
        
        cap.release()
        
        if len(skeletons) == 0:
            return None
        
        skeletons = np.array(skeletons)
        skeletons = self.resample_sequence(skeletons)
        skeletons = self.normalize_skeleton(skeletons)
        return skeletons
    
    def resample_sequence(self, sequence):
        current_frames = sequence.shape[0]
        if current_frames == self.target_frames:
            return sequence
        elif current_frames > self.target_frames:
            indices = np.linspace(0, current_frames - 1, self.target_frames, dtype=int)
            return sequence[indices]
        else:
            resampled = np.zeros((self.target_frames, sequence.shape[1]))
            resampled[:current_frames] = sequence
            for i in range(current_frames, self.target_frames):
                resampled[i] = sequence[-1]
            return resampled
    
    def normalize_skeleton(self, skeletons):
        normalized = skeletons.copy()
        for t in range(skeletons.shape[0]):
            frame = skeletons[t].reshape(33, 4)
            left_hip = frame[23, :2]
            right_hip = frame[24, :2]
            hip_center = (left_hip + right_hip) / 2
            left_shoulder = frame[11, :2]
            right_shoulder = frame[12, :2]
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
            if shoulder_width > 0:
                frame[:, :2] = (frame[:, :2] - hip_center) / shoulder_width
            normalized[t] = frame.flatten()
        return normalized
    
    def __del__(self):
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()


# ==================== Transformer模型 ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1)]


class SkeletonTransformer(nn.Module):
    def __init__(self, input_dim=132, d_model=128, nhead=4, num_layers=2, 
                 dim_feedforward=256, num_classes=5, dropout=0.1):  # 改为5类
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# ==================== 数据集类 ====================
class SkeletonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    train_losses, train_accs, val_accs = [], [], []
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        val_acc = 100. * correct / total
        val_accs.append(val_acc)
        
        scheduler.step()
        print(f'Epoch {epoch+1}: Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')
    
    return train_losses, train_accs, val_accs


# ==================== 主程序 ====================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 类别映射 - 使用连续的标签0-4
    class_mapping = {
        'backhand_drive': 0,      # 反手平抽
        'backhand_net_shot': 1,   # 反手网前球
        'forehand_clear': 2,      # 正手高远球
        'forehand_drive': 3,      # 正手平抽
        'forehand_net_shot': 4,   # 正手网前球
    }
    
    # 检查是否已有预处理数据
    if not os.path.exists('processed_data/X_train.npy'):
        print("\n=== 开始预处理数据 ===")
        os.makedirs('processed_data', exist_ok=True)
        extractor = SkeletonExtractor(target_frames=30)
        
        all_skeletons, all_labels = [], []
        
        for class_name, label in class_mapping.items():
            if not os.path.exists(class_name):
                print(f"跳过 {class_name} (文件夹不存在)")
                continue
            
            video_files = [f for f in os.listdir(class_name) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            print(f"处理 {class_name} (标签 {label}): {len(video_files)} 个视频")
            
            for video_file in tqdm(video_files, desc=class_name):
                video_path = os.path.join(class_name, video_file)
                try:
                    skeleton = extractor.extract_skeleton_from_video(video_path)
                    if skeleton is not None and skeleton.shape == (30, 132):
                        all_skeletons.append(skeleton)
                        all_labels.append(label)
                except Exception as e:
                    print(f"处理失败: {video_file}")
        
        if len(all_skeletons) == 0:
            print("错误: 没有成功提取任何骨架数据")
            return
        
        X = np.array(all_skeletons)
        y = np.array(all_labels)
        
        print(f"\n成功处理 {len(X)} 个视频样本")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        np.save('processed_data/X_train.npy', X_train)
        np.save('processed_data/y_train.npy', y_train)
        np.save('processed_data/X_test.npy', X_test)
        np.save('processed_data/y_test.npy', y_test)
        
        # 保存标签映射
        label_map = {v: k for k, v in class_mapping.items() if os.path.exists(k)}
        with open('processed_data/label_map.json', 'w') as f:
            json.dump(label_map, f, indent=2)
        
        print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
    else:
        print("\n=== 加载预处理数据 ===")
        X_train = np.load('processed_data/X_train.npy')
        y_train = np.load('processed_data/y_train.npy')
        X_test = np.load('processed_data/X_test.npy')
        y_test = np.load('processed_data/y_test.npy')
        
        with open('processed_data/label_map.json', 'r') as f:
            label_map = json.load(f)
            label_map = {int(k): v for k, v in label_map.items()}
        
        print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
    
    # 创建数据加载器
    train_loader = DataLoader(SkeletonDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(SkeletonDataset(X_test, y_test), batch_size=16, shuffle=False)
    
    # 创建模型
    model = SkeletonTransformer(num_classes=len(label_map))
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    print("\n=== 开始训练 ===")
    train_losses, train_accs, val_accs = train_model(
        model, train_loader, test_loader, epochs=20, lr=1e-3, device=device
    )
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    # 测试模型
    print("\n=== 测试模型 ===")
    model.eval()
    all_preds, all_labels_list = [], []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels_list.extend(batch_y.numpy())
    
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels_list)) / len(all_labels_list)
    print(f"\n测试集准确率: {accuracy:.2f}%")
    
    # 输出分类报告
    target_names = [label_map[i] for i in sorted(label_map.keys())]
    print("\n分类报告:")
    print(classification_report(all_labels_list, all_preds, target_names=target_names))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels_list, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # 保存模型
    torch.save(model.state_dict(), 'skeleton_transformer.pth')
    print("\n模型已保存为 skeleton_transformer.pth")
    
    print("\n=== 实验完成！===")

if __name__ == "__main__":
    main()