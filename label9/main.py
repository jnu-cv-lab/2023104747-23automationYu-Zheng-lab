import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# ===================== 环境 & 设备 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 数据集加载 =====================
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

full_train = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size
train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ===================== CNN 模型 =====================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ===================== 任务3：学习率对比（Adam）=====================
def train_with_lr(lr, epochs=5):
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(val_acc)

        print(f"[lr={lr}] Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # 测试阶段
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_acc = 100 * test_correct / test_total

    return model, train_loss_list, train_acc_list, val_loss_list, val_acc_list, test_acc

# 训练三个学习率
print("===== 开始训练 lr=0.1 =====")
_, tl_01, ta_01, vl_01, va_01, te_01 = train_with_lr(lr=0.1, epochs=5)

print("\n===== 开始训练 lr=0.01 =====")
_, tl_001, ta_001, vl_001, va_001, te_001 = train_with_lr(lr=0.01, epochs=5)

print("\n===== 开始训练 lr=0.001 =====")
model, tl_0001, ta_0001, vl_0001, va_0001, te_0001 = train_with_lr(lr=0.001, epochs=5)

# 绘制学习率对比曲线
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.plot(tl_01, marker='o', label='lr=0.1')
plt.plot(tl_001, marker='s', label='lr=0.01')
plt.plot(tl_0001, marker='^', label='lr=0.001')
plt.title("训练损失对比")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(vl_01, marker='o', label='lr=0.1')
plt.plot(vl_001, marker='s', label='lr=0.01')
plt.plot(vl_0001, marker='^', label='lr=0.001')
plt.title("验证损失对比")
plt.xlabel("Epoch")
plt.ylabel("Val Loss")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(ta_01, marker='o', label='lr=0.1')
plt.plot(ta_001, marker='s', label='lr=0.01')
plt.plot(ta_0001, marker='^', label='lr=0.001')
plt.title("训练准确率对比")
plt.xlabel("Epoch")
plt.ylabel("Train Acc (%)")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(va_01, marker='o', label='lr=0.1')
plt.plot(va_001, marker='s', label='lr=0.01')
plt.plot(va_0001, marker='^', label='lr=0.001')
plt.title("验证准确率对比")
plt.xlabel("Epoch")
plt.ylabel("Val Acc (%)")
plt.legend()
plt.grid(True)

plt.suptitle("任务3：不同学习率对比（Adam优化器）", fontsize=14)
plt.tight_layout()
plt.show()

print("\n===== 三种学习率测试集准确率 =====")
print(f"lr=0.1    → {te_01:.2f}%")
print(f"lr=0.01   → {te_001:.2f}%")
print(f"lr=0.001  → {te_0001:.2f}%")

# ===================== 关键：设置模型为评估模式 =====================
model.eval()

# =============================================================================
# ===================== 任务4：第一层卷积核可视化 =====================
# =============================================================================
print("\n===== 任务4：卷积核可视化 =====")
conv1_weight = model.conv1.weight.data.cpu()

plt.figure(figsize=(10, 5))
for i in range(8):
    plt.subplot(2, 4, i+1)
    kernel = conv1_weight[i, 0]
    plt.imshow(kernel, cmap="gray")
    plt.title(f"卷积核 {i+1}")
    plt.axis("off")
plt.suptitle("任务4：训练后的第一层卷积核", fontsize=14)
plt.tight_layout()
plt.show()

# =============================================================================
# ===================== 任务5：Feature map 可视化 =====================
# =============================================================================
print("\n===== 任务5：Feature map 可视化 =====")
test_img, test_label = test_dataset[0]
input_tensor = test_img.unsqueeze(0).to(device)

with torch.no_grad():
    feature_maps = model.conv1(input_tensor)
feature_maps = feature_maps.squeeze(0).cpu()

plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(feature_maps[i], cmap="gray")
    plt.title(f"Feature map {i+1}")
    plt.axis("off")
plt.suptitle("任务5：第一层卷积输出特征图", fontsize=14)
plt.tight_layout()
plt.show()

# =============================================================================
# ===================== 任务6：错误分类样本展示 =====================
# =============================================================================
print("\n===== 任务6：错误样本分析 =====")
error_images = []
error_true = []
error_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicts = torch.max(outputs, 1)
        
        for i in range(len(labels)):
            if labels[i] != predicts[i] and len(error_images) < 8:
                error_images.append(images[i].cpu())
                error_true.append(labels[i].item())
                error_pred.append(predicts[i].item())
        if len(error_images) >= 8:
            break

plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(error_images[i][0], cmap="gray")
    plt.title(f"真:{error_true[i]}\n预:{error_pred[i]}")
    plt.axis("off")
plt.suptitle("任务6：模型错误分类的样本", fontsize=14)
plt.tight_layout()
plt.show()

# =============================================================================
# ===================== 任务7：混淆矩阵 =====================
# =============================================================================
print("\n===== 任务7：混淆矩阵 =====")
all_true = []
all_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicts = torch.max(outputs, 1)
        all_true.extend(labels.numpy())
        all_pred.extend(predicts.cpu().numpy())

# 纯matplotlib绘制混淆矩阵，无需额外依赖
def plot_confusion_matrix(y_true, y_pred):
    num_classes = 10
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("任务7：测试集混淆矩阵", fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(all_true, all_pred)