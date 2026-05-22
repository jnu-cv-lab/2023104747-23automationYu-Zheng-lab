# ===================== 任务1：环境准备 =====================
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# 1.1 测试库导入
print("PyTorch 版本:", torch.__version__)
print("torchvision 版本:", torchvision.__version__)

# 1.2 判断是否支持GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 1.3 简单张量操作
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])
z = x + y
print("张量加法结果:\n", z)


# ===================== 任务2：加载MNIST图像数据集 =====================
# 数据预处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))  # MNIST均值方差
])

# 下载加载数据集
full_train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# 划分训练集和验证集（8:2）
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 显示至少8张样本+真实标签
examples = iter(train_loader)
images, labels = next(examples)

plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i][0], cmap="gray")
    plt.title(f"Label: {labels[i].item()}")
    plt.axis("off")
plt.tight_layout()
plt.show()


# ===================== 任务3：定义CNN卷积神经网络 =====================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层1: 输入1通道(MNIST灰度图),输出16通道,3×3卷积
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层：MNIST输入28×28，经过2次池化后变为7×7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 输出10类(0-9)

    def forward(self, x):
        # 卷积+激活+池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 展平
        x = x.view(-1, 32 * 7 * 7)
        # 全连接+激活
        x = F.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x

# 实例化模型
model = SimpleCNN().to(device)
print("\n===== CNN模型结构 =====")
print(model)


# ===================== 任务4：训练模型 + 任务5：验证模型 =====================
# 超参数设置
epochs = 5
criterion = nn.CrossEntropyLoss()  # 损失函数：多分类交叉熵
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 优化器：Adam

# 保存训练曲线数据
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(epochs):
    # ========== 训练阶段 ==========
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

    # ========== 验证阶段 ==========
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(val_acc)

    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {avg_val_loss:.4f}, Val Acc:   {val_acc:.2f}%\n")


# ===================== 任务6：测试模型 =====================
model.eval()
test_loss = 0.0
correct = 0
total = 0
test_images_show = []
test_labels_true = []
test_labels_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 保存前8张用于可视化
        if len(test_images_show) < 8:
            test_images_show.extend(images.cpu())
            test_labels_true.extend(labels.cpu())
            test_labels_pred.extend(predicted.cpu())

avg_test_loss = test_loss / len(test_loader)
test_acc = 100 * correct / total
print("===== 测试集结果 =====")
print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Acc:  {test_acc:.2f}%")

# 显示8张测试图像+真实标签+预测标签
plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(test_images_show[i][0], cmap="gray")
    plt.title(f"True:{test_labels_true[i].item()}, Pred:{test_labels_pred[i].item()}")
    plt.axis("off")
plt.tight_layout()
plt.show()


# ===================== 任务7：绘制训练曲线 =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文显示
plt.figure(figsize=(12, 5))

# 绘制Loss曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_loss_list, marker="o", label="Train Loss")
plt.plot(range(1, epochs+1), val_loss_list, marker="s", label="Val Loss")
plt.title("训练与验证损失曲线")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# 绘制Accuracy曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_acc_list, marker="o", label="Train Acc")
plt.plot(range(1, epochs+1), val_acc_list, marker="s", label="Val Acc")
plt.title("训练与验证准确率曲线")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()