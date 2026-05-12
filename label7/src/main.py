# 解决绘图闪退，强制开启交互模式
import matplotlib
matplotlib.use('TkAgg')

# 导入所有需要的库
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------- 任务1：数据准备 ----------------------
digits = load_digits()
X = digits.data
y = digits.target

print("===== 任务1：数据集信息 =====")
print(f"总样本数：{len(digits.images)}")
print(f"图像大小：{digits.images[0].shape[0]} × {digits.images[0].shape[1]}")
print(f"标签范围：{digits.target.min()} ~ {digits.target.max()}")

# 显示样本图
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(digits.images[i], cmap="gray")
    plt.title(f"Label:{y[i]}")
    plt.axis("off")
plt.show(block=True)  # 强制等待关闭窗口

# ---------------------- 任务2：训练集 / 测试集 划分 ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("\n===== 任务2：划分结果 =====")
print(f"训练集数量：{X_train.shape[0]}")
print(f"测试集数量：{X_test.shape[0]}")

# 可视化：训练集 vs 测试集分布（PCA降维）
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sc1 = plt.scatter(X_train_2d[:,0], X_train_2d[:,1], c=y_train, cmap="tab10", s=8, alpha=0.7)
plt.title("Training Set (训练集)")
plt.colorbar(sc1, label="Digit")

plt.subplot(1, 2, 2)
sc2 = plt.scatter(X_test_2d[:,0], X_test_2d[:,1], c=y_test, cmap="tab10", s=8, alpha=0.7)
plt.title("Test Set (测试集)")
plt.colorbar(sc2, label="Digit")
plt.tight_layout()
plt.show(block=True)

# ---------------------- 任务3：特征表示 ----------------------
print("\n===== 任务3：特征说明 =====")
print("8×8 图像 → 展平为 64 维特征向量")
print("传统机器学习必须用一维向量输入")

# ---------------------- 任务4：模型训练 ======================
print("\n===== 任务4：模型训练与准确率 =====")
models = {
    "KNN": KNeighborsClassifier(),
    "朴素贝叶斯": GaussianNB(),
    "逻辑回归": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "决策树": DecisionTreeClassifier(max_depth=4, random_state=42),
    "随机森林": RandomForestClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:10s} → 准确率：{acc:.4f}")

best = max(results, key=results.get)
print(f"\n最佳模型：{best}，准确率 {results[best]:.4f}")

# ---------------------- 可视化：决策树结构（缩小尺寸） ======================
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

print("\n" + "="*50)
print("📊 决策树 详细训练规则（文字版）")
print("="*50)
tree_rules = export_text(dt, feature_names=[f'pixel_{i}' for i in range(64)])
print(tree_rules)

plt.figure(figsize=(12, 6))  # 缩小画布，避免卡死
plot_tree(
    dt,
    feature_names=[f"px{i}" for i in range(64)],
    class_names=[str(i) for i in range(10)],
    filled=True,
    rounded=True,
    fontsize=7
)
plt.title("Decision Tree Structure (决策树结构图)", fontsize=16)
plt.show(block=True)

# ---------------------- 任务5 + 任务6：最优模型错误分析 ======================
print("\n===== 任务6：最优模型（SVM）错误分析 =====")
best_model = models["SVM"]
y_pred = best_model.predict(X_test)

# 混淆矩阵
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("SVM 混淆矩阵")
plt.show(block=True)

# 错误样本展示
error_idx = np.where(y_test != y_pred)[0]
print(f"\nSVM 共错误分类 {len(error_idx)} 个样本")

plt.figure(figsize=(12, 6))
for i, idx in enumerate(error_idx[:6]):
    plt.subplot(2, 3, i+1)
    img = X_test[idx].reshape(8, 8)
    plt.imshow(img, cmap="gray")
    plt.title(f"真实:{y_test[idx]}, 预测:{y_pred[idx]}")
    plt.axis("off")
plt.tight_layout()
plt.show(block=True)