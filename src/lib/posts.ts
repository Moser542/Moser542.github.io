export interface BlogPost {
  id: string
  title: string
  summary: string
  date: string
  readingTime: string
  tags: string[]
  content: string
}

export const posts: BlogPost[] = [
  {
    id: 'pytorch-loss-functions',
    title: 'PyTorch Loss函数深度解析',
    summary:
      '深入理解PyTorch中12种常用损失函数的数学原理、使用场景与代码实践，包括MSE、交叉熵、KL散度等核心损失函数的完整指南。',
    date: '2024-01-15',
    readingTime: '25分钟',
    tags: ['PyTorch', '深度学习', '损失函数', 'Python'],
    content: `
## 引言

损失函数（Loss Function）是深度学习中至关重要的组成部分，它衡量模型预测值与真实值之间的差异，指导模型参数的优化方向。PyTorch提供了丰富的损失函数库，本文将深入解析12种常用损失函数的数学原理、应用场景和代码实践。

---

## 1. nn.MSELoss - 均方误差损失

### 数学公式

均方误差损失（Mean Squared Error Loss）计算预测值与真实值之间差的平方的均值：

$$
\\text{MSE}(x, y) = \\frac{1}{n} \\sum_{i=1}^{n} (x_i - y_i)^2
$$

在PyTorch中，还可以设置\`reduction\`参数：
- \`'mean'\`（默认）：返回损失的平均值
- \`'sum'\`：返回损失的总和
- \`'none'\`：返回每个元素的损失

### 用途场景

1. **回归问题**：预测连续值，如房价预测、股票价格预测
2. **图像重建**：自编码器、图像超分辨率任务
3. **语音合成**：生成模型的重建损失
4. **信号处理**：噪声消除、信号恢复

### 代码示例

\`\`\`python
import torch
import torch.nn as nn

# 创建MSELoss实例
mse_loss = nn.MSELoss()
mse_loss_sum = nn.MSELoss(reduction='sum')
mse_loss_none = nn.MSELoss(reduction='none')

# 模拟预测值和真实值
predictions = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
targets = torch.tensor([1.5, 2.5, 2.8, 4.2])

# 计算损失
loss_mean = mse_loss(predictions, targets)
loss_sum = mse_loss_sum(predictions, targets)
loss_none = mse_loss_none(predictions, targets)

print(f"Mean MSE Loss: {loss_mean.item():.4f}")      # 0.1450
print(f"Sum MSE Loss: {loss_sum.item():.4f}")        # 0.5800
print(f"None MSE Loss: {loss_none}")                  # tensor values

# 反向传播
loss_mean.backward()
print(f"Gradients: {predictions.grad}")
\`\`\`

### 实际应用示例：线性回归

\`\`\`python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 生成模拟数据
torch.manual_seed(42)
X = torch.linspace(-5, 5, 100).reshape(-1, 1)
true_w, true_b = 2.5, 1.0
y_true = true_w * X + true_b + torch.randn(X.shape) * 0.5

# 定义模型参数
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# 定义优化器和损失函数
optimizer = torch.optim.SGD([w, b], lr=0.01)
criterion = nn.MSELoss()

# 训练循环
losses = []
for epoch in range(200):
    optimizer.zero_grad()
    y_pred = w * X + b
    loss = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, w = {w.item():.2f}, b = {b.item():.2f}")

print(f"\\nLearned: w = {w.item():.2f} (true: {true_w}), b = {b.item():.2f} (true: {true_b})")
\`\`\`

### 注意事项

1. **对异常值敏感**：由于平方操作，异常值会产生较大的梯度，可能导致训练不稳定
2. **梯度消失问题**：当预测值与真实值接近时，梯度会变得很小
3. **数值稳定性**：对于非常大的值，平方可能导致数值溢出
4. **学习率选择**：由于梯度与误差成正比，需要合理设置学习率

---

## 2. nn.CrossEntropyLoss - 交叉熵损失

### 数学公式

交叉熵损失（Cross-Entropy Loss）结合了\`LogSoftmax\`和\`NLLLoss\`：

$$
\\text{CE}(x, y) = -\\sum_{c=1}^{C} y_c \\log(\\text{softmax}(x)_c)
$$

对于单标签分类（真实标签为one-hot编码）：

$$
\\text{CE}(x, y) = -\\log\\left(\\frac{e^{x_{y}}}{\\sum_{j} e^{x_j}}\\right)
$$

### 用途场景

1. **多分类任务**：图像分类（ImageNet、CIFAR）、文本分类
2. **语言模型**：预测下一个词的概率分布
3. **命名实体识别**：序列标注任务
4. **目标检测**：类别预测分支

### 代码示例

\`\`\`python
import torch
import torch.nn as nn

# 创建CrossEntropyLoss实例
ce_loss = nn.CrossEntropyLoss()

# 模拟模型输出（batch_size=3, num_classes=5）
# 注意：CrossEntropyLoss期望未归一化的分数（logits）
logits = torch.tensor([
    [2.0, 1.0, 0.1, 0.5, 0.3],   # 样本1，预测类别0概率最高
    [0.5, 2.5, 0.2, 0.1, 0.3],   # 样本2，预测类别1概率最高
    [0.1, 0.2, 0.3, 3.0, 0.5]    # 样本3，预测类别3概率最高
], requires_grad=True)

# 真实标签（类别索引）
targets = torch.tensor([0, 1, 3])

# 计算损失
loss = ce_loss(logits, targets)
print(f"Cross Entropy Loss: {loss.item():.4f}")

# 带权重的交叉熵（处理类别不平衡）
class_weights = torch.tensor([1.0, 2.0, 1.5, 0.5, 1.0])  # 类别权重
weighted_ce_loss = nn.CrossEntropyLoss(weight=class_weights)
weighted_loss = weighted_ce_loss(logits, targets)
print(f"Weighted Cross Entropy Loss: {weighted_loss.item():.4f}")
\`\`\`

### 带标签平滑的交叉熵

\`\`\`python
# 标签平滑（Label Smoothing）可以防止过拟合
label_smoothing_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
smoothed_loss = label_smoothing_loss(logits, targets)
print(f"Label Smoothing Loss: {smoothed_loss.item():.4f}")
\`\`\`

### 多标签分类示例

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# 模拟数据
batch_size, input_dim, num_classes = 8, 64, 10
X = torch.randn(batch_size, input_dim)
y = torch.randint(0, 2, (batch_size, num_classes)).float()  # 多标签

model = MultiLabelClassifier(input_dim, num_classes)
logits = model(X)

# 多标签分类使用BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, y)
print(f"Multi-label Loss: {loss.item():.4f}")
\`\`\`

### 注意事项

1. **不要在输入前加Softmax**：该损失函数内部已经包含了Softmax操作
2. **目标标签格式**：类别索引（非one-hot编码）
3. **类别不平衡**：使用\`weight\`参数处理不平衡数据集
4. **标签平滑**：使用\`label_smoothing\`参数防止过拟合
5. **忽略特定标签**：使用\`ignore_index\`参数忽略填充标签

---

## 3. nn.BCELoss / nn.BCEWithLogitsLoss - 二分类交叉熵

### 数学公式

二分类交叉熵（Binary Cross-Entropy）：

$$
\\text{BCE}(x, y) = -[y \\log(x) + (1-y) \\log(1-x)]
$$

对于批量数据：
$$
\\text{BCE}(x, y) = -\\frac{1}{n}\\sum_{i=1}^{n} [y_i \\log(x_i) + (1-y_i) \\log(1-x_i)]
$$

### BCELoss vs BCEWithLogitsLoss

- **BCELoss**：输入需要先经过Sigmoid激活
- **BCEWithLogitsLoss**：内部包含Sigmoid，数值更稳定

### 用途场景

1. **二分类任务**：垃圾邮件检测、情感分析
2. **多标签分类**：每个样本属于多个类别
3. **图像分割**：二值分割掩码预测
4. **生成模型**：GAN中的判别器损失

### 代码示例

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 使用BCELoss（需要先Sigmoid）
bce_loss = nn.BCELoss()
predictions = torch.tensor([0.8, 0.2, 0.6, 0.9])  # 已经过Sigmoid
targets = torch.tensor([1.0, 0.0, 1.0, 1.0])
loss_bce = bce_loss(predictions, targets)
print(f"BCE Loss: {loss_bce.item():.4f}")

# 使用BCEWithLogitsLoss（推荐）
bce_logits_loss = nn.BCEWithLogitsLoss()
logits = torch.tensor([1.5, -1.2, 0.5, 2.3], requires_grad=True)  # 原始logits
loss_bce_logits = bce_logits_loss(logits, targets)
print(f"BCE with Logits Loss: {loss_bce_logits.item():.4f}")
\`\`\`

### 正样本权重

\`\`\`python
# 处理正负样本不平衡
pos_weight = torch.tensor([3.0])  # 正样本权重
weighted_bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 模拟不平衡数据
logits_imbalanced = torch.randn(1000)
targets_imbalanced = (torch.rand(1000) < 0.1).float()  # 只有10%正样本

loss_weighted = weighted_bce_loss(logits_imbalanced, targets_imbalanced)
print(f"Weighted BCE Loss: {loss_weighted.item():.4f}")
\`\`\`

### 图像分割示例

\`\`\`python
import torch
import torch.nn as nn

class BinarySegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        # pred: [B, 1, H, W], target: [B, 1, H, W]
        bce_loss = self.bce(pred, target)
        
        # Dice Loss（额外指标）
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = (2. * intersection) / (pred_sigmoid.sum() + target.sum() + 1e-8)
        dice_loss = 1 - dice
        
        return bce_loss + dice_loss

# 模拟分割数据
pred_mask = torch.randn(4, 1, 256, 256)
true_mask = (torch.rand(4, 1, 256, 256) > 0.5).float()

seg_loss = BinarySegmentationLoss()
loss = seg_loss(pred_mask, true_mask)
print(f"Segmentation Loss: {loss.item():.4f}")
\`\`\`

### 注意事项

1. **数值稳定性**：优先使用\`BCEWithLogitsLoss\`
2. **目标范围**：目标值必须在[0, 1]范围内
3. **输入范围**：BCELoss期望[0, 1]，BCEWithLogitsLoss接受任意实数
4. **样本不平衡**：使用\`pos_weight\`调整正负样本权重

---

## 4. nn.NLLLoss - 负对数似然损失

### 数学公式

负对数似然损失（Negative Log-Likelihood Loss）：

$$
\\text{NLL}(x, y) = -x_{y}
$$

其中\`x\`是已经过\`LogSoftmax\`处理的输入。

### 与CrossEntropyLoss的关系

\`\`\`python
# CrossEntropyLoss = LogSoftmax + NLLLoss
import torch.nn.functional as F

logits = torch.randn(3, 5)
targets = torch.tensor([0, 2, 4])

# 方法1：使用CrossEntropyLoss
ce_loss = nn.CrossEntropyLoss()
loss1 = ce_loss(logits, targets)

# 方法2：LogSoftmax + NLLLoss
log_probs = F.log_softmax(logits, dim=1)
nll_loss = nn.NLLLoss()
loss2 = nll_loss(log_probs, targets)

print(f"CrossEntropy: {loss1.item():.4f}")
print(f"LogSoftmax + NLL: {loss2.item():.4f}")  # 结果相同
\`\`\`

### 用途场景

1. **配合LogSoftmax使用**：用于分类任务的最后一层
2. **概率模型**：当模型输出对数概率时
3. **序列标注**：CRF等概率图模型

### 代码示例

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 创建NLLLoss实例
nll_loss = nn.NLLLoss()

# LogSoftmax输出（已经是概率的对数）
log_probs = torch.tensor([
    [-0.5, -1.2, -2.0, -1.8, -1.5],   # 类别0概率最高
    [-1.8, -0.3, -2.1, -1.5, -2.0],   # 类别1概率最高
    [-2.0, -1.8, -0.8, -1.0, -1.2]    # 类别2概率最高
], requires_grad=True)

targets = torch.tensor([0, 1, 2])

# 计算损失
loss = nll_loss(log_probs, targets)
print(f"NLL Loss: {loss.item():.4f}")

# 手动计算验证
# 对于第一个样本，target=0，loss = -log_probs[0, 0] = 0.5
\`\`\`

### 自定义LogSoftmax网络

\`\`\`python
class ClassifierWithLogSoftmax(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.log_softmax(x)

# 使用
model = ClassifierWithLogSoftmax(64, 10)
criterion = nn.NLLLoss()

X = torch.randn(8, 64)
y = torch.randint(0, 10, (8,))

log_probs = model(X)
loss = criterion(log_probs, y)
\`\`\`

### 注意事项

1. **输入要求**：输入必须是LogSoftmax的输出（对数概率）
2. **不要重复LogSoftmax**：如果模型已经包含LogSoftmax，不要再额外添加
3. **与CrossEntropyLoss等价**：大多数情况下推荐直接使用CrossEntropyLoss

---

## 5. nn.L1Loss - L1损失

### 数学公式

L1损失（平均绝对误差，Mean Absolute Error）：

$$
\\text{L1}(x, y) = \\frac{1}{n} \\sum_{i=1}^{n} |x_i - y_i|
$$

### 用途场景

1. **回归任务**：对异常值更鲁棒
2. **图像处理**：图像去噪、风格迁移
3. **稀疏编码**：L1正则化诱导稀疏解
4. **特征选择**：Lasso回归

### 代码示例

\`\`\`python
import torch
import torch.nn as nn

# 创建L1Loss实例
l1_loss = nn.L1Loss()
l1_loss_sum = nn.L1Loss(reduction='sum')

predictions = torch.tensor([1.0, 2.5, 3.2, 4.8], requires_grad=True)
targets = torch.tensor([1.2, 2.0, 3.5, 5.0])

loss = l1_loss(predictions, targets)
print(f"L1 Loss (mean): {loss.item():.4f}")

loss_sum = l1_loss_sum(predictions, targets)
print(f"L1 Loss (sum): {loss_sum.item():.4f}")

# 反向传播
loss.backward()
print(f"Gradients: {predictions.grad}")
\`\`\`

### L1 vs MSE 比较

\`\`\`python
import torch
import matplotlib.pyplot as plt

# 模拟误差范围
errors = torch.linspace(-3, 3, 100)

# L1损失
l1_values = torch.abs(errors)

# MSE损失
mse_values = errors ** 2

print("误差分析：")
print(f"误差=0时: L1={l1_values[50].item():.2f}, MSE={mse_values[50].item():.2f}")
print(f"误差=1时: L1={l1_values[66].item():.2f}, MSE={mse_values[66].item():.2f}")
print(f"误差=2时: L1={l1_values[83].item():.2f}, MSE={mse_values[83].item():.2f}")
print(f"误差=3时: L1={l1_values[99].item():.2f}, MSE={mse_values[99].item():.2f}")
\`\`\`

### 图像去噪示例

\`\`\`python
import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 使用L1损失进行去噪
model = DenoisingAutoencoder()
criterion = nn.L1Loss()  # L1对椒盐噪声更鲁棒

# 模拟噪声图像
noisy_images = torch.randn(8, 1, 64, 64)
clean_images = torch.randn(8, 1, 64, 64)

outputs = model(noisy_images)
loss = criterion(outputs, clean_images)
\`\`\`

### 注意事项

1. **梯度不连续**：在零点处梯度不连续，可能导致优化困难
2. **对异常值鲁棒**：相比MSE，异常值的影响较小
3. **稀疏性**：可以产生稀疏解，适合特征选择
4. **计算效率**：绝对值计算比平方更高效

---

## 6. nn.SmoothL1Loss - 平滑L1损失

### 数学公式

平滑L1损失（Huber Loss）结合了L1和L2的优点：

$$
\\text{SmoothL1}(x, y) = \\begin{cases}
\\frac{1}{2}(x - y)^2 / \\beta & \\text{if } |x - y| < \\beta \\\\
|x - y| - \\frac{1}{2}\\beta & \\text{otherwise}
\\end{cases}
$$

其中\`β\`（beta）是阈值参数，默认为1.0。

### 用途场景

1. **目标检测**：Fast R-CNN、Faster R-CNN的边界框回归
2. **关键点检测**：人体姿态估计
3. **深度估计**：单目深度预测
4. **机器人控制**：连续动作预测

### 代码示例

\`\`\`python
import torch
import torch.nn as nn

# 创建SmoothL1Loss实例
smooth_l1 = nn.SmoothL1Loss()
smooth_l1_beta = nn.SmoothL1Loss(beta=0.5)  # 自定义beta

predictions = torch.tensor([1.0, 3.0, 5.0, 10.0], requires_grad=True)
targets = torch.tensor([1.2, 2.5, 5.5, 8.0])

loss = smooth_l1(predictions, targets)
print(f"Smooth L1 Loss (beta=1.0): {loss.item():.4f}")

loss_beta = smooth_l1_beta(predictions, targets)
print(f"Smooth L1 Loss (beta=0.5): {loss_beta.item():.4f}")
\`\`\`

### 边界框回归示例

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BBoxRegressor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 4)  # tx, ty, tw, th
    
    def forward(self, x):
        return self.fc(x)

# 边界框回归损失
def bbox_regression_loss(pred_deltas, target_deltas):
    """
    pred_deltas: [N, 4] 预测的偏移量 (tx, ty, tw, th)
    target_deltas: [N, 4] 目标偏移量
    """
    smooth_l1 = nn.SmoothL1Loss(beta=1.0)
    return smooth_l1(pred_deltas, target_deltas)

# 模拟数据
feature_dim = 512
batch_size = 16

model = BBoxRegressor(feature_dim)
features = torch.randn(batch_size, feature_dim)
pred_deltas = model(features)
target_deltas = torch.randn(batch_size, 4) * 0.1  # 小偏移量

loss = bbox_regression_loss(pred_deltas, target_deltas)
print(f"BBox Regression Loss: {loss.item():.4f}")
\`\`\`

### L1 vs SmoothL1 vs MSE 对比

\`\`\`python
import torch
import numpy as np

def compare_losses():
    errors = torch.linspace(-3, 3, 100)
    
    l1 = torch.abs(errors)
    mse = errors ** 2
    
    # SmoothL1计算
    beta = 1.0
    smooth_l1 = torch.where(
        torch.abs(errors) < beta,
        0.5 * errors ** 2 / beta,
        torch.abs(errors) - 0.5 * beta
    )
    
    # 关键点对比
    for e in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        idx = int((e + 3) * 100 / 6)
        print(f"误差={e:+.1f}: L1={l1[idx]:.3f}, SmoothL1={smooth_l1[idx]:.3f}, MSE={mse[idx]:.3f}")

compare_losses()
\`\`\`

### 注意事项

1. **阈值选择**：\`beta\`参数决定L1和L2的切换点
2. **梯度稳定性**：在零点附近梯度连续，优化更稳定
3. **异常值处理**：对大误差使用L1，对小误差使用L2
4. **目标检测标配**：大多数检测框架的默认选择

---

## 7. nn.KLDivLoss - KL散度损失

### 数学公式

KL散度（Kullback-Leibler Divergence）衡量两个概率分布的差异：

$$
\\text{KL}(P || Q) = \\sum_{i} P(i) \\log\\left(\\frac{P(i)}{Q(i)}\\right)
$$

PyTorch中的实现：
$$
\\text{loss}(x, y) = y \\cdot (\\log y - x)
$$

### 用途场景

1. **知识蒸馏**：教师模型向学生模型传递知识
2. **变分自编码器（VAE）**：正则化隐空间分布
3. **强化学习**：策略优化（PPO、SAC）
4. **语言模型**：文本生成模型对齐

### 代码示例

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 创建KLDivLoss实例
kl_loss = nn.KLDivLoss(reduction='batchmean')

# 学生模型输出（log概率）
student_log_probs = F.log_softmax(torch.randn(4, 10), dim=1)

# 教师模型输出（概率）
teacher_probs = F.softmax(torch.randn(4, 10), dim=1)

# 计算KL散度
loss = kl_loss(student_log_probs, teacher_probs)
print(f"KL Divergence Loss: {loss.item():.4f}")
\`\`\`

### 知识蒸馏示例

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # 蒸馏损失权重
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        # 软标签损失（知识蒸馏）
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = self.kl_div(soft_student, soft_targets) * (self.temperature ** 2)
        
        # 硬标签损失（标准交叉熵）
        student_loss = self.ce_loss(student_logits, labels)
        
        # 组合损失
        return self.alpha * distillation_loss + (1 - self.alpha) * student_loss

# 使用示例
student_model = nn.Linear(64, 10)
teacher_model = nn.Linear(64, 10)

# 冻结教师模型
for param in teacher_model.parameters():
    param.requires_grad = False

features = torch.randn(8, 64)
labels = torch.randint(0, 10, (8,))

student_logits = student_model(features)
with torch.no_grad():
    teacher_logits = teacher_model(features)

criterion = DistillationLoss(temperature=4.0, alpha=0.7)
loss = criterion(student_logits, teacher_logits, labels)
print(f"Distillation Loss: {loss.item():.4f}")
\`\`\`

### VAE中的KL散度

\`\`\`python
import torch
import torch.nn as nn

def vae_kl_loss(mu, log_var):
    """
    VAE的KL散度损失
    KL(N(mu, sigma) || N(0, 1))
    """
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim * 2)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=1)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

# 训练VAE
vae = VAE(784, 20)
x = torch.randn(32, 784)
x_recon, mu, log_var = vae(x)

recon_loss = nn.MSELoss()(x_recon, x)
kl_loss = vae_kl_loss(mu, log_var)
total_loss = recon_loss + 0.001 * kl_loss

print(f"Reconstruction Loss: {recon_loss.item():.4f}")
print(f"KL Loss: {kl_loss.item():.4f}")
print(f"Total Loss: {total_loss.item():.4f}")
\`\`\`

### 注意事项

1. **输入格式**：第一个输入必须是log概率，第二个是概率
2. **reduction参数**：建议使用\`batchmean\`以获得正确的梯度缩放
3. **对称性**：KL散度不对称，\`KL(P||Q) ≠ KL(Q||P)\`
4. **数值稳定性**：注意避免log(0)，添加小常数\`eps\`

---

## 8. nn.MarginRankingLoss - 边际排序损失

### 数学公式

边际排序损失用于学习样本之间的相对排序：

$$
\\text{loss}(x_1, x_2, y) = \\max(0, -y \\cdot (x_1 - x_2) + \\text{margin})
$$

其中\`y ∈ {-1, 1}\`表示排序关系。

### 用途场景

1. **学习排序（LTR）**：搜索引擎结果排序
2. **推荐系统**：物品偏好排序
3. **人脸验证**：相似度学习
4. **信息检索**：文档相关性排序

### 代码示例

\`\`\`python
import torch
import torch.nn as nn

# 创建MarginRankingLoss实例
margin_loss = nn.MarginRankingLoss(margin=1.0)

# 输出1（更相关的样本得分应该更高）
output1 = torch.tensor([3.0, 2.0, 5.0], requires_grad=True)

# 输出2（较不相关的样本）
output2 = torch.tensor([1.0, 4.0, 3.0], requires_grad=True)

# 目标：1表示output1应该大于output2，-1表示相反
targets = torch.tensor([1.0, -1.0, 1.0])

# 计算损失
loss = margin_loss(output1, output2, targets)
print(f"Margin Ranking Loss: {loss.item():.4f}")

# 手动验证
# 第1个样本: max(0, -1*(3-1)+1) = max(0, -1) = 0 ✓
# 第2个样本: max(0, -(-1)*(2-4)+1) = max(0, 3) = 3
# 第3个样本: max(0, -1*(5-3)+1) = max(0, -1) = 0 ✓
\`\`\`

### 推荐系统示例

\`\`\`python
import torch
import torch.nn as nn

class RankingModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        return (user_emb * item_emb).sum(dim=1)

# 训练排序模型
num_users, num_items, embedding_dim = 1000, 5000, 64
model = RankingModel(num_users, num_items, embedding_dim)
criterion = nn.MarginRankingLoss(margin=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模拟训练数据：用户对正样本和负样本的偏好
batch_size = 64
user_ids = torch.randint(0, num_users, (batch_size,))
pos_item_ids = torch.randint(0, num_items, (batch_size,))  # 正样本
neg_item_ids = torch.randint(0, num_items, (batch_size,))  # 负样本

# 前向传播
pos_scores = model(user_ids, pos_item_ids)
neg_scores = model(user_ids, neg_item_ids)
targets = torch.ones(batch_size)  # 正样本得分应该更高

# 计算损失
loss = criterion(pos_scores, neg_scores, targets)
print(f"Ranking Loss: {loss.item():.4f}")
\`\`\`

### 注意事项

1. **标签取值**：目标标签必须是+1或-1
2. **边际选择**：\`margin\`越大，对错误排序的惩罚越大
3. **样本配对**：需要构造正负样本对进行训练
4. **计算效率**：训练时需要为每个正样本采样负样本

---

## 9. nn.TripletMarginLoss - 三元组边际损失

### 数学公式

三元组边际损失用于学习嵌入空间中的距离关系：

$$
\\text{loss}(a, p, n) = \\max(0, d(a, p) - d(a, n) + \\text{margin})
$$

其中：
- \`a\`（anchor）：锚点样本
- \`p\`（positive）：正样本，与锚点相似
- \`n\`（negative）：负样本，与锚点不相似
- \`d\`：距离度量（默认欧氏距离）

### 用途场景

1. **人脸识别**：FaceNet、DeepFace
2. **行人重识别**：跨摄像头追踪
3. **图像检索**：相似图像搜索
4. **文本相似度**：语义匹配

### 代码示例

\`\`\`python
import torch
import torch.nn as nn

# 创建TripletMarginLoss实例
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

# 锚点、正样本、负样本嵌入
anchor = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
positive = torch.tensor([[1.1, 2.1, 3.1]], requires_grad=True)  # 与anchor相似
negative = torch.tensor([[5.0, 6.0, 7.0]], requires_grad=True)  # 与anchor不相似

# 计算损失
loss = triplet_loss(anchor, positive, negative)
print(f"Triplet Margin Loss: {loss.item():.4f}")

# 手动验证
dist_pos = torch.norm(anchor - positive, p=2)
dist_neg = torch.norm(anchor - negative, p=2)
print(f"Distance to positive: {dist_pos.item():.4f}")
print(f"Distance to negative: {dist_neg.item():.4f}")
print(f"Loss = max(0, {dist_pos.item():.4f} - {dist_neg.item():.4f} + 1.0)")
\`\`\`

### 人脸识别示例

\`\`\`python
import torch
import torch.nn as nn

class FaceEmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, 128)  # 128维嵌入
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)  # L2归一化

# 训练
model = FaceEmbeddingNet()
criterion = nn.TripletMarginLoss(margin=0.2, p=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模拟三元组数据
anchor_imgs = torch.randn(16, 3, 128, 128)
positive_imgs = torch.randn(16, 3, 128, 128)  # 同一人的不同照片
negative_imgs = torch.randn(16, 3, 128, 128)  # 不同人的照片

anchor_emb = model(anchor_imgs)
positive_emb = model(positive_imgs)
negative_emb = model(negative_imgs)

loss = criterion(anchor_emb, positive_emb, negative_emb)
print(f"Face Recognition Triplet Loss: {loss.item():.4f}")
\`\`\`

### 半硬负样本挖掘

\`\`\`python
import torch
import torch.nn as nn

def semihard_negative_mining(anchor, positive, negative, margin=0.2):
    """
    半硬负样本挖掘：
    选择距离锚点比正样本远，但在边际内的负样本
    """
    batch_size = anchor.size(0)
    
    # 计算所有距离
    dist_pos = torch.norm(anchor - positive, p=2, dim=1)
    dist_neg = torch.norm(anchor - negative, p=2, dim=1)
    
    # 选择半硬负样本
    # 条件：dist_pos < dist_neg < dist_pos + margin
    mask = (dist_neg > dist_pos) & (dist_neg < dist_pos + margin)
    
    if mask.sum() == 0:
        # 如果没有半硬负样本，使用所有负样本
        mask = torch.ones(batch_size, dtype=torch.bool)
    
    return mask

# 使用半硬负样本挖掘
anchor = torch.randn(32, 128)
positive = torch.randn(32, 128)
negative = torch.randn(32, 128)

mask = semihard_negative_mining(anchor, positive, negative)
print(f"Selected {mask.sum()}/{len(mask)} semi-hard negatives")
\`\`\`

### 注意事项

1. **三元组采样**：好的三元组选择对训练效果至关重要
2. **边际设置**：太小难以学习区分性特征，太大可能导致收敛困难
3. **归一化**：嵌入向量通常需要L2归一化
4. **距离度量**：可以使用不同的p范数，p=2最常用

---

## 10. nn.CosineEmbeddingLoss - 余弦嵌入损失

### 数学公式

余弦嵌入损失基于余弦相似度：

$$
\\text{loss}(x_1, x_2, y) = \\begin{cases}
1 - \\cos(x_1, x_2) & \\text{if } y = 1 \\\\
\\max(0, \\cos(x_1, x_2) - \\text{margin}) & \\text{if } y = -1
\\end{cases}
$$

其中\`cos(x_1, x_2)\`是两个向量的余弦相似度。

### 用途场景

1. **语义相似度**：文本匹配、问答系统
2. **嵌入学习**：词向量、句子向量
3. **图像-文本匹配**：跨模态检索
4. **签名验证**：文档认证

### 代码示例

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 创建CosineEmbeddingLoss实例
cosine_loss = nn.CosineEmbeddingLoss(margin=0.5)

# 相似的向量对
x1_similar = torch.tensor([[1.0, 0.0, 0.0]])
x2_similar = torch.tensor([[0.9, 0.1, 0.0]])

# 不相似的向量对
x1_diff = torch.tensor([[1.0, 0.0, 0.0]])
x2_diff = torch.tensor([[0.0, 1.0, 0.0]])

# 标签：1表示相似，-1表示不相似
labels_similar = torch.tensor([1.0])
labels_diff = torch.tensor([-1.0])

# 计算损失
loss_similar = cosine_loss(x1_similar, x2_similar, labels_similar)
loss_diff = cosine_loss(x1_diff, x2_diff, labels_diff)

print(f"Loss for similar pair: {loss_similar.item():.4f}")  # 应该较低
print(f"Loss for different pair: {loss_diff.item():.4f}")    # 应该较低

# 计算余弦相似度
cos_sim = F.cosine_similarity(x1_similar, x2_similar)
print(f"Cosine similarity: {cos_sim.item():.4f}")
\`\`\`

### 语义匹配示例

\`\`\`python
import torch
import torch.nn as nn

class SemanticMatcher(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, embed_dim, batch_first=True)
    
    def forward(self, x):
        emb = self.embedding(x)
        _, (hidden, _) = self.encoder(emb)
        return hidden.squeeze(0)

# 语义匹配训练
vocab_size, embed_dim = 10000, 256
model = SemanticMatcher(vocab_size, embed_dim)
criterion = nn.CosineEmbeddingLoss(margin=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模拟句子对
# 格式：[batch_size, seq_len]
sentence1 = torch.randint(0, vocab_size, (16, 20))
sentence2_match = torch.randint(0, vocab_size, (16, 20))   # 匹配的句子
sentence2_mismatch = torch.randint(0, vocab_size, (16, 20))  # 不匹配的句子

# 获取嵌入
emb1 = model(sentence1)
emb2_match = model(sentence2_match)
emb2_mismatch = model(sentence2_mismatch)

# 计算损失
labels_match = torch.ones(16)
labels_mismatch = -torch.ones(16)

loss_match = criterion(emb1, emb2_match, labels_match)
loss_mismatch = criterion(emb1, emb2_mismatch, labels_mismatch)
total_loss = loss_match + loss_mismatch

print(f"Matching Loss: {loss_match.item():.4f}")
print(f"Mismatching Loss: {loss_mismatch.item():.4f}")
print(f"Total Loss: {total_loss.item():.4f}")
\`\`\`

### 注意事项

1. **向量归一化**：输入向量会自动归一化计算余弦相似度
2. **边际选择**：\`margin\`通常设置在0到1之间
3. **标签取值**：标签必须是1（相似）或-1（不相似）
4. **与欧氏距离的关系**：归一化后，余弦距离等价于欧氏距离的平方

---

## 11. nn.CTCLoss - CTC损失

### 数学公式

CTC（Connectionist Temporal Classification）损失用于序列到序列的学习，无需对齐：

$$
\\text{CTC}(x, y) = -\\log P(y|x) = -\\log \\sum_{\\pi \\in \\mathcal{B}^{-1}(y)} P(\\pi|x)
$$

其中\`π\`是对齐路径，\`B\`是合并相同字符的操作。

### 用途场景

1. **语音识别**：语音转文字
2. **手写识别**：手写文字识别
3. **OCR**：场景文字识别
4. **蛋白质序列预测**：生物信息学

### 代码示例

\`\`\`python
import torch
import torch.nn as nn

# 创建CTCLoss实例
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

# 模拟模型输出
# shape: [seq_len, batch_size, num_classes]
# 假设序列长度50，batch_size=4，类别数20（包括blank）
log_probs = torch.randn(50, 4, 20).log_softmax(2)
log_probs = log_probs.requires_grad_()

# 目标标签
# shape: [batch_size, max_target_length] 或展平形式
targets = torch.randint(1, 20, (4, 10))  # 每个样本最多10个字符

# 输入长度和目标长度
input_lengths = torch.tensor([50, 48, 52, 45])  # 每个样本的实际长度
target_lengths = torch.tensor([8, 7, 10, 6])    # 每个样本的目标长度

# 计算CTC损失
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
print(f"CTC Loss: {loss.item():.4f}")
\`\`\`

### 语音识别示例

\`\`\`python
import torch
import torch.nn as nn

class SpeechRecognizer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers=3, 
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x: [batch, time, features]
        output, _ = self.lstm(x)
        logits = self.fc(output)
        return logits.log_softmax(2).transpose(0, 1)  # [time, batch, classes]

# 模型配置
input_dim = 80    # MFCC特征维度
hidden_dim = 256
num_classes = 30  # 字母表大小 + blank

model = SpeechRecognizer(input_dim, hidden_dim, num_classes)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

# 模拟音频数据
batch_size = 8
audio_features = torch.randn(batch_size, 200, input_dim)  # 200帧音频
transcripts = torch.randint(1, num_classes, (batch_size, 20))  # 转录文本

# 前向传播
log_probs = model(audio_features)

# 长度信息
input_lengths = torch.full((batch_size,), 200, dtype=torch.long)
target_lengths = torch.randint(10, 20, (batch_size,))

# 计算损失
loss = criterion(log_probs, transcripts, input_lengths, target_lengths)
print(f"Speech Recognition CTC Loss: {loss.item():.4f}")
\`\`\`

### 解码（推理）

\`\`\`python
import torch

def ctc_greedy_decode(log_probs, blank=0):
    """
    贪婪解码CTC输出
    log_probs: [seq_len, num_classes]
    """
    # 获取每个时间步的预测类别
    best_paths = torch.argmax(log_probs, dim=1)
    
    # 合并重复字符
    decoded = []
    prev = None
    for p in best_paths:
        if p != prev and p != blank:
            decoded.append(p.item())
        prev = p
    
    return decoded

# 解码示例
log_probs_sample = torch.randn(100, 30).log_softmax(1)
decoded_sequence = ctc_greedy_decode(log_probs_sample, blank=0)
print(f"Decoded sequence: {decoded_sequence}")
\`\`\`

### 注意事项

1. **输入格式**：\`log_probs\`必须是[seq_len, batch, num_classes]形状
2. **长度信息**：必须提供准确的input_lengths和target_lengths
3. **空白符号**：blank通常设为0，放在字符表开头
4. **数值稳定性**：使用\`zero_infinity=True\`处理极端情况
5. **内存消耗**：长序列可能消耗大量内存

---

## 12. nn.HingeEmbeddingLoss - 铰链嵌入损失

### 数学公式

铰链嵌入损失：

$$
\\text{loss}(x, y) = \\begin{cases}
x & \\text{if } y = 1 \\\\
\\max(0, \\text{margin} - x) & \\text{if } y = -1
\\end{cases}
$$

### 用途场景

1. **度量学习**：学习样本间的相似度
2. **半监督学习**：利用未标注数据
3. **图嵌入**：节点相似度学习
4. **知识图谱**：实体关系学习

### 代码示例

\`\`\`python
import torch
import torch.nn as nn

# 创建HingeEmbeddingLoss实例
hinge_loss = nn.HingeEmbeddingLoss(margin=1.0)

# 相似度得分（通常是两个嵌入的点积或距离）
scores = torch.tensor([2.0, 0.5, 1.5, -0.5], requires_grad=True)

# 标签：1表示相似（正样本对），-1表示不相似（负样本对）
labels = torch.tensor([1.0, -1.0, 1.0, -1.0])

# 计算损失
loss = hinge_loss(scores, labels)
print(f"Hinge Embedding Loss: {loss.item():.4f}")

# 手动验证
# 第1个样本: y=1, loss = 2.0
# 第2个样本: y=-1, loss = max(0, 1-0.5) = 0.5
# 第3个样本: y=1, loss = 1.5
# 第4个样本: y=-1, loss = max(0, 1-(-0.5)) = 1.5
\`\`\`

### 图嵌入示例

\`\`\`python
import torch
import torch.nn as nn

class GraphEmbedding(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
    
    def get_similarity(self, node1, node2):
        emb1 = self.embedding(node1)
        emb2 = self.embedding(node2)
        return (emb1 * emb2).sum(dim=1)  # 点积相似度

# 训练
num_nodes, embed_dim = 1000, 128
model = GraphEmbedding(num_nodes, embed_dim)
criterion = nn.HingeEmbeddingLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 正样本对（有边连接的节点）
pos_pairs = torch.randint(0, num_nodes, (64, 2))
# 负样本对（无边连接的节点）
neg_pairs = torch.randint(0, num_nodes, (64, 2))

# 计算相似度
pos_sim = model.get_similarity(pos_pairs[:, 0], pos_pairs[:, 1])
neg_sim = model.get_similarity(neg_pairs[:, 0], neg_pairs[:, 1])

# 合并并计算损失
all_sim = torch.cat([pos_sim, neg_sim])
all_labels = torch.cat([torch.ones(64), -torch.ones(64)])

loss = criterion(all_sim, all_labels)
print(f"Graph Embedding Hinge Loss: {loss.item():.4f}")
\`\`\`

### 半监督学习示例

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(embeddings, labels, temperature=0.1):
    """
    使用HingeEmbeddingLoss的对比损失
    同类样本相似，不同类样本不相似
    """
    batch_size = embeddings.size(0)
    
    # 计算相似度矩阵
    sim_matrix = torch.mm(embeddings, embeddings.t()) / temperature
    
    # 构建标签矩阵
    label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    label_matrix[label_matrix == 0] = -1  # 不同类为-1
    label_matrix.fill_diagonal_(0)  # 忽略对角线
    
    # 只使用非对角线元素
    mask = ~torch.eye(batch_size, dtype=torch.bool)
    sim_flat = sim_matrix[mask]
    label_flat = label_matrix[mask]
    
    hinge_loss = nn.HingeEmbeddingLoss(margin=1.0)
    return hinge_loss(sim_flat, label_flat)

# 示例
embeddings = F.normalize(torch.randn(32, 128), dim=1)
labels = torch.randint(0, 5, (32,))  # 5个类别

loss = contrastive_loss(embeddings, labels)
print(f"Contrastive Loss with Hinge: {loss.item():.4f}")
\`\`\`

### 注意事项

1. **输入含义**：输入应该是相似度度量，而非原始嵌入
2. **标签取值**：1表示相似（正样本对），-1表示不相似（负样本对）
3. **边际设置**：\`margin\`控制负样本对的期望距离
4. **与TripletLoss的关系**：可以看作TripletLoss的简化版本

---

## 总结

### 损失函数选择指南

| 任务类型 | 推荐损失函数 | 备注 |
|---------|------------|------|
| 二分类 | BCEWithLogitsLoss | 数值稳定，包含Sigmoid |
| 多分类 | CrossEntropyLoss | 包含Softmax |
| 回归 | MSELoss / L1Loss | MSE对异常值敏感 |
| 边界框回归 | SmoothL1Loss | 目标检测标配 |
| 知识蒸馏 | KLDivLoss | 需要温度参数 |
| 度量学习 | TripletMarginLoss | 人脸识别、ReID |
| 语音识别 | CTCLoss | 无需对齐 |
| 排序学习 | MarginRankingLoss | 推荐系统 |

### 最佳实践

1. **数值稳定性**：优先使用包含内置激活函数的损失（如BCEWithLogitsLoss）
2. **类别不平衡**：使用class_weight或pos_weight参数
3. **梯度分析**：理解损失函数的梯度特性有助于选择合适的学习率
4. **组合损失**：复杂任务可以组合多种损失函数

\`\`\`python
# 示例：组合多种损失
def combined_loss(pred, target, alpha=0.5):
    mse = nn.MSELoss()(pred, target)
    l1 = nn.L1Loss()(pred, target)
    return alpha * mse + (1 - alpha) * l1
\`\`\`

希望本文能帮助你深入理解PyTorch损失函数，在实际项目中做出正确的选择！
`,
  },
]

export function getPost(id: string): BlogPost | undefined {
  return posts.find((post) => post.id === id)
}

export function getAllPosts(): BlogPost[] {
  return posts
}
