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
  {
    id: 'llm-pretraining-guide',
    title: '大模型预训练完全指南：从架构原理到工程实践',
    summary: '深入解析大语言模型预训练的完整流程，涵盖Transformer架构原理、分布式训练策略、数据处理流水线，以及从零开始预训练LLaMA的完整代码实现。',
    date: '2024-02-20',
    readingTime: '45分钟',
    tags: ['大模型', 'LLM', '预训练', 'Transformer', '分布式训练'],
    content: `
## 引言

大语言模型（Large Language Model, LLM）的预训练是当代人工智能领域最具变革性的技术之一。从GPT系列的惊艳表现到LLaMA的高效架构，预训练技术正在重新定义机器学习的边界。本文将全面深入地探讨大模型预训练的方方面面，从理论基础到工程实践，帮助你建立完整的知识体系。

### 预训练的意义

预训练的核心思想是让模型在海量无标注文本上学习通用的语言表示，然后通过微调适应下游任务。这种方法具有以下优势：

1. **数据效率**：无需大量标注数据即可获得优秀性能
2. **知识迁移**：预训练获得的知识可以有效迁移到各种任务
3. **泛化能力**：模型学习到通用的语言理解能力
4. **规模效应**：随着模型规模增大，性能持续提升

### 发展历程

大模型预训练的发展经历了几个重要阶段：

- **2017年**：Transformer架构提出，奠定了现代大模型的基础
- **2018年**：BERT和GPT-1发布，开启了预训练语言模型时代
- **2019年**：GPT-2展示了大规模语言模型的生成能力
- **2020年**：GPT-3证明了规模法则，1750亿参数惊艳世界
- **2022年**：ChatGPT发布，大模型走进大众视野
- **2023年**：LLaMA系列开源，降低了大模型的研究门槛
- **2024年**：多模态大模型蓬勃发展，AI Agent成为热点

---

## Transformer架构深度解析

Transformer是大语言模型的基石架构。与传统的RNN/LSTM相比，Transformer完全基于注意力机制，具有并行计算优势和更强的长距离依赖建模能力。

![Transformer架构](/images/transformer-architecture.png)

### 整体架构

Transformer采用编码器-解码器（Encoder-Decoder）结构，包含以下核心组件：

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerConfig:
    """Transformer模型配置"""
    def __init__(
        self,
        vocab_size=32000,
        d_model=4096,
        n_heads=32,
        n_layers=32,
        d_ff=11008,
        max_seq_len=2048,
        dropout=0.1,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout


class Transformer(nn.Module):
    """完整的Transformer模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 词嵌入和位置编码
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Dropout层
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Transformer层堆叠
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(config.d_model)
        
        # 输出层（语言模型头）
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重共享
        self.lm_head.weight = self.token_embedding.weight
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # 生成位置索引
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # 嵌入层
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)
        
        # 因果注意力掩码
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device), 
                diagonal=1
            ).bool()
        else:
            causal_mask = ~attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Transformer层
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        # 最终归一化
        x = self.final_norm(x)
        
        # 语言模型头
        logits = self.lm_head(x)
        
        return logits
\`\`\`

### 嵌入层详解

嵌入层将离散的词元（token）映射为连续的向量表示：

$$
\\text{Embedding}(x) = E_x + P_x
$$

其中 $E_x$ 是词嵌入，$P_x$ 是位置嵌入。

\`\`\`python
class EmbeddingLayer(nn.Module):
    """嵌入层：词嵌入 + 位置嵌入"""
    
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 可选：使用旋转位置编码（RoPE）
        self.use_rope = False
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        
        # 词嵌入
        token_embeds = self.token_embedding(input_ids)
        
        # 位置嵌入
        position_embeds = self.position_embedding(positions)
        
        # 组合
        embeddings = token_embeds + position_embeds
        embeddings = self.dropout(embeddings)
        
        return embeddings
\`\`\`

### 前馈神经网络（FFN）

每个Transformer层包含一个前馈神经网络，用于非线性变换：

$$
\\text{FFN}(x) = \\text{GELU}(xW_1 + b_1)W_2 + b_2
$$

\`\`\`python
class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # 使用SwiGLU激活函数（LLaMA风格）
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
    
    def forward(self, x):
        # SwiGLU: gate * up_proj
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        out = self.down_proj(gate * up)
        return self.dropout(out)
\`\`\`

### 层归一化

层归一化对于训练稳定性至关重要：

$$
\\text{LayerNorm}(x) = \\gamma \\cdot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta
$$

\`\`\`python
class RMSNorm(nn.Module):
    """RMS LayerNorm（LLaMA使用）"""
    
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x):
        # RMS归一化
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
\`\`\`

---

## 自注意力机制详解

自注意力机制是Transformer的核心创新，它允许模型在处理每个位置时关注输入序列的所有位置。

![自注意力机制](/images/self-attention.png)

### 数学原理

自注意力计算查询（Query）、键（Key）、值（Value）三个矩阵：

$$
\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V
$$

其中：
- $Q = XW_Q$：查询矩阵
- $K = XW_K$：键矩阵  
- $V = XW_V$：值矩阵
- $d_k$：键向量的维度

### 缩放点积注意力

\`\`\`python
class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    
    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.scale = math.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        # 计算注意力分数
        # q: [batch, heads, seq_len, d_k]
        # k: [batch, heads, seq_len, d_k]
        # 注意力分数: [batch, heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == True, float('-inf'))
        
        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights
\`\`\`

### 多头注意力

多头注意力允许模型同时关注不同位置的不同表示子空间：

$$
\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h)W^O
$$

其中每个head为：

$$
\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

\`\`\`python
class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Q, K, V投影层
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 注意力计算
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头形式
        # [batch, seq_len, d_model] -> [batch, n_heads, seq_len, d_k]
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attn_output, attn_weights = self.attention(q, k, v, mask)
        
        # 合并多头
        # [batch, n_heads, seq_len, d_k] -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 输出投影
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        return output, attn_weights
\`\`\`

### 因果注意力掩码

在语言模型中，为了防止模型看到未来的词元，需要使用因果注意力掩码：

\`\`\`python
def create_causal_mask(seq_len, device):
    """创建因果注意力掩码"""
    # 上三角矩阵（不包括对角线）设为True
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask

# 示例：4长度序列的因果掩码
# [[False, True,  True,  True ],
#  [False, False, True,  True ],
#  [False, False, False, True ],
#  [False, False, False, False]]
\`\`\`

### 旋转位置编码（RoPE）

RoPE是一种相对位置编码方法，被LLaMA等模型采用：

$$
f(x, m) = 
\\begin{pmatrix}
x_1 \\\\
x_2 \\\\
x_3 \\\\
x_4 \\\\
\\vdots
\\end{pmatrix}
\\otimes
\\begin{pmatrix}
\\cos(m\\theta_1) \\\\
\\cos(m\\theta_1) \\\\
\\cos(m\\theta_2) \\\\
\\cos(m\\theta_2) \\\\
\\vdots
\\end{pmatrix}
+
\\begin{pmatrix}
-x_2 \\\\
x_1 \\\\
-x_4 \\\\
x_3 \\\\
\\vdots
\\end{pmatrix}
\\otimes
\\begin{pmatrix}
\\sin(m\\theta_1) \\\\
\\sin(m\\theta_1) \\\\
\\sin(m\\theta_2) \\\\
\\sin(m\\theta_2) \\\\
\\vdots
\\end{pmatrix}
$$

\`\`\`python
class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码（RoPE）"""
    
    def __init__(self, d_model, max_seq_len=2048, base=10000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算cos和sin值
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # [seq_len, d_model/2]
        
        # 复制以匹配完整维度
        emb = torch.cat((freqs, freqs), dim=-1)
        # [seq_len, d_model]
        
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))
    
    def forward(self, x, seq_len):
        # x: [batch, n_heads, seq_len, d_k]
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        # 应用旋转
        return self._apply_rotary_emb(x, cos, sin)
    
    def _apply_rotary_emb(self, x, cos, sin):
        # 将x分成两部分
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        
        # 旋转
        rotated_x = torch.cat((-x2, x1), dim=-1)
        
        # 应用旋转位置编码
        return x * cos + rotated_x * sin
\`\`\`

### Flash Attention优化

Flash Attention是一种高效的注意力计算方法，显著减少内存访问：

\`\`\`python
# Flash Attention的核心思想：分块计算，避免存储完整的注意力矩阵
class FlashAttention(nn.Module):
    """Flash Attention简化实现"""
    
    def __init__(self, d_k, dropout=0.1, block_size=256):
        super().__init__()
        self.scale = math.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size
    
    def forward(self, q, k, v, mask=None):
        batch_size, n_heads, seq_len, d_k = q.shape
        
        # 输出张量
        output = torch.zeros_like(q)
        
        # 分块计算
        for i in range(0, seq_len, self.block_size):
            q_block = q[:, :, i:i+self.block_size, :]
            
            # 分块计算注意力
            block_output = self._compute_block_attention(q_block, k, v, mask, i)
            output[:, :, i:i+self.block_size, :] = block_output
        
        return output
    
    def _compute_block_attention(self, q_block, k, v, mask, start_idx):
        # 计算当前块的注意力分数
        scores = torch.matmul(q_block, k.transpose(-2, -1)) / self.scale
        
        # 应用因果掩码
        if mask is not None:
            end_idx = start_idx + q_block.shape[2]
            block_mask = mask[:, :, start_idx:end_idx, :]
            scores = scores.masked_fill(block_mask, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        return torch.matmul(attention_weights, v)
\`\`\`

---

## 主流大模型架构对比

不同的模型架构在设计和性能上各有特点。下面我们对主流大模型进行详细对比。

### GPT系列架构

GPT（Generative Pre-trained Transformer）采用仅解码器（Decoder-only）架构：

\`\`\`python
class GPTBlock(nn.Module):
    """GPT风格的Transformer块"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # Pre-LN结构（先归一化，后计算）
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x, mask=None):
        # 注意力残差连接
        x = x + self.attention(self.ln1(x), mask)[0]
        # FFN残差连接
        x = x + self.ffn(self.ln2(x))
        return x
\`\`\`

### BERT架构

BERT（Bidirectional Encoder Representations from Transformers）采用仅编码器架构：

\`\`\`python
class BERTBlock(nn.Module):
    """BERT风格的Transformer块"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # Post-LN结构（先计算，后归一化）
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # 注意力残差连接
        x = self.ln1(x + self.attention(x, mask)[0])
        # FFN残差连接
        x = self.ln2(x + self.ffn(x))
        return x
\`\`\`

### LLaMA架构特点

LLaMA对标准Transformer进行了多项优化：

\`\`\`python
class LLaMABlock(nn.Module):
    """LLaMA风格的Transformer块"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # RMSNorm替代LayerNorm
        self.ln1 = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.ln2 = RMSNorm(d_model)
        # SwiGLU激活的FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x, mask=None):
        # Pre-Norm结构
        x = x + self.attention(self.ln1(x), mask)[0]
        x = x + self.ffn(self.ln2(x))
        return x


class LLaMAModel(nn.Module):
    """LLaMA模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 嵌入层（不使用位置嵌入，依赖RoPE）
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer层
        self.layers = nn.ModuleList([
            LLaMABlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout
            ) for _ in range(config.n_layers)
        ])
        
        # 最终归一化
        self.final_norm = RMSNorm(config.d_model)
        
        # 语言模型头
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重共享
        self.lm_head.weight = self.token_embedding.weight
        
        # RoPE
        self.rope = RotaryPositionEmbedding(
            config.d_model // config.n_heads,
            config.max_seq_len
        )
    
    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        
        # 因果掩码
        causal_mask = create_causal_mask(
            input_ids.shape[1], 
            input_ids.device
        )
        
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits
\`\`\`

### 架构对比表

| 特性 | GPT系列 | BERT | LLaMA |
|-----|--------|------|-------|
| 架构类型 | Decoder-only | Encoder-only | Decoder-only |
| 注意力方向 | 单向（因果） | 双向 | 单向（因果） |
| 归一化位置 | Pre-LN | Post-LN | Pre-LN |
| 归一化类型 | LayerNorm | LayerNorm | RMSNorm |
| 位置编码 | 可学习/RoPE | 可学习 | RoPE |
| 激活函数 | GELU | GELU | SwiGLU |
| 偏置项 | 有 | 有 | 无 |
| 预训练任务 | 自回归 | MLM+NSP | 自回归 |

---

## 预训练任务设计

预训练任务的设计直接影响模型学习到的表示质量。

### 自回归语言建模（CLM）

自回归语言建模是GPT和LLaMA使用的预训练任务：

$$
\\mathcal{L}_{CLM} = -\\sum_{t=1}^{T} \\log P(x_t | x_{<t})
$$

\`\`\`python
class CausalLMObjective(nn.Module):
    """自回归语言建模目标"""
    
    def __init__(self, vocab_size, ignore_index=-100):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, logits, labels):
        # logits: [batch, seq_len, vocab_size]
        # labels: [batch, seq_len]
        
        # 移位：预测下一个词
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # 计算损失
        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
\`\`\`

### 掩码语言建模（MLM）

MLM是BERT使用的预训练任务，随机掩盖部分词元：

$$
\\mathcal{L}_{MLM} = -\\sum_{i \\in M} \\log P(x_i | x_{\\setminus M})
$$

\`\`\`python
class MLMMasker:
    """MLM掩码生成器"""
    
    def __init__(
        self, 
        vocab_size, 
        mask_token_id, 
        mask_prob=0.15,
        random_replace_prob=0.1,
        keep_prob=0.1
    ):
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.random_replace_prob = random_replace_prob
        self.keep_prob = keep_prob
    
    def __call__(self, input_ids):
        # 复制原始输入
        labels = input_ids.clone()
        
        # 随机选择掩码位置
        probability_matrix = torch.full(
            input_ids.shape, self.mask_prob
        )
        
        # 掩码
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 保存真实标签
        labels[~masked_indices] = -100  # 忽略
        
        # 80%替换为[MASK]
        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, 1 - self.random_replace_prob - self.keep_prob)
        ).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        # 10%替换为随机词
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, self.random_replace_prob)
        ).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            self.vocab_size, input_ids.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]
        
        # 10%保持不变
        
        return input_ids, labels
\`\`\`

### 混合预训练目标

现代大模型常使用多种预训练目标的组合：

\`\`\`python
class HybridPretrainingObjective(nn.Module):
    """混合预训练目标"""
    
    def __init__(
        self, 
        vocab_size, 
        mask_token_id,
        clm_weight=0.5,
        mlm_weight=0.5
    ):
        super().__init__()
        self.clm_weight = clm_weight
        self.mlm_weight = mlm_weight
        
        self.clm_loss = CausalLMObjective(vocab_size)
        self.mlm_masker = MLMMasker(vocab_size, mask_token_id)
    
    def forward(self, model, input_ids):
        # CLM损失
        clm_logits = model(input_ids)
        clm_loss = self.clm_loss(clm_logits, input_ids)
        
        # MLM损失
        masked_input, mlm_labels = self.mlm_masker(input_ids)
        mlm_logits = model(masked_input)
        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, mlm_logits.size(-1)),
            mlm_labels.view(-1),
            ignore_index=-100
        )
        
        # 组合损失
        total_loss = self.clm_weight * clm_loss + self.mlm_weight * mlm_loss
        
        return total_loss, {'clm_loss': clm_loss, 'mlm_loss': mlm_loss}
\`\`\`

---

## 大规模分布式训练

训练大模型需要分布式计算技术来处理海量参数和数据。

![分布式训练策略](/images/distributed-training.png)

### 数据并行（Data Parallelism）

数据并行是最简单的分布式策略：

\`\`\`python
import torch.distributed as dist
import torch.nn.parallel.DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    """初始化分布式环境"""
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def train_distributed(rank, world_size):
    """分布式训练"""
    setup_distributed(rank, world_size)
    
    # 创建模型并移到对应GPU
    model = LLaMAModel(config).to(rank)
    
    # 包装为DDP
    model = DDP(model, device_ids=[rank])
    
    # 数据加载器
    train_loader = get_dataloader(rank, world_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            batch = batch.to(rank)
            
            optimizer.zero_grad()
            outputs = model(batch)
            loss = compute_loss(outputs, batch)
            loss.backward()
            optimizer.step()
    
    dist.destroy_process_group()
\`\`\`

### 模型并行（Model Parallelism）

当模型太大无法放入单个GPU时，需要模型并行：

\`\`\`python
class PipelineParallelLLaMA(nn.Module):
    """流水线并行的LLaMA"""
    
    def __init__(self, config, num_stages=4):
        super().__init__()
        self.num_stages = num_stages
        layers_per_stage = config.n_layers // num_stages
        
        # 将层分配到不同设备
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            stage_layers = nn.ModuleList([
                LLaMABlock(config.d_model, config.n_heads, config.d_ff)
                for _ in range(layers_per_stage)
            ])
            self.stages.append(stage_layers)
        
        self.stage_devices = [f'cuda:{i}' for i in range(num_stages)]
    
    def forward(self, x):
        for i, stage in enumerate(self.stages):
            x = x.to(self.stage_devices[i])
            for layer in stage:
                x = layer(x)
        return x
\`\`\`

### 张量并行（Tensor Parallelism）

张量并行将单个层分割到多个GPU：

\`\`\`python
class TensorParallelAttention(nn.Module):
    """张量并行的多头注意力"""
    
    def __init__(self, d_model, n_heads, rank, world_size):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.rank = rank
        self.world_size = world_size
        
        # 每个GPU处理部分注意力头
        self.local_n_heads = n_heads // world_size
        self.d_k = d_model // n_heads
        
        # 分片的Q, K, V投影
        self.q_proj = nn.Linear(d_model, self.local_n_heads * self.d_k, bias=False)
        self.k_proj = nn.Linear(d_model, self.local_n_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, self.local_n_heads * self.d_k, bias=False)
        
        # 分片的输出投影
        self.out_proj = nn.Linear(self.local_n_heads * self.d_k, d_model, bias=False)
    
    def forward(self, x):
        # 本地计算
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头形式
        batch_size, seq_len, _ = q.shape
        q = q.view(batch_size, seq_len, self.local_n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.local_n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.local_n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # 重塑
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1
        )
        
        # 输出投影
        output = self.out_proj(output)
        
        # All-Reduce聚合结果
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        
        return output
\`\`\`

### ZeRO优化器

ZeRO（Zero Redundancy Optimizer）显著减少内存占用：

\`\`\`python
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

def configure_zero():
    """ZeRO配置"""
    return {
        "zero_optimization": {
            "stage": 3,  # ZeRO-3
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
        },
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": 1,
    }

# 计算内存需求
def estimate_memory(model, num_gpus):
    estimate_zero3_model_states_mem_needs_all_live(
        model, 
        num_gpus=num_gpus,
        num_parameters=model.num_parameters()
    )
\`\`\`

### DeepSpeed集成

\`\`\`python
import deepspeed

def train_with_deepspeed():
    """使用DeepSpeed训练"""
    model = LLaMAModel(config)
    
    # DeepSpeed配置
    ds_config = {
        "train_batch_size": 128,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-4,
                "warmup_num_steps": 2000,
                "total_num_steps": 100000
            }
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu"},
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8
        }
    }
    
    # 初始化DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config
    )
    
    # 训练循环
    for batch in train_loader:
        outputs = model_engine(batch)
        loss = compute_loss(outputs, batch)
        model_engine.backward(loss)
        model_engine.step()
\`\`\`

---

## 数据处理流水线

高质量的数据处理是预训练成功的关键。

![预训练流程](/images/pretraining-pipeline.png)

### 数据收集与清洗

\`\`\`python
import re
from bs4 import BeautifulSoup
from tqdm import tqdm

class DataCleaner:
    """数据清洗管道"""
    
    def __init__(self):
        self.min_length = 100
        self.max_length = 100000
    
    def clean_text(self, text):
        """清洗文本"""
        # 移除HTML标签
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # 标准化空白
        text = re.sub(r'\\s+', ' ', text)
        
        # 移除特殊字符（保留多语言支持）
        text = re.sub(r'[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\x9f]', '', text)
        
        # 移除过长的重复模式
        text = self._remove_repeated_patterns(text)
        
        return text.strip()
    
    def _remove_repeated_patterns(self, text, max_repeat=10):
        """移除重复模式"""
        # 检测并移除连续重复的短语
        pattern = re.compile(r'(.+?)\\1{' + str(max_repeat) + r',}')
        while True:
            new_text = pattern.sub(r'\\1', text)
            if new_text == text:
                break
            text = new_text
        return text
    
    def is_quality_text(self, text):
        """判断文本质量"""
        # 长度检查
        if len(text) < self.min_length or len(text) > self.max_length:
            return False
        
        # 字符多样性
        unique_chars = len(set(text))
        if unique_chars < 50:
            return False
        
        # 语言检测
        # ... 更多质量过滤
        
        return True


def process_corpus(input_files, output_file):
    """处理大规模语料库"""
    cleaner = DataCleaner()
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in tqdm(input_files):
            with open(input_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    text = cleaner.clean_text(line)
                    if cleaner.is_quality_text(text):
                        out_f.write(text + '\\n')
\`\`\`

### 分词器训练

\`\`\`python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor

def train_tokenizer(corpus_files, vocab_size=32000, output_path="tokenizer.json"):
    """训练BPE分词器"""
    # 初始化分词器
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    # 字节级预处理
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    # 训练器配置
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<pad>",
            "<s>",
            "</s>",
            "<unk>",
            "<mask>",
        ],
        show_progress=True
    )
    
    # 训练
    tokenizer.train(files=corpus_files, trainer=trainer)
    
    # 后处理
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    
    # 保存
    tokenizer.save(output_path)
    
    return tokenizer


class TokenizerWrapper:
    """分词器包装器"""
    
    def __init__(self, tokenizer_path):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.bos_token_id = self.tokenizer.token_to_id("<s>")
        self.eos_token_id = self.tokenizer.token_to_id("</s>")
    
    def encode(self, text, max_length=None):
        """编码文本"""
        encoding = self.tokenizer.encode(text)
        
        if max_length:
            # 截断
            encoding.ids = encoding.ids[:max_length]
        
        return encoding.ids
    
    def decode(self, ids):
        """解码ID序列"""
        return self.tokenizer.decode(ids)
    
    def batch_encode(self, texts, max_length=None, padding=True):
        """批量编码"""
        all_ids = []
        all_attention_masks = []
        
        for text in texts:
            ids = self.encode(text, max_length)
            attention_mask = [1] * len(ids)
            all_ids.append(ids)
            all_attention_masks.append(attention_mask)
        
        if padding:
            max_len = max(len(ids) for ids in all_ids)
            for i in range(len(all_ids)):
                padding_length = max_len - len(all_ids[i])
                all_ids[i] = all_ids[i] + [self.pad_token_id] * padding_length
                all_attention_masks[i] = all_attention_masks[i] + [0] * padding_length
        
        return {
            'input_ids': torch.tensor(all_ids),
            'attention_mask': torch.tensor(all_attention_masks)
        }
\`\`\`

### 数据加载器

\`\`\`python
from torch.utils.data import Dataset, DataLoader

class PretrainingDataset(Dataset):
    """预训练数据集"""
    
    def __init__(
        self, 
        data_path, 
        tokenizer, 
        max_length=2048,
        buffer_size=10000
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        
        # 加载并缓存数据
        self.data = self._load_and_cache_data(data_path)
    
    def _load_and_cache_data(self, data_path):
        """加载并缓存数据"""
        cached_data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            current_chunk = []
            current_length = 0
            
            for line in f:
                tokens = self.tokenizer.encode(line.strip())
                current_chunk.extend(tokens)
                current_length += len(tokens)
                
                # 当累积足够长度时，分割成训练样本
                while current_length >= self.max_length + 1:
                    sample = current_chunk[:self.max_length + 1]
                    cached_data.append(sample)
                    current_chunk = current_chunk[self.max_length:]
                    current_length = len(current_chunk)
        
        return cached_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ids = self.data[idx]
        return torch.tensor(ids, dtype=torch.long)


def create_dataloader(
    data_path, 
    tokenizer, 
    batch_size=32,
    max_length=2048,
    num_workers=4,
    shuffle=True
):
    """创建数据加载器"""
    dataset = PretrainingDataset(
        data_path, 
        tokenizer, 
        max_length
    )
    
    # 自定义批次处理
    def collate_fn(batch):
        # 填充到批次内最大长度
        max_len = max(len(item) for item in batch)
        
        input_ids = []
        attention_mask = []
        
        for item in batch:
            padding_length = max_len - len(item)
            input_ids.append(
                torch.cat([item, torch.zeros(padding_length, dtype=torch.long)])
            )
            attention_mask.append(
                torch.cat([torch.ones(len(item)), torch.zeros(padding_length)])
            )
        
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask)
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
\`\`\`

---

## 实战：从零预训练小型LLaMA

现在让我们实现一个完整的小型LLaMA预训练流程。

### 模型定义

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLaMAConfig:
    """LLaMA模型配置"""
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 1376  # 约 8/3 * d_model
    max_seq_len: int = 1024
    dropout: float = 0.1
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    """RMS LayerNorm"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class RotaryEmbedding(nn.Module):
    """旋转位置编码"""
    
    def __init__(self, d_k: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # 计算频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x, seq_len):
        # 生成位置索引
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        
        # 计算频率
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 应用旋转
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        
        return self._apply_rotary(x, cos, sin)
    
    def _apply_rotary(self, x, cos, sin):
        # x: [batch, n_heads, seq_len, d_k]
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


class LLaMAAttention(nn.Module):
    """LLaMA多头注意力"""
    
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        # 投影层（无偏置）
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # RoPE
        self.rope = RotaryEmbedding(
            self.d_k, 
            config.max_seq_len,
            config.rope_theta
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 应用RoPE
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算输出
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.o_proj(output)


class LLaMAFFN(nn.Module):
    """LLaMA前馈网络（SwiGLU）"""
    
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class LLaMATransformerBlock(nn.Module):
    """LLaMA Transformer块"""
    
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.attention = LLaMAAttention(config)
        self.ffn = LLaMAFFN(config)
        self.ln1 = RMSNorm(config.d_model)
        self.ln2 = RMSNorm(config.d_model)
    
    def forward(self, x, mask=None):
        # Pre-Norm结构
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class LLaMAForCausalLM(nn.Module):
    """LLaMA因果语言模型"""
    
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer层
        self.layers = nn.ModuleList([
            LLaMATransformerBlock(config) 
            for _ in range(config.n_layers)
        ])
        
        # 最终归一化
        self.final_norm = RMSNorm(config.d_model)
        
        # 语言模型头
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重共享
        self.lm_head.weight = self.token_embedding.weight
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # 嵌入
        x = self.token_embedding(input_ids)
        
        # 因果掩码
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device), 
            diagonal=1
        ).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Transformer层
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        # 最终归一化
        x = self.final_norm(x)
        
        # 语言模型头
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self, 
        input_ids, 
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        top_p=None
    ):
        """生成文本"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # 截断到最大长度
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            
            # 前向传播
            logits = self(idx_cond)
            
            # 只取最后一个位置的logits
            logits = logits[:, -1, :] / temperature
            
            # Top-k采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p采样
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 追加
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def count_parameters(self):
        """计算参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 创建模型
config = LLaMAConfig(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_layers=12,
    d_ff=1376,
    max_seq_len=1024
)

model = LLaMAForCausalLM(config)
print(f"模型参数量: {model.count_parameters() / 1e6:.2f}M")
\`\`\`

### 训练循环

\`\`\`python
import wandb
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

class Trainer:
    """预训练器"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        tokenizer,
        config,
        output_dir="checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = output_dir
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=config.warmup_ratio
        )
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        # 初始化wandb
        if config.use_wandb:
            wandb.init(project="llama-pretraining", config=vars(config))
    
    def train(self):
        """训练循环"""
        best_val_loss = float('inf')
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0
            
            progress_bar = tqdm(
                self.train_loader, 
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
            )
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    # 前向传播
                    logits = self.model(input_ids)
                    
                    # 计算损失（下一个词预测）
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    
                    loss = self.loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                
                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                # 更新权重
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # 更新学习率
                self.scheduler.step()
                
                # 记录
                epoch_loss += loss.item()
                global_step += 1
                
                if global_step % self.config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{lr:.2e}'
                    })
                    
                    if self.config.use_wandb:
                        wandb.log({
                            'train/loss': loss.item(),
                            'train/lr': lr,
                            'train/step': global_step
                        })
            
            # 验证
            val_loss = self.validate()
            avg_train_loss = epoch_loss / len(self.train_loader)
            
            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # 记录到wandb
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': avg_train_loss,
                    'val/loss': val_loss
                })
            
            # 保存检查点
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.save_checkpoint(epoch, val_loss, is_best=False)
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    logits = self.model(input_ids)
                    
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    
                    loss = self.loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': vars(self.config)
        }
        
        path = f"{self.output_dir}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = f"{self.output_dir}/best_model.pt"
            torch.save(checkpoint, best_path)
        
        print(f"Checkpoint saved: {path}")


@dataclass
class TrainingConfig:
    """训练配置"""
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    num_epochs: int = 10
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    log_interval: int = 100
    use_amp: bool = True
    use_wandb: bool = True


# 启动训练
if __name__ == "__main__":
    # 配置
    model_config = LLaMAConfig()
    training_config = TrainingConfig()
    
    # 创建模型
    model = LLaMAForCausalLM(model_config)
    
    # 创建数据加载器
    train_loader = create_dataloader("train.txt", tokenizer, batch_size=32)
    val_loader = create_dataloader("val.txt", tokenizer, batch_size=32)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=training_config
    )
    
    # 开始训练
    trainer.train()
\`\`\`

---

## 训练监控与调优

### 监控指标

\`\`\`python
import matplotlib.pyplot as plt
from collections import defaultdict

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def log(self, metric_name, value):
        """记录指标"""
        self.metrics[metric_name].append(value)
    
    def plot_metrics(self, save_path=None):
        """绘制指标曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 损失曲线
        if 'train_loss' in self.metrics:
            axes[0, 0].plot(self.metrics['train_loss'], label='Train')
            if 'val_loss' in self.metrics:
                axes[0, 0].plot(self.metrics['val_loss'], label='Val')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training/Validation Loss')
            axes[0, 0].legend()
        
        # 学习率曲线
        if 'learning_rate' in self.metrics:
            axes[0, 1].plot(self.metrics['learning_rate'])
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule')
        
        # 梯度范数
        if 'grad_norm' in self.metrics:
            axes[1, 0].plot(self.metrics['grad_norm'])
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_title('Gradient Norm')
        
        # Perplexity
        if 'train_ppl' in self.metrics:
            axes[1, 1].plot(self.metrics['train_ppl'], label='Train')
            if 'val_ppl' in self.metrics:
                axes[1, 1].plot(self.metrics['val_ppl'], label='Val')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Perplexity')
            axes[1, 1].set_title('Perplexity')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


def compute_perplexity(loss):
    """计算困惑度"""
    return math.exp(loss)


def compute_grad_norm(model):
    """计算梯度范数"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)
\`\`\`

### 超参数调优

\`\`\`python
import optuna

def objective(trial):
    """Optuna优化目标"""
    # 超参数搜索空间
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.01, 0.3)
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.01, 0.1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # 创建配置
    config = TrainingConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio
    )
    
    # 训练模型
    model = LLaMAForCausalLM(LLaMAConfig())
    trainer = Trainer(model, train_loader, val_loader, tokenizer, config)
    
    # 简化训练
    val_loss = trainer.quick_validate()
    
    return val_loss

# 运行优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print(f"Best trial: {study.best_trial.params}")
\`\`\`

---

## 总结与展望

### 核心要点回顾

本文全面介绍了大模型预训练的关键技术：

1. **架构设计**：Transformer是基础，LLaMA进行了多项优化（RoPE、SwiGLU、RMSNorm）
2. **自注意力机制**：多头注意力、因果掩码、位置编码是核心组件
3. **分布式训练**：数据并行、模型并行、张量并行、ZeRO是处理大规模训练的关键
4. **数据处理**：高质量的数据清洗、分词、加载器设计至关重要
5. **训练技巧**：学习率调度、梯度裁剪、混合精度训练是稳定训练的法宝

### 未来发展趋势

大模型预训练领域正在快速发展：

1. **更长的上下文**：从2048到100K+，长上下文成为标配
2. **多模态融合**：文本、图像、音频、视频的统一建模
3. **高效架构**：Mamba、RWKV等线性注意力架构崭露头角
4. **推理优化**：量化、蒸馏、剪枝技术日趋成熟
5. **AI Agent**：大模型作为Agent的核心大脑，与工具深度集成

### 进一步学习资源

- **论文**：Attention Is All You Need, GPT-3, LLaMA, FlashAttention
- **开源项目**：Hugging Face Transformers, DeepSpeed, Megatron-LM
- **课程**：Stanford CS224N, DeepLearning.AI LLM系列

希望本指南能帮助你建立对大模型预训练的完整理解，并在实践中取得成功！
`,
  },
]

export function getPost(id: string): BlogPost | undefined {
  return posts.find((post) => post.id === id)
}

export function getAllPosts(): BlogPost[] {
  return posts
}
