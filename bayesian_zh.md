# 从鱼类到分类器：贝叶斯决策理论和朴素贝叶斯实用指南

在机器学习的世界中，并非所有算法都需要深度神经网络或千兆字节的数据才能发光发热。最古老、最简单且最强大的范式之一植根于*贝叶斯决策理论*，这是一种统计分类方法，支撑着**贝叶斯分类器**、**高斯混合模型（GMMs）**和广受欢迎的**朴素贝叶斯**分类器等算法。

本笔记本通过真实世界的类比、必要的数学知识和可立即测试的Python代码，带您了解这些概念。

## 1. 贝叶斯决策理论：鱼类故事

想象您正在分类鱼类——有些是鲑鱼，有些是海鲈鱼。您的目标是建立一个规则，根据鱼的特征（如颜色或亮度）自动分类鱼类。

每条鱼属于一个类别：
- ω₁ = 海鲈鱼
- ω₂ = 鲑鱼

贝叶斯决策理论首先假设每个类别都有先验概率，$ p(\omega_1) $ 和 $ p(\omega_2) $，表示在没有额外信息的情况下遇到每种鱼类的可能性。如果 $ p(\omega_1) = p(\omega_2) = 0.5 $，则两个类别同样可能。在实践中，先验概率通常从数据或领域知识中估计。
- **先验概率**：在任何数据之前，鱼类来自每个类别的可能性
- **类别条件分布**：特征值（如"亮度"）在每个类别中的分布方式

当有额外信息可用时，如亮度测量值（$ x $）等特征，贝叶斯决策理论会整合这些证据来计算后验概率 $ p(\omega_j | x) $，这会根据特征更新每个类别的可能性。然后，决策规则将样本分配给具有最高后验概率的类别。

要计算**后验概率**：

$$
p(\omega_j | x) = \frac{p(x | \omega_j) p(\omega_j)}{p(x)}
$$

其中：
- $ p(\omega_j | x) $：给定特征 $ x $ 时类别 $\omega_j$ 的后验概率。
- $ p(x | \omega_j) $：类别条件概率密度，描述类别 $\omega_j$ 中特征 $ x $ 的分布。
- $ p(\omega_j) $：类别 $\omega_j$ 的先验概率。
- $ p(x) $：证据，计算为 $ p(x) = \sum_{j=1}^c p(x | \omega_j) p(\omega_j) $，其中 $ c $ 是类别数量。

对于两类问题，决策规则是：

- 如果 $ p(\omega_1 | x) > p(\omega_2 | x) $，选择 $\omega_1$。
- 否则选择 $\omega_2$。

由于 $ p(x) $ 是公共缩放因子，决策可以简化为比较 $ p(x | \omega_1) p(\omega_1) $ 和 $ p(x | \omega_2) p(\omega_2) $。
**贝叶斯决策规则**选择具有最高后验概率的类别。

### 示例：鱼类分类

假设我们有先验概率 $ p(\omega_1) = 2/3 $（海鲈鱼）和 $ p(\omega_2) = 1/3 $（鲑鱼）。给定亮度测量值 $ x $，我们使用类别条件密度函数 $ p(x | \omega_1) $ 和 $ p(x | \omega_2) $ 计算后验概率，这些函数描述了每种鱼类类型的亮度分布。

## 推广到多个特征和类别

对于实际应用，我们经常处理多个特征（形成特征向量 $\mathbf{x}$）和多个类别（$\omega_1, \omega_2, \ldots, \omega_c$）。后验概率变为：

$$ p(\omega_j | \mathbf{x}) = \frac{p(\mathbf{x} | \omega_j) p(\omega_j)}{p(\mathbf{x})} $$

其中 $ p(\mathbf{x}) = \sum_{j=1}^c p(\mathbf{x} | \omega_j) p(\omega_j) $。分类器将 $\mathbf{x}$ 分配给最大化判别函数的类别 $\omega_i$：

$$ g_i(\mathbf{x}) = p(\omega_i | \mathbf{x}) $$

替代判别函数包括：

$$ g_i(\mathbf{x}) = p(\mathbf{x} | \omega_i) p(\omega_i) $$
$$ g_i(\mathbf{x}) = \log p(\mathbf{x} | \omega_i) + \log p(\omega_i) $$

这些对于分类是等价的，因为它们保持概率的顺序。

## 2. 判别函数和后验概率

在更复杂的设置中，鱼类可能由特征向量 **x** 描述（例如，颜色、长度、亮度）。贝叶斯分类器推广到多个类别并使用**判别函数**：

$$
g_i(x) = p(x|\omega_i) \cdot p(\omega_i)
$$

或者为了稳定性使用对数形式：
$$
g_i(x) = \log p(x|\omega_i) + \log p(\omega_i)
$$

## 3. 高斯假设和参数估计

```python

import numpy as np

def estimate_parameters(X):
    mu = X.mean(axis=0)
    sigma = np.cov(X, rowvar=False)
    return mu, sigma

```

## 4. 高斯混合模型（GMMs）

GMM将数据建模为**高斯混合**：

$$
p(x) = \sum_{i=1}^{m} \alpha_i \cdot \mathcal{N}(x | \mu_i, \Sigma_i)
$$

我们使用EM算法来估计参数：
- **E步骤**：计算责任
- **M步骤**：更新 $ \mu_i, \Sigma_i, \alpha_i $

### 4.1 生成假数据

```python
import numpy as np
import matplotlib.pyplot as plt

# 为了可重现性
np.random.seed(0)

# 为二分类生成两个高斯聚类
n_train = 120
n_test = 30
n_features = 2

# 类别0：以(0, 0)为中心
X0 = np.random.randn(n_train // 2, n_features) + np.array([0, 0])
y0 = np.zeros(n_train // 2)

# 类别1：以(3, 3)为中心
X1 = np.random.randn(n_train // 2, n_features) + np.array([3, 3])
y1 = np.ones(n_train // 2)

# 合并为训练集
X_train = np.vstack([X0, X1])
y_train = np.hstack([y0, y1])

# 打乱训练集
idx = np.random.permutation(n_train)
X_train = X_train[idx]
y_train = y_train[idx]

# 测试集：类似分布
X0_test = np.random.randn(n_test // 2, n_features) + np.array([0, 0])
X1_test = np.random.randn(n_test // 2, n_features) + np.array([3, 3])
X_test = np.vstack([X0_test, X1_test])
# 绘制训练数据
plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='类别0（训练）', alpha=0.7)
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='类别1（训练）', alpha=0.7)

# 绘制测试数据
plt.scatter(X_test[:, 0], X_test[:, 1], c='k', marker='x', label='测试数据', alpha=0.8)

plt.title('假数据：两个高斯聚类')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.legend()
plt.grid(True)
plt.show()
```

```python

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(X_train)

predicted_probs = gmm.predict_proba(X_test)
print(predicted_probs[:10])
```

### 结果解释
`GaussianMixture`中的`predict_proba`返回X_test中每个样本属于混合中每个高斯组件的预测概率（**责任**）。

- predicted_probs中的每一行对应于X_test中的一个样本。
- 每一列对应于一个组件（这里，2列对应2个组件）。
- 值是概率（在0和1之间），每行总和为1。

## 5. 朴素贝叶斯分类器

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# 高斯朴素贝叶斯示例
gnb = GaussianNB()
gnb.fit(X_train, y_train)
preds = gnb.predict(X_test)
print(preds)
```

## 5.1 经典朴素贝叶斯示例：约翰应该打高尔夫吗？

"打高尔夫"数据集是说明朴素贝叶斯分类的经典示例。目标是根据天气条件预测约翰是否会打高尔夫。

**特征：**
- 天气：晴天、多云、雨天
- 温度：热、温和、凉爽
- 湿度：高、正常
- 风：弱、强

**目标：**
- 打高尔夫：是、否

让我们看看朴素贝叶斯如何用于这个问题。

```python
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

# 经典打高尔夫数据集
data = {
    '天气':    ['晴天', '晴天', '多云', '雨天', '雨天', '雨天', '多云', '晴天', '晴天', '雨天', '晴天', '多云', '多云', '雨天'],
    '温度':['热',   '热',   '热',      '温和','凉爽', '凉爽',  '凉爽',     '温和',  '凉爽',  '温和', '温和',  '温和',     '热',      '温和'],
    '湿度':   ['高',  '高',  '高',     '高','正常','正常','正常',  '高',  '正常','正常','正常','高',     '正常',   '高'],
    '风':       ['弱',  '强','弱',     '弱','弱',  '强','强',  '弱',  '弱',  '弱', '强','强',   '弱',     '强'],
    '打高尔夫':   ['否',    '否',    '是',      '是', '是',  '否',    '是',      '否',    '是',   '是', '是',   '是',      '是',      '否']
}

df = pd.DataFrame(data)

# 编码分类特征
le_outlook = LabelEncoder()
le_temp = LabelEncoder()
le_humidity = LabelEncoder()
le_wind = LabelEncoder()
le_play = LabelEncoder()

df['天气_编码'] = le_outlook.fit_transform(df['天气'])
df['温度_编码'] = le_temp.fit_transform(df['温度'])
df['湿度_编码'] = le_humidity.fit_transform(df['湿度'])
df['风_编码'] = le_wind.fit_transform(df['风'])
df['打高尔夫_编码'] = le_play.fit_transform(df['打高尔夫'])

X = df[['天气_编码', '温度_编码', '湿度_编码', '风_编码']]
y = df['打高尔夫_编码']

# 训练朴素贝叶斯分类器
model = CategoricalNB()
model.fit(X, y)

# 预测新的一天：天气=晴天，温度=凉爽，湿度=高，风=强
sample = [[le_outlook.transform(['晴天'])[0],
           le_temp.transform(['凉爽'])[0],
           le_humidity.transform(['高'])[0],
           le_wind.transform(['强'])[0]]]

pred = model.predict(sample)
print("预测结果 [晴天, 凉爽, 高, 强]:", le_play.inverse_transform(pred)[0])
```

**解释：**  
模型根据天气条件预测约翰是否会打高尔夫。您可以更改样本输入来尝试不同的天气组合。

## 5.2 手动解决"打高尔夫"示例

为了理解朴素贝叶斯的工作原理，让我们手动计算给定天气条件下约翰会打高尔夫的概率：  
**天气=晴天，温度=凉爽，湿度=高，风=强**

### 步骤1：计算先验概率

计算打高尔夫列中"是"和"否"的数量：

- $P(\text{是}) = \frac{\text{是的数量}}{\text{总数}} = \frac{9}{14}$
- $P(\text{否}) = \frac{\text{否的数量}}{\text{总数}} = \frac{5}{14}$

### 步骤2：计算似然

对于每个特征，计算给定打高尔夫=是和打高尔夫=否时特征值的概率。

#### 示例：$P(\text{天气=晴天}|\text{是})$

- 打高尔夫=是时"晴天"的天数：2
- 总"是"：9  
$\Rightarrow P(\text{晴天}|\text{是}) = \frac{2}{9}$

- 打高尔夫=否时"晴天"的天数：3
- 总"否"：5  
$\Rightarrow P(\text{晴天}|\text{否}) = \frac{3}{5}$

对所有特征重复：

| 特征         | 值   | $P(\cdot|\text{是})$ | $P(\cdot|\text{否})$ |
|-----------------|---------|----------------------|----------------------|
| 天气         | 晴天   | 2/9                  | 3/5                  |
| 温度     | 凉爽    | 3/9                  | 1/5                  |
| 湿度        | 高    | 3/9                  | 4/5                  |
| 风            | 强  | 3/9                  | 3/5                  |

### 步骤3：应用朴素贝叶斯公式

计算未归一化概率：

- 对于"是"：
  $$
  P(\text{是}) \times P(\text{晴天}|\text{是}) \times P(\text{凉爽}|\text{是}) \times P(\text{高}|\text{是}) \times P(\text{强}|\text{是}) \\
  = \frac{9}{14} \times \frac{2}{9} \times \frac{3}{9} \times \frac{3}{9} \times \frac{3}{9}
  $$

- 对于"否"：
  $$
  P(\text{否}) \times P(\text{晴天}|\text{否}) \times P(\text{凉爽}|\text{否}) \times P(\text{高}|\text{否}) \times P(\text{强}|\text{否}) \\
  = \frac{5}{14} \times \frac{3}{5} \times \frac{1}{5} \times \frac{4}{5} \times \frac{3}{5}
  $$

### 步骤4：计算和比较

- 对于"是"：
  $$
  \frac{9}{14} \times \frac{2}{9} \times \frac{3}{9} \times \frac{3}{9} \times \frac{3}{9} = \frac{2 \times 3 \times 3 \times 3}{14 \times 9 \times 9 \times 9} = \frac{54}{10206} \approx 0.0053
  $$

- 对于"否"：
  $$
  \frac{5}{14} \times \frac{3}{5} \times \frac{1}{5} \times \frac{4}{5} \times \frac{3}{5} = \frac{3 \times 1 \times 4 \times 3}{14 \times 5 \times 5 \times 5} = \frac{36}{8750} \approx 0.0041
  $$

### 步骤5：归一化（可选）

由于我们只需要比较，值较高的类别就是预测结果。

**结论：**  
由于 $0.0053 > 0.0041$，预测**是**（约翰会打高尔夫）。

---

**注意：**  
如果任何概率为零，可以使用拉普拉斯平滑来避免乘以零。

## 6. 可视化3D GMM聚类

```python

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成合成3D数据
np.random.seed(42)
mean1 = [0, 0, 0]
cov1 = np.diag([1, 2, 3])
data1 = np.random.multivariate_normal(mean1, cov1, 100)

mean2 = [5, 5, 5]
cov2 = np.diag([2, 1, 2])
data2 = np.random.multivariate_normal(mean2, cov2, 100)

X = np.vstack((data1, data2))

# 拟合GMM
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(X)
labels = gmm.predict(X)

# 3D绘图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', s=40)
ax.set_title('3D高斯混合模型聚类')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

``` 

![png](bayesian_files/bayesian_22_0.png)