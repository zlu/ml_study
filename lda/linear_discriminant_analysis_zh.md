# 线性判别分析（LDA）全面指南

线性判别分析（LDA）是一种强大的统计技术，广泛应用于机器学习和模式识别中，用于根据特征将对象分类到不同的组中。它在二分类或多分类问题中尤为有效，其目标是在特征空间中找到一个最佳的线性边界（二维为直线，三维为平面，更高维为超平面）以最大程度地区分各类别。 

## 什么是线性判别分析？

LDA是一种有监督学习算法，旨在将数据点投影到低维空间，同时最大化类别之间的分离度。它假设每个类别的数据服从正态分布且具有相似的协方差结构。其主要目标是找到特征的线性组合，使得两个或多个类别之间的区分度最大，从而便于对新数据点进行分类。与主成分分析（PCA）只关注方差最大化且不考虑类别标签不同，LDA明确利用类别信息来优化分离效果。

LDA广泛应用于图像识别、语音处理和生物信息学等领域。例如，它可以将人脸图像分类到不同个体，或将基因表达数据分类为不同疾病类型。其简洁性和高效性使其成为机器学习工具箱中的常用方法，尤其适用于特征数量多但类别数量有限的场景。

## LDA的数学基础

### 线性判别函数

LDA的核心是线性判别函数，其表达式为：

$$ g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + w_0 $$

其中，$\mathbf{x}$为特征向量，$\mathbf{w}$为决定分界超平面方向的权重向量，$w_0$为偏置项（或阈值），用于调整超平面的位置。对于二分类问题，判别规则如下：

- 如果 $ g(\mathbf{x}) > 0 $，判为类别$\omega_1$。
- 如果 $ g(\mathbf{x}) < 0 $，判为类别$\omega_2$。
- 如果 $ g(\mathbf{x}) = 0 $，则点位于决策边界上。

由 $ g(\mathbf{x}) = 0 $ 定义的超平面将特征空间分为$R_1$（$\omega_1$）和$R_2$（$\omega_2$）两部分。权重向量$\mathbf{w}$垂直于超平面，即对于超平面上的任意两点$\mathbf{x}_1$和$\mathbf{x}_2$，有$\mathbf{w}^T (\mathbf{x}_1 - \mathbf{x}_2) = 0$。这保证了$\mathbf{w}$定义了边界的法向方向。

### 到超平面的距离计算

LDA还提供了从任意点$\mathbf{x}$到超平面的距离计算方法。设$\mathbf{x}_p$为超平面上的一点，即$ g(\mathbf{x}_p) = 0 $。任意点$\mathbf{x}$可表示为：

$$ \mathbf{x} = \mathbf{x}_p + r \frac{\mathbf{w}}{\|\mathbf{w}\|} $$

其中$r$为$\mathbf{x}$到超平面的有符号距离，$\frac{\mathbf{w}}{\|\mathbf{w}\|}$为单位法向量。代入判别函数得：

$$ g(\mathbf{x}) = \mathbf{w}^T ( \mathbf{x}_p + r \frac{\mathbf{w}}{\|\mathbf{w}\|} ) + w_0 = \mathbf{w}^T \mathbf{x}_p + r \frac{\mathbf{w}^T \mathbf{w}}{\|\mathbf{w}\|} + w_0 = r \|\mathbf{w}\| $$

由于 $ g(\mathbf{x}_p) = 0 $，可得：

$$ r = \frac{g(\mathbf{x})}{\|\mathbf{w}\|} $$

当$\mathbf{x}$在超平面正侧（$R_1$）时距离为正，反之为负。原点到超平面的距离为：

$$ r = \frac{w_0}{\|\mathbf{w}\|} $$

这表明$w_0$控制超平面相对于原点的位置：
- $ w_0 > 0 $时，原点在正侧；
- $ w_0 < 0 $时，原点在负侧；
- $ w_0 = 0 $时，超平面经过原点。

### 优化类别分离

LDA的目标是找到最大化类别分离度的权重向量$\mathbf{w}$。对于二分类问题，假设每类样本均值为$\mathbf{m}_1$和$\mathbf{m}_2$。样本$\mathbf{x}$在$\mathbf{w}$方向上的投影为$\mathbf{w}^T \mathbf{x}$，两类投影均值之差为：

$$ |\tilde{m}_1 - \tilde{m}_2| = |\mathbf{w}^T (\mathbf{m}_1 - \mathbf{m}_2)| $$

为保证良好分离，该差值应远大于各类投影数据的离散度。第$i$类的类内散度为：

$$ \tilde{s}_i^2 = \sum_{\mathbf{x} \in D_i} ( \mathbf{w}^T \mathbf{x} - \mathbf{w}^T \mathbf{m}_i )^2 $$

总类内散度为：

$$ \tilde{s}^2 = \tilde{s}_1^2 + \tilde{s}_2^2 $$

LDA采用Fisher准则，最大化：

$$ J(\mathbf{w}) = \frac{(\tilde{m}_1 - \tilde{m}_2)^2}{\tilde{s}_1^2 + \tilde{s}_2^2} $$

将其用散度矩阵表示：
- 第$i$类的类内散度矩阵：

$$ \mathbf{S}_i = \sum_{\mathbf{x} \in D_i} (\mathbf{x} - \mathbf{m}_i)(\mathbf{x} - \mathbf{m}_i)^T $$

- 总类内散度矩阵：

$$ \mathbf{S}_W = \mathbf{S}_1 + \mathbf{S}_2 $$

- 类间散度矩阵：

$$ \mathbf{S}_B = (\mathbf{m}_1 - \mathbf{m}_2)(\mathbf{m}_1 - \mathbf{m}_2)^T $$

投影后的散度可写为：

$$ \tilde{s}_i^2 = \mathbf{w}^T \mathbf{S}_i \mathbf{w} $$

$$ \tilde{s}^2 = \mathbf{w}^T \mathbf{S}_W \mathbf{w} $$

$$ (\tilde{m}_1 - \tilde{m}_2)^2 = \mathbf{w}^T \mathbf{S}_B \mathbf{w} $$

因此Fisher准则为：

$$ J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}} $$

这是广义Rayleigh商，最优$\mathbf{w}$可通过广义特征值问题求解：

$$ \mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w} $$

若$\mathbf{S}_W$非奇异，则化为：

$$ \mathbf{S}_W^{-1} \mathbf{S}_B \mathbf{w} = \lambda \mathbf{w} $$

对于二分类问题，有更简便的解：

$$ \mathbf{w} = \mathbf{S}_W^{-1} (\mathbf{m}_1 - \mathbf{m}_2) $$

若$\mathbf{S}_W$奇异，可用正则化：

$$ \mathbf{w} = (\mathbf{S}_W + \beta \mathbf{I})^{-1} (\mathbf{m}_1 - \mathbf{m}_2) $$

其中$\beta$为小正数（如0.001），$\mathbf{I}$为单位矩阵。

偏置项$w_0$通常设为使超平面位于两类均值中点：

$$ w_0 = -\frac{\mathbf{w}^T (\mathbf{m}_1 + \mathbf{m}_2)}{2} $$

若两类散度不同，$w_0$可进一步调整以优化性能。

## 多分类扩展

对于$c$类问题，LDA可推广为多判别分析（MDA），将数据从$d$维空间投影到$c-1$维空间（假设$d \geq c$）。类内散度矩阵为：

$$ \mathbf{S}_W = \sum_{i=1}^c \mathbf{S}_i, \quad \mathbf{S}_i = \sum_{\mathbf{x} \in D_i} (\mathbf{x} - \mathbf{m}_i)(\mathbf{x} - \mathbf{m}_i)^T $$

总均值向量：

$$ \mathbf{m} = \frac{1}{n} \sum_{i=1}^c n_i \mathbf{m}_i $$

类间散度矩阵：

$$ \mathbf{S}_B = \sum_{i=1}^c n_i (\mathbf{m}_i - \mathbf{m})(\mathbf{m}_i - \mathbf{m})^T $$

投影由$c-1$个判别函数定义：

$$ g_i(\mathbf{x}) = \mathbf{w}_i^T \mathbf{x}, \quad i = 1, 2, \ldots, c-1 $$

这些函数组合成投影矩阵$\mathbf{W}$，$\mathbf{g}(\mathbf{x}) = \mathbf{W}^T \mathbf{x}$。目标是最大化类间与类内散度比值，通常用行列式衡量：

$$ J(\mathbf{W}) = \frac{|\mathbf{W}^T \mathbf{S}_B \mathbf{W}|}{|\mathbf{W}^T \mathbf{S}_W \mathbf{W}|} $$

$\mathbf{W}$的列为对应最大特征值的广义特征向量：

$$ \mathbf{S}_B \mathbf{w}_i = \lambda_i \mathbf{S}_W \mathbf{w}_i $$

若$\mathbf{S}_W$非奇异，则：

$$ \mathbf{S}_W^{-1} \mathbf{S}_B \mathbf{w}_i = \lambda_i \mathbf{w}_i $$

对于$c$类问题，最多有$c-1$个非零特征向量，对应所需判别函数的数量。

## 实践实现

### 二分类示例

考虑一个二分类问题，均值向量为：

$$ \mathbf{m}_1 = \begin{bmatrix} 0.1083 \\ -0.0653 \end{bmatrix}, \quad \mathbf{m}_2 = \begin{bmatrix} 1.8945 \\ 2.9026 \end{bmatrix} $$

类内散度矩阵：

$$ \mathbf{S}_W = \begin{bmatrix} 228.9365 & 10.9883 \\ 10.9883 & 189.216 \end{bmatrix} $$

直接法求解：

$$ \mathbf{w} = \mathbf{S}_W^{-1} (\mathbf{m}_1 - \mathbf{m}_2) = \begin{bmatrix} -0.0071 \\ -0.0153 \end{bmatrix} $$

$$ w_0 = 0.0288 $$

判别函数为：

$$ g(\mathbf{x}) = -0.0071 x_1 - 0.0153 x_2 + 0.0288 $$

也可通过特征值问题$\mathbf{S}_W^{-1} \mathbf{S}_B \mathbf{w} = \lambda \mathbf{w}$求解，结果方向一致，验证了两种方法的稳健性。

### 三分类示例

对于三分类问题，设有如下散度矩阵：

$$ \mathbf{S}_1 = \begin{bmatrix} 12640.28 & 1134.9 \\ 1134.9 & 10031.8 \end{bmatrix}, \quad \mathbf{S}_2 = \begin{bmatrix} 10253.37 & -36.08 \\ -36.08 & 8880.35 \end{bmatrix}, \quad \mathbf{S}_3 = \begin{bmatrix} 7250.18 & 47.82 \\ 47.82 & 9523.04 \end{bmatrix} $$

$$ \mathbf{S}_W = \begin{bmatrix} 301.4383 & 11.4665 \\ 11.4665 & 284.352 \end{bmatrix}, \quad \mathbf{S}_B = \begin{bmatrix} 439.7196 & -56.5992 \\ -56.5992 & 809.7615 \end{bmatrix} $$

解$\mathbf{S}_B \mathbf{W} = \lambda \mathbf{S}_W \mathbf{W}$得特征向量：

$$ \mathbf{W} = \begin{bmatrix} -0.0564 & -0.012 \\ -0.0101 & 0.0585 \end{bmatrix} $$

特征值$\Lambda = \begin{bmatrix} 1.4155 & 0 \\ 0 & 2.9127 \end{bmatrix}$。判别函数为：

$$ g_1(\mathbf{x}) = \begin{bmatrix} -0.012 \\ 0.0585 \end{bmatrix}^T \mathbf{x} - 0.0719 $$

$$ g_2(\mathbf{x}) = \begin{bmatrix} -0.0564 \\ -0.0101 \end{bmatrix}^T \mathbf{x} + 0.0708 $$

决策规则基于$g_1(\mathbf{x})$和$g_2(\mathbf{x})$的符号，确保三类的有效分离。

## 应用与局限

LDA广泛应用于人脸识别（如Fisherfaces）、医学诊断、文本分类等领域。其优点包括简洁、高效，且在类别数量较少、特征维度较高时表现良好。但LDA假设各类别服从高斯分布且协方差矩阵相等，这在实际数据中未必成立。此外，LDA难以处理非线性可分数据，此时可采用支持向量机（SVM）核方法或神经网络等更复杂模型。

## 与其他方法的比较

与PCA相比，LDA更适合分类任务，因为它利用了类别标签。SVM可处理非线性边界，但计算量更大。逻辑回归也是一种选择，但其假设特征与类别概率之间为对数关系，而LDA假设为高斯分布。

## 总结

LDA是分类算法中的基石，提供了一种数学优雅且高效的线性分界方法。通过最大化类间与类内散度比值，确保了投影空间中的良好分离。无论是二分类还是多分类问题，只要数据基本满足其假设，LDA都是极具价值的工具。

本文涵盖了LDA的理论基础、数学推导和实践步骤，并辅以讲义中的示例。理解并应用LDA，能有效应对各类分类挑战。


## 练习题

### 说明
- 请清晰、简明地回答所有问题。
- 数学推导题请写出详细步骤。
- 编程题请附带注释和简要说明。
- 可将答案整理为PDF或Jupyter Notebook提交。

---

### 第一部分：概念题

1. **LDA与PCA的区别**  
   a. 简述线性判别分析（LDA）与主成分分析（PCA）的主要区别。  
   b. 在什么情况下你会优先选择LDA而不是PCA？请说明理由。

2. **LDA的基本假设**  
   列举并简要说明LDA的主要假设。如果这些假设在实际中不成立，可能会带来什么影响？

3. **判别函数**  
   给定线性判别函数 $ g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + w_0 $，请解释向量 $ \mathbf{w} $ 和标量 $ w_0 $ 的几何意义。

4. **散度矩阵**  
   a. 定义二分类问题下的类内散度矩阵 $ \mathbf{S}_W $ 和类间散度矩阵 $ \mathbf{S}_B $。  
   b. 为什么LDA要最大化 $ \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}} $ 这个比值？

---

### 第二部分：数学推导

5. **最优权重向量**  
   从Fisher判别准则出发，推导二分类情况下最优权重向量 $ \mathbf{w} $ 的公式，并说明如何得到  
   $ \mathbf{w} = \mathbf{S}_W^{-1} (\mathbf{m}_1 - \mathbf{m}_2) $

6. **到超平面的距离**  
   证明点 $ \mathbf{x} $ 到LDA判别边界的有符号距离为  
   $ r = \frac{g(\mathbf{x})}{\|\mathbf{w}\|} $  
   其中 $ g(\mathbf{x}) $ 为判别函数。

---

### 第三部分：实践应用

7. **手算示例（二分类）**  
   已知：
   - $ \mathbf{m}_1 = \begin{bmatrix} 0.1083 \\ -0.0653 \end{bmatrix} $
   - $ \mathbf{m}_2 = \begin{bmatrix} 1.8945 \\ 2.9026 \end{bmatrix} $
   - $ \mathbf{S}_W = \begin{bmatrix} 228.9365 & 10.9883 \\ 10.9883 & 189.216 \end{bmatrix} $

   a. 计算最优权重向量 $ \mathbf{w} $。  
   b. 计算使判别边界位于两类均值中点的偏置项 $ w_0 $。  
   c. 写出判别函数 $ g(\mathbf{x}) $ 的显式表达式。

8. **编程：LDA在真实数据集上的应用**  
   a. 加载一个合适的数据集（如Iris鸢尾花数据集，或任意包含两类及以上、多特征的数据集）。  
   b. 手动实现LDA（不要直接用scikit-learn的LDA主算法，但可用于对比）。  
   c. 将数据投影到LDA方向上并可视化。  
   d. 与scikit-learn的LDA分类准确率进行对比。

---

### 第四部分：思考题

9. **局限性与扩展**  
   a. 讨论LDA在实际数据应用中的一个局限性。  
   b. 针对该局限性，提出一种可能的扩展或替代方法。

---

**附加题（可选）：**  
- 对于三分类问题，说明判别函数的数量如何确定，以及投影空间的形态。

---

## 示例：LDA在基因表达数据疾病分类中的应用

线性判别分析（LDA）在生物信息学中被广泛用于根据基因表达谱将样本（如患者）分类为不同的疾病类型。下面是一个使用Python的实际示例，采用模拟的基因表达数据进行说明。在实际研究中，通常会使用真实的数据集，如白血病或乳腺癌的基因表达数据。

### 模拟示例：疾病类型分类

```python
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 模拟基因表达数据
np.random.seed(42)
n_samples = 100  # 样本数量（患者数）
n_genes = 50    # 基因数量（特征数）

# 模拟两种疾病类型（0 和 1）
X_class0 = np.random.normal(loc=0.0, scale=1.0, size=(n_samples//2, n_genes))
X_class1 = np.random.normal(loc=1.0, scale=1.0, size=(n_samples//2, n_genes))
X = np.vstack([X_class0, X_class1])
y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 拟合LDA模型
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# 预测与评估
y_pred = lda.predict(X_test)
print("准确率:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 将数据投影到LDA轴上进行可视化
X_lda = lda.transform(X)
plt.figure(figsize=(8, 4))
plt.hist(X_lda[y==0], bins=20, alpha=0.7, label='疾病类型 0')
plt.hist(X_lda[y==1], bins=20, alpha=0.7, label='疾病类型 1')
plt.xlabel('LDA投影')
plt.ylabel('样本数量')
plt.title('模拟基因表达数据的LDA投影')
plt.legend()
plt.show()
```

**说明：**
- 我们为两种疾病类型分别模拟了基因表达数据，每种类型有50个基因。
- LDA被训练用于寻找最佳区分两种疾病类型的基因线性组合。
- 输出的准确率和分类报告展示了模型的性能。
- 直方图可视化了LDA在投影空间中对两类的区分效果。

**注意：** 在实际分析中，通常会使用真实的基因表达数据集（如GEO或TCGA），这些数据集通常包含成千上万个基因，并需要更复杂的预处理（如归一化、特征选择等）。但整体流程与本示例类似。

---

## 练习题参考答案

### 第一部分：概念题

1. **LDA与PCA的区别**  
a. **区别：** LDA（线性判别分析）是一种有监督的降维方法，利用类别标签最大化类别可分性；而PCA（主成分分析）是一种无监督方法，只最大化方差，不考虑类别标签。  
b. **何时优先LDA：** 当任务是分类且有类别标签时，LDA更优，因为它能找到最佳区分类别的方向。PCA更适合探索性数据分析或无标签场景。

2. **LDA的基本假设**  
- 每个类别服从正态分布（高斯分布）。
- 所有类别具有相同的协方差矩阵（同方差性）。
- 各样本独立。
- 特征与类别标签线性相关。
  
若假设不成立，LDA可能效果不佳，甚至产生误导性结果，如类别分离度差或过拟合。

3. **判别函数**  
- $\mathbf{w}$为决策超平面的法向量，决定类别分界的方向。
- $w_0$为偏置项（截距），决定超平面相对于原点的位置。

4. **散度矩阵**  
a. 类内散度矩阵$\mathbf{S}_W$为各类别协方差矩阵之和，衡量类内样本的离散度。类间散度矩阵$\mathbf{S}_B$衡量类别均值之间的离散度。
  
b. 最大化$\frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$，可使类别投影均值尽量远离（分子），同时类内方差最小（分母），从而提升分类效果。

---

### 第二部分：数学推导

5. **最优权重向量**  
Fisher准则为$J(\mathbf{w}) = \frac{(\mathbf{w}^T(\mathbf{m}_1-\mathbf{m}_2))^2}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$。对$\mathbf{w}$求极值，得到广义特征值问题，解为$\mathbf{w} = \mathbf{S}_W^{-1}(\mathbf{m}_1-\mathbf{m}_2)$。

6. **到超平面的距离**  
$g(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + w_0$，点$\mathbf{x}$到超平面的有符号距离为$r = \frac{g(\mathbf{x})}{\|\mathbf{w}\|}$，因为$g(\mathbf{x})$为$\mathbf{x}$在$\mathbf{w}$方向上的投影，除以$\|\mathbf{w}\|$即为距离。

---

### 第三部分：实践应用

7. **手算示例（二分类）**  
已知：
- $\mathbf{m}_1 = \begin{bmatrix} 0.1083 \\ -0.0653 \end{bmatrix}$
- $\mathbf{m}_2 = \begin{bmatrix} 1.8945 \\ 2.9026 \end{bmatrix}$
- $\mathbf{S}_W = \begin{bmatrix} 228.9365 & 10.9883 \\ 10.9883 & 189.216 \end{bmatrix}$

a. $\mathbf{w} = \mathbf{S}_W^{-1}(\mathbf{m}_1-\mathbf{m}_2) = \begin{bmatrix} -0.0071 \\ -0.0153 \end{bmatrix}$

b. $w_0 = -\frac{\mathbf{w}^T(\mathbf{m}_1+\mathbf{m}_2)}{2} = 0.0288$

c. $g(\mathbf{x}) = -0.0071 x_1 - 0.0153 x_2 + 0.0288$

8. **编程：LDA在真实数据集上的应用**  
- a. 加载如Iris等数据集。
- b. 计算类别均值、类内/类间散度，按上述公式求$\mathbf{w}$，并将数据投影到$\mathbf{w}$上。
- c. 用matplotlib等可视化投影结果。
- d. 用sklearn的LDA对比分类准确率。

---

### 第四部分：思考题

9. **局限性与扩展**  
a. LDA假设各类协方差矩阵相等且线性可分，实际数据常不满足，导致效果下降。
  
b. 二次判别分析（QDA）可放宽协方差假设，或采用核LDA、SVM、神经网络等非线性方法。

---

**附加题（可选）：**  
- 三分类问题最多有$c-1=2$个判别函数，数据从$d$维空间投影到二维，每个轴对应一个判别函数，便于可视化和分类。

---
