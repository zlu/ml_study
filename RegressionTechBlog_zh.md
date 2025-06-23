# 理解机器学习中的回归：全面指南

## 引言
回归是监督式机器学习的基石，使我们能够根据输入数据预测数值结果，例如房价或股票价值。与处理类别标签的分类不同，回归关注的是连续变量。本文将探讨回归的基本原理，从简单线性模型到高级技术如岭回归和套索回归，并介绍如何评估和验证这些模型。文末附有测验及答案，帮助你检验理解。

## 简单与多元线性回归
线性回归通过一条直线或超平面来建模因变量（目标）与一个或多个自变量（预测因子）之间的关系。

### 简单线性回归
想象一下根据工作年限预测某人的薪资。简单线性回归使用单一预测因子，通过如下公式估计目标：
$$ y = \theta_0 + \theta_1 x + \varepsilon $$
其中，$ y $为目标（薪资），$ x $为预测因子（经验），$ \theta_0 $为截距，$ \theta_1 $为斜率，$ \varepsilon $为捕捉未建模因素的误差项。目标是找到最适合数据的$ \theta_0 $和$ \theta_1 $。

### 多元线性回归
当多个因素影响目标时，如经验、学历和地区共同影响薪资，我们使用多元线性回归：
$$ y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_m x_m + \varepsilon $$
该模型扩展到$ m $个预测因子（$ x_1, x_2, \ldots, x_m $），系数为$ \theta_1, \theta_2, \ldots, \theta_m $。挑战在于利用训练数据准确估计这些系数。

## 普通最小二乘（OLS）估计
OLS是拟合线性回归模型的标准方法，通过最小化预测值与实际值之间的平方误差和。对于$ N $个数据点$ \{(x_i, y_i)\} $，模型预测：
$$ y_i = \theta_0 + \theta_1 x_{i1} + \cdots + \theta_m x_{im} + \varepsilon_i $$
矩阵形式为：
$$ \mathbf{y} = \mathbf{\Phi} \boldsymbol{\theta} + \boldsymbol{\varepsilon} $$
其中$ \mathbf{y} $为目标向量，$ \mathbf{\Phi} $为设计矩阵（含截距的全1列），$ \boldsymbol{\theta} $为参数向量，$ \boldsymbol{\varepsilon} $为误差向量。损失函数为：
$$ J(\boldsymbol{\theta}) = \frac{1}{2} \sum_{i=1}^N \varepsilon_i^2 = \frac{1}{2} (\mathbf{y} - \mathbf{\Phi} \boldsymbol{\theta})^T (\mathbf{y} - \mathbf{\Phi} \boldsymbol{\theta}) $$
最小化$ J $得到OLS估计：
$$ \widehat{\boldsymbol{\theta}} = (\mathbf{\Phi}^T \mathbf{\Phi})^{-1} \mathbf{\Phi}^T \mathbf{y} $$
当$ \mathbf{\Phi}^T \mathbf{\Phi} $可逆时该解有效，但在高维数据或自变量高度相关时会出现问题。

## 岭回归
当预测因子数量多于观测数或预测因子高度相关时，OLS可能表现不佳，导致模型不稳定或过拟合。岭回归通过在损失函数中加入L2惩罚项来解决这一问题：
$$ J_{\text{ridge}}(\boldsymbol{\theta}) = \frac{1}{2} (\mathbf{y} - \mathbf{\Phi} \boldsymbol{\theta})^T (\mathbf{y} - \mathbf{\Phi} \boldsymbol{\theta}) + \lambda \boldsymbol{\theta}^T \boldsymbol{\theta} $$
惩罚项$ \lambda \boldsymbol{\theta}^T \boldsymbol{\theta} $将系数收缩至零，提高了模型的稳定性和泛化能力。解为：
$$ \widehat{\boldsymbol{\theta}}_{\text{ridge}} = (\mathbf{\Phi}^T \mathbf{\Phi} + \lambda \mathbf{I})^{-1} \mathbf{\Phi}^T \mathbf{y} $$
其中$ \lambda $控制正则化强度，$ \lambda $越大，系数越小，但可能导致欠拟合。

## 套索回归（Lasso）
套索回归采用L1惩罚项，促进稀疏性，将部分系数精确地压缩为零：
$$ J_{\text{lasso}}(\boldsymbol{\theta}) = \frac{1}{2} (\mathbf{y} - \mathbf{\Phi} \boldsymbol{\theta})^T (\mathbf{y} - \mathbf{\Phi} \boldsymbol{\theta}) + \lambda \sum_{j=1}^m |\theta_j| $$
这使得套索回归非常适合特征选择，能够识别最重要的预测因子。与岭回归不同，套索回归没有解析解，需要数值优化方法如ISTA或FISTA。$ \lambda $的选择至关重要：值越大，稀疏性越强，值越小，保留的预测因子越多。

## 模型评估指标
评估回归模型需衡量预测与实际值的匹配程度。常用指标包括：

- **均方误差（MSE）**：预测值与实际值之差的平方的平均值：
  $$ \text{MSE} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2 $$
  由于平方运算，对大误差惩罚更重。
- **均方根误差（RMSE）**：MSE的平方根，单位与目标一致：
  $$ \text{RMSE} = \sqrt{\text{MSE}} $$
  由于易于解释，应用广泛。
- **平均绝对误差（MAE）**：绝对误差的平均值，对异常值不敏感：
  $$ \text{MAE} = \frac{1}{n} \sum_{i=1}^n |\hat{y}_i - y_i| $$
- **决定系数（R平方，$ R^2 $）**：衡量模型解释的方差比例：
  $$ R^2 = 1 - \frac{\sum_{i=1}^n (\hat{y}_i - y_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2} $$
  取值范围为0到1，越高表示拟合越好。但$ R^2 $会随着预测因子增多而增加，即使这些变量无关。
- **调整后的R平方**：对$ R^2 $进行调整，考虑预测因子数量：
  $$ \text{Adjusted } R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - m - 1} $$
  它惩罚不必要的变量，有助于模型选择。

## 模型验证
线性回归假设：
1. **线性关系**：预测因子与目标之间为线性关系，可通过散点图验证。
2. **残差正态性**：误差应服从正态分布，可用直方图或Q-Q图检查。
3. **残差均值为零**：平均误差应接近零。
4. **多元正态性**：预测因子应服从多元正态分布，可用Q-Q图评估。
5. **同方差性**：残差在不同预测因子取值下方差应恒定。异方差性（方差变化）可通过残差与预测值的散点图检测，常表现为漏斗形。
若假设被违反，可能需对数据进行变换或采用其他模型。

## 非线性回归
当关系不是线性时，非线性回归可建模任意函数：
$$ y = f(x, \boldsymbol{\theta}) + \varepsilon $$
例如多项式模型（$ y = \theta_0 + \theta_1 x + \theta_2 x^2 + \varepsilon $）或有理模型。尽管功能强大，但由于神经网络能够有效处理复杂关系，非线性回归如今应用较少。

## 测验：回归知识自测
以下问题帮助你巩固理解。答案见下一节。

### 概念题
1. 简单线性回归与多元线性回归的关键区别是什么？
2. 当预测因子数量超过观测数时，为什么OLS估计会失效？
3. Lasso回归中的L1惩罚与Ridge回归中的L2惩罚在模型结果上有何不同？
4. 为什么模型选择时更推荐使用调整后的$ R^2 $而不是$ R^2 $？
5. 回归模型中的异方差性说明了什么，如何检测？

### 编程题
请使用`scikit-learn`、`numpy`和`matplotlib`，基于[Auto MPG数据集](https://archive.ics.uci.edu/ml/datasets/Auto+MPG)回答下列问题。
1. 用`displacement`预测`mpg`，拟合简单线性回归模型，报告截距和斜率。
2. 用`displacement`、`horsepower`和`weight`预测`mpg`，拟合多元线性回归模型，并计算训练集RMSE。
3. 对上述三个预测因子应用岭回归，$ \lambda = 1.0 $，并与OLS系数对比。
4. 用$ \lambda = 0.1 $对同样的预测因子做Lasso回归，指出哪些系数被压缩为零。
5. 绘制多元线性回归模型的残差与预测`mpg`的散点图，检查是否存在异方差性。

## 测验答案

### 概念题
1. **答案**：简单线性回归只用一个预测因子建模目标，多元线性回归用多个预测因子，可建模更复杂的关系。
2. **答案**：当预测因子多于观测数时，$ \mathbf{\Phi}^T \mathbf{\Phi} $变为奇异矩阵（不可逆），导致系数无唯一解。
3. **答案**：Lasso的L1惩罚促使部分系数为零，实现特征选择；Ridge的L2惩罚则将所有系数收缩至零但不会消除，提升模型稳定性。
4. **答案**：调整后的$ R^2 $会惩罚不必要的预测因子，防止过拟合，而$ R^2 $会随变量增多而增加。
5. **答案**：异方差性说明残差方差随预测因子变化，违反线性回归假设。可通过残差与预测值的散点图检测，常见漏斗形。

### 编程题
以下为基于Auto MPG数据集的Python示例代码。运行前请确保数据已清洗（如处理`horsepower`中的缺失值）。

1. **用`displacement`预测`mpg`，拟合简单线性回归模型，报告截距和斜率。**
   ```python
   import pandas as pd
   from sklearn.linear_model import LinearRegression

   # 加载并清洗数据集
   df = pd.read_csv('auto-mpg.csv')
   df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

   # 简单线性回归
   X = df[['displacement']]
   y = df['mpg']
   model = LinearRegression().fit(X, y)
   print(f"截距: {model.intercept_:.4f}, 斜率: {model.coef_[0]:.4f}")
   ```
   **预期输出**：截距和斜率取决于数据，通常约为35.0和-0.06。

2. **用`displacement`、`horsepower`和`weight`预测`mpg`，拟合多元线性回归模型，并计算训练集RMSE。**
   ```python
   import numpy as np
   from sklearn.metrics import mean_squared_error

   # 多元线性回归
   X = df[['displacement', 'horsepower', 'weight']]
   y = df['mpg']
   model = LinearRegression().fit(X, y)
   y_pred = model.predict(X)
   rmse = np.sqrt(mean_squared_error(y, y_pred))
   print(f"RMSE: {rmse:.4f}")
   ```
   **预期输出**：RMSE通常在4.0–5.0之间，具体取决于数据预处理。

3. **对上述三个预测因子应用岭回归，$ \lambda = 1.0 $，并与OLS系数对比。**
   ```python
   from sklearn.linear_model import Ridge

   # 岭回归
   ridge = Ridge(alpha=1.0).fit(X, y)
   print("OLS系数:", model.coef_)
   print("Ridge系数:", ridge.coef_)
   ```
   **预期输出**：由于L2惩罚，Ridge系数略小于OLS系数。

4. **用$ \lambda = 0.1 $对同样的预测因子做Lasso回归，指出哪些系数被压缩为零。**
   ```python
   from sklearn.linear_model import Lasso

   # Lasso回归
   lasso = Lasso(alpha=0.1).fit(X, y)
   print("Lasso系数:", lasso.coef_)
   ```
   **预期输出**：部分系数（如`horsepower`）可能为零，表明Lasso进行了特征选择。

5. **绘制多元线性回归模型的残差与预测`mpg`的散点图，检查是否存在异方差性。**
   ```python
   import matplotlib.pyplot as plt

   # 残差图
   residuals = y - y_pred
   plt.scatter(y_pred, residuals)
   plt.axhline(0, color='red', linestyle='--')
   plt.xlabel('预测MPG')
   plt.ylabel('残差')
   plt.title('残差与预测MPG散点图')
   plt.savefig('residual_plot.png')
   plt.close()
   ```
   **预期输出**：若散点无明显模式（如漏斗形），则为同方差性；若呈漏斗形，则为异方差性。

## 结论
回归是机器学习中用途广泛的工具，从简单线性模型到正则化技术如岭回归和套索回归。理解其假设、评估指标和验证方法对于构建稳健模型至关重要。上述测验有助于巩固知识，`scikit-learn`等工具让这些概念易于实践。多用Auto MPG等数据集练习，提升你的技能！ 