# Understanding Regression in Machine Learning: A Comprehensive Guide

## Introduction
Regression is a cornerstone of supervised machine learning, enabling predictions of numerical outcomes, such as house prices or stock values, based on input data. Unlike classification, which deals with categorical labels, regression focuses on continuous variables. This blog explores the essentials of regression, from simple linear models to advanced techniques like ridge and lasso regression, and covers how to evaluate and validate these models. A quiz at the end tests your understanding, with answers provided.

## Simple and Multiple Linear Regression
Linear regression models the relationship between a dependent variable (the target) and one or more independent variables (predictors) using a straight line or hyperplane.

### Simple Linear Regression
Imagine predicting someone's salary based on their years of experience. Simple linear regression uses a single predictor to estimate the target with the equation:
$$ y = \theta_0 + \theta_1 x + \varepsilon $$
Here, $ y $ is the target (salary), $ x $ is the predictor (experience), $ \theta_0 $ is the intercept, $ \theta_1 $ is the slope, and $ \varepsilon $ is the error term capturing unmodeled factors. The goal is to find $ \theta_0 $ and $ \theta_1 $ that best fit the data.

### Multiple Linear Regression
When multiple factors influence the target, such as experience, education, and location affecting salary, we use multiple linear regression:
$$ y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_m x_m + \varepsilon $$
This extends the model to $ m $ predictors ($ x_1, x_2, \ldots, x_m $), with coefficients $ \theta_1, \theta_2, \ldots, \theta_m $. The challenge is to estimate these coefficients accurately using training data.

## Ordinary Least Squares (OLS) Estimation
OLS is the standard method for fitting linear regression models by minimizing the sum of squared errors between predicted and actual values. For $ N $ data points $ \{(x_i, y_i)\} $, the model predicts:
$$ y_i = \theta_0 + \theta_1 x_{i1} + \cdots + \theta_m x_{im} + \varepsilon_i $$
In matrix form, this becomes:
$$ \mathbf{y} = \mathbf{\Phi} \boldsymbol{\theta} + \boldsymbol{\varepsilon} $$
where $ \mathbf{y} $ is the vector of targets, $ \mathbf{\Phi} $ is the design matrix (with a column of ones for the intercept), $ \boldsymbol{\theta} $ is the parameter vector, and $ \boldsymbol{\varepsilon} $ is the error vector. The loss function is:
$$ J(\boldsymbol{\theta}) = \frac{1}{2} \sum_{i=1}^N \varepsilon_i^2 = \frac{1}{2} (\mathbf{y} - \mathbf{\Phi} \boldsymbol{\theta})^T (\mathbf{y} - \mathbf{\Phi} \boldsymbol{\theta}) $$
Minimizing $ J $ yields the OLS estimate:
$$ \widehat{\boldsymbol{\theta}} = (\mathbf{\Phi}^T \mathbf{\Phi})^{-1} \mathbf{\Phi}^T \mathbf{y} $$
This solution works well when $ \mathbf{\Phi}^T \mathbf{\Phi} $ is invertible, but issues arise with high-dimensional data or collinear predictors.

## Ridge Regression
OLS can struggle when predictors outnumber observations or when predictors are highly correlated, leading to unstable or overfitting models. Ridge regression addresses this by adding an L2 penalty to the loss function:
$$ J_{\text{ridge}}(\boldsymbol{\theta}) = \frac{1}{2} (\mathbf{y} - \mathbf{\Phi} \boldsymbol{\theta})^T (\mathbf{y} - \mathbf{\Phi} \boldsymbol{\theta}) + \lambda \boldsymbol{\theta}^T \boldsymbol{\theta} $$
The penalty term $ \lambda \boldsymbol{\theta}^T \boldsymbol{\theta} $ shrinks coefficients toward zero, improving stability and generalization. The solution is:
$$ \widehat{\boldsymbol{\theta}}_{\text{ridge}} = (\mathbf{\Phi}^T \mathbf{\Phi} + \lambda \mathbf{I})^{-1} \mathbf{\Phi}^T \mathbf{y} $$
Here, $ \lambda $ controls the strength of regularization; larger $ \lambda $ values reduce coefficient magnitudes but may underfit the data.

## Lasso Regression
Lasso regression uses an L1 penalty instead, promoting sparsity by driving some coefficients exactly to zero:
$$ J_{\text{lasso}}(\boldsymbol{\theta}) = \frac{1}{2} (\mathbf{y} - \mathbf{\Phi} \boldsymbol{\theta})^T (\mathbf{y} - \mathbf{\Phi} \boldsymbol{\theta}) + \lambda \sum_{j=1}^m |\theta_j| $$
This makes lasso ideal for feature selection, as it identifies the most important predictors. Unlike ridge, lasso lacks an analytical solution and requires numerical optimization techniques like ISTA or FISTA. The choice of $ \lambda $ is critical: larger values increase sparsity, while smaller ones retain more predictors.

## Model Evaluation Metrics
Evaluating regression models involves measuring how well predictions match actual values. Common metrics include:

- **Mean Squared Error (MSE)**: Averages squared differences between predictions and actuals:
  $$ \text{MSE} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2 $$
  It penalizes large errors heavily due to squaring.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, in the same units as the target:
  $$ \text{RMSE} = \sqrt{\text{MSE}} $$
  It’s widely used for its interpretability.
- **Mean Absolute Error (MAE)**: Averages absolute differences, less sensitive to outliers:
  $$ \text{MAE} = \frac{1}{n} \sum_{i=1}^n |\hat{y}_i - y_i| $$
- **R-squared ($ R^2 $)**: Measures the proportion of variance explained by the model:
  $$ R^2 = 1 - \frac{\sum_{i=1}^n (\hat{y}_i - y_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2} $$
  Values range from 0 to 1, with higher values indicating better fit. However, $ R^2 $ increases with more predictors, even if they’re irrelevant.
- **Adjusted R-squared**: Adjusts $ R^2 $ for the number of predictors:
  $$ \text{Adjusted } R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - m - 1} $$
  It penalizes unnecessary variables, aiding model selection.

## Model Validation
Linear regression assumes:
1. **Linearity**: The relationship between predictors and target is linear, verifiable via scatter plots.
2. **Normality of Residuals**: Errors should be normally distributed, checked with histograms or Q-Q plots.
3. **Zero Mean Residuals**: The average error should be near zero.
4. **Multivariate Normality**: Predictors should follow a multivariate normal distribution, assessed with Q-Q plots.
5. **Homoscedasticity**: Residuals should have constant variance across predictor values. Heteroscedasticity (varying variance) can be detected in residual vs. predicted value scatter plots, often showing a funnel shape.
Violations of these assumptions may require data transformations or alternative models.

## Nonlinear Regression
When relationships are not linear, nonlinear regression models arbitrary functions:
$$ y = f(x, \boldsymbol{\theta}) + \varepsilon $$
Examples include polynomial models ($ y = \theta_0 + \theta_1 x + \theta_2 x^2 + \varepsilon $) or rational models. While powerful, nonlinear regression is less common today due to the rise of neural networks, which handle complex relationships effectively.

## Quiz: Test Your Regression Knowledge
Below are questions to reinforce your understanding. Answers follow in the next section.

### Conceptual Questions
1. What is the key difference between simple and multiple linear regression?
2. Why does OLS estimation fail when the number of predictors exceeds the number of observations?
3. How does the L1 penalty in lasso regression differ from the L2 penalty in ridge regression in terms of model outcomes?
4. Why is adjusted $ R^2 $ preferred over $ R^2 $ for model selection?
5. What does heteroscedasticity indicate in a regression model, and how can it be detected?

### Coding Questions
Use Python with `scikit-learn`, `numpy`, and `matplotlib` to answer these questions based on the [Auto MPG dataset](https://archive.ics.uci.edu/ml/datasets/Auto+MPG).
1. Fit a simple linear regression model to predict `mpg` using `displacement`. Report the intercept and slope.
2. Fit a multiple linear regression model using `displacement`, `horsepower`, and `weight` to predict `mpg`. Calculate the RMSE on the training data.
3. Apply ridge regression to the same predictors as in question 2 with $ \lambda = 1.0 $. Compare the coefficients with those from OLS.
4. Use lasso regression with $ \lambda = 0.1 $ on the same predictors. Identify which coefficients are set to zero.
5. Create a scatter plot of residuals vs. predicted `mpg` from the multiple linear regression model to check for heteroscedasticity.

## Quiz Answers

### Conceptual Questions
1. **Answer**: Simple linear regression uses one predictor variable to model the target, while multiple linear regression uses several predictors, allowing for more complex relationships.
2. **Answer**: OLS fails when predictors exceed observations because $ \mathbf{\Phi}^T \mathbf{\Phi} $ becomes singular (non-invertible), leading to no unique solution for the coefficients.
3. **Answer**: The L1 penalty in lasso promotes sparsity by setting some coefficients to zero, selecting key predictors, whereas the L2 penalty in ridge shrinks all coefficients toward zero without eliminating any, improving stability.
4. **Answer**: Adjusted $ R^2 $ penalizes the addition of unnecessary predictors, preventing overfitting, unlike $ R^2 $, which always increases with more variables.
5. **Answer**: Heteroscedasticity indicates non-constant residual variance across predictor values, violating linear regression assumptions. It’s detected via scatter plots of residuals vs. predicted values, often showing a funnel shape.

### Coding Questions
Below are sample solutions using Python and the Auto MPG dataset. Ensure the dataset is cleaned (e.g., handle missing values in `horsepower`) before running the code.

1. **Fit a simple linear regression model to predict `mpg` using `displacement`. Report the intercept and slope.**
   ```python
   import pandas as pd
   from sklearn.linear_model import LinearRegression

   # Load and clean dataset
   df = pd.read_csv('auto-mpg.csv')
   df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

   # Simple linear regression
   X = df[['displacement']]
   y = df['mpg']
   model = LinearRegression().fit(X, y)
   print(f"Intercept: {model.intercept_:.4f}, Slope: {model.coef_[0]:.4f}")
   ```
   **Expected Output**: Intercept and slope values depend on the dataset but might be around 35.0 and -0.06, respectively.

2. **Fit a multiple linear regression model using `displacement`, `horsepower`, and `weight` to predict `mpg`. Calculate the RMSE on the training data.**
   ```python
   import numpy as np
   from sklearn.metrics import mean_squared_error

   # Multiple linear regression
   X = df[['displacement', 'horsepower', 'weight']]
   y = df['mpg']
   model = LinearRegression().fit(X, y)
   y_pred = model.predict(X)
   rmse = np.sqrt(mean_squared_error(y, y_pred))
   print(f"RMSE: {rmse:.4f}")
   ```
   **Expected Output**: RMSE typically around 4.0–5.0, depending on data preprocessing.

3. **Apply ridge regression to the same predictors as in question 2 with $ \lambda = 1.0 $. Compare the coefficients with those from OLS.**
   ```python
   from sklearn.linear_model import Ridge

   # Ridge regression
   ridge = Ridge(alpha=1.0).fit(X, y)
   print("OLS Coefficients:", model.coef_)
   print("Ridge Coefficients:", ridge.coef_)
   ```
   **Expected Output**: Ridge coefficients are slightly smaller than OLS due to the L2 penalty shrinking them toward zero.

4. **Use lasso regression with $ \lambda = 0.1 $ on the same predictors. Identify which coefficients are set to zero.**
   ```python
   from sklearn.linear_model import Lasso

   # Lasso regression
   lasso = Lasso(alpha=0.1).fit(X, y)
   print("Lasso Coefficients:", lasso.coef_)
   ```
   **Expected Output**: Some coefficients (e.g., `horsepower`) may be zero, indicating feature selection by lasso.

5. **Create a scatter plot of residuals vs. predicted `mpg` from the multiple linear regression model to check for heteroscedasticity.**
   ```python
   import matplotlib.pyplot as plt

   # Residual plot
   residuals = y - y_pred
   plt.scatter(y_pred, residuals)
   plt.axhline(0, color='red', linestyle='--')
   plt.xlabel('Predicted MPG')
   plt.ylabel('Residuals')
   plt.title('Residuals vs Predicted MPG')
   plt.savefig('residual_plot.png')
   plt.close()
   ```
   **Expected Output**: A scatter plot with no clear pattern (e.g., funnel shape) indicates homoscedasticity; a funnel shape suggests heteroscedasticity.

## Conclusion
Regression is a versatile tool in machine learning, from simple linear models to regularized techniques like ridge and lasso. Understanding their assumptions, evaluation metrics, and validation methods is crucial for building robust models. The quiz above helps solidify your knowledge, and tools like `scikit-learn` make it easy to apply these concepts in practice. Keep experimenting with datasets like Auto MPG to deepen your skills!