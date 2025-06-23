import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================================
# 1. Enhanced Missing Value Handling
# ===========================================
"""
Imputation is the process of replacing missing data with substituted values.
It's crucial because most ML algorithms cannot handle missing values directly.
"""

def demonstrate_imputation(df):
    print("=== Missing Values Before Imputation ===")
    print(df.isna().sum())
    
    # 1. Mean/Median Imputation
    mean_imputer = SimpleImputer(strategy='mean')
    df['horsepower_mean'] = mean_imputer.fit_transform(df[['horsepower']])
    
    # 2. Group-based Imputation (using numpy)
    group_means = df.groupby('cylinders')['horsepower'].transform('mean')
    df['horsepower_group'] = df['horsepower'].fillna(group_means)
    
    # 3. KNN Imputation (more advanced)
    knn_imputer = KNNImputer(n_neighbors=2)
    df['horsepower_knn'] = knn_imputer.fit_transform(df[['horsepower']])
    
    print("\n=== After Imputation ===")
    print(df[['horsepower', 'horsepower_mean', 'horsepower_group', 'horsepower_knn']])
    return df

# ===========================================
# 2. Enhanced Outlier Handling
# ===========================================
"""
Outliers can significantly affect model performance. Let's explore detection and handling methods.
"""

def handle_outliers(df, column):
    print(f"\n=== Handling Outliers in {column} ===")
    
    # Calculate bounds using IQR method
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Detect outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Detected {len(outliers)} outliers in {column}")
    
    # Visualization
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df[column])
    plt.title(f"Before Outlier Handling - {column}")
    
    # Method 1: Capping (Winsorization)
    df[f"{column}_capped"] = np.where(df[column] > upper_bound, upper_bound,
                                     np.where(df[column] < lower_bound, lower_bound, df[column]))
    
    # Method 2: Log Transformation (for right-skewed data)
    if (df[column] > 0).all():
        df[f"{column}_log"] = np.log1p(df[column])
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[f"{column}_capped"])
    plt.title(f"After Capping - {column}")
    plt.tight_layout()
    plt.show()
    
    return df

# ===========================================
# 3. Enhanced Feature Scaling
# ===========================================
"""
Feature scaling is crucial for algorithms that are sensitive to the scale of features.
"""

def demonstrate_scaling(df, columns):
    print("\n=== Feature Scaling ===")
    
    # Original data
    print("Original data:")
    print(df[columns].head())
    
    # 1. Standardization (Z-score)
    scaler = StandardScaler()
    df_std = df.copy()
    df_std[columns] = scaler.fit_transform(df[columns])
    
    # 2. Min-Max Scaling
    minmax = MinMaxScaler()
    df_minmax = df.copy()
    df_minmax[columns] = minmax.fit_transform(df[columns])
    
    # 3. Robust Scaling (for data with outliers)
    robust = RobustScaler()
    df_robust = df.copy()
    df_robust[columns] = robust.fit_transform(df[columns])
    
    print("\nStandardized data (mean=0, std=1):")
    print(df_std[columns].head())
    
    print("\nWhen to use which scaler?")
    print("- StandardScaler: When data is ~normally distributed")
    print("- MinMaxScaler: When you know the distribution is not Gaussian")
    print("- RobustScaler: When data contains many outliers")
    
    return df_std, df_minmax, df_robust

# ===========================================
# 4. Dimensionality Reduction
# ===========================================
"""
Dimensionality reduction helps reduce the number of features while preserving 
important information. It's useful for:
1. Reducing overfitting
2. Speeding up training
3. Visualizing high-dimensional data
4. Removing correlated features
"""

def demonstrate_dimensionality_reduction(df, target_col='mpg'):
    print("\n=== Dimensionality Reduction ===")
    
    # Prepare data
    X = df.drop(columns=[target_col, 'car_name'])
    y = df[target_col]
    
    # Handle categorical columns
    X = pd.get_dummies(X, columns=['origin'])
    
    # 1. PCA (Principal Component Analysis)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total explained variance:", sum(pca.explained_variance_ratio_))
    
    # Plot PCA results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(label=target_col)
    plt.title('PCA of Car Features')
    plt.show()
    
    return X_pca

# ===========================================
# 5. Feature Selection
# ===========================================
"""
Feature selection helps identify the most important features for your model.
Why it's important:
1. Reduces overfitting
2. Improves model interpretability
3. Reduces training time
4. May improve model performance
"""

def demonstrate_feature_selection(df, target_col='mpg'):
    print("\n=== Feature Selection ===")
    
    # Prepare data
    X = df.drop(columns=[target_col, 'car_name'])
    y = df[target_col]
    
    # Handle categorical columns
    X = pd.get_dummies(X, columns=['origin'])
    
    # 1. Univariate Selection (SelectKBest)
    selector = SelectKBest(score_func=f_regression, k=3)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected features
    selected_features = X.columns[selector.get_support()]
    print("Top features (SelectKBest):", list(selected_features))
    
    # 2. Recursive Feature Elimination (RFE)
    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=3)
    rfe.fit(X, y)
    
    # Get selected features
    selected_features_rfe = X.columns[rfe.support_]
    print("Top features (RFE):", list(selected_features_rfe))
    
    return selected_features

# ===========================================
# Main Execution
# ===========================================
if __name__ == "__main__":
    # Create sample data
    data = {
        "car_name": ["car_a", "car_b", "car_c", "car_d", "car_e", "car_f"],
        "cylinders": [4, 6, 8, 4, 4, 8],
        "displacement": [140, 200, 360, 150, 130, 3700],
        "horsepower": [90, 105, 215, 92, np.nan, 220],
        "weight": [2400, 3000, 4300, 2500, 2200, 4400],
        "acceleration": [15.5, 14.0, 12.5, 16.0, 15.0, 11.0],
        "model_year": [80, 78, 76, 82, 81, 77],
        "origin": [1, 1, 1, 2, 3, 1],
        "mpg": [30.5, 24.0, 13.0, 29.5, 32.0, 10.0]
    }
    df = pd.DataFrame(data)
    
    # 1. Handle missing values
    print("\n" + "="*50)
    print("1. HANDLING MISSING VALUES")
    print("="*50)
    df = demonstrate_imputation(df)
    
    # 2. Handle outliers
    print("\n" + "="*50)
    print("2. HANDLING OUTLIERS")
    print("="*50)
    df = handle_outliers(df, 'displacement')
    
    # 3. Feature scaling
    print("\n" + "="*50)
    print("3. FEATURE SCALING")
    print("="*50)
    numeric_cols = ['weight', 'acceleration', 'displacement']
    df_std, df_minmax, df_robust = demonstrate_scaling(df, numeric_cols)
    
    # 4. Dimensionality reduction
    print("\n" + "="*50)
    print("4. DIMENSIONALITY REDUCTION")
    print("="*50)
    X_pca = demonstrate_dimensionality_reduction(df)
    
    # 5. Feature selection
    print("\n" + "="*50)
    print("5. FEATURE SELECTION")
    print("="*50)
    selected_features = demonstrate_feature_selection(df)
