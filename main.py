import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load and Explore Data

# Load the data
df = pd.read_csv("./data.csv")

# Check for missing values
missing_values = df.isnull().sum()

# Verify no missing values
print("Missing values in each column:\n", missing_values)

# Get the number of rows and features excluding the column names
num_rows = df.shape[0]
num_features = df.shape[1]

# Print the number of rows and features
print(f"The number of rows in the dataset: {num_rows}")
print(f"The number of features in the dataset: {num_features}")

# Select only numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Step 2: Feature Correlation Analysis

# Compute the correlation matrix
corr_matrix = numeric_df.corr()

# Focus on correlations with 'TCPOutputJitter'
jitter_correlation = corr_matrix['TCPOutputJitter'].sort_values(ascending=False)

# Set a threshold for strong correlations
threshold = 0.3
strong_correlations = jitter_correlation[abs(jitter_correlation) > threshold]

# Print the features most correlated with 'TCPOutputJitter'
print("Top features correlated with TCPOutputJitter:\n", strong_correlations)

# Plot heatmap of top correlations with jitter
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix.loc[strong_correlations.index, strong_correlations.index], annot=True, cmap='Pastel1', fmt='.2f', annot_kws={"size": 8}, linewidths=.5)
plt.title('Correlation Matrix Heatmap (Strong Correlations with Jitter)', fontsize=16)
plt.xticks(rotation=0, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.show()

# Step 3: Data Visualization

# Plot histograms for selected features
for col in strong_correlations.index:
    if col != 'TCPOutputJitter':  # Exclude the target variable itself
        plt.figure(figsize=(8, 6))
        df[col].plot(kind='hist', bins=20, color=plt.get_cmap('tab10')(0.6))
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title(f'Histogram Distribution of: \n {col}', fontsize=16)
        plt.show()

# Plot scatter plots between selected features and jitter
for col in strong_correlations.index:
    if col != 'TCPOutputJitter':  # Exclude the target variable itself
        plt.figure(figsize=(10, 6))
        plt.scatter(df[col], df['TCPOutputJitter'], alpha=0.5, color=plt.get_cmap('tab10')(0.6), label=f'{col}')
        plt.title(f'Scatter Plot of: \n {col} vs TCPOutputJitter', fontsize=16)
        plt.xlabel(col)
        plt.ylabel('TCPOutputJitter')
        plt.legend()
        plt.show()

# Plot box plots for selected features with annotations
for col in strong_correlations.index:
    if col != 'TCPOutputJitter':  # Exclude the target variable itself
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, y=col, color=plt.get_cmap('tab10')(0.6))
        plt.title(f'Box Plot of: \n {col}', fontsize=16)
        plt.ylabel(col)
        
        # Get quartile values
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        median = df[col].median()
        IQR = Q3 - Q1
        lower_whisker = max(df[col].min(), Q1 - 1.5 * IQR)
        upper_whisker = min(df[col].max(), Q3 + 1.5 * IQR)
        
        # Add annotations
        plt.annotate('Q1', xy=(0.1, Q1), xytext=(0.15, Q1), arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)
        plt.annotate('Median', xy=(0.1, median), xytext=(0.15, median), arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)
        plt.annotate('Q3', xy=(0.1, Q3), xytext=(0.15, Q3), arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)
        plt.annotate('Lower Whisker', xy=(0.1, lower_whisker), xytext=(0.15, lower_whisker), arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)
        plt.annotate('Upper Whisker', xy=(0.1, upper_whisker), xytext=(0.15, upper_whisker), arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)
        
        plt.show()

# Step 4: Data Preprocessing

# Select the relevant features
selected_features = ['TCPOutputJitter', 'TCPInputJitter', 'TCPInputDelay', '0_InterATimesReq', 'AvgVideoBitRate', 'AvgQualityIndex']

# Standardize the selected features
scaler = StandardScaler()
df_selected = pd.DataFrame(scaler.fit_transform(df[selected_features]), columns=selected_features)

# Verify the result of normalization
print(df_selected.describe())

# Define the upper and lower quantiles for outlier capping
lower_quantile = 0.01
upper_quantile = 0.99

# Cap outliers
for col in df_selected.columns:
    lower_cap = df_selected[col].quantile(lower_quantile)
    upper_cap = df_selected[col].quantile(upper_quantile)
    df_selected[col] = np.where(df_selected[col] < lower_cap, lower_cap, df_selected[col])
    df_selected[col] = np.where(df_selected[col] > upper_cap, upper_cap, df_selected[col])

# Verify the result after capping outliers
print("Data after capping outliers:")
print(df_selected.describe())

# Step 5: Model Training and Evaluation

# Define the target and features
X = df_selected.drop(columns=['TCPOutputJitter'])
y = df_selected['TCPOutputJitter']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
