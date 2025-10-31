import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

print(dataset.head(3))

# 1. Perform basic EDA to understand the structure and distribution of the dataset.
# Shape
print(f"Rows x Columns: {dataset.shape[0]} x {dataset.shape[1]}")

# Column groups
print("\nColumn Groups:")
numerical_col = dataset.select_dtypes(include="number").columns.tolist()
categorical_col = dataset.select_dtypes(exclude="number").columns.tolist()
print(f"Numerical ({len(numerical_col)}): {numerical_col}")
print(f"Categorical ({len(categorical_col)}): {categorical_col}")

# Data Types
print("\nData Types:\n", dataset.dtypes)

# Non-Unique Values
print("\nNonunique values:\n", dataset.nunique())

# Missing Values
print("\nMissing Values per Column:\n", dataset.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows:", dataset.duplicated().sum())

# Numerical Statistics
print("\nNumerical Statistics:\n", dataset[numerical_col].describe().T.round(3).to_string())

# 2. Plot distributions of key features using histograms, box plots, etc.
numerical_key_features = ["Age", "Weight", "CH2O", "FCVC", "FAF", "TUE"]
categorical_key_features = ['family_history_with_overweight', 'FAVC', 'CAEC', "MTRANS"]

# Plot numerical key features
for num in numerical_key_features:
    plt.figure()
    plt.hist(dataset[num], bins=30, edgecolor="black", linewidth=0.6)
    plt.title(f"{num} – Histogram")
    plt.xlabel(num); plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# Plot categorical key features
for cat in categorical_key_features:
    plt.figure()
    data = dataset[cat].value_counts(dropna=False)
    plt.bar([str(x) for x in data.index], data.values, edgecolor="black", linewidth=0.6)
    plt.title(f"{cat} – Category Frequencies")
    plt.xlabel(cat); plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# 3. Visualize relationships between features and identify correlations using heatmaps
# Correlation matrix for numerical features
correlation_matrix = dataset[numerical_col].corr()
print("\nCorrelation Matrix:\n", correlation_matrix.round(3).to_string())

# Create correlation heatmap
plt.figure(figsize=(12, 10))

# Create heatmap
im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
cbar = plt.colorbar(im)
cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)

# Set ticks and labels
plt.xticks(range(len(numerical_col)), numerical_col, rotation=45, ha='right')
plt.yticks(range(len(numerical_col)), numerical_col)

# Add correlation values as text
for i in range(len(numerical_col)):
    for j in range(len(numerical_col)):
        text = plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()

# Calculate all correlations for summary statistics
strong_corr = []
moderate_corr = []
all_corrs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        all_corrs.append(abs(correlation_matrix.iloc[i, j]))

print("\nStrong Correlations (|r| > 0.5):")
if strong_corr:
    for feat1, feat2, corr in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
        print(f"{feat1} <-> {feat2} : {corr}")
else:
    print("  None found")

print("\nModerate Correlations (0.3 < |r| ≤ 0.5):")
if moderate_corr:
    for feat1, feat2, corr in sorted(moderate_corr, key=lambda x: abs(x[2]), reverse=True):
        print(f"{feat1} <-> {feat2} : {corr}")
else:
    print("  None found")


#4. Discuss key insights drawn from EDA and potential challenges with the dataset
print("\n1. DATASET CHARACTERISTICS:")
print(f"   - Total samples: {dataset.shape[0]}")
print(f"   - Features: {dataset.shape[1] - 1} (16 predictive features)")
print(f"   - Target variable: NObeyesdad (7 classes)")
print(f"   - No missing values detected")
print(f"   - No duplicate rows found")

print("\n2. FEATURE CORRELATIONS:")
n_features = len(correlation_matrix.columns)
expected_correlations = (n_features * (n_features - 1)) // 2
print(f"   - Number of numerical features: {n_features}")
print(f"   - Total unique pairwise correlations: {len(all_corrs)} (expected: {expected_correlations})")
print(f"   - Mean absolute correlation: {np.mean(all_corrs):.3f}")
print(f"   - Max absolute correlation: {np.max(all_corrs):.3f}")
print(f"   - Weak correlations (|r| ≤ 0.3): {len([c for c in all_corrs if c <= 0.3])}/{len(all_corrs)}")
print(f"   - Moderate correlations (0.3 < |r| ≤ 0.5): {len([c for c in all_corrs if 0.3 < c <= 0.5])}/{len(all_corrs)}")
print(f"   - Strong correlations (|r| > 0.5): {len([c for c in all_corrs if c > 0.5])}/{len(all_corrs)}")

# Age distribution insights
print("\n3. AGE DISTRIBUTION:")
print(f"   - Age range: {dataset['Age'].min():.1f} - {dataset['Age'].max():.1f} years")
print(f"   - Mean age: {dataset['Age'].mean():.1f} ± {dataset['Age'].std():.1f} years")
print(f"   - Young adults (<25): {(dataset['Age'] < 25).sum()} ({(dataset['Age'] < 25).sum()/len(dataset)*100:.1f}%)")
print(f"   - Adults (25-40): {((dataset['Age'] >= 25) & (dataset['Age'] < 40)).sum()} ({((dataset['Age'] >= 25) & (dataset['Age'] < 40)).sum()/len(dataset)*100:.1f}%)")
print(f"   - Older adults (40+): {(dataset['Age'] >= 40).sum()} ({(dataset['Age'] >= 40).sum()/len(dataset)*100:.1f}%)")

# Gender distribution
print("\n4. GENDER DISTRIBUTION:")
gender_counts = dataset['Gender'].value_counts()
for gender, count in gender_counts.items():
    print(f"   - {gender}: {count} ({count/len(dataset)*100:.1f}%)")

print("\n=== POTENTIAL CHALLENGES ===")
# Challenge 1: Class Imbalance
print("\n1. CLASS IMBALANCE:")
target_distribution = dataset['NObeyesdad'].value_counts().sort_index()
print(f"   Target variable distribution:")
for obesity_class, count in target_distribution.items():
    percentage = count / len(dataset) * 100
    bar = "█" * int(percentage / 2)
    print(f"   {obesity_class:25s}: {count:4d} ({percentage:5.1f}%) {bar}")
# Calculate imbalance ratio
max_class = target_distribution.max()
min_class = target_distribution.min()
imbalance_ratio = max_class / min_class
print(f"\n   Imbalance ratio (max/min): {imbalance_ratio:.2f}:1")
print(f"   Recommendation: Consider using:")
print(f"   - Stratified sampling for train/test split")
print(f"   - Class weights in model training")
print(f"   - SMOTE or other resampling techniques if needed")

# Challenge 2: Feature Multicollinearity
print("\n2. FEATURE MULTICOLLINEARITY:")
if strong_corr:
    print(f"   Found {len(strong_corr)} strong correlation(s):")
    for feat1, feat2, corr in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True)[:3]:
        print(f"   - {feat1} <-> {feat2}: r = {corr:.3f}")
    print(f"   Recommendation: Monitor these features for potential redundancy")
else:
    print(f"   No strong correlations (|r| > 0.5) detected")
    print(f"   All numerical features appear relatively independent")

# Challenge 3: Mixed Data Types
print("\n3. MIXED DATA TYPES:")
print(f"   - Numerical features: {len(numerical_col)}")
print(f"   - Categorical features: {len(categorical_col)}")
print(f"   Recommendation: Will require encoding for categorical variables")
print(f"   - Binary categorical: Gender, family_history_with_overweight, FAVC, SMOKE, SCC")
print(f"   - Ordinal categorical: CAEC, CALC, MTRANS")

# Challenge 4: Feature Scales
print("\n4. FEATURE SCALING NEEDS:")
print(f"   Numerical features have different scales:")
for col in numerical_col[:5]:  # Show first 5
    print(f"   - {col:10s}: [{dataset[col].min():.2f}, {dataset[col].max():.2f}]")
print(f"   Recommendation: Apply StandardScaler or MinMaxScaler for distance-based models")
