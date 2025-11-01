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
print("\nCorrelation Matrix (Numerical Features):")
print(correlation_matrix.round(3).to_string())

# Create correlation heatmap
plt.figure(figsize=(10, 8))
im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
cbar = plt.colorbar(im)
cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)

# Set ticks and labels
plt.xticks(range(len(numerical_col)), numerical_col, rotation=45, ha='right')
plt.yticks(range(len(numerical_col)), numerical_col)

# Add correlation values as text
for i in range(len(numerical_col)):
    for j in range(len(numerical_col)):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                ha="center", va="center", color="black", fontsize=8)

plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()

# Calculate correlation statistics
all_corrs = []
strong_corr = []
moderate_corr = []
weak_corr = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        feat1 = correlation_matrix.columns[i]
        feat2 = correlation_matrix.columns[j]
        corr = correlation_matrix.iloc[i, j]
        abs_corr = abs(corr)
        
        all_corrs.append(abs_corr)
        
        if abs_corr > 0.5:
            strong_corr.append((feat1, feat2, corr))
        elif abs_corr > 0.3:
            moderate_corr.append((feat1, feat2, corr))
        else:
            weak_corr.append((feat1, feat2, corr))

# Display correlation categories
print(f"\nStrong Correlations (|r| > 0.5): {len(strong_corr)} pairs")
if strong_corr:
    for feat1, feat2, corr in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
        direction = "positive" if corr > 0 else "negative"
        print(f"• {feat1} <-> {feat2}: {corr:.3f} ({direction})")
else:
    print("• None found")

print(f"\nModerate Correlations (0.3 < |r| ≤ 0.5): {len(moderate_corr)} pairs")
if moderate_corr:
    for feat1, feat2, corr in sorted(moderate_corr, key=lambda x: abs(x[2]), reverse=True):
        print(f"• {feat1} <-> {feat2}: {corr:.3f}")
else:
    print("• None found")

# Display top 5 overall correlations
print(f"\nTop 5 correlations:")
all_pairwise_corrs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        feat1 = correlation_matrix.columns[i]
        feat2 = correlation_matrix.columns[j]
        corr = correlation_matrix.iloc[i, j]
        all_pairwise_corrs.append((feat1, feat2, corr))

top_5_correlations = sorted(all_pairwise_corrs, key=lambda x: abs(x[2]), reverse=True)[:5]

for feat1, feat2, corr in top_5_correlations:
    direction = "positive" if corr > 0 else "negative"
    strength = "STRONG" if abs(corr) > 0.5 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
    print(f"• {feat1} <-> {feat2}: {corr:.3f} ({direction}, {strength})")

# Correlation summary statistics
print(f"\nCorrelation statistics summary:")
n_features = len(correlation_matrix.columns)
expected_correlations = (n_features * (n_features - 1)) // 2
print(f"• Mean absolute correlation: {np.mean(all_corrs):.3f}")
print(f"• Maximum correlation: {np.max(all_corrs):.3f}")
print(f"• Correlation distribution:")
print(f"  - Strong (|r| > 0.5): {len(strong_corr)} pairs ({len(strong_corr)/len(all_corrs)*100:.1f}%)")
print(f"  - Moderate (0.3 < |r| ≤ 0.5): {len(moderate_corr)} pairs ({len(moderate_corr)/len(all_corrs)*100:.1f}%)")
print(f"  - Weak (|r| ≤ 0.3): {len(weak_corr)} pairs ({len(weak_corr)/len(all_corrs)*100:.1f}%)")

# 4. Discuss key insights drawn from EDA and potential challenges with the dataset
print("\nKEY INSIGHTS FROM EDA")
print("\nDataset Overview:")
print(f"• Samples: {dataset.shape[0]}, Features: {dataset.shape[1] - 1}")
print(f"• Target: NObeyesdad (7 obesity classes)")
print(f"• Data Quality: No missing values, {dataset.duplicated().sum()} duplicates")

print("\nDemographic Profile:")
gender_counts = dataset['Gender'].value_counts()
print(f"• Age Range: {dataset['Age'].min():.0f}-{dataset['Age'].max():.0f} years")
print(f"• Mean Age: {dataset['Age'].mean():.1f} ± {dataset['Age'].std():.1f} years")
print(f"• Gender: {gender_counts['Male']} Male, {gender_counts['Female']} Female")
print(f"• Young Adults (<25): {(dataset['Age'] < 25).sum()/len(dataset)*100:.1f}%")

print("\nPhysical Measurements:")
print(f"• Height: {dataset['Height'].min():.2f}-{dataset['Height'].max():.2f} meters")
print(f"• Weight: {dataset['Weight'].min():.1f}-{dataset['Weight'].max():.1f} kg")

print("\nKey Correlations:")
for feat1, feat2, corr in top_5_correlations:
    direction = "positive" if corr > 0 else "negative"
    print(f"• {feat1} <-> {feat2}: {corr:.3f} ({direction})")

print("\nPotential Challenges")

# Challenge 1: Weak Correlation between most features
print("\n1. Weak Feature Correlations:")
print(f"• Mean absolute correlation: {np.mean(all_corrs):.3f}")
print(f"• Strongest correlation: Height ↔ Weight (r = {correlation_matrix.loc['Height', 'Weight']:.3f})")
print(f"• {len(weak_corr)} of {len(all_corrs)} feature pairs show weak correlations (|r| ≤ 0.3)")

# Challenge 2: Duplicate Rows
print("\n2. Duplicate Rows:")
print(f"• Found: {dataset.duplicated().sum()} duplicate rows")

# Challenge 3: Feature Scaling Requirements
print("\n3. Feature Scaling Requirements:")
print("• Features have very different ranges:")
print(f"  - Weight: {dataset['Weight'].min():.0f}-{dataset['Weight'].max():.0f}")
print(f"  - TUE: {dataset['TUE'].min():.0f}-{dataset['TUE'].max():.0f}")
print(f"  - Age: {dataset['Age'].min():.0f}-{dataset['Age'].max():.0f}")

# Key Insights summary 
print("\n1. We notice that the target variable is well-distributed across "
"the obesity categories, which reduces the risk of bias towards any class.")
print("2. There are no missing values which simplifies the preprocessing stage.")
print("3. Some features such as Age, Height, and Weight have a wide range of values, " \
"that can help model diverse patterns.")
print("4. Feature correlations mostly have weak correlations with each other and the target. Weak correlations" \
" imply that individual features alone might not strongly predict obesity categories, highlighting the " \
"need for using multiple features to capture the relationship.")
print("5. There are 24 duplicate rows, which must be removed as they can add a slight bias analysis.")



# Data Preprocessing Requirements 
print("Datapreprocessing Requirements")
# 1. Handle missing values appropriately (e.g., imputation, removal)
# Check for duplicates
duplicate_count_before = dataset.duplicated().sum()
print(f"\nDuplicate rows found: {duplicate_count_before}")
print(f"Dataset size before removing duplicates: {len(dataset)} rows")

if duplicate_count_before > 0:
    # Remove duplicates
    dataset = dataset.drop_duplicates()
    # Verify removal
    duplicate_count_after = dataset.duplicated().sum()
    rows_removed = duplicate_count_before - duplicate_count_after
    print(f"Dataset size after removing duplicates: {len(dataset)} rows")
    print(f"Removed {rows_removed} duplicate rows")
else:
    print("No duplicate rows found")

# Reset index after removing duplicates
dataset = dataset.reset_index(drop=True)

#Handle Missing Values
missing_values = dataset.isnull().sum()
total_missing = missing_values.sum()

if total_missing > 0:
    print(f"\nMissing values found:")
    print(missing_values[missing_values > 0])
    # Add your imputation/removal logic here
else:
    print("\nNo missing values found in the dataset")

#2. 
