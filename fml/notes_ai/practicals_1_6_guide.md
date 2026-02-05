# FML Practicals 1-6: Complete Study Guide

> **Mode:** Teaching + Notes | **Goal:** Understand, then implement

---

# Practical 1: NumPy & Matplotlib

## Concept Map
```
Python Lists (slow, flexible)
        ↓
    NumPy Arrays (fast, homogeneous)
        ↓
    ┌─────────────────┬─────────────────┐
    ↓                 ↓                 ↓
 Creation         Operations        Visualization
 (zeros, ones,    (math, stats,     (Matplotlib)
  arange, random)  reshape, slice)
```

## Core Definitions

| Term             | Definition                                                               |
| ---------------- | ------------------------------------------------------------------------ |
| **ndarray**      | N-dimensional array. Fixed size, same data type throughout.              |
| **Shape**        | Tuple describing dimensions. `(3,4)` = 3 rows, 4 columns.                |
| **dtype**        | Data type of elements. `int64`, `float64`, etc.                          |
| **Broadcasting** | Auto-expansion of smaller array to match larger for element-wise ops.    |
| **Axis**         | Direction of operation. `axis=0` = along rows, `axis=1` = along columns. |
| **Figure**       | Container for all plot elements. One figure can have multiple subplots.  |
| **Axes**         | Single plot area within a figure. Contains the actual visualization.     |

## Intuition: Why NumPy?

**Problem:** Python lists are slow for numerical operations.
```python
# Python list: loops through each element
result = [a[i] + b[i] for i in range(len(a))]  # Slow

# NumPy: vectorized, runs in C
result = a + b  # 100x faster
```

**Why it matters:** ML deals with millions of numbers. Speed is critical.

## NumPy Operations

### Array Creation
```python
import numpy as np

# From list
arr = np.array([1, 2, 3, 4, 5])

# Filled arrays
zeros = np.zeros((3, 3))        # 3x3 matrix of 0s
ones = np.ones((2, 4))          # 2x4 matrix of 1s
empty = np.empty((2, 2))        # Uninitialized (garbage values)

# Sequences
range_arr = np.arange(0, 10, 2)       # [0, 2, 4, 6, 8] - like range()
linspace = np.linspace(0, 1, 5)       # [0, 0.25, 0.5, 0.75, 1] - evenly spaced

# Random
random_int = np.random.randint(1, 100, size=(3, 3))  # 3x3 random integers
random_float = np.random.rand(3, 3)                   # 3x3 uniform [0,1)
random_normal = np.random.randn(3, 3)                 # 3x3 standard normal
```

### Indexing & Slicing
```python
arr = np.array([10, 20, 30, 40, 50])

arr[0]      # 10 (first element)
arr[-1]     # 50 (last element)
arr[1:4]    # [20, 30, 40] (index 1 to 3)
arr[::2]    # [10, 30, 50] (every 2nd element)
arr[::-1]   # [50, 40, 30, 20, 10] (reversed)

# 2D Arrays
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

matrix[0, 0]     # 1 (row 0, col 0)
matrix[1, :]     # [4, 5, 6] (entire row 1)
matrix[:, 2]     # [3, 6, 9] (entire column 2)
matrix[0:2, 0:2] # [[1,2], [4,5]] (submatrix)
```

### Mathematical Operations
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise
a + b    # [5, 7, 9]
a * b    # [4, 10, 18]
a ** 2   # [1, 4, 9]
np.sqrt(a)  # [1.0, 1.41, 1.73]

# Statistical
np.sum(a)      # 6
np.mean(a)     # 2.0
np.std(a)      # 0.816
np.min(a)      # 1
np.max(a)      # 3
np.argmax(a)   # 2 (index of max)
```

### Reshaping
```python
arr = np.arange(12)  # [0,1,2,...,11]

arr.reshape(3, 4)    # 3 rows, 4 columns
arr.reshape(4, -1)   # 4 rows, auto-calculate columns
arr.flatten()        # Back to 1D
arr.T                # Transpose (swap rows/columns)
```

## Matplotlib Visualization

### Basic Structure
```python
import matplotlib.pyplot as plt

# Method 1: Simple
plt.plot(x, y)
plt.title('Title')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Method 2: Object-oriented (recommended)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Title')
plt.show()
```

### Plot Types with Examples

```python
import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# LINE PLOT - for trends
plt.figure(figsize=(8, 4))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.title('Line Plot')
plt.legend()
plt.grid(True)
plt.show()

# BAR CHART - for comparisons
categories = ['A', 'B', 'C', 'D']
values = [25, 40, 30, 55]
plt.bar(categories, values, color='skyblue', edgecolor='black')
plt.title('Bar Chart')
plt.show()

# SCATTER PLOT - for correlations
x = np.random.randn(50)
y = x + np.random.randn(50) * 0.5
plt.scatter(x, y, c='red', alpha=0.7)
plt.title('Scatter Plot')
plt.show()

# HISTOGRAM - for distributions
data = np.random.normal(50, 15, 1000)
plt.hist(data, bins=30, edgecolor='black')
plt.axvline(np.mean(data), color='red', linestyle='--')
plt.title('Histogram')
plt.show()

# PIE CHART - for proportions
sizes = [30, 25, 20, 15, 10]
labels = ['A', 'B', 'C', 'D', 'E']
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Pie Chart')
plt.show()
```

### Saving Figures
```python
plt.savefig('output.png', dpi=150, bbox_inches='tight')
```

## Common Traps

| Mistake             | Problem            | Fix                               |
| ------------------- | ------------------ | --------------------------------- |
| `arr * 2` on list   | Error              | Convert to numpy first            |
| `arr[1,2]` on 1D    | Error              | Use `arr[1]`                      |
| Forgot `plt.show()` | No display         | Add at end                        |
| Wrong reshape size  | Error              | Product must match total elements |
| In-place vs copy    | Unexpected changes | Use `.copy()` when needed         |

## Condensed Notes

```
NumPy:
- Create: array(), zeros(), ones(), arange(), linspace(), random
- Index: arr[i], arr[start:end:step], arr[row, col]
- Stats: sum(), mean(), std(), min(), max(), argmax()
- Shape: reshape(), flatten(), T

Matplotlib:
- plt.plot() = line
- plt.bar() = comparison
- plt.scatter() = correlation
- plt.hist() = distribution
- plt.pie() = proportion
- Always: title(), xlabel(), ylabel(), legend(), show()
```

---

# Practical 2: Pandas

## Concept Map
```
Data Source (CSV, Excel, JSON, Database)
              ↓
         pd.read_*()
              ↓
    ┌─────────────────────┐
    │     DataFrame       │
    │  (rows × columns)   │
    │  - labeled axes     │
    │  - mixed types OK   │
    └─────────────────────┘
              ↓
    ┌─────────┬───────────┬──────────┐
    ↓         ↓           ↓          ↓
  Explore   Select     Transform   Export
  (info,    (loc,      (fillna,    (to_csv,
   describe) iloc)     groupby)    to_excel)
```

## Core Definitions

| Term          | Definition                                                      |
| ------------- | --------------------------------------------------------------- |
| **DataFrame** | 2D labeled data structure. Rows have index, columns have names. |
| **Series**    | 1D labeled array. Single column from DataFrame.                 |
| **Index**     | Row labels. Default: 0, 1, 2... Can be custom.                  |
| **loc**       | Label-based selection. `df.loc[row_label, col_label]`           |
| **iloc**      | Integer position-based selection. `df.iloc[row_num, col_num]`   |
| **NaN**       | Not a Number. Represents missing data.                          |

## Intuition: Why Pandas?

**Problem:** Real data is messy - mixed types, missing values, needs filtering.
**Solution:** Pandas provides Excel-like operations in Python.

| Excel       | Pandas               |
| ----------- | -------------------- |
| Open file   | `pd.read_csv()`      |
| Filter rows | `df[df['col'] > 50]` |
| VLOOKUP     | `df.merge()`         |
| Pivot table | `df.pivot_table()`   |
| Save as     | `df.to_csv()`        |

## DataFrame Operations

### Creating DataFrames
```python
import pandas as pd

# From dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['NY', 'LA', 'Chicago']
}
df = pd.DataFrame(data)

# From list of lists
df = pd.DataFrame(
    [[1, 2], [3, 4]],
    columns=['A', 'B']
)
```

### Import/Export
```python
# CSV
df = pd.read_csv('file.csv')
df.to_csv('output.csv', index=False)  # index=False removes row numbers

# Excel (needs openpyxl)
df = pd.read_excel('file.xlsx')
df.to_excel('output.xlsx', index=False)

# JSON
df = pd.read_json('file.json')
df.to_json('output.json')
```

### Exploration
```python
df.shape          # (rows, columns)
df.columns        # Column names
df.dtypes         # Data types per column
df.head(5)        # First 5 rows
df.tail(5)        # Last 5 rows
df.info()         # Summary: types, non-null counts
df.describe()     # Statistics: mean, std, min, max, quartiles
df.isnull().sum() # Missing values per column
```

### Selection
```python
# Column selection
df['Name']              # Single column (Series)
df[['Name', 'Age']]     # Multiple columns (DataFrame)

# Row selection by position
df.iloc[0]              # First row
df.iloc[0:3]            # Rows 0, 1, 2
df.iloc[0, 1]           # Row 0, Column 1

# Row selection by label
df.loc[0]               # Row with index 0
df.loc[df['Age'] > 25]  # Rows where Age > 25
```

### Filtering
```python
# Single condition
df[df['Age'] > 25]

# Multiple conditions (use & for AND, | for OR)
df[(df['Age'] > 25) & (df['City'] == 'NY')]

# isin for multiple values
df[df['City'].isin(['NY', 'LA'])]

# String contains
df[df['Name'].str.contains('li')]
```

### Transformation
```python
# Add column
df['Score'] = [85, 90, 78]

# Apply function
df['Age_Group'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Old')

# Sorting
df.sort_values('Age', ascending=False)

# Grouping
df.groupby('City')['Age'].mean()       # Mean age per city
df.groupby('City').agg({'Age': 'mean', 'Score': 'sum'})
```

### Handling Missing Values
```python
df.isnull().sum()                        # Count nulls per column
df.dropna()                              # Remove rows with any null
df.fillna(0)                             # Fill all nulls with 0
df['Age'].fillna(df['Age'].mean())       # Fill with mean
df['City'].fillna(df['City'].mode()[0])  # Fill with mode
```

## Common Traps

| Mistake                           | Problem            | Fix                 |
| --------------------------------- | ------------------ | ------------------- |
| `df.dropna()` alone               | Doesn't modify df  | `df = df.dropna()`  |
| `df['col'] = df['col'].fillna(0)` | Works              | Assign back         |
| `index=True` when saving          | Extra index column | Use `index=False`   |
| Filter without parentheses        | Syntax error       | `(cond1) & (cond2)` |
| Using `and` instead of `&`        | Error              | Use `&` for Series  |

## Condensed Notes

```
Load: read_csv(), read_excel(), read_json()
Save: to_csv(index=False), to_excel(), to_json()

Explore:
- shape, columns, dtypes
- head(), tail(), info(), describe()
- isnull().sum()

Select:
- Column: df['col'], df[['col1','col2']]
- Row by position: df.iloc[0], df.iloc[0:3]
- Row by label: df.loc[condition]

Filter: df[df['col'] > value]
Group: df.groupby('col').agg_func()
Missing: fillna(value), dropna()
```

---

# Practical 3: Scikit-learn Introduction

## Concept Map
```
                    SCIKIT-LEARN
                         │
    ┌────────────────────┼────────────────────┐
    ↓                    ↓                    ↓
 DATASETS            PREPROCESSING         MODELS
 - load_iris()       - train_test_split    - fit()
 - load_digits()     - StandardScaler      - predict()
 - make_*()          - LabelEncoder        - score()
                                               │
                                               ↓
                                           METRICS
                                           - accuracy_score
                                           - confusion_matrix
```

## Core Definitions

| Term            | Definition                                                      |
| --------------- | --------------------------------------------------------------- |
| **Feature (X)** | Input variables. What you use to predict. Columns in your data. |
| **Target (y)**  | Output variable. What you predict. Usually last column.         |
| **fit()**       | Learn patterns from training data. Adjusts model parameters.    |
| **transform()** | Apply learned transformation to data.                           |
| **predict()**   | Use trained model to make predictions on new data.              |
| **train set**   | Data used to teach the model. Typically 80%.                    |
| **test set**    | Data used to evaluate. Model never sees during training. 20%.   |

## Intuition: The ML Workflow

```
1. GET DATA         →  What do we have?
2. SPLIT            →  Separate train/test (prevent cheating)
3. PREPROCESS       →  Scale, encode, clean
4. TRAIN            →  Model learns patterns from train data
5. PREDICT          →  Apply to test data
6. EVALUATE         →  How good? Accuracy, errors, etc.
```

**Why split?** If model sees test data during training, it memorizes, not learns.

## Built-in Datasets

```python
from sklearn import datasets

# Classification
iris = datasets.load_iris()       # 150 samples, 4 features, 3 classes
digits = datasets.load_digits()   # 1797 images, 64 features, 10 classes

# Regression
diabetes = datasets.load_diabetes()
boston = datasets.load_boston()   # Deprecated, use fetch_*

# Access data
X = iris.data           # Features (150, 4)
y = iris.target         # Labels (150,)
names = iris.feature_names
targets = iris.target_names
```

## Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42     # Reproducibility
)

# Result:
# X_train: 80% of features for training
# X_test: 20% of features for testing
# y_train: 80% of labels for training
# y_test: 20% of labels for evaluation
```

## Preprocessing

### StandardScaler (Z-score normalization)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fit_transform on train: learn mean/std AND transform
X_train_scaled = scaler.fit_transform(X_train)

# transform only on test: use train's mean/std
X_test_scaled = scaler.transform(X_test)

# Result: mean ≈ 0, std ≈ 1
```

**Why scale?**
- Algorithms like KNN, SVM use distance. Large values dominate.
- Gradient descent converges faster with scaled features.

### LabelEncoder
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
labels = ['cat', 'dog', 'cat', 'bird']
encoded = le.fit_transform(labels)  # [1, 2, 1, 0]
decoded = le.inverse_transform([0, 1, 2])  # ['bird', 'cat', 'dog']
```

## Training & Prediction

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Step 1: Create model
model = DecisionTreeClassifier()

# Step 2: Train (fit)
model.fit(X_train, y_train)

# Step 3: Predict
predictions = model.predict(X_test)

# Step 4: Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2%}")
```

## Complete Pipeline Example

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 5. Predict
predictions = model.predict(X_test)

# 6. Evaluate
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(classification_report(y_test, predictions))
```

## Common Traps

| Mistake                        | Why Wrong                  | Fix                             |
| ------------------------------ | -------------------------- | ------------------------------- |
| `scaler.fit_transform(X_test)` | Data leakage               | `scaler.transform(X_test)` only |
| No `random_state`              | Different results each run | Set `random_state=42`           |
| Forgot scaling for KNN/SVM     | Poor performance           | Always scale for distance-based |
| Using scaled data for trees    | Unnecessary                | Trees don't need scaling        |

## Condensed Notes

```
Workflow: Load → Split → Scale → Train → Predict → Evaluate

Split: train_test_split(X, y, test_size=0.2, random_state=42)

Scale:
- scaler.fit_transform(X_train)  # Learn + apply
- scaler.transform(X_test)       # Apply only

Train: model.fit(X_train, y_train)
Predict: model.predict(X_test)
Evaluate: accuracy_score(y_test, predictions)

Key rule: Never fit on test data!
```

---

# Practical 4: Fruit Prediction (ML Introduction)

## Concept Map
```
Raw Data (text labels)
       ↓
   Encoding (text → numbers)
       ↓
   ML Model learns patterns
       ↓
   New input → Prediction
```

## The Dataset

| Age Group | Gender | Favorite Fruit |
| --------- | ------ | -------------- |
| 10-15     | Male   | Apple          |
| 10-15     | Female | Banana         |
| 16-20     | Male   | Orange         |
| 16-20     | Female | Mango          |
| 21-25     | Male   | Banana         |
| 21-25     | Female | Mango          |

## Intuition: How ML Learns

**Human approach:** Look at data, find pattern.
- "If Female and 16+, output is Mango"
- "If Male and 10-15, output is Apple"

**ML approach:** Same, but automated.
- Decision Tree finds these IF-THEN rules by itself
- No explicit programming needed

## Step-by-Step Implementation

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# 1. Create dataset
data = {
    'Age_Group': ['10-15', '10-15', '16-20', '16-20', '21-25', '21-25'],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Fruit': ['Apple', 'Banana', 'Orange', 'Mango', 'Banana', 'Mango']
}
df = pd.DataFrame(data)

# 2. Encode categorical columns
le_age = LabelEncoder()
le_gender = LabelEncoder()
le_fruit = LabelEncoder()

df['Age_Enc'] = le_age.fit_transform(df['Age_Group'])
df['Gender_Enc'] = le_gender.fit_transform(df['Gender'])
df['Fruit_Enc'] = le_fruit.fit_transform(df['Fruit'])

# Check encoding:
# Age: 10-15=0, 16-20=1, 21-25=2
# Gender: Female=0, Male=1
# Fruit: Apple=0, Banana=1, Mango=2, Orange=3

# 3. Prepare X and y
X = df[['Age_Enc', 'Gender_Enc']]
y = df['Fruit_Enc']

# 4. Train
model = DecisionTreeClassifier()
model.fit(X, y)

# 5. Predict new input
def predict_fruit(age_group, gender):
    age_enc = le_age.transform([age_group])[0]
    gender_enc = le_gender.transform([gender])[0]
    pred_enc = model.predict([[age_enc, gender_enc]])[0]
    return le_fruit.inverse_transform([pred_enc])[0]

# Test
print(predict_fruit('16-20', 'Female'))  # Mango
print(predict_fruit('10-15', 'Male'))    # Apple
```

## Key Learning

| Concept  | Explanation                     |
| -------- | ------------------------------- |
| Features | Age_Group, Gender (inputs)      |
| Target   | Fruit (output to predict)       |
| Encoding | Must convert text to numbers    |
| Fit      | Model learns the patterns       |
| Predict  | Apply learned rules to new data |

## Condensed Notes

```
1. Create DataFrame with categorical data
2. Encode ALL text columns with LabelEncoder
3. X = input columns (encoded)
4. y = output column (encoded)
5. model.fit(X, y) - trains
6. model.predict([[encoded_inputs]]) - predicts
7. inverse_transform() - get original label back
```

---

# Practical 5: Dataset Preprocessing

## Concept Map
```
Raw Data (messy)
      ↓
  ┌───┴───┐
  ↓       ↓
Explore  Identify Issues
  ↓       ↓
  └───┬───┘
      ↓
  Handle Missing → Encode → Scale → Select Features
      ↓
  Clean Data (ready for ML)
```

## Core Definitions

| Term                  | Definition                                             |
| --------------------- | ------------------------------------------------------ |
| **Missing Value**     | Empty cell. Shows as NaN, None, or blank.              |
| **Imputation**        | Filling missing values with estimated values.          |
| **Mode**              | Most frequent value. Used for categorical columns.     |
| **Correlation**       | Relationship strength between two variables. -1 to +1. |
| **Feature Selection** | Choosing most useful columns. Reduces noise.           |

## Step-by-Step Process

### 1. Load & Explore
```python
import pandas as pd
import numpy as np

df = pd.read_csv('Student.csv')

# Explore
print(df.shape)           # Dimensions
print(df.dtypes)          # Data types
print(df.head())          # First rows
print(df.describe())      # Statistics
print(df.isnull().sum())  # Missing per column
```

### 2. Handle Missing Values
```python
# For numeric columns: use mean
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Marks'] = df['Marks'].fillna(df['Marks'].mean())

# For categorical columns: use mode
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

# Verify
print(df.isnull().sum())  # Should be all 0
```

**Decision guide:**
| Data Type                    | Missing Strategy |
| ---------------------------- | ---------------- |
| Numeric, normal distribution | Mean             |
| Numeric, skewed              | Median           |
| Categorical                  | Mode             |
| Too many missing (>50%)      | Drop column      |

### 3. Encode Categorical Data
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Encode Gender: F→0, M→1
df['Gender_Enc'] = le.fit_transform(df['Gender'])

# Encode Passed: No→0, Yes→1
df['Passed_Enc'] = le.fit_transform(df['Passed'])
```

### 4. Feature Selection
```python
# Method 1: Correlation with target
# Higher absolute correlation = more important
corr = df[['Age', 'Hours_Studied', 'Marks', 'Passed_Enc']].corr()
print(corr['Passed_Enc'].sort_values(ascending=False))

# Method 2: SelectKBest
from sklearn.feature_selection import SelectKBest, f_classif

X = df[['Age', 'Gender_Enc', 'Hours_Studied', 'Marks']]
y = df['Passed_Enc']

selector = SelectKBest(f_classif, k=2)
selector.fit(X, y)
print(dict(zip(X.columns, selector.scores_)))
```

### 5. Final Dataset
```python
# Select only useful columns
final_df = df[['Hours_Studied', 'Marks', 'Passed_Enc']]
final_df.to_csv('Student_Clean.csv', index=False)
```

## Common Traps

| Mistake                        | Problem               | Fix                       |
| ------------------------------ | --------------------- | ------------------------- |
| `fillna(mean)` on categorical  | Wrong type            | Use `mode()[0]`           |
| `mode()` returns Series        | Can't use directly    | Use `mode()[0]`           |
| Encode after split             | Inconsistent encoding | Encode on full data first |
| Include ID/Name in correlation | Meaningless           | Drop non-feature columns  |

## Condensed Notes

```
Explore: shape, dtypes, describe(), isnull().sum()

Missing Values:
- Numeric: fillna(mean()) or fillna(median())
- Categorical: fillna(mode()[0])

Encode: LabelEncoder().fit_transform(column)

Feature Selection:
- Correlation: df.corr()['target']
- SelectKBest: scores features statistically

Output: Clean CSV ready for ML
```

---

# Practical 6: Confusion Matrix & Performance Metrics

## Concept Map
```
Actual Labels    Predicted Labels
      ↓                ↓
      └───────┬────────┘
              ↓
       Compare each pair
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
  Correct             Wrong
  (TP, TN)           (FP, FN)
              ↓
         Confusion Matrix
              ↓
        Calculate Metrics
```

## Core Definitions

| Term                    | Definition                  | Example                           |
| ----------------------- | --------------------------- | --------------------------------- |
| **TP (True Positive)**  | Predicted YES, Actually YES | Said celebrity, was celebrity     |
| **TN (True Negative)**  | Predicted NO, Actually NO   | Said not celebrity, wasn't        |
| **FP (False Positive)** | Predicted YES, Actually NO  | Said celebrity, wasn't (Type I)   |
| **FN (False Negative)** | Predicted NO, Actually YES  | Said not celebrity, was (Type II) |

## Confusion Matrix Layout

```
                    PREDICTED
                  No      Yes
              ┌───────┬───────┐
    ACTUAL No │  TN   │  FP   │
              ├───────┼───────┤
           Yes│  FN   │  TP   │
              └───────┴───────┘
```

**Memory trick:** 
- T/F = Was prediction correct?
- P/N = What did we predict?

## All Metrics with Formulas

| Metric                   | Formula               | In Words                            |
| ------------------------ | --------------------- | ----------------------------------- |
| **Accuracy**             | (TP+TN)/(TP+TN+FP+FN) | Correct / Total                     |
| **Error Rate**           | (FP+FN)/(Total)       | Wrong / Total                       |
| **Precision**            | TP/(TP+FP)            | Of predicted YES, how many correct? |
| **Recall (Sensitivity)** | TP/(TP+FN)            | Of actual YES, how many found?      |
| **Specificity**          | TN/(TN+FP)            | Of actual NO, how many correct?     |
| **F1 Score**             | 2×(P×R)/(P+R)         | Harmonic mean of Precision & Recall |

## Worked Example

```
Actual:    [1, 1, 0, 1, 0, 0, 1, 1, 0, 1]
Predicted: [1, 0, 0, 1, 1, 0, 1, 1, 0, 0]
```

Step 1: Compare each position
```
Position:  0  1  2  3  4  5  6  7  8  9
Actual:    1  1  0  1  0  0  1  1  0  1
Predicted: 1  0  0  1  1  0  1  1  0  0
Result:   TP FN TN TP FP TN TP TP TN FN
```

Step 2: Count
- TP = 4 (positions 0, 3, 6, 7)
- TN = 3 (positions 2, 5, 8)
- FP = 1 (position 4)
- FN = 2 (positions 1, 9)

Step 3: Calculate
```
Accuracy    = (4+3)/(4+3+1+2) = 7/10 = 0.70 = 70%
Precision   = 4/(4+1) = 4/5 = 0.80 = 80%
Recall      = 4/(4+2) = 4/6 = 0.67 = 67%
Specificity = 3/(3+1) = 3/4 = 0.75 = 75%
F1          = 2×(0.80×0.67)/(0.80+0.67) = 0.73
```

## Python Implementation

```python
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

actual = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
predicted = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 0])

# Confusion Matrix
cm = confusion_matrix(actual, predicted)
tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Manual calculations
accuracy = (tp + tn) / len(actual)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)
f1 = 2 * (precision * recall) / (precision + recall)

# Verify with sklearn
print(f"Accuracy:  {accuracy:.2f} == {accuracy_score(actual, predicted):.2f}")
print(f"Precision: {precision:.2f} == {precision_score(actual, predicted):.2f}")
print(f"Recall:    {recall:.2f} == {recall_score(actual, predicted):.2f}")
print(f"F1:        {f1:.2f} == {f1_score(actual, predicted):.2f}")
```

## When to Use Which Metric

| Scenario           | Priority Metric | Why                           |
| ------------------ | --------------- | ----------------------------- |
| Balanced classes   | Accuracy        | Fair overall measure          |
| Imbalanced classes | F1 or AUC       | Accuracy misleading           |
| Spam detection     | Precision       | Don't mark real email as spam |
| Disease detection  | Recall          | Don't miss sick patients      |
| General comparison | F1              | Balances precision & recall   |

## Common Traps

| Mistake                       | Problem              | Fix              |
| ----------------------------- | -------------------- | ---------------- |
| Confusing P/N with actual     | Wrong interpretation | P/N = prediction |
| Wrong sklearn matrix order    | TN,FP,FN,TP          | Use `cm.ravel()` |
| Using accuracy for imbalanced | Misleading           | Use F1           |
| FP vs FN confusion            | Wrong conclusions    | Draw the matrix  |

## Condensed Notes

```
Matrix Layout: [[TN, FP], [FN, TP]]
Extract: tn, fp, fn, tp = cm.ravel()

Formulas:
- Accuracy = (TP+TN) / Total
- Precision = TP / (TP+FP)   → Of predicted+, how many correct?
- Recall = TP / (TP+FN)      → Of actual+, how many found?
- Specificity = TN / (TN+FP) → Of actual-, how many correct?
- F1 = 2PR / (P+R)           → Balance of P and R

sklearn: accuracy_score, precision_score, recall_score, f1_score
```

---

# Quick Reference Card

```python
# Universal imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
df = pd.read_csv('file.csv')

# Explore
df.shape, df.info(), df.describe(), df.isnull().sum()

# Clean
df['col'].fillna(df['col'].mean())  # numeric
df['col'].fillna(df['col'].mode()[0])  # categorical
df['col_enc'] = LabelEncoder().fit_transform(df['col'])

# ML Pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
```
