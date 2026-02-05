# FML Practical Tracker
**Subject:** Fundamental of Machine Learning (DI04016031)  
**Total Practicals:** 11 | **Total Hours:** 30

---

## Practicals Checklist

| #   | Practical                                               | Unit | Hours | Status |
| --- | ------------------------------------------------------- | ---- | ----- | ------ |
| 1   | NumPy & Matplotlib Operations                           | VI   | 4     | ⬜      |
| 2   | Pandas - Data Import/Export (Excel, CSV)                | VI   | 4     | ⬜      |
| 3   | Scikit-learn Introduction                               | VI   | 4     | ⬜      |
| 4   | ML Introduction - Predicting Favorite Fruit             | I    | 2     | ⬜      |
| 5   | Dataset Preprocessing (Student.csv)                     | II   | 2     | ⬜      |
| 6   | Confusion Matrix & Performance Metrics (Celebrity Game) | III  | 2     | ⬜      |
| 7   | K-Nearest Neighbor Classification                       | IV   | 2     | ⬜      |
| 8   | ML Project - Music Genre Prediction (SVM)               | IV   | 4     | ⬜      |
| 9   | Logistic Regression (vgsales.csv)                       | IV   | 2     | ⬜      |
| 10  | Linear Regression Project (home_data.csv)               | IV   | 2     | ⬜      |
| 11  | K-Means Clustering                                      | V    | 2     | ⬜      |

---

## Detailed Practicals

### Practical 1: NumPy & Matplotlib Operations
**Unit:** VI | **Hours:** 4
- [ ] Create arrays using `zeros`, `ones`, `arange`, `linspace`, and `random`.
- [ ] Perform array indexing and slicing (1D and 2D).
- [ ] Perform element-wise mathematical operations `(+, -, *, /, **)`.
- [ ] Calculate statistics `(sum, mean, std, min, max)`.
- [ ] Reshape arrays using `reshape`, `flatten`, and `T`.
- [ ] Create a Line Plot with title, labels, and grid.
- [ ] Create a Bar Chart for comparing categories.
- [ ] Create a Scatter Plot to show correlations.
- [ ] Create a Histogram to visualize distributions.
- [ ] Create a Pie Chart for proportions.

---

### Practical 2: Pandas - Data Import/Export
**Unit:** VI | **Hours:** 4
- [ ] Create DataFrames manually from dictionaries and lists.
- [ ] Load data using `read_csv`, `read_excel`, and `read_json`.
- [ ] Export data using `to_csv` (with `index=False`), `to_excel`, and `to_json`.
- [ ] Explore data: check `shape`, `columns`, `dtypes`, and use `head()`, `info()`, `describe()`.
- [ ] Select columns (Series) and rows using `loc` (label) and `iloc` (position).
- [ ] Filter rows based on single and multiple conditions.
- [ ] Handle missing values: detect with `isnull()`, drop with `dropna()`, fill with `fillna()`.
- [ ] Group data using `groupby` and aggregate functions.

---

### Practical 3: Scikit-learn Introduction
**Unit:** VI | **Hours:** 4
- [ ] Load built-in datasets (e.g., `iris`, `digits`).
- [ ] Split data into training and testing sets using `train_test_split`.
- [ ] Preprocess data: Scale features using `StandardScaler`.
- [ ] Preprocess data: Encode targets using `LabelEncoder` (if needed).
- [ ] Initialize and Train a model (e.g., `KNeighborsClassifier`).
- [ ] Predict outcomes on the test set.
- [ ] Evaluate performance using `accuracy_score`.

---

### Practical 4: ML Introduction - Predicting Favorite Fruit
**Unit:** I | **Hours:** 2
- [ ] Create the dataset (Age Group, Gender, Favorite Fruit) as a DataFrame.
- [ ] Encode categorical columns (Age, Gender, Fruit) using `LabelEncoder`.
- [ ] Define Features `X` (Age, Gender) and Target `y` (Fruit).
- [ ] Train a `DecisionTreeClassifier` on the data.
- [ ] Implement a function to predict fruit for new inputs (e.g., '16-20', 'Female').
- [ ] Verify predictions against expected logic.

---

### Practical 5: Dataset Preprocessing (Student.csv)
**Unit:** II | **Hours:** 2
- [ ] Load `Student.csv` and explore structure/stats.
- [ ] Check for missing values using `isnull().sum()`.
- [ ] Fill missing numeric values with the column `mean`.
- [ ] Fill missing categorical values with the column `mode`.
- [ ] Encode categorical features (e.g., Gender, Passed) to numbers.
- [ ] Perform Feature Selection (e.g., Correlation Matrix or SelectKBest).
- [ ] Export the cleaned and processed data to `Student_Clean.csv`.

---

### Practical 6: Confusion Matrix & Performance Metrics
**Unit:** III | **Hours:** 2
- [ ] Create/Simulate Actual vs Predicted data arrays.
- [ ] Generate the Confusion Matrix using `confusion_matrix`.
- [ ] Extract TN, FP, FN, TP values.
- [ ] Calculate Metrics Manually: Accuracy, Precision, Recall, F1, Specificity.
- [ ] Calculate Metrics using Sklearn: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`.
- [ ] Compare manual calculations with Sklearn results.

---

### Practical 7: K-Nearest Neighbor Classification
**Unit:** IV | **Hours:** 2
- Implement KNN algorithm
- Predict class labels of test data
- Provide explicit training and test data

---

### Practical 8: ML Project - Music Genre Prediction (SVM)
**Unit:** IV | **Hours:** 4
- Use music.csv dataset
- Split data into input (age, gender) and output (genre)
- Apply SVM model from sklearn
- Calculate model accuracy
- Generate synthetic dataset for verification

---

### Practical 9: Logistic Regression (vgsales.csv)
**Unit:** IV | **Hours:** 2
- Import vgsales.csv from Kaggle
- Find rows/columns, basic info using `describe`
- Apply logistic regression for price prediction

---

### Practical 10: Linear Regression Project (home_data.csv)
**Unit:** IV | **Hours:** 2
- Import home_data.csv from Kaggle
- Explore data using `head`, `info`, `describe`
- Plot house price vs area using matplotlib
- Apply linear regression model
- Evaluate using suitable metrics

---

### Practical 11: K-Means Clustering
**Unit:** V | **Hours:** 2
- Implement K-means algorithm
- Cluster a set of points
- Provide explicit training and test data

---

## Projects (Choose One)
> Duration: 14-16 hours | Submit by end of semester

1. **BigMart Sales Prediction** - Build regression model to predict sales of 1559 products across 10 outlets
2. **Stock Price Prediction** - Predict future stock values using ML
