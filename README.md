# 📊 Heart Attack Risk Prediction Using Machine Learning

<p align="center">

**A Comparative Machine Learning Framework for Heart Attack Risk Prediction**

Predicting cardiovascular risk using data preprocessing, multiple machine learning algorithms, hyperparameter optimization, class-imbalance techniques, and cross-validation.

<br>

<a href="#-project-overview">
  <img src="https://img.shields.io/badge/Project-Overview-blue?style=for-the-badge">
</a>
<a href="#-dataset">
  <img src="https://img.shields.io/badge/Dataset-Kaggle-orange?style=for-the-badge">
</a>
<a href="#-machine-learning-models">
  <img src="https://img.shields.io/badge/Models-6-green?style=for-the-badge">
</a>
<a href="#-results">
  <img src="https://img.shields.io/badge/Results-Analysis-purple?style=for-the-badge">
</a>

</p>

---

# 📊 Project Dashboard

| 🔍 Category                     | 📌 Details                                                                      |
| ------------------------------- | ------------------------------------------------------------------------------- |
| **Project Type**                | Machine Learning / Healthcare Analytics                                         |
| **Domain**                      | Cardiovascular Risk Prediction                                                  |
| **Task**                        | Binary Classification                                                           |
| **Target Variable**             | `Heart Attack Risk`                                                             |
| **Programming Language**        | Python                                                                          |
| **Development Environment**     | Jupyter Notebook                                                                |
| **Dataset Source**              | Kaggle                                                                          |
| **Train/Test Split**            | 80% / 20%                                                                       |
| **Cross-Validation**            | 5-Fold Stratified Cross-Validation                                              |
| **Preprocessing**               | Imputation, Scaling, One-Hot Encoding                                           |
| **Class Balancing**             | Class Weighting, SMOTE, Random Undersampling                                    |
| **Hyperparameter Optimization** | RandomizedSearchCV                                                              |
| **Evaluation**                  | Accuracy, Precision, Recall, F1, ROC-AUC, Balanced Accuracy, MCC, Cohen's Kappa |

---

# 📌 Project Overview

Heart attack is a major cardiovascular health problem, and identifying individuals who may be at elevated risk is an important challenge in healthcare analytics.

This project develops and evaluates a **machine learning framework for heart attack risk prediction** using demographic, clinical, behavioral, and lifestyle-related variables.

The project compares multiple supervised machine learning algorithms and investigates how preprocessing, class-imbalance handling, hyperparameter optimization, and cross-validation affect predictive performance.

The analysis is implemented in Python using a complete machine learning pipeline.

The research report identifies factors such as **age, sex, cholesterol, systolic blood pressure, diastolic blood pressure, smoking, and diabetes** as important cardiovascular-related variables considered in the study.

---

# 🎯 Objectives

**The main objectives of this project are to:**

* 🧹 Prepare and preprocess the heart attack risk dataset.
* 🔎 Identify relevant features for predictive modelling.
* 🤖 Implement multiple machine learning classification algorithms.
* ⚖️ Investigate class-imbalance handling techniques.
* 🔧 Optimize model hyperparameters.
* 🔄 Apply stratified 5-fold cross-validation.
* 📊 Compare models using multiple evaluation metrics.
* 🧪 Analyze confusion matrices and classification reports.
* 📈 Examine ROC-AUC performance.
* 🧠 Evaluate the trade-off between overall accuracy and detection of positive heart-attack-risk cases.

---

# 📂 Dataset

**The dataset used in this project is the **Heart Attack Risk Prediction Dataset** obtained from Kaggle.**

### 🔗 Dataset Source

**Kaggle:**
https://www.kaggle.com/datasets/ghnshymsaini/heart-attack-risk-prediction-dataset

The report describes `Heart Attack Risk` as the binary dependent variable, where:

* `0` = Absence of heart attack risk
* `1` = Presence of heart attack risk

The report also identifies demographic and clinical variables including age, sex, cholesterol, systolic blood pressure, diastolic blood pressure, smoking, and diabetes.

---

# 🧬 Dataset Features

The notebook works with a broad set of demographic, clinical, behavioral, lifestyle, and geographic variables.

### 👤 Demographic Features

* `Age`
* `Sex`

### ❤️ Clinical & Health Features

* `Cholesterol`
* `Heart Rate`
* `Diabetes`
* `Family History`
* `Previous Heart Problems`
* `Medication Use`
* `BMI`
* `Triglycerides`
* `Blood Pressure`
* `Systolic_BP`
* `Diastolic_BP`

### 🚬 Lifestyle Features

* `Smoking`
* `Obesity`
* `Alcohol Consumption`
* `Exercise Hours Per Week`
* `Diet`
* `Stress Level`
* `Sedentary Hours Per Day`
* `Physical Activity Days Per Week`
* `Sleep Hours Per Day`

### 💰 Socioeconomic / Geographic Features

* `Income`
* `Country`
* `Continent`
* `Hemisphere`

### 🎯 Target

```text
Heart Attack Risk
```

---

# 🔄 Machine Learning Workflow

**The project follows a structured machine learning pipeline:**

```text
                 ┌──────────────────────┐
                 │   Kaggle Dataset     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Data Preparation   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Feature Processing   │
                 │ • Missing Values     │
                 │ • Scaling            │
                 │ • Encoding           │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Train/Test Split     │
                 │      80 / 20         │
                 └──────────┬───────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │ Multiple ML Algorithms      │
              │ • Logistic Regression       │
              │ • Decision Tree             │
              │ • Random Forest             │
              │ • Gradient Boosting         │
              │ • LightGBM                  │
              │ • MLP Neural Network        │
              └──────────────┬──────────────┘
                             │
                             ▼
                 ┌──────────────────────┐
                 │ Hyperparameter       │
                 │ Optimization         │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Class Imbalance      │
                 │ • Class Weighting    │
                 │ • SMOTE              │
                 │ • Undersampling      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ 5-Fold Stratified CV │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Model Evaluation     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Comparative Results  │
                 └──────────────────────┘
```

---

# 🧹 Data Preprocessing

The notebook implements preprocessing using a `ColumnTransformer` and separate pipelines for numerical and categorical variables.

### Numerical Features

The numerical preprocessing pipeline includes:

* Median imputation
* Standard scaling

```python
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
```

### Categorical Features

Categorical variables are processed using:

* Most-frequent-value imputation
* One-hot encoding

```python
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
```

**This approach allows the preprocessing operations to be integrated directly into the machine learning pipeline.**

---

# 🩺 Blood Pressure Feature Engineering

The original:

```text
Blood Pressure
```

field is divided into two separate numerical variables:

```text
Systolic_BP
Diastolic_BP
```

This transformation allows the two blood-pressure measurements to be independently processed by the machine learning models.

The original `Patient ID` field is also removed because it does not represent a predictive clinical feature.

---

# 🎯 Target Variable

The prediction target is:

```python
TARGET = 'Heart Attack Risk'
```

The problem is treated as a **binary classification task**:

| Value | Meaning              |
| ----: | -------------------- |
|   `0` | No Heart Attack Risk |
|   `1` | Heart Attack Risk    |

---

# 🤖 Machine Learning Models

The project compares six major machine learning algorithms.

## 1. Logistic Regression

A linear classification model used as a baseline for predicting the probability of heart attack risk.

```python
LogisticRegression(max_iter=5000)
```

---

## 2. Decision Tree

A tree-based model capable of learning non-linear relationships and decision rules.

```python
DecisionTreeClassifier(random_state=42)
```

---

## 3. Random Forest

An ensemble of decision trees designed to improve predictive stability and reduce overfitting.

```python
RandomForestClassifier(random_state=42)
```

---

## 4. Gradient Boosting

A sequential ensemble-learning technique that builds multiple weak learners to improve predictive performance.

```python
GradientBoostingClassifier(random_state=42)
```

---

## 5. LightGBM

A gradient-boosting framework designed for efficient tree-based learning.

```python
LGBMClassifier(
    random_state=42,
    verbose=-1
)
```

---

## 6. MLP Neural Network

A Multi-Layer Perceptron neural network used to model potentially complex non-linear relationships.

```python
MLPClassifier(
    max_iter=1000,
    random_state=42
)
```

---

# 🔧 Hyperparameter Optimization

The notebook applies **RandomizedSearchCV** to investigate different hyperparameter combinations.

The optimization uses:

```text
RandomizedSearchCV
```

with:

* 5-fold cross-validation
* 10 sampled parameter combinations
* F1-score as the optimization metric
* `random_state = 42`
* Parallel processing using `n_jobs=-1`

The use of F1-score for tuning is particularly relevant because the project evaluates the ability of models to identify positive heart-attack-risk cases rather than relying only on overall accuracy.

---

# ⚖️ Class Imbalance Handling

The project investigates multiple strategies for handling class imbalance.

## 1. Class Weighting

Models such as Logistic Regression, Decision Tree, and Random Forest are evaluated using balanced class weights.

```python
class_weight='balanced'
```

This approach increases the importance of the underrepresented class during model training.

---

## 2. SMOTE

The project also implements the:

**Synthetic Minority Oversampling Technique (SMOTE)**

SMOTE generates synthetic examples of the minority class to create a more balanced training dataset.

The notebook demonstrates the transformation from:

```text
Original Training Distribution
0 → 4,499
1 → 2,511
```

to:

```text
After SMOTE
0 → 4,499
1 → 4,499
```

---

## 3. Random Undersampling

The project additionally evaluates random undersampling to reduce the number of observations from the majority class.

The resulting training distribution is:

```text
0 → 2,511
1 → 2,511
```

---

# 🔄 Cross-Validation

To examine model stability and generalization, the notebook applies:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The cross-validation procedure evaluates:

* Accuracy
* Precision
* Recall
* F1-score

### Cross-Validation Results

| Metric    | Mean Score |
| --------- | ---------: |
| Accuracy  | **0.6036** |
| Precision | **0.3504** |
| Recall    | **0.1262** |
| F1 Score  | **0.1854** |

The corresponding standard deviations are also calculated in the notebook.

---

# 📏 Evaluation Metrics

The project uses several metrics instead of relying only on accuracy.

| Metric                | Purpose                                                           |
| --------------------- | ----------------------------------------------------------------- |
| **Accuracy**          | Overall proportion of correct predictions                         |
| **Precision**         | Proportion of predicted positive cases that are actually positive |
| **Recall**            | Proportion of actual positive cases correctly detected            |
| **F1 Score**          | Harmonic balance between precision and recall                     |
| **ROC-AUC**           | Measures ranking/discrimination ability                           |
| **Balanced Accuracy** | Accounts for both classes                                         |
| **MCC**               | Measures correlation between predicted and actual classes         |
| **Cohen's Kappa**     | Measures agreement beyond chance                                  |
| **Confusion Matrix**  | Shows TP, TN, FP, and FN                                          |

---

# 📊 Baseline Model Results

The notebook's initial XGBoost implementation produced the following test-set results:

| Metric                           |      Score |
| -------------------------------- | ---------: |
| Accuracy                         | **0.6229** |
| Precision                        | **0.4298** |
| Recall                           | **0.1608** |
| F1 Score                         | **0.2341** |
| Balanced Accuracy                | **0.5209** |
| Matthews Correlation Coefficient | **0.0587** |
| Cohen's Kappa                    | **0.0484** |
| ROC-AUC                          | **0.5233** |

### Classification Performance

For the positive class (`Heart Attack Risk = 1`), the baseline model achieved:

```text
Precision : 0.43
Recall    : 0.16
F1 Score  : 0.23
```

The results demonstrate why multiple evaluation metrics are important for this healthcare classification problem.

---

# 📈 Model Comparison

The notebook generates comparative visualizations for:

* Model Accuracy
* Model F1 Score
* Model ROC-AUC
* Precision
* Recall
* Confusion Matrices
* Tuned-model performance

The research report similarly emphasizes that accuracy alone does not provide a complete assessment for healthcare prediction, particularly when identifying positive cases is important.

---

# 🧪 Confusion Matrix Analysis

Confusion matrices are generated for the evaluated models to examine:

```text
True Negatives
False Positives
False Negatives
True Positives
```

For example, the notebook reports the following LightGBM confusion matrix:

```text
[[1049   76]
 [ 581   47]]
```

This corresponds to:

|                    | Predicted No Risk | Predicted Risk |
| ------------------ | ----------------: | -------------: |
| **Actual No Risk** |              1049 |             76 |
| **Actual Risk**    |               581 |             47 |

The relatively high number of false negatives illustrates the importance of evaluating recall in addition to accuracy.

The report also discusses this behavior and notes that LightGBM identified many non-risk cases while detecting fewer positive risk cases.

---

# 🧠 Hyperparameter Tuning Results

The notebook performs model-specific hyperparameter optimization.

### Decision Tree

```text
max_depth = None
min_samples_split = 5
min_samples_leaf = 2
```

### Random Forest

```text
n_estimators = 200
max_depth = None
min_samples_split = 10
min_samples_leaf = 2
```

### Gradient Boosting

```text
n_estimators = 300
max_depth = 7
learning_rate = 0.05
```

### MLP Neural Network

```text
hidden_layer_sizes = (50,)
activation = tanh
alpha = 0.01
learning_rate_init = 0.001
```

### LightGBM

```text
num_leaves = 50
n_estimators = 300
max_depth = 5
learning_rate = 0.1
```

The notebook's F1-based randomized search identified the **MLP Neural Network** as the highest-scoring tuned model in that particular search, with a cross-validation F1 score of approximately **0.366**.

---

# 🏆 Tuned MLP Evaluation

Using the tuned MLP configuration, the notebook reports:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **0.5203** |
| Precision | **0.3296** |
| Recall    | **0.3280** |
| F1 Score  | **0.3288** |

This demonstrates an important trade-off in the project: a model can have lower overall accuracy while detecting a larger proportion of positive cases.

The research report similarly emphasizes that models with higher sensitivity/recall can be important in healthcare prediction even when their overall accuracy is lower.

---

# 🔍 Key Findings

The analysis produces several important observations.

### 📌 1. Accuracy Alone Is Not Enough

Models with relatively higher accuracy do not necessarily provide the strongest detection of positive heart-attack-risk cases.

### 📌 2. Positive-Class Detection Is Challenging

The baseline evaluation shows considerably lower recall for the positive class.

### 📌 3. F1 Score Provides Additional Insight

F1-score helps evaluate the balance between precision and recall.

### 📌 4. Ensemble Models Show Competitive Accuracy

The analysis report reports approximately:

```text
Logistic Regression → ~64%
Random Forest       → ~64%
Gradient Boosting   → ~63%
LightGBM            → ~62%
Decision Tree       → ~54%
MLP                 → ~54%
```

### 📌 5. Cross-Validation Indicates Limited Positive-Class Recall

The notebook reports an average cross-validation recall of approximately:

```text
12.62%
```

This indicates that identifying positive cases remains a major challenge in the current modelling setup.

---

# 📊 Research Findings

The accompanying analysis report found that individuals with heart attack presence had higher mean values for:

* Total Cholesterol
* Systolic Blood Pressure
* Diastolic Blood Pressure

For example:

| Variable          | No Heart Attack | Heart Attack |
| ----------------- | --------------: | -----------: |
| Total Cholesterol |          198.68 |       221.87 |
| Systolic BP       |          119.39 |       128.25 |
| Diastolic BP      |           79.62 |        85.47 |

These descriptive findings provide additional context for the machine learning analysis.

---

# 📉 ROC-AUC Analysis

The analysis report indicates that the evaluated models produced ROC-AUC values close to **0.50**, suggesting limited discriminatory capability in the current modelling setup.

Reported approximate values include:

| Model               | ROC-AUC |
| ------------------- | ------: |
| Random Forest       |   ~0.52 |
| Gradient Boosting   |   ~0.51 |
| LightGBM            |   ~0.51 |
| Logistic Regression |   ~0.50 |
| Decision Tree       |   ~0.50 |
| MLP                 |   ~0.47 |

---

# 🛠️ Technologies & Libraries

The project uses the following Python ecosystem:

### Programming

* 🐍 Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost
* LightGBM
* Imbalanced-learn

### Visualization

* Matplotlib
* Seaborn

### Notebook Environment

* Jupyter Notebook

### Key Techniques

* Pipeline
* ColumnTransformer
* StandardScaler
* OneHotEncoder
* SimpleImputer
* StratifiedKFold
* RandomizedSearchCV
* SMOTE
* Random Undersampling

---

# 📁 Repository Structure

```text
Heart-Attack-Risk-Prediction/
│
├── 📓 Dataset Implimentation.ipynb
│
├── 📊 heart_attack_prediction_dataset.csv
│
├── 📈 model_comparison_results.csv
│
├── 📄 Analysis Report.docx
│
├── 📜 README.md
│
└── 📁 Results/
    ├── confusion_matrices/
    ├── model_comparison/
    ├── cross_validation/
    └── visualizations/
```

> **Note:** The exact contents of the repository should match the files you have uploaded to GitHub. The `Results/` folders can be added if you decide to save the generated figures separately.

---

# ▶️ How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/heart-attack-risk-prediction.git
```

## 2. Navigate to the Project

```bash
cd heart-attack-risk-prediction
```

## 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm catboost
```

## 4. Open the Notebook

```bash
jupyter notebook
```

Then open:

```text
Dataset Implimentation.ipynb
```

## 5. Place the Dataset

Make sure:

```text
heart_attack_prediction_dataset.csv
```

is located in the same directory as the notebook unless the dataset path is changed in the notebook.

## 6. Run the Notebook

Run the cells sequentially to reproduce:

* Data preprocessing
* Feature transformation
* Train/test splitting
* Model training
* Performance evaluation
* Cross-validation
* Hyperparameter tuning
* Class-balancing experiments
* Visualization
* Model comparison

---

# 📊 Generated Visualizations

The notebook generates several visual analyses, including:

### 📌 Cross-Validation Metrics

Comparison of:

```text
Accuracy
Precision
Recall
F1 Score
```

### 📌 Model Accuracy Comparison

Visual comparison of model accuracy values.

### 📌 F1-Score Comparison

Comparison of F1 performance across models.

### 📌 ROC-AUC Comparison

Comparison of model discrimination performance.

### 📌 Confusion Matrices

Individual confusion matrices for the evaluated machine learning algorithms.

### 📌 Class Distribution

Visualization of class distribution before and after balancing techniques.

---

# 🔬 Research Methodology

The overall methodology can be summarized as:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Numerical & Categorical Preprocessing
      ↓
Train/Test Split
      ↓
Baseline Models
      ↓
Model Comparison
      ↓
Hyperparameter Optimization
      ↓
Class Imbalance Experiments
      ↓
SMOTE / Class Weighting / Undersampling
      ↓
5-Fold Stratified Cross-Validation
      ↓
Performance Evaluation
      ↓
Comparative Analysis
```

The research report describes the framework as incorporating preprocessing, feature selection, class-imbalance handling, model optimization, and cross-validation.

---

# ⚠️ Limitations

The current analysis has several limitations.

### Dataset Representation

The dataset may not fully represent the diversity of heart attack patients across different healthcare environments.

### Positive-Class Detection

The models demonstrate relatively limited recall for positive heart-attack-risk cases.

### Predictive Discrimination

ROC-AUC values close to 0.50 indicate limited discrimination in the current modelling configuration.

### Missing Variables

The research report notes that additional factors such as:

* Physical activity
* Genetic information
* Psychological stress

could provide additional information for future modelling.

### Clinical Application

The models are developed for research and educational machine-learning analysis and should **not be interpreted as a clinical diagnostic system**.

---

# 🚀 Future Improvements

Future versions of this project could investigate:

* 🔬 External validation using independent datasets
* 🧠 Explainable AI techniques such as SHAP
* ❤️ Additional cardiovascular biomarkers
* 🧬 Genetic and family-history information
* 🏃 Physical activity measurements
* 🧘 Psychological and stress-related variables
* 🌍 Larger and more diverse healthcare datasets
* ⚙️ More advanced hyperparameter optimization
* 🤖 Ensemble and stacking approaches
* 📊 Calibration analysis
* 🔍 Feature importance analysis
* 🏥 External clinical validation

The research report specifically recommends improving the diversity of variables and addressing the continuing difficulty of accurately predicting the minority/positive class.

---

# 📚 Research Report

The repository also contains the detailed analysis report describing:

* Problem Statement
* Dataset Collection
* Data Preprocessing
* Proposed Solution
* Experimental Setup
* Descriptive Analysis
* Model Evaluation
* Comparative Results
* Conclusion
* Recommendations
* References

---

# 📌 Important Interpretation

This project demonstrates that **model evaluation in healthcare should not depend exclusively on accuracy**.

A model may achieve relatively high accuracy while failing to identify a substantial proportion of positive cases.

Therefore, this repository evaluates multiple complementary metrics, particularly:

```text
Precision
Recall
F1 Score
ROC-AUC
Balanced Accuracy
Confusion Matrix
```

The report similarly concludes that multiple performance measures provide a more meaningful assessment of healthcare machine-learning models than accuracy alone.

---

# 👨‍💻 Author

**Shahid Azam**

🎓 Computer Science / Machine Learning Researcher
🔬 Research Interests: Machine Learning, Artificial Intelligence, Healthcare Analytics, Data Science

### 🔗 Connect

<p align="center">

<a href="https://github.com/shahidazam2020-oss">
<img src="https://img.shields.io/badge/GitHub-Shahid%20Azam-black?style=for-the-badge&logo=github">
</a>

</p>

---

# ⭐ If You Find This Project Useful

If this project helps you understand machine learning for healthcare prediction, consider:

⭐ **Starring the repository**

🍴 **Forking the project**

🐛 **Opening an issue**

💡 **Sharing suggestions**

🤝 **Contributing improvements**

---

# 📜 Disclaimer

This project is intended for **academic, research, and educational purposes**.

The machine learning models presented in this repository are experimental predictive models and should not be used as a substitute for professional medical diagnosis, clinical decision-making, or medical advice.

---

<p align="center">

### 🫀 Machine Learning • Healthcare Analytics • Cardiovascular Risk Prediction

**Built with Python, Scikit-learn, and Machine Learning**

</p>
