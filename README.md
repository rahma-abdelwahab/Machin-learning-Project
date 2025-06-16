# 🧠 Human Activity Recognition using MHEALTH Dataset

This machine learning project leverages the **MHEALTH dataset** to classify human physical activities using various ML models. It covers **data preprocessing, training multiple classifiers, hyperparameter tuning, and performance evaluation**.

---

## 📌 Project Summary

This notebook walks through the process of:

* Loading and exploring human activity sensor data
* Preprocessing the dataset (handling missing values, feature scaling, encoding)
* Training and evaluating multiple ML models
* Performing hyperparameter tuning for performance optimization
* Comparing model results to select the most effective algorithm

---

## 🚀 Features

### 🔹 1. Data Loading & Initial Exploration

* Loads `mhealth_raw_data.csv` into a DataFrame
* Displays data structure using:

  * `.shape`, `.info()`, and `.describe()`

### 🔹 2. Data Preprocessing

* **Missing Value Handling**: Drops rows with missing values
* **Scaling**: Standardizes numerical features using `StandardScaler`
* **Encoding**: Transforms categorical features like `subject` using `LabelEncoder`
* **Data Splitting**: Splits data into training and testing sets using `train_test_split`

### 🔹 3. Machine Learning Models

Implements and evaluates multiple ML classifiers:

* **K-Nearest Neighbors (KNN)**

  * Evaluates with accuracy, precision, recall, F1-score, and confusion matrix

* **Logistic Regression**

  * Simple baseline model for classification

* **Support Vector Machine (SVM)**

  * Hyperparameter tuning via `GridSearchCV`
  * Uses optimized parameters for evaluation

* **Multi-Layer Perceptron (MLP) / Neural Network**

  * Built using `MLPClassifier` from `sklearn` or Keras API
  * Effective in handling complex non-linear patterns

### 🔹 4. Model Evaluation & Comparison

* Calculates:

  * Accuracy
  * Precision
  * Recall
  * F1-score
* Highlights **Model\_1** as the top-performing model (Accuracy: **0.9537**)
* Emphasizes importance of:

  * Feature engineering
  * Data normalization
  * Hyperparameter tuning

---

## 📊 Dataset

* **File Name:** `mhealth_raw_data.csv`
* **Description:** Raw sensor data from wearable devices used for recognizing physical activities like walking, running, etc.
* **Attributes:** Includes data from accelerometers, gyroscopes, and ECG sensors.

---

## 🛠️ Technologies Used

* **Language:** Python 3.x
* **Core Libraries:**

  * `pandas`, `numpy`
  * `scikit-learn`
  * `matplotlib`, `seaborn`
  * `keras` (for deep learning models)
  * `mlxtend` (for visualization utilities)
  * `tqdm` (for progress bars)

---

## ▶️ How to Run

1. **Install Python 3.x**
2. **Install Dependencies**

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn keras mlxtend tqdm
   ```
3. **Place the Dataset**

   * Ensure `mhealth_raw_data.csv` is in the same directory as the notebook
4. **Open the Notebook**

   * Use Jupyter Lab, Jupyter Notebook, or Google Colab
   * Open `Machin_learning_Project.ipynb`
5. **Run Cells**

   * Execute each cell sequentially to replicate the workflow


