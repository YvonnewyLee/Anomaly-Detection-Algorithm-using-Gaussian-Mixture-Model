# Anomaly Detection with Boston Housing Dataset using Gaussian Mixture Model

## Overview
This project applies an anomaly detection algorithm using a Gaussian Mixture Model (GMM) on the Boston Housing Dataset. The dataset provides housing data with features such as crime rates, average number of rooms, and property tax rates, aimed at predicting the median value of homes.

## Project Structure
1. **Data Loading and Examination**: The Boston Housing dataset is loaded and examined to understand its structure.
2. **Feature Engineering**: The target variable is categorized based on its median value to identify whether housing prices are above or below the median.
3. **Modeling**:
   - **KNN Classifier**: Initially, a K-Nearest Neighbors (KNN) classifier is used without standardization to assess model performance.
   - **Feature Selection**: Using a decision tree classifier, less important features are iteratively removed, and KNN models are evaluated.
   - **Standardization**: The data is standardized and the effect of standardization on the model is analyzed.
4. **Evaluation**: A comparison of KNN and decision tree models is done to determine the best performing model. The best model is then applied to test data.
   
## Requirements
- Python 3.x
- Required libraries:
  - `scikit-learn`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/your-username/repository-name.git
   ```
   
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   ```
   jupyter notebook Anomaly_Detection.ipynb
   ```

## Data Description
The dataset contains 506 samples with 13 features. These features include:
- `CRIM`: Crime rate
- `ZN`: Proportion of residential land zoned for large lots
- `INDUS`: Proportion of non-retail business acres
- `CHAS`: Charles River dummy variable (1 if tract bounds river)
- `NOX`: Nitrogen oxide concentration
- `RM`: Average number of rooms per dwelling
- `AGE`: Proportion of owner-occupied units built before 1940
- `DIS`: Weighted distances to Boston employment centers
- `RAD`: Index of accessibility to radial highways
- `TAX`: Property tax rate
- `PTRATIO`: Pupil-teacher ratio
- `B`: Proportion of black residents
- `LSTAT`: Proportion of lower-status population
- `MEDV`: Median value of homes (target variable)

## Anomaly Detection Process

### Part 1: Dataset Examination
- The dataset is examined and transformed into a Pandas DataFrame.
- The target variable (`MEDV`) is split into two categories: above or below the median value.

### Part 2: KNN Classifier without Standardization
- A KNN model is trained on the dataset without standardization.
- Cross-validation is used to determine the optimal number of neighbors (`k`).

### Part 3: Feature Selection
- Decision trees are used to assess the importance of features.
- Features are removed iteratively, and the KNN classifier is retrained to evaluate the effect on accuracy.

### Part 4: Standardization
- Standardization of the features is performed using `StandardScaler`.
- The model is evaluated again with standardized data, and the performance is compared to the non-standardized version.

### Part 5: Decision Tree Classifier
- A decision tree classifier is trained and tuned to compare its performance with KNN.
- Hyperparameters (`max_depth`, `min_samples_split`) are tuned using cross-validation.

### Part 6: Test Data Evaluation
- The best performing model (KNN with standardized features) is evaluated on the test data.

## Results
- **Best Model**: The best model was KNN with `k=6` after feature selection and standardization, achieving a test accuracy of **78.29%**.
- **Feature Importance**: It was found that the top two features were the most influential in the classification.

## Conclusion
This project demonstrates anomaly detection using Gaussian Mixture Models, evaluates multiple machine learning models, and showcases the impact of feature selection and standardization on model performance.
