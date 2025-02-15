# Boston Housing Dataset Classification Using KNN and Decision Tree

## Overview
This project uses the Boston Housing Dataset to classify housing prices based on whether the median value of homes is above or below the median. The algorithm explores several classification techniques, including K-Nearest Neighbors (KNN) and Decision Trees, while evaluating the effect of feature selection and data standardization on model performance.

## Project Structure
1. **Data Loading and Examination**: The Boston Housing dataset is loaded and examined. A new target variable is created based on whether the median home price is above or below the dataset's median.
2. **Modeling**:
   - **KNN Classifier**: Initially, a K-Nearest Neighbors (KNN) classifier is used to classify whether home prices are above or below the median.
   - **Feature Selection**: The importance of each feature is assessed using a decision tree classifier, and less important features are iteratively removed to improve model performance.
   - **Standardization**: The dataset is standardized to observe the effects on the KNN model and its performance.
3. **Evaluation**: The performance of both KNN and Decision Tree models is compared, with cross-validation used to tune model parameters. The best model is applied to the test set for final evaluation.

## Requirements
- Python 3.x
- Required libraries:
  - `scikit-learn`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`

## Data Description
The Boston Housing dataset consists of 506 samples and 13 features. These features include:
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

The target variable (`MEDV`) is transformed into a binary classification target based on whether the value is above or below the median.

## Workflow

### Part 1: Dataset Examination
- The dataset is loaded from `openml` and examined. A new target variable (`new_target`) is created where values above the median are labeled as `1`, and values below the median are labeled as `0`.

### Part 2: KNN Classifier without Standardization
- A K-Nearest Neighbors (KNN) classifier is trained on the dataset without any feature standardization.
- Cross-validation is performed to determine the optimal value of `k` (number of neighbors). The training and validation accuracy are plotted for different values of `k`.

### Part 3: Feature Selection
- The importance of features is evaluated using a Decision Tree classifier. The least important features are iteratively removed based on the Decision Tree’s feature importance scores.
- The KNN model is retrained after removing features to evaluate how the performance changes.

### Part 4: Standardization
- The dataset is standardized using `StandardScaler` to improve the model performance, as distance-based algorithms like KNN are sensitive to feature scaling.
- The KNN model is trained again on the standardized data, and the effect of standardization on performance is analyzed by comparing it to the non-standardized version.

### Part 5: Decision Tree Classifier
- A Decision Tree classifier is trained on the standardized data. Hyperparameters like `max_depth` and `min_samples_split` are tuned using cross-validation to find the best performing tree model.
- The performance of the Decision Tree is compared with the KNN model to evaluate which classifier yields better accuracy.

### Part 6: Test Data Evaluation
- The best performing model (KNN with standardized features and `k=6`) is applied to the test data to evaluate the final accuracy.

## Results
- **Best Model**: The best model was KNN with `k=6`, achieving a test accuracy of **78.29%** after feature selection and standardization.
- **Feature Importance**: Feature selection using the Decision Tree identified the most important features for classification, improving the model's accuracy.

## Conclusion
This project demonstrates how to preprocess data, train classification models (KNN and Decision Tree), and evaluate model performance with cross-validation. It also highlights the impact of feature selection and data standardization on model performance. The KNN model with standardized data achieved the best results on this classification task.
