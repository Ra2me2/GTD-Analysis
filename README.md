# Global Terrorism Database (GTD) - Number of Kills Prediction

This repository presents an analysis of the Global Terrorism Database (GTD) for predicting the number of kills in terrorist incidents. The project compares the performance of several machine learning and deep learning models to determine the most accurate approach for this task.


## Dataset
The dataset used for this project is the Global Terrorism Database (GTD) . It contains:
- Features such as attack type, weapon type, target type, location, casualty numbers, etc.

## Objective
The primary goal of this project is to develop and compare various machine learning and deep learning models to predict the number of kills in terrorist incidents.


## Models Compared
- **Machine Learning Models**:
  - Linear Regression
  - Ridge Regression
  - Random Forrest
  - Gradient Boosting
- **Deep Learning Models**:
  - Tabnet
  - Multi-Layer Perceptron (MLP)
  - Graph Neural Network (GNN)


## Results
| Model                          | R2    | RMSE  |
|--------------------------------|-------|-------|
| Linear Regression              | 0.41  | 5.50  |
| Ridge Regression               | 0.39  | 5.60  |
| Random Forrest                 | 0.78  | 3.38  |
| Gradient Boosting              | 0.71  | 3.88  |
| Tabnet                         | 0.74  | 3.63  |
| Multi-Layer Perceptron (MLP)   | 0.80  | 3.17  |
| Graph Neural Network (GNN)     | 0.76  | 2.72  |

