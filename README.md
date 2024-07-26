# DA_PRO
# Airbnb Price Prediction Model Evaluation

## Introduction

This project involves evaluating various machine learning models for predicting Airbnb prices. The evaluation includes different types of models, ranging from traditional regression techniques to advanced neural networks. The goal is to determine which models perform best in terms of accuracy and error metrics.

## Models Evaluated

### Traditional Regression Models
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Generalized Linear Model**

### Ensemble Models
- **Random Forest**
- **Gradient Boosting**
- **AdaBoost**
- **XGBoost**
- **Extra Trees**
- **Decision Tree**

### Neural Network Models
- **Deep Neural Network (DNN)**
- **Convolutional Neural Network (CNN)**
- **Long Short-Term Memory Network (LSTM)**
- **Bidirectional LSTM**
- **Gated Recurrent Unit (GRU)**

## Performance Metrics

The models were evaluated using the following metrics:
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in predictions, without considering their direction.
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values, giving more weight to larger errors.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, providing error magnitudes in the same units as the target variable.
- **R² Score**: Indicates the proportion of variance in the dependent variable predictable from the independent variables.

## Results

### Traditional Regression Models

| Model                        | MAE    | MSE        | RMSE   | R²    |
|------------------------------|--------|------------|--------|-------|
| **Linear Regression**       | 52.85  | 15399.20   | 124.09 | -0.138|
| **Ridge Regression**         | 52.85  | 14679.28   | 121.16 | -0.085|
| **Lasso Regression**         | 52.56  | 15105.97   | 122.91 | -0.117|
| **Generalized Linear Model** | 51.60  | 11389.55   | 106.72 | 0.158 |

### Ensemble Models

| Model                   | MAE    | MSE        | RMSE   | R²    |
|-------------------------|--------|------------|--------|-------|
| **XGBoost**             | 43.47  | 9785.74    | 98.92  | 0.277 |
| **Extra Trees**         | 42.82  | 9962.12    | 99.81  | 0.264 |
| **Random Forest**       | 43.23  | 9875.94    | 99.38  | 0.270 |
| **Gradient Boosting**   | 43.48  | 9924.37    | 99.62  | 0.266 |
| **AdaBoost**            | 53.51  | 11542.53   | 107.44 | 0.147 |
| **Decision Tree**       | 44.72  | 10983.40   | 104.80 | 0.188 |

### Neural Network Models

| Model                   | MAE    | MSE        | RMSE   | R²    |
|-------------------------|--------|------------|--------|-------|
| **LSTM**                | 48.66  | 12229.34   | 110.59 | 0.096 |
| **Bidirectional LSTM**  | 48.25  | 12305.77   | 110.93 | 0.090 |
| **GRU**                 | 49.84  | 13009.76   | 114.06 | 0.038 |
| **DNN**                 | 50.93  | 14135.44   | 118.89 | -0.045|
| **CNN**                 | 59.13  | 14606.07   | 120.86 | -0.080|
| **Gaussian Naive Bayes**| 156.80 | 47961.58   | 219.00 | -2.546|

## Analysis

### Best Performing Models

- **XGBoost**: Achieved the best R² score of 0.277, indicating the highest proportion of variance explained by the model. It also had a low MAE and RMSE, making it the most reliable model for this task.
- **Extra Trees**, **Random Forest**, and **Gradient Boosting**: All performed very well with competitive MAE and RMSE scores. They have relatively high R² scores, suggesting good predictive power.

### Neural Network Models

- **LSTM** and **Bidirectional LSTM** showed better performance compared to standard DNN and CNN models, particularly in terms of MAE and RMSE. This suggests that recurrent networks are better suited for capturing complex patterns in time-series or sequential data.
- **GRU** also performed reasonably well but did not outperform LSTM or Bidirectional LSTM in this case.
- **CNN** and **DNN** models underperformed compared to ensemble methods and other neural networks.

### Underperforming Models

- **Gaussian Naive Bayes**: Showed the worst performance with very high MAE and RMSE, and a negative R² score. This indicates that it is not suitable for this regression task.
- **Linear Regression**, **Ridge Regression**, and **Lasso Regression**: While they are traditional models, they did not perform as well as the top ensemble models and advanced neural networks.


## Conclusion

Based on the evaluation, XGBoost, Extra Trees, Random Forest, and Gradient Boosting are the top-performing models for predicting Airbnb prices. Neural networks, particularly LSTM models, also perform well but do not outperform the top ensemble methods in this case. Gaussian Naive Bayes is not recommended for this regression problem due to its poor performance.

## Future Work

- Further tuning of hyperparameters for the top-performing models.
- Exploring other advanced regression techniques or combining models to improve prediction accuracy.
- Investigating additional features or data transformations that could enhance model performance.

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
