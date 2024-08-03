
# Airbnb Price Prediction Model Evaluation


## Introduction


This project involves evaluating various machine learning models for predicting Airbnb prices. The evaluation includes different types of models, ranging from regression techniques to advanced neural networks. As part of the analysis, Exploratory Data Analysis (EDA) was conducted to gather valuable insights that address our research questions. The goal is to determine which models perform best in terms of accuracy and error metrics.



## Models Evaluated

## Tree-based and Probalistic models
- **Decision Tree**
- **Extra Trees**
- **Gaussian Naive Bayes**

### Regression Models
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Generalized Linear Model**

### Ensemble Models
- **Random Forest**
- **Gradient Boosting**
- **XGBoost**
- **AdaBoost**


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

### Tree-Based and Probabilistic Models Comparison

|**Model**                  | **MAE**  | **MSE** | **RMSE** |  **R² Score** |
|----------------------|----------------------|---------------------|--------------------------|------------|
| Extra Trees          | 42.82                | 9962.12             | 99.81                    | 0.26       |
| Decision Tree        | 44.72                | 10983.40            | 104.80                   | 0.19       |
| Gaussian Naive Bayes | 156.80               | 47961.58            | 219.00                   | -2.55      |


### Regression Models

| **Model**                | **MAE**    | **MSE**       | **RMSE**  | **R² Score** |
|--------------------------|------------|---------------|-----------|--------------|
| Linear Regression          | 52.85    | 15399.20  | 124.09   | -0.14    |
| Generalized Linear Model   | 51.60    | 11388.78  | 106.72   | 0.16     |
| Ridge Regression           | 52.85    | 14679.28  | 121.16   | -0.09    |
| Lasso Regression           | 52.56    | 15105.96  | 122.91   | -0.12    |

### Ensemble Models

| **Model**                | **MAE**    | **MSE**       | **RMSE**  | **R² Score** |
|--------------------------|------------|---------------|-----------|--------------|
| Random Forest          | 43.47 | 9898.75  | 99.49  | 0.27    |
| Gradient Boosting      | 43.48 | 9924.37  | 99.62  | 0.27    |
| XGBoost                | 43.47 | 9785.74  | 98.92  | 0.28    |
| AdaBoost               | 53.51 | 11542.53 | 107.44 | 0.15    |



### Deep learning based Models
| **Model**              | **MAE** | **MSE**    | **RMSE** | **R² Score** |
|------------------------|---------|------------|----------|--------------|
| **DNN**                | 49.55   | 13995.21   | 118.30   | -0.03        |
| **CNN**                | 63.41   | 15308.48   | 123.73   | -0.13        |
| **LSTM**               | 48.48   | 13220.95   | 114.98   | 0.02         |
| **Bidirectional LSTM** | 48.83   | 12545.84   | 112.01   | 0.07         |
| **GRU**                | 48.81   | 13551.13   | 116.41   | -0.00        |

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
- **Linear Regression**, **Ridge Regression**, and **Lasso Regression**: While they are Regression models, they did not perform as well as the top ensemble models and advanced neural networks.


## Conclusion

Based on the evaluation, XGBoost, Extra Trees, Random Forest, and Gradient Boosting are the top-performing models for predicting Airbnb prices. Neural networks, particularly LSTM models, also perform well but do not outperform the top ensemble methods in this case. Gaussian Naive Bayes is not recommended for this regression problem due to its poor performance.

## Future Work

- Further tuning of hyperparameters for the top-performing models.
- Exploring other advanced regression techniques or combining models to improve prediction accuracy.
- Investigating additional features or data transformations that could enhance model performance.


## References

### Tree-Based and Probabilistic Models
- **Decision Tree**: [Scikit-learn Documentation - Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
- **Extra Trees**: [Scikit-learn Documentation - Extra Trees](https://scikit-learn.org/stable/modules/ensemble.html#extra-trees)
- **Gaussian Naive Bayes**: [Scikit-learn Documentation - Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes)

### Regression Models
- **Linear Regression**: [Scikit-learn Documentation - Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
- **Ridge Regression**: [Scikit-learn Documentation - Ridge Regression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)
- **Lasso Regression**: [Scikit-learn Documentation - Lasso Regression](https://scikit-learn.org/stable/modules/linear_model.html#lasso)
- **Generalized Linear Model**: [Scikit-learn Documentation - Generalized Linear Models](https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-model)

### Ensemble Models
- **Random Forest**: [Scikit-learn Documentation - Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- **Gradient Boosting**: [Scikit-learn Documentation - Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- **XGBoost**: [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- **AdaBoost**: [Scikit-learn Documentation - AdaBoost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)

### Neural Network Models
- **Deep Neural Network (DNN)**: [TensorFlow Documentation - Keras](https://www.tensorflow.org/guide/keras)
- **Convolutional Neural Network (CNN)**: [TensorFlow Documentation - CNN](https://www.tensorflow.org/guide/keras/functional)
- **Long Short-Term Memory Network (LSTM)**: [TensorFlow Documentation - LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- **Bidirectional LSTM**: [TensorFlow Documentation - Bidirectional LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional)
- **Gated Recurrent Unit (GRU)**: [TensorFlow Documentation - GRU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)
