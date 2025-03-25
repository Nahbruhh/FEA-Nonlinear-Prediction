## Authors

-   Copyright (c) 2025 Nahbruhh - Nguyen Manh Tuan (https://github.com/Nahbruhh)

# FEA Nonlinear Prediction Using Machine Learning

This project uses machine learning to predict nonlinear FEA results (stress and strain) based on linear FEA outputs. This approach aims to provide faster and more cost-effective predictions compared to traditional nonlinear FEA simulations.

## Project Description

The goal is to develop a machine learning model that accurately predicts nonlinear FEA outputs using linear FEA outputs as input features.

**Input Features (X):**

- Linear Equivalent von Mises stress
- Linear Maximum Principal stress
- Linear Equivalent Strain

**Output Labels (Y):**

- Nonlinear Equivalent von Mises stress
- Nonlinear Maximum Principal stress
- Nonlinear Total Strain
- Nonlinear Plastic Strain
- Nonlinear Elastic Strain

## Pros and Cons

**Pros:**

- **Faster Predictions:** ML models can significantly reduce prediction time compared to nonlinear FEA simulations.
- **Cost-Effective:** Reduces the computational resources.
- **Generalizable:** Well-trained models can be applied to similar materials and geometries.

**Cons:**

- **Accuracy:** Heavily dependent on the quality and diversity of training data.
- **Limited Extrapolation:** May struggle with unseen scenarios or extreme conditions.
- **Complexity:** Capturing complex nonlinear behaviors like plasticity can be challenging.

## Difficulty



## Modeling Approach

Supervised learning using regression models.

## Project Structure
```
FEA-Nonlinear-Prediction/
├── README.md
├── data/
│   └── data_training.csv
├── notebooks/
│   └── FEA_Nonlinear_Prediction.ipynb
├── models/
│   ├── random_forest.joblib
│   ├── xgboost.joblib 
│   ├── scaler_X.joblib
│   └── scaler_y.joblib
├── streamlit_app/
│   ├── app.py
│   └── requirements.txt
├── requirements.txt
├── LICENSE
└── .gitignore 
```


## Files

-   **`data/data_training.csv`:** The dataset containing linear and nonlinear FEA results.
-   **`notebooks/FEA_Nonlinear_Prediction.ipynb`:** Jupyter Notebook with data loading, preprocessing, modeling, and evaluation.
-   **`models/`:** Directory containing saved models and scalers.
-   **`streamlit_app/app.py`:** Streamlit application for interactive predictions.
-   **`requirements.txt`:** Project dependencies.
-   **.gitignore:** Files that are ignored by git.

## Notebook Steps

1.  **Data Loading:** Loading the dataset containing linear and nonlinear FEA results.
2.  **Data Preprocessing:** Scaling and splitting the data for training and testing.
3.  **Exploratory Data Analysis (EDA):** Visualizing data distributions and correlations.
4.  **Model Selection, Training and Saving:** Choosing, training regression models (Random Forest, XGBoost). And saving the trained models and scalers for deployment.
5.  **Model Evaluation:** Assessing the performance of the trained models.
6.  **Deploy UI via Streamlit:** Create an interactive web app.

## Dependencies

-   pandas
-   scikit-learn
-   matplotlib
-   seaborn
-   xgboost 
-   joblib
-   streamlit

## Usage

1.  Run the Jupyter Notebook to train and evaluate the models.
2.  Use the Streamlit app to make interactive predictions.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.


## License


This project is licensed under the MIT License - see the `LICENSE` file for details.

---

