# FEA Nonlinear Prediction Using Machine Learning

This project uses machine learning to predict nonlinear Finite Element Analysis (FEA) results (stress and strain) based on linear FEA outputs. This approach aims to provide faster and more cost-effective predictions compared to traditional nonlinear FEA simulations.

## Project Description

The goal is to develop a machine learning model that accurately predicts nonlinear FEA outputs (stress and strain) using linear FEA outputs as input features.

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
- **Cost-Effective:** Reduces the computational resources required for complex analyses.
- **Generalizable:** Well-trained models can be applied to similar materials and geometries.

**Cons:**

- **Accuracy:** Heavily dependent on the quality and diversity of training data.
- **Limited Extrapolation:** May struggle with unseen scenarios or extreme conditions.
- **Complexity:** Capturing complex nonlinear behaviors like plasticity can be challenging.

## Difficulty

Moderate to High. Requires domain knowledge and careful data handling.

## Modeling Approach

Supervised learning using regression models.

## Project Structure
EA-Nonlinear-Prediction/
├── README.md             
├── data/
│   └── data_training_2.csv 
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
└── .gitignore            

## Setup and Installation

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone [repository_url]
    cd FEA-Nonlinear-Prediction
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate     # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter Notebook:**

    ```bash
    jupyter notebook notebooks/FEA_Nonlinear_Prediction.ipynb
    ```

5.  **Run the Streamlit application:**

    ```bash
    cd streamlit_app
    streamlit run app.py
    ```

## Files

-   **`data/data_training_2.csv`:** The dataset containing linear and nonlinear FEA results.
-   **`notebooks/FEA_Nonlinear_Prediction.ipynb`:** Jupyter Notebook with data loading, preprocessing, modeling, and evaluation.
-   **`models/`:** Directory containing saved models and scalers.
-   **`streamlit_app/app.py`:** Streamlit application for interactive predictions.
-   **`requirements.txt`:** Project dependencies.
-   **.gitignore:** Files that are ignored by git.

## Notebook Steps

1.  **Load the Data:** Load the dataset using pandas.
2.  **Preprocessing:** Scale features and labels, split data into train and test sets.
3.  **Data Visualization:** Visualize correlations, scatter plots, and distributions.
4.  **Hyperparameter Definition for Model Selection:** Define hyperparameter search spaces.
5.  **Model Selection:** Train and evaluate Random Forest and XGBoost.
6.  **Model Evaluation and Refinement:** Evaluate models, inverse transform predictions, and visualize residuals.
7.  **Evaluate Results:** Summarize and compare model performance.
8.  **Save Models:** Save trained models and scalers.
9.  **Deploy UI via Streamlit:** Create an interactive web app.

## Dependencies

-   pandas
-   scikit-learn
-   matplotlib
-   seaborn
-   xgboost (optional)
-   joblib
-   streamlit

## Usage

1.  Run the Jupyter Notebook to train and evaluate the models.
2.  Use the Streamlit app to make interactive predictions.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Authors

-   Copyright (c) 2025 Nahbruhh - Nguyen Manh Tuan (https://github.com/Nahbruhh)

## License


This project is licensed under the MIT License - see the `LICENSE` file for details.

---

