<div style="text-align: center; padding: 20px; background-color:rgb(37, 152, 37); box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);">
  <h1>Diabetes Prediction Project</h1>
  <div>
    <a href="https://opensource.org/licenses/MIT">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    </a>
    <a href="https://greensoftware.foundation/">
      <img src="https://img.shields.io/badge/Green%20Code-Certified-brightgreen.svg" alt="Green Code: Certified">
    </a>
  </div>
  <p>A machine learning project for predicting diabetes based on diagnostic measurements using various classification algorithms.</p>
</div>


<div style="box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19); padding: 20px; background-color:rgb(23, 210, 26);">
  <img src="Green Coded Certified.png" alt="Green Coding Certified" width="100%">
</div>



## Overview

This project uses machine learning models to predict the likelihood of diabetes in patients based on diagnostic measurements. The project includes data preprocessing, exploratory data analysis, model training, and evaluation of different machine learning algorithms.

## Project Structure

- `diabetes-dataset.csv` - The dataset containing patient diagnostic measurements
- `Diabetics.ipynb` - Jupyter notebook with the complete analysis and model development
- `KNN_best_model.pkl` - Saved K-Nearest Neighbors model
- `model_accuracies.csv` - CSV file containing performance metrics of different models
- `LICENSE` - MIT License file

## Dataset

The dataset contains the following features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)²)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)
- **Outcome**: Class variable (0 or 1) indicating whether the patient has diabetes

<div style="box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);">
  <table style="width:100%; border-collapse: collapse;">
    <tr>
      <th style="padding: 8px; background-color: #4CAF50; color: white; text-align: left; border: 1px solid #ddd;">Feature</th>
      <th style="padding: 8px; background-color: #4CAF50; color: white; text-align: left; border: 1px solid #ddd;">Mean</th>
      <th style="padding: 8px; background-color: #4CAF50; color: white; text-align: left; border: 1px solid #ddd;">Std</th>
      <th style="padding: 8px; background-color: #4CAF50; color: white; text-align: left; border: 1px solid #ddd;">Min</th>
      <th style="padding: 8px; background-color: #4CAF50; color: white; text-align: left; border: 1px solid #ddd;">Max</th>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">Pregnancies</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">3.85</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">3.37</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">0.00</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">17.00</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">Glucose</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">120.89</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">31.97</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">0.00</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">199.00</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">BMI</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">31.99</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">7.88</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">0.00</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">67.10</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">Age</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">33.24</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">11.76</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">21.00</td>
      <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">81.00</td>
    </tr>
  </table>
</div>

## Methodology

The project follows these steps:

1. **Data Loading and Preprocessing**:
   - Loading the dataset from diabetes-dataset.csv
   - Handling missing values using KNN Imputation
   - Feature scaling using MinMaxScaler

2. **Exploratory Data Analysis**:
   - Statistical summary of features
   - Visualization of feature distributions and relationships

3. **Model Training and Evaluation**:
   - Train-test split
   - Training multiple classification models:
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM)
     - Naive Bayes
     - Decision Tree
     - Random Forest

4. **Model Comparison**:
   - Evaluation using accuracy score, confusion matrix, and classification report

<div style="box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(19, 243, 56, 0.94);">
  <div style="padding: 15px; background-color:rgba(240, 54, 3, 0.87);">
    <h4>Libraries Used</h4>
    <ul>
      <li>Pandas for data manipulation</li>
      <li>NumPy for numerical operations</li>
      <li>Matplotlib and Seaborn for visualization</li>
      <li>Scikit-learn for machine learning algorithms</li>
      <li>Joblib for model persistence</li>
      <li>Rich for formatted console output</li>
    </ul>
  </div>
</div>

## Usage

### Requirements

- Python 3.x
- Required packages:
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn
  - joblib
  - rich

### Running the Project

1. Clone this repository
   ```
   git clone https://github.com/niloydebbarma-code/Diabetics-Prediction.git
   ```

2. Install the required packages:
   ```
   pip install pandas numpy seaborn matplotlib scikit-learn joblib rich
   ```
3. Run the Jupyter notebook:
   ```
   jupyter notebook Diabetics.ipynb
   ```

### Using the Trained Model

To use the saved KNN model for predictions:

```python
import joblib
import numpy as np

# Load the model
model = joblib.load('KNN_best_model.pkl')

# Create a sample input (features should be in the same order as the dataset)
# [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
sample = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Get prediction
prediction = model.predict(sample)
print("Prediction (0: No Diabetes, 1: Diabetes):", prediction[0])
```

## Green Coding Certification

<div style="box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19); padding: 15px; background-color:rgb(5, 57, 228); border-left: 5px solid #4CAF50;">
  <h3 style="color: #2E7D32;">Green Coded Project</h3>
  <p>This project follows green coding principles, optimizing computational resources and minimizing environmental impact:</p>
  <ul>
    <li>Efficient data preprocessing to reduce memory usage</li>
    <li>Optimized machine learning algorithms implementation</li>
    <li>Model persistence to avoid redundant computation</li>
  </ul>
</div>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.