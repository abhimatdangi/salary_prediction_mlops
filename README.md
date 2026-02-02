# 📊 Salary Prediction MLOps Pipeline

This project is an end-to-end MLOps system that predicts salaries of data professionals using structured HR data. It shows how raw CSV data can be converted into a fully automated machine-learning product that trains models, serves predictions through an API, and continuously monitors performance over time.

The main goal of this project is to help companies and job seekers understand what a fair salary should be based on experience, job role, performance, and other factors. Everything is built in a production-style workflow instead of simple notebooks.

---

## What the system does

The system automatically takes a salary dataset, stores it safely in a database, cleans and prepares it, trains multiple machine-learning models, selects the best one, deploys it as a web API, and then keeps watching the data to make sure the model stays accurate as new data arrives.

The pipeline runs using Apache Airflow, so once it is triggered it executes every step in the correct order without manual work.

---

## Dataset

The project uses the **Salary Prediction of Data Professions** dataset from Kaggle.  
It contains 2,639 employee records and 13 columns including age, gender, job role, department, past experience, performance ratings, leave data, and salary. Each row represents one employee.

This dataset is ideal because it directly connects to the real business problem of salary planning and HR analytics.

---

## System design

The system follows an ELT (Extract-Load-Transform) approach. The raw CSV file is first loaded into MariaDB without any modification. This keeps the original data safe and reproducible. All cleaning, feature engineering, and transformations happen later in Python using Pandas and Scikit-learn.

Each employee remains one row throughout the pipeline, which is important for machine-learning consistency and drift monitoring.

---

## How the pipeline works

First, the CSV file is loaded into MariaDB as a raw table. Column names are normalized and the schema is validated so the structure is always correct.

Next, the data is cleaned and prepared. Duplicate rows are removed, invalid values like infinity are replaced, missing values are handled using statistical imputation, and a new feature called tenure is created from joining date and current date. Categorical fields such as gender, designation, and unit are converted into numerical features using one-hot encoding. The final cleaned dataset is saved and split into training and testing sets.

After preprocessing, the data is validated again using a custom Python checker. This step looks for missing values, infinite values, duplicate rows, invalid salary values, and suspicious columns. A validation report is created and the pipeline stops if serious problems are found.

Then the model training stage begins. Four regression models are trained: Linear Regression, Decision Tree, Random Forest, and Gradient Boosting. Their performance is measured using MAE, RMSE, and R². All experiments, parameters, and metrics are logged in MLflow. The best model, which is Gradient Boosting in this project, is selected and saved as a file.

The selected model is then deployed. It is copied into the API directory and loaded by a FastAPI application. The API exposes endpoints to check health, see required features, and send prediction requests. A user can open Swagger UI in the browser and test salary predictions easily.

After deployment, the system continuously monitors the model. Evidently is used to compare new data with the training data. Data drift checks whether the feature distributions have changed. Concept drift checks whether salary values or predictions are shifting over time. The system generates HTML and CSV reports so changes can be inspected.

---

## Model results

Gradient Boosting was selected as the best model. It achieved:

- Mean Absolute Error (MAE): 4,647  
- Root Mean Squared Error (RMSE): 10,039  
- R² Score: 0.94  

This means the model explains about 94% of the variation in salaries, which is very strong for a real-world dataset.

---

## API usage

The trained model is served using FastAPI.  
Users can send a JSON request with employee features and receive a predicted salary in response. Swagger UI is available at:

