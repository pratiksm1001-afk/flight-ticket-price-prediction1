✈ Flight Ticket Price Prediction Using XGBoost

📌 Project Overview

Flight Ticket Price Prediction is a Machine Learning mini project that predicts the approximate price of a flight ticket based on different flight-related features.

The project uses the XGBoost Regression algorithm to learn patterns from historical flight ticket data and predict ticket prices.

A Tkinter GUI application is provided so users can enter flight details and receive predicted prices through a simple desktop interface.

---

🎯 Objectives

The main objectives of this project are:

- To analyze historical flight ticket data.
- To preprocess categorical and numerical data.
- To train an XGBoost regression model.
- To predict approximate flight ticket prices.
- To compare predicted prices for different airlines.
- To provide an easy-to-use graphical user interface.
- To generate a CSV report containing prediction results.

---

🛠 Technologies Used

Technology| Purpose
Python| Programming language
Pandas| Data processing
NumPy| Numerical operations
Scikit-learn| Data preprocessing and evaluation
XGBoost| Machine Learning prediction
Matplotlib| Graphs and visualization
Tkinter| GUI application
Pickle| Saving trained ML objects

---

🤖 Machine Learning Algorithm

XGBoost Regressor

XGBoost (Extreme Gradient Boosting) is a gradient boosting algorithm used for regression and classification problems.

In this project, "XGBRegressor" is used because the target variable, price, is a continuous numerical value.

The model learns relationships between flight information and ticket prices.

---

📊 Dataset

The project uses the "Clean_Dataset.csv" dataset.

Dataset size

- Records: 300,153
- Columns: 12

Features

The dataset contains:

- Airline
- Flight
- Source City
- Departure Time
- Stops
- Arrival Time
- Destination City
- Class
- Duration
- Days Left

Target

price

The target represents the flight ticket price.

---

🔄 Project Workflow

Dataset
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Categorical Encoding
   ↓
Train/Test Split
   ↓
XGBoost Regression
   ↓
Model Evaluation
   ↓
Save Trained Model
   ↓
GUI Application
   ↓
User Flight Details
   ↓
Price Prediction
   ↓
CSV Report

---

📁 Project Structure

flight ticket price prediction/
│
├── data/
│   └── Clean_Dataset.csv
│
├── models/
│   ├── xgboost_flight_model.pkl
│   ├── ordinal_encoder.pkl
│   └── feature_info.pkl
│
├── outputs/
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   ├── evaluation_results.pkl
│   └── prediction_*.csv
│
├── train_model.py
├── predict_price.py
├── gui.py
├── requirements.txt
└── README.md

---

⚙️ Installation

1. Install Python

Install Python 3.x on your computer.

Check the installation:

python --version

---

2. Install Required Libraries

Open PowerShell or Command Prompt in the project folder:

pip install -r requirements.txt

Required packages:

pandas
numpy
scikit-learn
xgboost
matplotlib

Tkinter is normally included with standard Python installations on Windows.

---

▶️ How to Run the Project

Step 1 — Train the Model

Run:

python train_model.py

This performs:

- Dataset loading
- Data preprocessing
- Train/test splitting
- XGBoost model training
- Model evaluation
- Graph generation
- Model saving

The trained model is saved inside:

models/

---

Step 2 — Test Prediction

Run:

python predict_price.py

Enter:

From City
To City
Class
Days Left

The program displays predicted prices for available airlines.

---

Step 3 — Run GUI

Run:

python gui.py

The graphical application allows the user to enter flight information and predict the ticket price.

---

📈 Model Performance

The trained XGBoost model achieved the following results on the test dataset:

Metric| Result
MAE| ₹1,344.64
RMSE| ₹2,518.01
R² Score| 0.9877

Interpretation

- MAE: The average absolute prediction error was approximately ₹1,344.64.
- RMSE: The RMSE was approximately ₹2,518.01.
- R² Score: The model obtained an R² score of approximately 0.9877 on the test set.

These results describe performance on this particular dataset and test split.

---

🧪 Example Prediction

Example input:

From City: Bangalore
To City: Delhi
Class: Business
Days Left: 3

Example output:

Airline       Predicted Price
--------------------------------
Air_India     ₹47,871.81
Vistara       ₹65,669.30

The application also saves the prediction results as a CSV file in the "outputs" folder.

---

📊 Visualizations

The project generates:

Actual vs Predicted Price

outputs/actual_vs_predicted.png

This graph compares actual ticket prices with prices predicted by the model.

Feature Importance

outputs/feature_importance.png

This graph shows the relative importance assigned by the XGBoost model to the input features.

---

🖥️ GUI Features

The desktop GUI provides:

- Source city selection
- Destination city selection
- Travel class selection
- Days-left input
- Airline-wise predictions
- Historical average price
- Predicted price
- Lowest predicted fare
- CSV report generation
- Input validation

---

⚠️ Limitations

This project predicts prices based on historical data.

The prediction should not be interpreted as a live airline booking price because:

- Airline prices can change dynamically.
- Real-time availability is not connected.
- Taxes and additional fees may change.
- The model depends on the quality and coverage of the dataset.
- Predictions outside the dataset's range may be less reliable.

---

🚀 Future Scope

The project can be improved by:

- Connecting to real-time flight APIs.
- Adding date and time selection.
- Adding departure and arrival airport information.
- Adding more recent flight data.
- Developing a web application.
- Deploying the model online.
- Adding interactive charts.
- Comparing multiple Machine Learning algorithms.
- Adding automatic model retraining.
- Providing price trend analysis.

---

👨‍💻 Project Type

Mini Project – Artificial Intelligence and Machine Learning

Project Title

Flight Ticket Price Prediction Using XGBoost

Domain

Machine Learning / Artificial Intelligence

Algorithm

XGBoost Regression

Interface

Python Tkinter GUI

---

📜 License

This project is developed for educational and academic purposes.For your college submission, this README is enough as the project documentation file. You can later add screenshots of your professional GUI, actual-vs-predicted graph, feature-importance graph, and terminal output to make it more complete.