🌿 Air Quality Health Risk Prediction using Machine Learning


A machine learning-based system that predicts health risk levels (Low, Moderate, High, Very High) using real-time or historical air pollution data. This project uses multiple supervised learning algorithms, evaluates their performance, and deploys the best model using a Streamlit web application.

📌 Project Overview
Air pollution is a major environmental concern that severely impacts public health. Pollutants like PM2.5, PM10, NO₂, CO, SO₂, and O₃ contribute to respiratory, cardiovascular, and long-term health risks.
This project builds an intelligent system that:

Analyzes air quality parameters

Predicts the associated health risk category

Visualizes model performance

Offers an interactive user dashboard

The system uses various ML models, with Random Forest delivering the highest accuracy.

🎯 Objectives
Predict air quality health risk levels based on pollution indicators.

Evaluate multiple ML algorithms (LR, KNN, SVM, RF, GB).

Visualize results using heatmaps, scatter plots, displots, and confusion matrices.

Deploy the best-performing model in an intuitive Streamlit dashboard.

Provide data-driven insights for environmental monitoring.

🗂️ Dataset
The dataset includes 4000+ rows with cleaned and engineered features such as:

CO(GT), NOx(GT), NO₂(GT), O₃(GT), SO₂(GT)

PM2.5, PM10

Temperature, Humidity, Pressure

Wind Speed, Wind Direction

Derived features: Ratio fields, Moving Averages, THI

Target Output: Health_Risk_Level

Source: Custom-processed dataset based on Kaggle air quality data.

🛠️ Technologies Used
Technology	Purpose
Python	Core development
Scikit-learn	Model training
Pandas & NumPy	Data preprocessing
Matplotlib & Seaborn	Visualization
Streamlit	Web app deployment
Pickle	Model saving

🤖 Machine Learning Models Used
Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Random Forest (Best performance)

Gradient Boosting

Performance metrics include:

Accuracy

Confusion Matrix

Scatter Plot (Actual vs Predicted)

Displot Distribution

Feature Importance (RF)

📈 Model Performance
Model	Accuracy
Random Forest	⭐ 98.91%
SVM	96.43%
KNN	94.12%
Logistic Regression	91.67%
Gradient Boosting	92.84%

Random Forest was selected for deployment.

🖥️ Streamlit Web App
The Streamlit UI allows users to:

Input pollutant values using sliders

View predicted health risk level

Visualize outputs with charts

Interact with a clean and modern UI

▶ Run the App Locally
bash
Copy code
streamlit run app.py
📂 Project Structure
bash
Copy code
├── AirQuality_Model.ipynb        # Jupyter notebook with training & analysis
├── app.py                         # Streamlit dashboard
├── scaler.pkl                     # StandardScaler object
├── label_encoder.pkl              # LabelEncoder for target
├── aqi_model.pkl                  # Trained Random Forest model
├── README.md                      # Project documentation
└── requirements.txt               # Dependencies
🚀 How to Run the Project
1️⃣ Clone the repository
bash
Copy code
git clone https://github.com/your-username/air-quality-health-risk.git
cd air-quality-health-risk
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run the Streamlit app
bash
Copy code
streamlit run app.py
🌐 Features
✔ Smart ML-powered prediction

✔ Multi-model comparison

✔ Interactive UI with charts

✔ Easy deployment

✔ Clean and well-documented pipeline


