# 🚛 AI-Based Freight Demand Forecasting System

An AI-powered freight demand prediction system designed to improve logistics planning, transportation efficiency, and supply chain management using Machine Learning.

## 📌 Project Overview

Freight transportation plays a critical role in logistics and supply chain management. Traditional freight forecasting methods often rely on manual estimation and static historical assumptions, making them inefficient for handling changing transportation patterns, seasonal demand fluctuations, fuel price changes, and weather impacts.

This project uses **Machine Learning (XGBoost)** to analyze freight-related data and forecast future freight demand accurately. The system helps logistics companies make proactive decisions for better warehouse planning, transportation scheduling, and cost reduction.

---

## 🎯 Problem Statement

Traditional freight forecasting systems face challenges such as:

- Inaccurate demand prediction
- Poor warehouse utilization
- Delayed shipments
- High transportation costs
- Inability to adapt to dynamic logistics conditions
- Lack of real-time prediction support

This system solves these problems using **AI and predictive analytics**.

---

## ✨ Features

✅ Freight demand prediction using Machine Learning  
✅ Data preprocessing and cleaning  
✅ Feature engineering for better accuracy  
✅ XGBoost-based forecasting model  
✅ Real-time prediction support via Flask API  
✅ Interactive frontend dashboard  
✅ Transportation trend analysis  
✅ Logistics planning optimization  
✅ User-friendly interface

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Machine Learning
- XGBoost
- Scikit-learn

### Data Processing
- Pandas
- NumPy

### Backend
- Flask

### Frontend
- HTML
- CSS
- JavaScript

### Development Tools
- Visual Studio Code
- GitHub

---

## 📊 Dataset Used

The project initially uses a **synthetic freight dataset** generated using Python scripts to simulate real-world transportation scenarios.

### Dataset Features:
- Shipment Volume
- Cargo Type
- Transportation Mode
- Fuel Prices
- Weather Conditions
- Seasonal Trends
- Freight Volume

> Future versions can integrate real-world logistics datasets for improved prediction accuracy.

---

## ⚙️ System Architecture

The project follows a modular architecture:

```text
Data Collection
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
XGBoost Model Training
       ↓
Flask Backend API
       ↓
Frontend Dashboard
       ↓
Real-Time Freight Prediction
```

---

## 🔍 Workflow

1. Collect freight-related data
2. Clean and preprocess dataset
3. Perform feature engineering
4. Train XGBoost model
5. Save trained model using Joblib
6. Connect model with Flask backend
7. Send prediction requests
8. Display outputs on dashboard

---

## 📈 Machine Learning Model

The forecasting system uses:

### **XGBoost Regressor**

Why XGBoost?
- High prediction accuracy
- Fast training
- Handles large datasets efficiently
- Reduces overfitting
- Performs well on logistics forecasting

### Evaluation Metrics
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Square Error)**

---

## 📂 Project Structure

```text
Freight-Demand-Forecasting/
│── dataset/
│   ├── freight_data.csv
│
│── model/
│   ├── trained_model.pkl
│
│── backend/
│   ├── app.py
│   ├── requirements.txt
│
│── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│
│── notebooks/
│   ├── preprocessing.ipynb
│   ├── model_training.ipynb
│
│── screenshots/
│
│── README.md
```

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/bharathm1307/Freight-Demand-Forecasting.git
```

### Step 2: Navigate to Project Folder

```bash
cd Freight-Demand-Forecasting
```

### Step 3: Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate virtual environment:

#### Windows
```bash
venv\Scripts\activate
```

#### Mac/Linux
```bash
source venv/bin/activate
```

### Step 4: Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### Start Flask Backend

```bash
python app.py
```

or

```bash
flask run
```

### Open Frontend

Run the HTML file:

```text
index.html
```

Then open in browser:

```text
http://127.0.0.1:5000
```

---

## 📸 Screenshots

### Dashboard View

_Add your dashboard screenshots here_

Example:

```md
![Dashboard](screenshots/dashboard.png)
```

---

## 🧪 Testing

The system was tested using:

- Functional Testing
- Integration Testing
- Performance Testing
- Model Testing
- Frontend Dashboard Testing
- API Testing

All core functionalities passed successfully.

---

## 📌 Results

The system successfully achieved:

✔️ Improved freight demand prediction  
✔️ Better transportation planning  
✔️ Reduced operational inefficiencies  
✔️ Faster prediction generation  
✔️ Real-time forecasting support

---

## 🔮 Future Improvements

- Integration with **real-world freight datasets**
- Live **weather API integration**
- **Traffic analytics**
- GPS tracking integration
- IoT-enabled logistics sensors
- Cloud deployment
- Deep Learning models (**LSTM, Neural Networks**)
- Mobile application support

---

## 👨‍💻 Author

**Bharath M**  
Artificial Intelligence Engineering Student  
Sri Venkateshwara College of Engineering, Bengaluru

GitHub:  
https://github.com/bharathm1307

---

## 📄 License

This project is developed for **academic and internship purposes**.

---

## ⭐ Support

If you found this project useful, consider giving it a **star ⭐ on GitHub**.