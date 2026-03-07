# ?? Yulara Solar Digital Twin

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-lightgrey?logo=flask)](https://flask.palletsprojects.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![AWS](https://img.shields.io/badge/AWS-Elastic%20Beanstalk-FF9900?logo=amazonaws)](https://aws.amazon.com)
[![Netlify](https://img.shields.io/badge/Netlify-Deployed-00C7B7?logo=netlify)](https://netlify.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A **cloud-based digital twin platform** for the Yulara Solar Farm (Northern Territory, Australia).

This system integrates:

- Machine learning forecasting  
- Electricity price prediction  
- Anomaly detection  
- Interactive 3D visualization  

into a **scalable cloud architecture** deployed on AWS and Netlify.

The platform demonstrates how **AI + cloud infrastructure can optimize renewable energy operations** through predictive analytics and real-time monitoring.

---

# ?? Live Deployment

| Service | URL |
|------|------|
| Frontend (Netlify) | https://musical-tapioca-408587.netlify.app |
| Backend API (AWS Elastic Beanstalk) | http://yulara-backend-env.eba-mt7aim3j.us-east-1.elasticbeanstalk.com |

---

# ??? System Architecture

The platform follows a **data ? ML ? API ? visualization pipeline**.
DATA INGESTION
(AEMO NEM CSV + Solar Farm + Weather Data)
¦
?
DATA PIPELINE
preprocess.py
feature engineering
yulara_master.csv
¦
?
ML TRAINING
train_models.py

• Prophet ? Solar power forecasting
• XGBoost ? Electricity price prediction
• Isolation Forest ? Anomaly detection
¦
?
BACKEND — Flask REST API (AWS Elastic Beanstalk)

GET /api/stats
GET /api/alerts
POST /api/forecast/prophet
POST /api/forecast/price
POST /api/anomalies
POST /api/historical
POST /api/simulation/3d
POST /api/predict/revenue
¦
?
FRONTEND — React Dashboard (Netlify)

• Power Forecast Dashboard
• Price Forecast Dashboard
• Anomaly Detection Panel
• 3D Solar Farm Simulation
• Revenue Prediction Tool
---

# ?? Repository Structure


solar-digital-twin/

app.py
Flask REST API serving all /api endpoints

preprocess.py
Data preprocessing and feature engineering pipeline

train_models.py
Trains ML models and stores artifacts

download_price_data.py
Downloads electricity price data from AEMO

image_gen.py
Generates model evaluation plots

requirements.txt
Python dependencies

models/
best_power_model.pkl
best_price_model.pkl
isolation_forest_model.pkl
anomaly_scaler.pkl

power_metrics.json
price_metrics.json
anomaly_metrics.json
revenue_config.json
training_summary.json

yulara-frontend/

src/
    api/
        index.js
    components/
    App.jsx

public/
    _redirects

package.json

---

# ?? Machine Learning Model Performance

| Model | Task | R² Score | Accuracy | MAE |
|------|------|------|------|------|
| Prophet | Solar Power Forecast | 0.9905 | 99.05% | 24.33 kW |
| XGBoost | Electricity Price Prediction | 0.8986 | 89.86% | 14.73 $/MWh |
| Isolation Forest | Anomaly Detection | 0.97 | 97% | ~3% anomaly rate |

---

# ?? Local Setup

## Prerequisites

- Python 3.10+
- Node.js 18+
- pip

---

## Clone Repository
git clone https://github.com/HUNT-001/solar-digital-twin.git

cd solar-digital-twin


---

# Backend Setup


pip install -r requirements.txt

python preprocess.py
python train_models.py

python app.py


Backend runs at:


http://localhost:5000


---

# Frontend Setup


cd yulara-frontend
npm install
npm start


Frontend runs at:


http://localhost:3000


---

# ?? API Reference

| Method | Endpoint | Description | Example Payload |
|------|------|------|------|
| GET | /api/stats | Solar farm statistics | — |
| GET | /api/alerts | Active anomaly alerts | — |
| POST | /api/forecast/prophet | Solar power forecast | { "hours": 24 } |
| POST | /api/forecast/price | Electricity price forecast | { "hours": 24 } |
| POST | /api/anomalies | Retrieve anomaly records | { "n_records": 100 } |
| POST | /api/historical | Historical power data | { "hours": 168 } |
| POST | /api/simulation/3d | Data for 3D solar simulation | { "hours": 24 } |
| POST | /api/predict/revenue | Revenue prediction | { "power_kw": 500 } |

---

# ?? Cloud Deployment

## Backend — AWS Elastic Beanstalk


zip -r yulara-deploy.zip app.py requirements.txt models/
eb deploy
---

## Frontend — Netlify


cd yulara-frontend
npm run build


Upload the **build/** folder to Netlify.

---

# ?? Dataset Sources

| Dataset | Source | Description |
|------|------|------|
| Solar Generation | Desert Gardens Solar Farm | 15-minute power generation data |
| Weather Data | Bureau of Meteorology | Temperature, irradiance, humidity |
| Electricity Price | AEMO NEM | 5-minute settlement prices |

Raw datasets are not included due to size (~400MB).

---

# ?? Technology Stack

| Layer | Technology |
|------|------|
| Frontend | React, Plotly.js, Three.js |
| Backend | Python, Flask |
| ML Forecasting | Prophet |
| ML Price Prediction | XGBoost |
| ML Anomaly Detection | Isolation Forest |
| Cloud | AWS Elastic Beanstalk |
| Hosting | Netlify |
| Data Processing | Pandas, NumPy |

---

# ?? Project Context

This project demonstrates a **cloud-based digital twin architecture for renewable energy infrastructure**, integrating:

- Machine learning forecasting  
- anomaly detection  
- real-time APIs  
- interactive visualization  
- scalable cloud deployment

The system showcases how **AI-driven analytics can improve monitoring and optimization of solar energy systems.**

