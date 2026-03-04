# ☀️ Yulara Solar Digital Twin

The **Yulara Solar Digital Twin** is a cloud-based analytics platform that integrates machine learning with interactive 3D visualizations to optimize solar farm operations. It provides actionable insights through real-time monitoring, 7-day power forecasting, and automated anomaly detection.

🔗 **[Live Deployment](https://musical-tapioca-408587.netlify.app/)** | 🔌 **[Backend API](https://yulara-backend-env.eba-mt7aim3j.us-east-1.elasticbeanstalk.com)**

---

## 🚀 Key Capabilities

***Real-time Monitoring:** Tracks power generation for a 1.8 MW solar installation.
***Predictive Power Forecasting:** 7-day output predictions using the **Prophet** time-series model.
***Financial Analytics:** Electricity spot price prediction via **XGBoost** and revenue estimation.
***Automated Anomaly Detection:** Identifies system faults using **Isolation Forest** with 97% accuracy.
***3D Visualization:** High-fidelity digital twin rendered in **WebGL/Three.js**.

---

## 🏗️ System Architecture

The platform utilizes a **Three-Tier Cloud Architecture** to process over 1 million historical records:

| Layer | Technology | Role |
| --- | --- | --- |
| **Frontend** | React 18, Three.js, Plotly.js | User dashboard and 3D visualization |
| **Backend** | Flask 3.0, Gunicorn, AWS EB | RESTful API and ML model execution |
| **Data** | AWS S3, Boto3 | Streaming storage for 1M+ CSV records |

### Data Flow Architecture

The system streams data from **AWS S3** using `boto3` to minimize memory overhead on cloud instances. ML predictions are executed in-memory and returned as JSON to the React frontend.

---

## 🤖 Machine Learning Performance

The system employs four specialized models to manage different aspects of the solar farm.

| Model | Algorithm | Metric | Accuracy/Score |
| --- | --- | --- | --- |
| **Power Forecast** | Prophet | $R^2$ Score | 98.05% |
| **Price Forecast** | XGBoost | $R^2$ Score | 89.86% |
| **Anomaly Detection** | Isolation Forest | Accuracy | 97.00% |
| **Revenue Model** | Formula-based | Accuracy | 100.00% |

### Anomaly Detection Logic

The **Isolation Forest** model identifies anomalies by isolating samples with shorter path lengths in a tree structure. The anomaly score is calculated as:

$$s(x,n) = 2^{-\frac{E(h(x))}{c(n)}}$$

Where $E(h(x))$ is the average path length of sample $x$ and $c(n)$ is the average path length of a binary tree for dataset size $n$ .

---

## 🛠️ Installation & Setup

### Prerequisites

* Python 3.11.

* Node.js 18.x or higher 

* AWS Account (for S3 and Elastic Beanstalk) 

### Backend Setup

```bash
cd yulara-backend
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py

```

Note: Ensure `.pkl` model artifacts and `yulara_price_input.csv` are in the root directory.

### Frontend Setup

```bash
cd yulara-frontend
npm install
npm start

```

---

## 📡 API Reference

**Base URL:** `https://yulara-backend-env.eba-mt7aim3j.us-east-1.elasticbeanstalk.com` 

* `GET /api/stats`: Retrieve aggregate statistics (average power, efficiency, etc.).

* `GET /api/alerts`: Get real-time system alerts and warnings.

* `POST /api/forecast/prophet`: Generate a 7-day power forecast.

* `POST /api/anomalies`: Detect faults in the most recent 1,000 data points.

---
