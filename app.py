from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
import boto3
from collections import deque

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))

def p(rel):
    return os.path.join(BASE, rel)

# ── Health check (REQUIRED for Elastic Beanstalk)
@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

# ── Model loading
print("Loading models...")
try:
    prophet_power  = joblib.load(p('models/best_power_model.pkl'))
    price_model    = joblib.load(p('models/best_price_model.pkl'))
    iso_forest     = joblib.load(p('models/isolation_forest_model.pkl'))
    anomaly_scaler = joblib.load(p('models/anomaly_scaler.pkl'))
    print("✅ All models loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    prophet_power = price_model = iso_forest = anomaly_scaler = None

# ── Lazy tail-loader from S3
_df = None

def get_df():
    global _df
    if _df is not None:
        return _df
    print("Loading master dataset from S3 (tail mode)...")
    try:
        s3  = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
        obj = s3.get_object(Bucket="yulara-data-bucket", Key="yulara_master.csv")
        usecols   = ["timestamp", "power_kw", "irradiance", "temp_c", "wind_speed", "efficiency", "site"]
        tail_rows = int(os.getenv("TAIL_ROWS", "50000"))
        chunksize = int(os.getenv("CHUNK_SIZE", "200000"))
        keep = deque()
        kept = 0
        for chunk in pd.read_csv(obj["Body"], usecols=usecols, chunksize=chunksize):
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
            keep.append(chunk)
            kept += len(chunk)
            while kept > tail_rows and len(keep) > 1:
                dropped = len(keep[0])
                keep.popleft()
                kept -= dropped
        _df = pd.concat(list(keep), ignore_index=True).dropna(subset=["timestamp"])
        _df = _df.sort_values("timestamp").reset_index(drop=True)
        print(f"✅ Loaded tail df rows: {len(_df):,}")
        print(f"   Range: {_df['timestamp'].min()} → {_df['timestamp'].max()}")
    except Exception as e:
        print(f"⚠️ S3 load failed: {e}")
        _df = pd.DataFrame(columns=["timestamp","power_kw","irradiance","temp_c","wind_speed","efficiency","site"])
    return _df

# ── Price CSV (safe load)
price_df = pd.DataFrame(columns=['timestamp'])
try:
    price_df = pd.read_csv(p('yulara_price_input.csv'))
    print(f"price_df columns: {price_df.columns.tolist()}")
    time_col = next(
        (c for c in ['timestamp','Timestamp','date','Date','time','Time','datetime','ds']
         if c in price_df.columns), None
    )
    if time_col:
        if time_col != 'timestamp':
            price_df = price_df.rename(columns={time_col: 'timestamp'})
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], errors='coerce')
        price_df = price_df.sort_values('timestamp').reset_index(drop=True)
        print(f"✅ price_df loaded: {len(price_df)} rows")
    else:
        print(f"⚠️ No timestamp col found. Cols: {price_df.columns.tolist()}")
        price_df['timestamp'] = pd.NaT
except Exception as e:
    print(f"⚠️ price_df load failed: {e}")
    price_df = pd.DataFrame(columns=['timestamp'])

# ── Pre-warm the S3 dataframe at startup (avoids first-request timeout)
get_df()

# ─────────────────────────────────────────────
@app.route('/api/stats', methods=['GET'])
def get_stats():
    df     = get_df()
    recent = df.tail(1000)
    return jsonify({
        'avg_power'     : float(recent['power_kw'].mean()),
        'max_power'     : float(recent['power_kw'].max()),
        'min_power'     : float(recent['power_kw'].min()),
        'total_records' : len(df),
        'avg_efficiency': float(recent['efficiency'].mean()),
        'sites'         : df['site'].unique().tolist(),
        'date_range'    : {
            'start': df['timestamp'].min().strftime('%Y-%m-%d'),
            'end'  : df['timestamp'].max().strftime('%Y-%m-%d')
        }
    })

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    recent    = get_df().tail(100)
    avg_power = recent['power_kw'].mean()
    max_power = recent['power_kw'].max()
    avg_temp  = recent['temp_c'].mean()
    avg_eff   = recent['efficiency'].mean()
    alerts    = []
    if avg_power < 80:
        alerts.append({'type': 'warning', 'severity': 'medium',
            'message': f'Below optimal power output: {avg_power:.1f} kW',
            'timestamp': datetime.now().isoformat()})
    if avg_temp > 35:
        alerts.append({'type': 'warning', 'severity': 'high',
            'message': f'High temperature: {avg_temp:.1f}°C',
            'timestamp': datetime.now().isoformat()})
    elif avg_temp > 30:
        alerts.append({'type': 'info', 'severity': 'low',
            'message': f'Temperature: {avg_temp:.1f}°C',
            'timestamp': datetime.now().isoformat()})
    if avg_eff < 0.12:
        alerts.append({'type': 'critical', 'severity': 'high',
            'message': f'Critical efficiency: {avg_eff*100:.1f}%',
            'timestamp': datetime.now().isoformat()})
    elif avg_eff < 0.16:
        alerts.append({'type': 'warning', 'severity': 'medium',
            'message': f'Low efficiency: {avg_eff*100:.1f}%',
            'timestamp': datetime.now().isoformat()})
    if not alerts:
        alerts.append({'type': 'info', 'severity': 'low',
            'message': 'All systems normal',
            'timestamp': datetime.now().isoformat()})
    return jsonify({'alerts': alerts, 'total': len(alerts),
                    'timestamp': datetime.now().isoformat()})

@app.route('/api/forecast/prophet', methods=['POST'])
def prophet_forecast():
    data  = request.json or {}
    hours = data.get('hours', 168)
    try:
        future   = prophet_power.make_future_dataframe(periods=hours, freq='h')
        forecast = prophet_power.predict(future)
        result   = forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(hours).copy()
        result['yhat']       = result['yhat'].clip(lower=0)
        result['yhat_lower'] = result['yhat_lower'].clip(lower=0)
        result['yhat_upper'] = result['yhat_upper'].clip(lower=0)
        return jsonify({
            'timestamps'  : result['ds'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
            'forecast'    : result['yhat'].tolist(),
            'lower'       : result['yhat_lower'].tolist(),
            'upper'       : result['yhat_upper'].tolist(),
            'avg_forecast': float(result['yhat'].mean()),
            'model'       : 'Prophet Power'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast/price', methods=['POST'])
def price_forecast():
    data  = request.json or {}
    hours = data.get('hours', 168)
    try:
        last = price_df.iloc[-1] if len(price_df) > 0 else pd.Series()
        base = datetime.now()
        rows = []
        # Replace the rows building block with:
        for i in range(hours):
            t = base + timedelta(hours=i)
            h = t.hour
            dow = t.weekday()
            rows.append([
                float(last.get('lag_1h',   100)),
                float(last.get('lag_24h',  100)),
                float(last.get('lag_48h',  100)),
                float(last.get('lag_168h', 100)),
                float(last.get('rolling_mean_6h',  100)),
                float(last.get('rolling_mean_24h', 100)),
                float(last.get('rolling_std_6h',   20.0)),
                float(last.get('rolling_std_24h',  20.0)),
                h, dow, t.month,
                int(dow >= 5),
                np.sin(2*np.pi*h/24),
                np.cos(2*np.pi*h/24),
                np.sin(2*np.pi*dow/7),
                np.cos(2*np.pi*dow/7)
            ])
        X_fut      = pd.DataFrame(rows)
        model_cols = price_model.feature_names_in_ if hasattr(price_model, 'feature_names_in_') else X_fut.columns.tolist()
        preds      = np.clip(price_model.predict(X_fut[model_cols]), 0, 15000)
        timestamps = [(base + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M') for i in range(hours)]
        return jsonify({
            'timestamps': timestamps,
            'forecast'  : preds.tolist(),
            'lower'     : (preds * 0.75).tolist(),
            'upper'     : (preds * 1.25).tolist(),
            'avg_price' : float(preds.mean()),
            'model'     : 'RandomForest Price'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/revenue', methods=['POST'])
def predict_revenue():
    data     = request.json or {}
    power_kw = float(data.get('power_kw', 500))
    ppa_rate = 250.0
    rev_hr   = (power_kw / 1000) * ppa_rate
    return jsonify({
        'revenue_usd_per_hour': round(rev_hr, 2),
        'daily_revenue'       : round(rev_hr * 24, 2),
        'monthly_revenue'     : round(rev_hr * 24 * 30, 2),
        'model'               : 'Formula (PPA $250/MWh)',
        'inputs'              : data
    })

@app.route('/api/anomalies', methods=['POST'])
def detect_anomalies():
    data      = request.json or {}
    n_records = data.get('n_records', 1000)
    try:
        recent   = get_df().tail(n_records).copy()
        features = ['power_kw','irradiance','temp_c','wind_speed','efficiency']
        X        = recent[features].dropna()
        X_scaled    = anomaly_scaler.transform(X)
        predictions = iso_forest.predict(X_scaled)
        scores      = iso_forest.score_samples(X_scaled)
        anomaly_indices = np.where(predictions == -1)[0]
        anomalies = []
        for idx in anomaly_indices[:100]:
            row = X.iloc[idx]
            anomalies.append({
                'power_kw'     : float(row['power_kw']),
                'irradiance'   : float(row['irradiance']),
                'temp_c'       : float(row['temp_c']),
                'efficiency'   : float(row['efficiency']),
                'anomaly_score': float(scores[idx])
            })
        return jsonify({
            'total_checked'  : len(X),
            'anomalies_found': len(anomaly_indices),
            'anomaly_rate'   : float(len(anomaly_indices)/len(X)*100),
            'anomalies'      : anomalies,
            'model'          : 'Isolation Forest'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical', methods=['POST'])
def get_historical():
    data   = request.json or {}
    hours  = data.get('hours', 168)
    df     = get_df()
    recent = df.tail(min(hours, len(df)))
    return jsonify({
        'timestamps': recent['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
        'power'     : recent['power_kw'].tolist(),
        'irradiance': recent['irradiance'].tolist(),
        'efficiency': recent['efficiency'].tolist(),
        'temp'      : recent['temp_c'].tolist()
    })

@app.route('/api/comparison', methods=['GET'])
def model_comparison():
    return jsonify({
        'models'  : ['Prophet Power','XGBoost Power','RF Power','RF Price','Isolation Forest'],
        'mae'     : [24.33, 56.07, 88.93, 37.99, None],
        'r2'      : [0.9905, 0.9138, 0.7602, 0.5593, None],
        'accuracy': [0.9905, 0.9138, 0.7602, 0.5593, 0.95]
    })

if __name__ == '__main__':
    print("\n🚀 Starting Flask API...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
