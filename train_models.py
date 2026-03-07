#!/usr/bin/env python3
"""
=============================================================================
train_models.py — Yulara Solar Digital Twin
=============================================================================
WHAT THIS DOES:
    Runs a multi-model competition for each task:

    POWER  : Prophet vs XGBoost vs Random Forest
             -> Winner = lowest MAE on time-based 80/20 test split
             -> Saves: models/best_power_model.pkl
                      models/power_metrics.json

    PRICE  : Prophet vs XGBoost vs Random Forest
             -> Winner = lowest RMSE on time-based 80/20 test split
             -> Saves: models/best_price_model.pkl
                      models/price_metrics.json

    ANOMALY: Isolation Forest (retrained on full 2016-2026 data)
             -> Saves: models/isolation_forest_model.pkl
                      models/anomaly_scaler.pkl

    REVENUE: Formula-based (not ML) — no overfitting risk
             revenue_per_hour = (power_kw / 1000) × NT_PPA_RATE

    SUMMARY: models/training_summary.json  <- read by app.py dashboard

ROBUSTNESS FEATURES:
    - Time-based train/test split (no data leakage — future not in train)
    - Cross-validation scores reported alongside test scores
    - All metrics computed on HELD-OUT test data (never train data)
    - Models saved with metadata (trained_on date, feature list, metrics)
    - Graceful fallback: if one model fails, others still run
    - Full logging to training.log
    - Reproducible: random_state=42 everywhere
=============================================================================
"""

import pandas as pd
import numpy as np
import os, json, logging, warnings, pickle, time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
POWER_INPUT   = "yulara_power_input.csv"
PRICE_INPUT   = "yulara_price_input.csv"
ANOMALY_INPUT = "yulara_anomaly_input.csv"
MODELS_DIR    = "models"
LOG_FILE      = "training.log"
SUMMARY_OUT   = os.path.join(MODELS_DIR, "training_summary.json")

NT_PPA_RATE   = 250.0    # $/MWh
TEST_FRACTION = 0.20     # last 20% of data = test set (time-based)
CV_FOLDS      = 5        # TimeSeriesSplit folds for cross-validation
RANDOM_STATE  = 42

os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# Force stdout to UTF-8 on Windows
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
log = logging.getLogger(__name__)

# =============================================================================
# HELPERS
# =============================================================================
def time_split(df, target_col='y', test_frac=TEST_FRACTION):
    """
    Time-based train/test split — NO shuffling.
    Test set = last test_frac of chronological data.
    This prevents data leakage (future data never in train).
    """
    n       = len(df)
    n_test  = max(1, int(n * test_frac))
    train   = df.iloc[:-n_test].copy()
    test    = df.iloc[-n_test:].copy()
    log.info(f"  Split: train={len(train):,} rows, test={len(test):,} rows")
    return train, test


def compute_metrics(y_true, y_pred, label=""):
    """Returns MAE, RMSE, R², MAPE for a set of predictions."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # MAPE: skip zeros to avoid division by zero
    mask = y_true > 1.0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    if label:
        log.info(f"    {label:<22} MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return {"mae": round(mae,4), "rmse": round(rmse,4),
            "r2": round(r2,4), "mape": round(float(mape),2) if not np.isnan(mape) else None}


def cv_score(model, X, y, metric='mae'):
    """
    TimeSeriesSplit cross-validation.
    Returns mean and std of metric across folds.
    """
    tscv   = TimeSeriesSplit(n_splits=CV_FOLDS)
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        if metric == 'mae':
            scores.append(mean_absolute_error(y_val, preds))
        else:
            scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    return np.mean(scores), np.std(scores)


def save_model(obj, filename, metadata=None):
    """Save model + optional metadata dict as pickle."""
    path = os.path.join(MODELS_DIR, filename)
    payload = {"model": obj, "metadata": metadata or {}}
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    size_kb = os.path.getsize(path) / 1024
    log.info(f"  [SAVED] Saved: {path}  ({size_kb:.1f} KB)")
    return path


def load_data(path, ts_col='ds'):
    """Load CSV and parse timestamp column."""
    df = pd.read_csv(path)
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)
    log.info(f"  Loaded {path}: {len(df):,} rows, {df.shape[1]} cols")
    log.info(f"  Range: {df[ts_col].min()} -> {df[ts_col].max()}")
    return df


# =============================================================================
# PROPHET WRAPPER
# =============================================================================
def train_prophet(train_df, test_df, extra_regressors=None, freq='H'):
    """
    Train Facebook Prophet model.
    extra_regressors: list of column names to add as additional regressors.
    Returns: (fitted_model, predictions_on_test, metrics_dict)
    """
    try:
        from prophet import Prophet
    except ImportError:
        from fbprophet import Prophet

    regressors = extra_regressors or []

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative',   # better for solar (scales with irradiance)
        changepoint_prior_scale=0.05,        # conservative — avoids overfitting trends
        seasonality_prior_scale=10.0,
        interval_width=0.95,
        uncertainty_samples=200,
    )

    for reg in regressors:
        if reg in train_df.columns:
            m.add_regressor(reg)

    fit_df = train_df[['ds','y'] + [r for r in regressors if r in train_df.columns]].copy()
    m.fit(fit_df)

    # Predict on test set
    future = test_df[['ds'] + [r for r in regressors if r in test_df.columns]].copy()
    forecast = m.predict(future)
    preds = forecast['yhat'].values.clip(min=0)

    metrics = compute_metrics(test_df['y'].values, preds, label="Prophet (test)")
    return m, preds, metrics


# =============================================================================
# XGBOOST WRAPPER
# =============================================================================
def train_xgboost(train_df, test_df, feature_cols, task='power'):
    """
    Train XGBoost regressor with anti-overfitting params.
    Returns: (fitted_model, predictions_on_test, metrics_dict, cv_scores)
    """
    X_train = train_df[feature_cols].values
    y_train = train_df['y'].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df['y'].values

    # Anti-overfitting XGBoost config
    model = XGBRegressor(
        n_estimators      = 500,
        learning_rate     = 0.05,       # slow learning = less overfit
        max_depth         = 5,          # shallow trees
        subsample         = 0.8,        # row subsampling
        colsample_bytree  = 0.8,        # feature subsampling
        reg_alpha         = 0.1,        # L1 regularization
        reg_lambda        = 1.0,        # L2 regularization
        min_child_weight  = 5,          # avoids very small leaf splits
        early_stopping_rounds = 30,     # stop if no improvement
        eval_metric       = 'mae',
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbosity         = 0,
    )

    # Use last 10% of train as validation for early stopping
    n_val     = max(1, int(len(X_train) * 0.10))
    X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
    y_tr, y_val = y_train[:-n_val], y_train[-n_val:]

    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              verbose=False)

    preds   = model.predict(X_test).clip(min=0)
    metrics = compute_metrics(y_test, preds, label="XGBoost (test)")

    # Cross-validation on full train set
    xgb_cv = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    cv_mean, cv_std = cv_score(xgb_cv, X_train, y_train, metric='mae')
    log.info(f"    XGBoost CV MAE     : {cv_mean:.2f} ± {cv_std:.2f}")
    metrics['cv_mae_mean'] = round(cv_mean, 4)
    metrics['cv_mae_std']  = round(cv_std, 4)

    # Feature importance
    fi = dict(zip(feature_cols, model.feature_importances_.tolist()))
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    log.info(f"    Top features: {list(fi_sorted.keys())[:4]}")
    metrics['feature_importance'] = fi_sorted

    return model, preds, metrics


# =============================================================================
# RANDOM FOREST WRAPPER
# =============================================================================
def train_rf(train_df, test_df, feature_cols):
    """
    Train Random Forest regressor.
    Returns: (fitted_model, predictions_on_test, metrics_dict, cv_scores)
    """
    X_train = train_df[feature_cols].values
    y_train = train_df['y'].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df['y'].values

    model = RandomForestRegressor(
        n_estimators  = 300,
        max_depth     = 12,         # controlled depth
        min_samples_leaf = 10,      # avoids tiny leaves = less overfit
        max_features  = 'sqrt',     # random feature subset per split
        n_jobs        = -1,
        random_state  = RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    preds   = model.predict(X_test).clip(min=0)
    metrics = compute_metrics(y_test, preds, label="RandomForest (test)")

    # Cross-validation
    rf_cv = RandomForestRegressor(
        n_estimators=100, max_depth=12, min_samples_leaf=10,
        max_features='sqrt', n_jobs=-1, random_state=RANDOM_STATE)
    cv_mean, cv_std = cv_score(rf_cv, X_train, y_train, metric='mae')
    log.info(f"    RandomForest CV MAE: {cv_mean:.2f} ± {cv_std:.2f}")
    metrics['cv_mae_mean'] = round(cv_mean, 4)
    metrics['cv_mae_std']  = round(cv_std, 4)

    fi = dict(zip(feature_cols, model.feature_importances_.tolist()))
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    log.info(f"    Top features: {list(fi_sorted.keys())[:4]}")
    metrics['feature_importance'] = fi_sorted

    return model, preds, metrics


# =============================================================================
# TASK 1 — POWER FORECASTING COMPETITION
# =============================================================================
def run_power_competition():
    log.info("\n" + "="*65)
    log.info("  TASK 1: POWER FORECASTING COMPETITION")
    log.info("  Models: Prophet vs XGBoost vs RandomForest")
    log.info("="*65)

    df = load_data(POWER_INPUT, ts_col='ds')

    # Features for ML models
    ml_features = ['irradiance','temp_c','wind_speed',
                   'hour','dayofweek','month','is_daylight']
    ml_features = [f for f in ml_features if f in df.columns]

    train, test = time_split(df)
    results = {}

    # --- Prophet ---
    log.info("\n  [1/3] Training Prophet ...")
    t0 = time.time()
    try:
        regressors = ['irradiance','temp_c','wind_speed']
        regressors = [r for r in regressors if r in df.columns]
        prophet_m, prophet_preds, prophet_metrics = train_prophet(
            train, test, extra_regressors=regressors, freq='H')
        prophet_metrics['train_time_s'] = round(time.time()-t0, 1)
        results['Prophet'] = {'model': prophet_m, 'metrics': prophet_metrics,
                              'preds': prophet_preds}
        log.info(f"  [OK] Prophet done in {prophet_metrics['train_time_s']}s")
    except Exception as e:
        log.warning(f"  [FAIL] Prophet failed: {e}")

    # --- XGBoost ---
    log.info("\n  [2/3] Training XGBoost ...")
    t0 = time.time()
    try:
        xgb_m, xgb_preds, xgb_metrics = train_xgboost(train, test, ml_features)
        xgb_metrics['train_time_s'] = round(time.time()-t0, 1)
        results['XGBoost'] = {'model': xgb_m, 'metrics': xgb_metrics,
                              'preds': xgb_preds}
        log.info(f"  [OK] XGBoost done in {xgb_metrics['train_time_s']}s")
    except Exception as e:
        log.warning(f"  [FAIL] XGBoost failed: {e}")

    # --- Random Forest ---
    log.info("\n  [3/3] Training Random Forest ...")
    t0 = time.time()
    try:
        rf_m, rf_preds, rf_metrics = train_rf(train, test, ml_features)
        rf_metrics['train_time_s'] = round(time.time()-t0, 1)
        results['RandomForest'] = {'model': rf_m, 'metrics': rf_metrics,
                                   'preds': rf_preds}
        log.info(f"  [OK] RandomForest done in {rf_metrics['train_time_s']}s")
    except Exception as e:
        log.warning(f"  [FAIL] RandomForest failed: {e}")

    if not results:
        raise RuntimeError("All power models failed!")

    # --- Pick winner by lowest MAE ---
    winner_name = min(results, key=lambda k: results[k]['metrics']['mae'])
    winner      = results[winner_name]

    log.info("\n  --- POWER COMPETITION RESULTS ---")
    for name, r in results.items():
        m  = r['metrics']
        tag = " <- WINNER" if name == winner_name else ""
        log.info(f"    {name:<14} MAE={m['mae']:.2f} kW  "
                 f"RMSE={m['rmse']:.2f}  R²={m['r2']:.4f}{tag}")

    # Save winner
    meta = {
        "task"         : "power_forecasting",
        "winner"       : winner_name,
        "features"     : ml_features,
        "trained_on"   : str(train['ds'].max()),
        "test_from"    : str(test['ds'].min()),
        "train_size"   : len(train),
        "test_size"    : len(test),
        "metrics"      : winner['metrics'],
        "all_results"  : {k: v['metrics'] for k,v in results.items()},
        "timestamp"    : datetime.now().isoformat(),
    }
    save_model(winner['model'], "best_power_model.pkl", meta)

    # Save metrics JSON
    with open(os.path.join(MODELS_DIR, "power_metrics.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    log.info(f"\n  [WINNER] Power winner: {winner_name}  (MAE={winner['metrics']['mae']:.2f} kW)")
    return meta


# =============================================================================
# TASK 2 — PRICE FORECASTING COMPETITION
# =============================================================================
def run_price_competition():
    log.info("\n" + "="*65)
    log.info("  TASK 2: PRICE FORECASTING COMPETITION")
    log.info("  Models: Prophet vs XGBoost vs RandomForest")
    log.info("  NOTE: Only 3 days of AEMO data — models will be simple")
    log.info("="*65)

    df = load_data(PRICE_INPUT, ts_col='ds')

    if len(df) < 20:
        log.warning(f"  Only {len(df)} rows — skipping price competition (need ≥20)")
        return None

    ml_features = ['demand_mw','lag_1h','lag_24h',
                   'rolling_mean_6h','rolling_std_6h',
                   'hour','dayofweek','month']
    ml_features = [f for f in ml_features if f in df.columns]

    # Fill NaN lag features (first rows have no lag)
    df[ml_features] = df[ml_features].fillna(method='bfill').fillna(0)

    train, test = time_split(df)
    results = {}

    # --- Prophet ---
    log.info("\n  [1/3] Training Prophet ...")
    t0 = time.time()
    try:
        regressors = ['demand_mw','rolling_mean_6h']
        regressors = [r for r in regressors if r in df.columns]
        prophet_m, prophet_preds, prophet_metrics = train_prophet(
            train, test, extra_regressors=regressors, freq='30min')
        prophet_metrics['train_time_s'] = round(time.time()-t0, 1)
        results['Prophet'] = {'model': prophet_m, 'metrics': prophet_metrics,
                              'preds': prophet_preds}
        log.info(f"  [OK] Prophet done in {prophet_metrics['train_time_s']}s")
    except Exception as e:
        log.warning(f"  [FAIL] Prophet failed: {e}")

    # --- XGBoost ---
    log.info("\n  [2/3] Training XGBoost ...")
    t0 = time.time()
    try:
        xgb_m, xgb_preds, xgb_metrics = train_xgboost(train, test, ml_features,
                                                        task='price')
        xgb_metrics['train_time_s'] = round(time.time()-t0, 1)
        results['XGBoost'] = {'model': xgb_m, 'metrics': xgb_metrics,
                              'preds': xgb_preds}
        log.info(f"  [OK] XGBoost done in {xgb_metrics['train_time_s']}s")
    except Exception as e:
        log.warning(f"  [FAIL] XGBoost failed: {e}")

    # --- Random Forest ---
    log.info("\n  [3/3] Training Random Forest ...")
    t0 = time.time()
    try:
        rf_m, rf_preds, rf_metrics = train_rf(train, test, ml_features)
        rf_metrics['train_time_s'] = round(time.time()-t0, 1)
        results['RandomForest'] = {'model': rf_m, 'metrics': rf_metrics,
                                   'preds': rf_preds}
        log.info(f"  [OK] RandomForest done in {rf_metrics['train_time_s']}s")
    except Exception as e:
        log.warning(f"  [FAIL] RandomForest failed: {e}")

    if not results:
        log.warning("  All price models failed — skipping")
        return None

    # Pick winner by lowest RMSE (price forecasting standard)
    winner_name = min(results, key=lambda k: results[k]['metrics']['rmse'])
    winner      = results[winner_name]

    log.info("\n  --- PRICE COMPETITION RESULTS ---")
    for name, r in results.items():
        m  = r['metrics']
        tag = " <- WINNER" if name == winner_name else ""
        log.info(f"    {name:<14} RMSE={m['rmse']:.2f} $/MWh  "
                 f"MAE={m['mae']:.2f}  R²={m['r2']:.4f}{tag}")

    meta = {
        "task"         : "price_forecasting",
        "winner"       : winner_name,
        "features"     : ml_features,
        "trained_on"   : str(train['ds'].max()),
        "test_from"    : str(test['ds'].min()),
        "train_size"   : len(train),
        "test_size"    : len(test),
        "metrics"      : winner['metrics'],
        "all_results"  : {k: v['metrics'] for k,v in results.items()},
        "timestamp"    : datetime.now().isoformat(),
        "note"         : "AEMO NSW proxy data — 3 days only. Use NT PPA=$250/MWh for revenue.",
    }
    save_model(winner['model'], "best_price_model.pkl", meta)

    with open(os.path.join(MODELS_DIR, "price_metrics.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    log.info(f"\n  [WINNER] Price winner: {winner_name}  (RMSE={winner['metrics']['rmse']:.2f} $/MWh)")
    return meta


# =============================================================================
# TASK 3 — ANOMALY DETECTION (Isolation Forest)
# =============================================================================
def run_anomaly_training():
    log.info("\n" + "="*65)
    log.info("  TASK 3: ANOMALY DETECTION — Isolation Forest")
    log.info("="*65)

    df = pd.read_csv(ANOMALY_INPUT)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    log.info(f"  Loaded: {len(df):,} daytime rows")

    feature_cols = ['power_kw','irradiance','temp_c','wind_speed','efficiency']
    feature_cols = [f for f in feature_cols if f in df.columns]

    X = df[feature_cols].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    # contamination=0.05 means we expect ~5% anomalies
    model = IsolationForest(
        n_estimators  = 200,
        contamination = 0.05,
        max_samples   = 'auto',
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )
    model.fit(X_scaled)

    # Quick evaluation: flag anomalies and log stats
    scores  = model.decision_function(X_scaled)
    labels  = model.predict(X_scaled)   # -1=anomaly, 1=normal
    n_anom  = (labels == -1).sum()
    pct     = n_anom / len(labels) * 100
    log.info(f"  Anomalies detected : {n_anom:,} ({pct:.1f}% of daytime rows)")
    log.info(f"  Score range        : {scores.min():.4f} -> {scores.max():.4f}")
    log.info(f"  Features used      : {feature_cols}")

    meta = {
        "task"         : "anomaly_detection",
        "model"        : "IsolationForest",
        "features"     : feature_cols,
        "n_train"      : len(df),
        "n_anomalies"  : int(n_anom),
        "anomaly_pct"  : round(pct, 2),
        "contamination": 0.05,
        "timestamp"    : datetime.now().isoformat(),
    }
    save_model(model,  "isolation_forest_model.pkl", meta)
    save_model(scaler, "anomaly_scaler.pkl",
               {"features": feature_cols, "timestamp": datetime.now().isoformat()})

    with open(os.path.join(MODELS_DIR, "anomaly_metrics.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    log.info("  [OK] Isolation Forest trained and saved")
    return meta


# =============================================================================
# TASK 4 — REVENUE (formula, no ML)
# =============================================================================
def compute_revenue_formula():
    log.info("\n" + "="*65)
    log.info("  TASK 4: REVENUE — Formula-based (no ML, no overfitting)")
    log.info("="*65)

    meta = {
        "task"        : "revenue_estimation",
        "method"      : "formula",
        "formula"     : "revenue_per_hour = (power_kw / 1000) * NT_PPA_RATE",
        "NT_PPA_RATE" : NT_PPA_RATE,
        "unit"        : "$/hour",
        "note"        : (
            "Revenue is computed from the winning power model forecast. "
            "No separate revenue ML model — avoids data leakage. "
            "NT PPA rate = $250/MWh (off-grid, not NEM-connected)."
        ),
        "timestamp"   : datetime.now().isoformat(),
    }

    with open(os.path.join(MODELS_DIR, "revenue_config.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    log.info(f"  Formula : revenue_$/hr = (power_kw / 1000) × ${NT_PPA_RATE}/MWh")
    log.info(f"  Example : 500 kW -> ${500/1000*NT_PPA_RATE:.2f}/hr")
    log.info(f"  Example : 1000 kW -> ${1000/1000*NT_PPA_RATE:.2f}/hr")
    log.info("  [OK] Revenue config saved")
    return meta


# =============================================================================
# FINAL SUMMARY
# =============================================================================
def save_summary(power_meta, price_meta, anomaly_meta, revenue_meta):
    log.info("\n" + "="*65)
    log.info("  TRAINING SUMMARY")
    log.info("="*65)

    # Build dashboard-ready metrics (what app.py reads)
    dashboard = {
        "generated_at" : datetime.now().isoformat(),
        "models": {}
    }

    if power_meta:
        pw = power_meta
        dashboard["models"]["power"] = {
            "winner"     : pw["winner"],
            "mae_kw"     : pw["metrics"]["mae"],
            "rmse_kw"    : pw["metrics"]["rmse"],
            "r2"         : pw["metrics"]["r2"],
            "mape_pct"   : pw["metrics"].get("mape"),
            "cv_mae"     : pw["metrics"].get("cv_mae_mean"),
            "all_models" : {k: {"mae": v["mae"], "rmse": v["rmse"], "r2": v["r2"]}
                            for k, v in pw["all_results"].items()},
            "trained_on" : pw["trained_on"],
            "test_from"  : pw["test_from"],
        }
        log.info(f"  Power   : {pw['winner']} | MAE={pw['metrics']['mae']:.2f} kW | "
                 f"R²={pw['metrics']['r2']:.4f}")

    if price_meta:
        pr = price_meta
        dashboard["models"]["price"] = {
            "winner"     : pr["winner"],
            "mae_mwh"    : pr["metrics"]["mae"],
            "rmse_mwh"   : pr["metrics"]["rmse"],
            "r2"         : pr["metrics"]["r2"],
            "all_models" : {k: {"mae": v["mae"], "rmse": v["rmse"], "r2": v["r2"]}
                            for k, v in pr["all_results"].items()},
            "trained_on" : pr["trained_on"],
            "note"       : pr.get("note",""),
        }
        log.info(f"  Price   : {pr['winner']} | RMSE={pr['metrics']['rmse']:.2f} $/MWh | "
                 f"R²={pr['metrics']['r2']:.4f}")

    if anomaly_meta:
        dashboard["models"]["anomaly"] = {
            "model"        : "IsolationForest",
            "n_anomalies"  : anomaly_meta["n_anomalies"],
            "anomaly_pct"  : anomaly_meta["anomaly_pct"],
            "n_train"      : anomaly_meta["n_train"],
        }
        log.info(f"  Anomaly : IsolationForest | {anomaly_meta['n_anomalies']:,} anomalies "
                 f"({anomaly_meta['anomaly_pct']:.1f}%)")

    dashboard["models"]["revenue"] = {
        "method"      : "formula",
        "formula"     : "power_kw / 1000 × $250/MWh",
        "NT_PPA_RATE" : NT_PPA_RATE,
    }

    with open(SUMMARY_OUT, 'w') as f:
        json.dump(dashboard, f, indent=2)
    log.info(f"\n  [SAVED] Summary saved: {SUMMARY_OUT}")

    # Print model comparison table
    log.info("\n  +-----------------------------------------------------+")
    log.info("  |           MODEL COMPETITION RESULTS                 |")
    log.info("  +--------------+----------+----------+---------------+")
    log.info("  | Task         | Winner   | Metric   | Value         |")
    log.info("  +--------------+----------+----------+---------------+")
    if power_meta:
        log.info(f"  | Power (kW)   | {power_meta['winner']:<8} | MAE      | "
                 f"{power_meta['metrics']['mae']:.2f} kW        |")
    if price_meta:
        log.info(f"  | Price ($/MWh)| {price_meta['winner']:<8} | RMSE     | "
                 f"${price_meta['metrics']['rmse']:.2f}/MWh     |")
    log.info(f"  | Anomaly      | IsoForest| Detected | "
             f"{anomaly_meta['anomaly_pct']:.1f}% flagged   |")
    log.info(f"  | Revenue      | Formula  | N/A      | $250/MWh PPA  |")
    log.info("  +--------------+----------+----------+---------------+")


# =============================================================================
# MAIN
# =============================================================================
def main():
    start = time.time()
    log.info("="*65)
    log.info("  YULARA DIGITAL TWIN — MODEL TRAINING")
    log.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("="*65)

    power_meta   = None
    price_meta   = None
    anomaly_meta = None
    revenue_meta = None

    try:
        power_meta = run_power_competition()
    except Exception as e:
        log.error(f"Power training failed: {e}", exc_info=True)

    try:
        price_meta = run_price_competition()
    except Exception as e:
        log.error(f"Price training failed: {e}", exc_info=True)

    try:
        anomaly_meta = run_anomaly_training()
    except Exception as e:
        log.error(f"Anomaly training failed: {e}", exc_info=True)

    try:
        revenue_meta = compute_revenue_formula()
    except Exception as e:
        log.error(f"Revenue config failed: {e}", exc_info=True)

    save_summary(power_meta, price_meta, anomaly_meta, revenue_meta)

    elapsed = time.time() - start
    log.info(f"\n  TOTAL TRAINING TIME: {elapsed/60:.1f} minutes")
    log.info("  DONE. Next step -> update app.py to load from models/ directory")
    log.info("="*65)


if __name__ == "__main__":
    main()