#!/usr/bin/env python3
"""
preprocess.py — Yulara Solar Digital Twin
Derives power_kw from Active_Energy_Delivered_Received cumulative meter.
"""

import pandas as pd
import numpy as np
import os, glob, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR         = "."
MASTER_OUT       = "yulara_master.csv"
POWER_OUT        = "yulara_power_input.csv"
PRICE_OUT        = "yulara_price_input.csv"
ANOMALY_OUT      = "yulara_anomaly_input.csv"
TIMESTAMP_FMT    = "%Y-%m-%d %H:%M:%S"
NT_PPA_RATE      = 250.0    # $/MWh
SOLAR_CAPACITY   = 1800.0   # kW total
PANEL_EFFICIENCY = 0.80

SITE_CAPACITY = {
    "Desert Gardens": 1100.0,
    "Sails"         :  700.0,
}

# Columns to read from sails_desert_gardens.csv
POWER_COLS_NEEDED = [
    "timestamp",
    "Active_Energy_Delivered_Received",
    "Global_Horizontal_Radiation",
    "Weather_Temperature_Celsius",
    "Wind_Speed",
]

WEATHER_COLS_NEEDED = [
    "timestamp",
    "Global_Horizontal_Radiation",
    "Weather_Temperature_Celsius",
    "Wind_Speed",
]

# =============================================================================
# HELPERS
# =============================================================================
def parse_ts(series):
    return pd.to_datetime(series, format='mixed', dayfirst=True)

def clean_num(df, col, lo=None, hi=None):
    if col not in df.columns:
        return df
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].ffill().bfill().fillna(0)
    if lo is not None: df[col] = df[col].clip(lower=lo)
    if hi is not None: df[col] = df[col].clip(upper=hi)
    return df

def add_time_feats(df, col='timestamp'):
    t = df[col]
    df['hour']        = t.dt.hour
    df['dayofweek']   = t.dt.dayofweek
    df['month']       = t.dt.month
    df['is_daylight'] = ((t.dt.hour >= 5) & (t.dt.hour <= 19)).astype(int)
    return df

# =============================================================================
# STEP 1 — LOAD POWER DATA
# =============================================================================
def load_power_data():
    print("\n[1] Loading sails_desert_gardens.csv ...")
    path = os.path.join(DATA_DIR, "sails_desert_gardens.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")

    header    = pd.read_csv(path, nrows=0)
    available = header.columns.tolist()

    # Build usecols from what actually exists in the file
    usecols = [c for c in POWER_COLS_NEEDED if c in available]
    if 'site' in available:
        usecols.append('site')

    has_energy = "Active_Energy_Delivered_Received" in available
    print(f"  Columns found  : {usecols}")
    print(f"  Energy column  : {'YES' if has_energy else 'NO — will use irradiance fallback'}")

    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    # Rename
    df.rename(columns={
        'Active_Energy_Delivered_Received': 'energy_kwh_cum',
        'Global_Horizontal_Radiation'     : 'irradiance',
        'Weather_Temperature_Celsius'     : 'temp_c',
        'Wind_Speed'                      : 'wind_speed',
    }, inplace=True)

    df['timestamp'] = parse_ts(df['timestamp'])
    df = clean_num(df, 'irradiance', lo=0, hi=1500)
    df = clean_num(df, 'temp_c',     lo=-10, hi=60)
    df = clean_num(df, 'wind_speed', lo=0, hi=50)

    # --- Detect site ---
    # Desert Gardens cumulative energy ~152,000 kWh (large farm)
    # Sails          cumulative energy   ~2,267 kWh (small farm)
    if 'site' not in df.columns:
        if 'energy_kwh_cum' in df.columns:
            df['energy_kwh_cum'] = pd.to_numeric(
                df['energy_kwh_cum'], errors='coerce').fillna(0)
            df['site'] = np.where(
                df['energy_kwh_cum'] > 50000, 'Desert Gardens', 'Sails')
            print("  Site detection : energy threshold > 50,000 kWh = Desert Gardens")
        else:
            df['site'] = 'Desert Gardens'

    # --- Derive power_kw: irradiance × site_capacity × efficiency ---
    # Active_Energy_Delivered_Received resets periodically (billing meter)
    # so energy deltas are near-zero between resets — not reliable for power.
    # Irradiance-based model is physically correct:
    #   power_kw = (irradiance W/m² / 1000) × capacity_kW × efficiency
    #   Peak: DG=(1000/1000)*1100*0.80=880kW  Sails=(1000/1000)*700*0.80=560kW
    cap_map        = df['site'].map(SITE_CAPACITY).fillna(700.0)
    df['power_kw'] = (df['irradiance'] / 1000.0) * cap_map * PANEL_EFFICIENCY
    df['power_kw'] = df['power_kw'].clip(lower=0)
    if 'energy_kwh_cum' in df.columns:
        df.drop(columns=['energy_kwh_cum'], inplace=True)
    print('  Power method   : irradiance × capacity × efficiency ✅')


    df = df.sort_values('timestamp').reset_index(drop=True)
    daytime = df[df['timestamp'].dt.hour.between(6, 18)]['power_kw']

    print(f"  Rows           : {len(df):,}")
    print(f"  Columns        : {list(df.columns)}")
    print(f"  Range          : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Sites          : {df['site'].unique()}")
    print(f"  power_kw range : {df['power_kw'].min():.1f} → {df['power_kw'].max():.1f} kW")
    print(f"  Daytime mean   : {daytime.mean():.1f} kW")
    return df

# =============================================================================
# STEP 2 — LOAD WEATHER DATA
# =============================================================================
def load_weather_data():
    print("\n[2] Loading weather_data.csv ...")
    path = os.path.join(DATA_DIR, "weather_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")

    header    = pd.read_csv(path, nrows=0)
    available = header.columns.tolist()
    usecols   = [c for c in WEATHER_COLS_NEEDED if c in available]
    print(f"  Columns found : {usecols}")

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df.rename(columns={
        'Global_Horizontal_Radiation' : 'irradiance_w',
        'Weather_Temperature_Celsius' : 'temp_c_w',
        'Wind_Speed'                  : 'wind_speed_w',
    }, inplace=True)

    df['timestamp'] = parse_ts(df['timestamp'])
    df = clean_num(df, 'irradiance_w', lo=0, hi=1500)
    df = clean_num(df, 'temp_c_w',     lo=-10, hi=60)
    df = clean_num(df, 'wind_speed_w', lo=0, hi=50)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    print(f"  Rows  : {len(df):,}")
    print(f"  Range : {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df

# =============================================================================
# STEP 3 — LOAD PRICE DATA
# =============================================================================
def load_price_data():
    print("\n[3] Loading AEMO price data ...")
    candidates = glob.glob(os.path.join(DATA_DIR, "NEMPRICEANDDEMAND_SA1*.csv"))
    if not candidates:
        print("  WARNING: No AEMO file found. Skipping price pipeline.")
        return None
    path = candidates[0]

    col_map = {
        "SETTLEMENTDATE"        : "timestamp",
        "Settlement Date"       : "timestamp",
        "RRP"                   : "spot_price",
        "Spot Price ($/MWh)"    : "spot_price",
        "TOTALDEMAND"           : "demand_mw",
        "Scheduled Demand (MW)" : "demand_mw",
    }

    header    = pd.read_csv(path, nrows=0)
    available = header.columns.tolist()
    usecols   = [c for c in col_map.keys() if c in available]
    print(f"  Columns found : {usecols}")

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df.rename(columns=col_map, inplace=True)

    if 'Type' in df.columns:
        df = df[df['Type'] == 'ACTUAL']

    df['timestamp'] = parse_ts(df['timestamp'])
    df = clean_num(df, 'spot_price', lo=0, hi=15000)
    df = clean_num(df, 'demand_mw',  lo=0)
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"  Rows  : {len(df):,}")
    print(f"  Range : {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df

# =============================================================================
# STEP 4 — BUILD MASTER
# =============================================================================
def build_master(power_df, weather_df):
    print("\n[4] Building master dataset ...")

    power_df['timestamp']   = power_df['timestamp'].dt.round('5min')
    weather_df['timestamp'] = weather_df['timestamp'].dt.round('5min')

    master = pd.merge(power_df, weather_df, on='timestamp', how='left')

    # Fill gaps using weather file backup
    if 'irradiance_w' in master.columns:
        master['irradiance'] = master['irradiance'].fillna(master['irradiance_w'])
        master['temp_c']     = master['temp_c'].fillna(master['temp_c_w'])
        master['wind_speed'] = master['wind_speed'].fillna(master['wind_speed_w'])
        master.drop(columns=['irradiance_w','temp_c_w','wind_speed_w'], inplace=True)

    for col in ['irradiance','temp_c','wind_speed']:
        master[col] = master[col].ffill().bfill().fillna(0)

    master = add_time_feats(master)

    # Efficiency = actual / theoretical
    theoretical    = (master['irradiance'] / 1000.0) * SOLAR_CAPACITY
    master['efficiency'] = np.where(
        theoretical > 0,
        (master['power_kw'] / theoretical).clip(0, 1), 0.0)

    # Total power per timestamp (sum of both sites)
    total = (master.groupby('timestamp')['power_kw']
                   .sum().reset_index()
                   .rename(columns={'power_kw':'total_power_kw'}))
    master = pd.merge(master, total, on='timestamp', how='left')

    # Revenue at NT PPA rate
    master['ppa_revenue_hr'] = (master['total_power_kw'] / 1000.0) * NT_PPA_RATE * (5/60)

    # Final column order
    cols = ['timestamp','site','power_kw','total_power_kw','irradiance',
            'temp_c','wind_speed','hour','dayofweek','month','is_daylight',
            'efficiency','ppa_revenue_hr']
    master = master[[c for c in cols if c in master.columns]]
    master = master.sort_values('timestamp').reset_index(drop=True)

    print(f"  Shape  : {master.shape}")
    print(f"  Columns: {list(master.columns)}")
    print(f"  Range  : {master['timestamp'].min()} → {master['timestamp'].max()}")
    print(f"  Nulls  : {master.isnull().sum().sum()}")
    return master

# =============================================================================
# STEP 5a — POWER INPUT (hourly, Prophet/XGBoost/RF)
# =============================================================================
def build_power_input(master):
    print("\n[5a] Building yulara_power_input.csv ...")

    # Sum both sites per 5-min timestamp first
    per5 = master.groupby('timestamp').agg(
        total_power_kw = ('power_kw',   'sum'),
        irradiance     = ('irradiance', 'mean'),
        temp_c         = ('temp_c',     'mean'),
        wind_speed     = ('wind_speed', 'mean'),
    ).reset_index()

    # Resample 5-min → hourly
    hourly = (per5.set_index('timestamp')
                  .resample('1H').mean()
                  .reset_index()
                  .rename(columns={'timestamp':'ds','total_power_kw':'y'}))

    hourly = add_time_feats(hourly, col='ds')
    hourly = hourly.dropna(subset=['y']).sort_values('ds').reset_index(drop=True)

    cols = ['ds','y','irradiance','temp_c','wind_speed',
            'hour','dayofweek','month','is_daylight']
    hourly = hourly[[c for c in cols if c in hourly.columns]]

    print(f"  Shape         : {hourly.shape}")
    print(f"  Range         : {hourly['ds'].min()} → {hourly['ds'].max()}")
    print(f"  y mean        : {hourly['y'].mean():.1f} kW")
    print(f"  y max         : {hourly['y'].max():.1f} kW")
    print(f"  Daytime mean  : {hourly[hourly['hour'].between(6,18)]['y'].mean():.1f} kW")
    return hourly

# =============================================================================
# STEP 5b — PRICE INPUT (30-min, Prophet/SARIMA/XGBoost)
# =============================================================================
def build_price_input(price_df):
    if price_df is None:
        return None
    print("\n[5b] Building yulara_price_input.csv ...")

    keep = ['timestamp','spot_price'] + (['demand_mw'] if 'demand_mw' in price_df.columns else [])
    df   = price_df[keep].copy()

    df = (df.set_index('timestamp').resample('30min').mean().reset_index())
    df = df.dropna(subset=['spot_price'])

    df['lag_1h']          = df['spot_price'].shift(2)
    df['lag_24h']         = df['spot_price'].shift(48)
    df['rolling_mean_6h'] = df['spot_price'].rolling(12, min_periods=1).mean()
    df['rolling_std_6h']  = df['spot_price'].rolling(12, min_periods=1).std().fillna(0)

    df = add_time_feats(df)
    df.rename(columns={'timestamp':'ds','spot_price':'y'}, inplace=True)

    cols = ['ds','y','demand_mw','lag_1h','lag_24h',
            'rolling_mean_6h','rolling_std_6h','hour','dayofweek','month']
    df = df[[c for c in cols if c in df.columns]].sort_values('ds').reset_index(drop=True)

    print(f"  Shape : {df.shape}")
    print(f"  Range : {df['ds'].min()} → {df['ds'].max()}")
    print(f"  y mean: ${df['y'].mean():.2f}/MWh  max: ${df['y'].max():.2f}/MWh")
    return df

# =============================================================================
# STEP 5c — ANOMALY INPUT (5-min daytime)
# =============================================================================
def build_anomaly_input(master):
    print("\n[5c] Building yulara_anomaly_input.csv ...")
    cols = ['timestamp','power_kw','irradiance','temp_c',
            'wind_speed','efficiency','hour','site']
    df = master[[c for c in cols if c in master.columns]].copy()
    df = df[(df['hour'].between(5, 19)) & (df['irradiance'] > 50)]
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"  Shape : {df.shape}")
    print(f"  Range : {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df

# =============================================================================
# STEP 6 — ALIGNMENT REPORT
# =============================================================================
def print_report(master, power_in, price_in, anomaly_in):
    print("\n" + "="*65)
    print("  TIMESTAMP ALIGNMENT REPORT")
    print("="*65)
    for fname, df, col in [
        ("yulara_master.csv",        master,     'timestamp'),
        ("yulara_power_input.csv",   power_in,   'ds'),
        ("yulara_price_input.csv",   price_in,   'ds'),
        ("yulara_anomaly_input.csv", anomaly_in, 'timestamp'),
    ]:
        if df is None:
            print(f"\n  {fname}: SKIPPED"); continue
        ts   = pd.to_datetime(df[col])
        freq = pd.infer_freq(ts[:200]) if len(ts) >= 3 else "?"
        print(f"\n  {fname}")
        print(f"    Start    : {ts.min()}")
        print(f"    End      : {ts.max()}")
        print(f"    Records  : {len(df):,}")
        print(f"    Frequency: {freq}")
        print(f"    Nulls    : {ts.isnull().sum()}")
    print("\n  NOTE: Power & Price do NOT share timestamps.")
    print("        Revenue = power_kw/1000 × $250/MWh (NT PPA rate).")
    print("="*65)

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*65)
    print("  YULARA — PREPROCESSING PIPELINE (LEAN)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65)

    power_df      = load_power_data()
    weather_df    = load_weather_data()
    price_df      = load_price_data()

    master        = build_master(power_df, weather_df)
    power_input   = build_power_input(master)
    price_input   = build_price_input(price_df)
    anomaly_input = build_anomaly_input(master)

    print_report(master, power_input, price_input, anomaly_input)

    print("\n[6] Saving outputs ...")
    master.to_csv(MASTER_OUT,   index=False, date_format=TIMESTAMP_FMT)
    power_input.to_csv(POWER_OUT,   index=False, date_format=TIMESTAMP_FMT)
    if price_input is not None:
        price_input.to_csv(PRICE_OUT, index=False, date_format=TIMESTAMP_FMT)
    anomaly_input.to_csv(ANOMALY_OUT, index=False, date_format=TIMESTAMP_FMT)

    print(f"  ✅ {MASTER_OUT}")
    print(f"  ✅ {POWER_OUT}")
    if price_input is not None:
        print(f"  ✅ {PRICE_OUT}")
    print(f"  ✅ {ANOMALY_OUT}")
    print("\n  DONE. Next step → run train_models.py")

if __name__ == "__main__":
    main()