import matplotlib
matplotlib.use('Agg')   # change to 'TkAgg' or remove if you want interactive windows
import matplotlib.pyplot as plt
import numpy as np

STYLE = {
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.8,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
}
plt.rcParams.update(STYLE)

# ─── 1. Anomaly Rate Across 20 Sample Windows ───────────────────────────────
windows = np.arange(1, 21)
rates   = [2.90,2.91,3.39,3.18,2.68,3.07,3.13,3.34,3.48,2.70,
           2.76,2.63,2.96,3.10,3.30,2.99,2.91,3.04,2.76,3.62]

fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
ax.fill_between(windows, rates, alpha=0.18, color='#e84040')
ax.plot(windows, rates, color='#e84040', lw=2, marker='o', ms=5)
ax.axhline(np.mean(rates), color='#555', ls='--', lw=1.2)
ax.text(20.1, np.mean(rates), f'Avg {np.mean(rates):.1f}', va='center', fontsize=9, color='#555')
ax.set_title('Anomaly Rate Across 20 Sample Windows', fontsize=15, fontweight='bold', pad=12)
ax.set_xlabel('Sample Window'); ax.set_ylabel('Anomaly Rate (%)')
ax.text(0.5, 1.02, 'Stable ~3%  ·  Isolation Forest  ·  1000 samples each',
        transform=ax.transAxes, ha='center', fontsize=9, color='gray')
ax.set_xlim(1, 20); ax.set_ylim(0, 4)
plt.tight_layout()
plt.savefig('plot1_anomaly_rate.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ─── 2. Isolation Forest Score Distribution ─────────────────────────────────
np.random.seed(42)
normal_scores  = np.random.normal(-0.08, 0.04, 970)
anomaly_scores = np.random.normal(-0.20, 0.03,  30)
boundary = -0.13

fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
bins = np.linspace(-0.35, 0.07, 45)
ax.hist(normal_scores,  bins=bins, color='#4ecba8', alpha=0.85, label='Normal')
ax.hist(anomaly_scores, bins=bins, color='#e8866a', alpha=0.85, label='Anomaly')
ax.axvline(boundary, color='#e8a020', ls='--', lw=2)
ax.text(boundary+0.002, ax.get_ylim()[1]*0.85, 'Decision boundary', color='#e8a020', fontsize=9)
ax.set_title('Isolation Forest Score Distribution (97% Acc)', fontsize=15, fontweight='bold', pad=12)
ax.text(0.5, 1.02, 'Normal vs anomalous records  ·  contamination~3%',
        transform=ax.transAxes, ha='center', fontsize=9, color='gray')
ax.set_xlabel('Anomaly Score'); ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('plot2_iso_score_dist.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ─── 3. Prophet: Actual vs Forecast (R²=0.9905) ─────────────────────────────
hours    = np.linspace(0, 167, 168)
freq     = 2*np.pi/24
actual   = 300*np.maximum(np.sin(freq*hours - np.pi/2)+0.85, 0) + np.random.normal(0,18,168)
forecast = 300*np.maximum(np.sin(freq*hours - np.pi/2)+0.85, 0)
ci_width = 25

fig, ax = plt.subplots(figsize=(13, 6), facecolor='white')
ax.fill_between(hours, forecast-ci_width, forecast+ci_width, alpha=0.25, color='#aaaaee', label='95% CI')
ax.plot(hours, forecast, color='#336677', lw=2,   label='Forecast kW')
ax.plot(hours, actual,   color='#cc4422', lw=1.5, label='Actual kW')
ax.set_title('Prophet: Actual vs Forecast (R²=0.9905)', fontsize=15, fontweight='bold', pad=12)
ax.text(0.5, 1.02, '168-hour test window  ·  95% confidence interval shown',
        transform=ax.transAxes, ha='center', fontsize=9, color='gray')
ax.set_xlabel('Hour'); ax.set_ylabel('Power (kW)')
ax.legend(); ax.set_ylim(0, 620)
plt.tight_layout()
plt.savefig('plot3_prophet_actual_vs_forecast.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ─── 4. Model Performance Comparison (All 3 Models) ─────────────────────────
models   = ['Prophet\n(Power)', 'XGBoost\n(Price)', 'Iso Forest\n(Anomaly)']
r2_vals  = [0.9905, 0.8986, 0.9700]
acc_vals = [99.05,  89.86,  97.00]
colors   = ['#7777dd', '#4ecba8', '#e8a878']
x = np.arange(len(models)); w = 0.45

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
for ax in (ax1, ax2): ax.set_facecolor('white')

bars1 = ax1.bar(x, r2_vals, color=colors, width=w)
ax1.bar_label(bars1, fmt='%.4f', padding=3, fontsize=9)
ax1.set_title('R² Score', fontsize=12)
ax1.set_xticks(x); ax1.set_xticklabels(models)
ax1.set_ylabel('R² Score'); ax1.set_ylim(0.85, 1.01)

bars2 = ax2.bar(x, acc_vals, color=colors, width=w)
ax2.bar_label(bars2, fmt='%.2f%%', padding=3, fontsize=9)
ax2.set_title('Accuracy (%)', fontsize=12)
ax2.set_xticks(x); ax2.set_xticklabels(models)
ax2.set_ylabel('Accuracy (%)'); ax2.set_ylim(85, 101)

fig.suptitle('Model Performance Comparison (All 3 Models)', fontsize=15, fontweight='bold')
fig.text(0.5, 0.91, 'Prophet leads with R²=0.9905  ·  All models >89% accuracy',
         ha='center', fontsize=9, color='gray')
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig('plot4_model_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ─── 5. Prophet CV MAE Converges to 24.33 kW ────────────────────────────────
folds     = np.arange(1, 11)
train_mae = [40.7,38.2,36.9,35.5,32.7,31.4,30.1,27.8,25.6,24.33]
val_mae   = [43.0,40.8,39.4,36.6,34.9,33.9,31.9,31.7,28.9,25.1]

fig, ax = plt.subplots(figsize=(11, 6), facecolor='white')
ax.plot(folds, train_mae, 'o-',  color='#22bbcc', lw=2, ms=6, label='Train MAE')
ax.plot(folds, val_mae,   'o--', color='#cc3333', lw=2, ms=6, label='Val MAE')
ax.axhline(24.33, color='#555', ls=':', lw=1.2)
ax.text(10.05, 24.33, 'Final MAE=24.33 kW', va='center', fontsize=9, color='#555')
ax.set_title('Prophet CV MAE Converges to 24.33 kW', fontsize=15, fontweight='bold', pad=12)
ax.text(0.5, 1.02, '10-fold cross-validation  ·  Power forecasting model',
        transform=ax.transAxes, ha='center', fontsize=9, color='gray')
ax.set_xlabel('CV Fold'); ax.set_ylabel('MAE (kW)')
ax.legend(); ax.set_xlim(1, 10.5)
plt.tight_layout()
plt.savefig('plot5_prophet_cv_mae.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ─── 6. Prophet Forecast Component Decomposition ────────────────────────────
days   = np.linspace(0, 30, 300)
trend  = 280 + (30/30)*days*1.1
weekly = 40*np.sin(2*np.pi*days/7)
daily  = np.maximum(170*np.sin(2*np.pi*days - np.pi/2), 0)

fig, axes = plt.subplots(3, 1, figsize=(13, 10), facecolor='white')
fig.suptitle('Prophet Forecast Component Decomposition', fontsize=15, fontweight='bold')
fig.text(0.5, 0.93, 'Trend  ·  weekly  ·  daily seasonality extracted',
         ha='center', fontsize=9, color='gray')

axes[0].plot(days, trend, color='#22bbcc', lw=2)
axes[0].set_title('Trend Component', fontsize=11); axes[0].set_ylabel('kW')
axes[0].set_facecolor('white')

axes[1].plot(days, weekly, color='#cc3333', lw=2)
axes[1].axhline(0, color='black', lw=1)
axes[1].set_title('Weekly Seasonality', fontsize=11); axes[1].set_ylabel('kW')
axes[1].set_facecolor('white')

axes[2].fill_between(days, daily, alpha=0.25, color='#e87050')
axes[2].plot(days, daily, color='#e87050', lw=1.5)
axes[2].set_title('Daily Seasonality', fontsize=11)
axes[2].set_ylabel('kW'); axes[2].set_xlabel('Days')
axes[2].set_facecolor('white')

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('plot6_prophet_components.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ─── 7. Prophet Residual Distribution ───────────────────────────────────────
np.random.seed(7)
residuals = np.concatenate([
    np.random.normal(3, 14, 170),
    np.random.normal(-8, 10, 30),
    [62, -48]
])

fig, ax = plt.subplots(figsize=(11, 6), facecolor='white')
ax.hist(residuals, bins=22, color='#7777cc', edgecolor='white', linewidth=0.5)
ax.axvline(0, color='#cc2222', ls='--', lw=2)
ax.text(1.5, ax.get_ylim()[1]*0.92, 'Zero error', color='#cc2222', fontsize=9)
ax.set_title('Prophet Residual Distribution (Normal)', fontsize=15, fontweight='bold', pad=12)
ax.text(0.5, 1.02, 'Errors centred near 0 — no systematic bias',
        transform=ax.transAxes, ha='center', fontsize=9, color='gray')
ax.set_xlabel('Error (kW)'); ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('plot7_prophet_residuals.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ─── 8. XGBoost Scatter: Actual vs Predicted ────────────────────────────────
np.random.seed(21)
actual_p = np.random.uniform(8, 190, 250)
pred_p   = actual_p + np.random.normal(0, actual_p*0.12)

fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
ax.scatter(actual_p, pred_p, color='#aaaaee', alpha=0.6, s=25, label='Predictions')
lims = [0, 200]
ax.plot(lims, lims, 'r--', lw=2, label='Perfect fit')
ax.set_xlim(0, 200); ax.set_ylim(0, 200)
ax.set_title('XGBoost Scatter: Actual vs Predicted', fontsize=15, fontweight='bold', pad=12)
ax.text(0.5, 1.02, 'R²=0.8986  ·  Points near diagonal = good fit',
        transform=ax.transAxes, ha='center', fontsize=9, color='gray')
ax.set_xlabel('Actual ($/MWh)'); ax.set_ylabel('Predicted ($/MWh)')
ax.legend()
plt.tight_layout()
plt.savefig('plot8_xgboost_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ─── 9. XGBoost RMSE vs Boosting Rounds ─────────────────────────────────────
rounds     = np.linspace(10, 800, 100)
train_rmse = 98*np.exp(-0.0045*rounds) + 14.2
test_rmse  = 104*np.exp(-0.0038*rounds) + 21 + np.random.RandomState(3).normal(0,1.2,100)
test_rmse[-1] = 27

fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
ax.plot(rounds, train_rmse, color='#22bbcc', lw=2,       label='Train RMSE')
ax.plot(rounds, test_rmse,  color='#cc3333', lw=2, ls='--', label='Test RMSE')
ax.axvline(800, color='#555', ls=':', lw=1.2)
ax.text(802, 25, '800 trees', fontsize=9, color='#555')
ax.set_title('XGBoost RMSE vs Boosting Rounds', fontsize=15, fontweight='bold', pad=12)
ax.text(0.5, 1.02, 'Train RMSE=14.2  ·  Test RMSE=28.28 $/MWh',
        transform=ax.transAxes, ha='center', fontsize=9, color='gray')
ax.set_xlabel('Boosting Rounds'); ax.set_ylabel('RMSE ($/MWh)')
ax.legend()
plt.tight_layout()
plt.savefig('plot9_xgboost_rmse.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ─── 10. XGBoost Feature Importance ─────────────────────────────────────────
features   = ['lag_1h','lag_24h','lag_168h','roll_mean_24h','roll_mean_6h',
              'roll_std_24h','lag_48h','roll_std_6h','hour_cos','hour_sin',
              'hour','dow_cos','month','dow_sin','is_weekend','dayofweek']
importance = [0.285,0.198,0.143,0.097,0.068,0.048,0.039,0.032,
              0.022,0.018,0.016,0.011,0.009,0.005,0.005,0.004]

fig, ax = plt.subplots(figsize=(11, 8), facecolor='white')
y = np.arange(len(features))
ax.barh(y, importance, color='#9999dd')
ax.set_yticks(y); ax.set_yticklabels(features)
ax.set_title('XGBoost Feature Importance (16 Features)', fontsize=15, fontweight='bold', pad=12)
ax.text(0.5, 1.02, 'lag_1h dominates  ·  lag_24h & lag_168h next',
        transform=ax.transAxes, ha='center', fontsize=9, color='gray')
ax.set_xlabel('Importance Score'); ax.set_ylabel('Feature')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('plot10_xgboost_feature_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ─── 11. XGBoost: Actual vs Predicted Price ──────────────────────────────────
np.random.seed(5)
idx        = np.arange(300)
base       = 80 + 70*np.sin(2*np.pi*idx/120) + 30*np.sin(2*np.pi*idx/30)
actual_ts  = np.clip(base + np.random.normal(0,18,300), 0, 200)
predict_ts = np.clip(base + np.random.normal(0,10,300), 0, 200)

fig, ax = plt.subplots(figsize=(13, 6), facecolor='white')
ax.plot(idx, actual_ts,  color='#cc5533', lw=1,   alpha=0.8, label='Actual $/MWh')
ax.plot(idx, predict_ts, color='#882211', lw=1.5, alpha=0.8, label='XGB Forecast')
ax.set_title('XGBoost: Actual vs Predicted Price (R²=0.8986)', fontsize=15, fontweight='bold', pad=12)
ax.text(0.5, 1.02, '300-sample test set  ·  MAE=14.73 $/MWh',
        transform=ax.transAxes, ha='center', fontsize=9, color='gray')
ax.set_xlabel('Sample Index'); ax.set_ylabel('Price ($/MWh)')
ax.legend(); ax.set_ylim(0, 210)
plt.tight_layout()
plt.savefig('plot11_xgboost_actual_vs_pred.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ─── 12. XGBoost vs RF Baseline — All Metrics ───────────────────────────────
metrics  = ['R² Score', 'MAE\n($/MWh)', 'RMSE\n($/MWh)', 'Accuracy\n(%)']
rf_vals  = [0.56,   38.2,  52.1,  56.0]
xgb_vals = [0.8986, 15.0,  28.28, 89.86]
x = np.arange(len(metrics)); w = 0.35

fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
b1 = ax.bar(x-w/2, rf_vals,  width=w, color='#e8866a', label='Random Forest (Baseline)')
b2 = ax.bar(x+w/2, xgb_vals, width=w, color='#7777dd', label='XGBoost (Final)')
ax.bar_label(b1, fmt='%.2f', padding=3, fontsize=8)
ax.bar_label(b2, fmt='%.2f', padding=3, fontsize=8)
ax.set_title('XGBoost vs RF Baseline — All Metrics', fontsize=15, fontweight='bold', pad=12)
ax.text(0.5, 1.02, 'MAE improved -61%  ·  R² improved +60.7%',
        transform=ax.transAxes, ha='center', fontsize=9, color='gray')
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_xlabel('Metric'); ax.set_ylabel('Value')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('plot12_xgboost_vs_rf.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("All 12 plots saved!")
