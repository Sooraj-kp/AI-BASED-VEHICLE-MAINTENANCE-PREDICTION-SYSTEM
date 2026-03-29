"""
Advanced ML Models — VehicleAI
Trains:
  1. RUL (Remaining Useful Life) Regressor
  2. Anomaly Detection — Isolation Forest
  3. Risk Score Calibrator
  + Generates 8 additional analysis graphs
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ─── Paths ───
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
GRAPH_DIR = os.path.join(BASE_DIR, "static", "images", "graphs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# ─── Plot Style ───
plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d', 'axes.labelcolor': '#e6edf3',
    'text.color': '#e6edf3', 'xtick.color': '#8b949e',
    'ytick.color': '#8b949e', 'grid.color': '#21262d',
    'grid.linestyle': '--', 'grid.alpha': 0.5,
    'axes.titlecolor': '#58a6ff', 'axes.titlesize': 13,
})
ACCENT  = '#58a6ff'; SUCCESS = '#3fb950'; DANGER = '#f85149'
WARNING = '#d29922'; PURPLE  = '#bc8cff'; CAUTION = '#f0883e'
PAL = [ACCENT, SUCCESS, DANGER, WARNING, PURPLE, CAUTION]

def save_fig(name):
    path = os.path.join(GRAPH_DIR, name)
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  ✓  {path}")

# ═══════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════
df = pd.read_csv(os.path.join(DATA_DIR, "engine_data.csv"))
df.columns = df.columns.str.strip()
FEATURES = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure',
            'Coolant pressure', 'lub oil temp', 'Coolant temp']
TARGET   = 'Engine Condition'

print("\n" + "═"*60)
print("  ADVANCED MODEL TRAINING")
print("═"*60)

# ═══════════════════════════════════════════════════
# PART 1 — ANOMALY DETECTION (Isolation Forest)
# ═══════════════════════════════════════════════════
print("\n[1] Training Anomaly Detection — Isolation Forest")

# Train on NORMAL data only (condition == 0)
X_normal = df[df[TARGET] == 0][FEATURES].values
scaler_anom = StandardScaler()
X_normal_sc = scaler_anom.fit_transform(X_normal)

iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # expect ~5% anomalies in production
    max_samples='auto',
    random_state=42,
    n_jobs=-1
)
iso.fit(X_normal_sc)

# Evaluate on full dataset
X_all_sc = scaler_anom.transform(df[FEATURES].values)
anom_scores = iso.decision_function(X_all_sc)   # higher = more normal
anom_labels = iso.predict(X_all_sc)             # -1 = anomaly, 1 = normal

# Normalize scores to 0-100 (higher = more anomalous)
mms = MinMaxScaler()
anom_risk = 1 - mms.fit_transform(anom_scores.reshape(-1,1)).flatten()

joblib.dump(iso,         os.path.join(MODEL_DIR, "anomaly_model.pkl"))
joblib.dump(scaler_anom, os.path.join(MODEL_DIR, "anomaly_scaler.pkl"))
joblib.dump(mms,         os.path.join(MODEL_DIR, "anomaly_normalizer.pkl"))
print(f"  Anomalies detected in training set: {(anom_labels == -1).sum()} / {len(anom_labels)}")

# ── GRAPH A1 — Anomaly Score Distribution ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(anom_scores[df[TARGET]==0], bins=60, color=SUCCESS, alpha=0.7, label='Normal', density=True)
axes[0].hist(anom_scores[df[TARGET]==1], bins=60, color=DANGER,  alpha=0.7, label='Fault',  density=True)
axes[0].axvline(0, color=WARNING, lw=1.5, linestyle='--', label='Anomaly threshold')
axes[0].set_xlabel('Anomaly Score (higher = more normal)')
axes[0].set_ylabel('Density'); axes[0].set_title('Isolation Forest Score Distribution')
axes[0].legend(framealpha=0)

# 2D scatter: RPM vs Coolant Temp colored by anomaly
sc = axes[1].scatter(df['Engine rpm'], df['Coolant temp'],
                     c=anom_risk, cmap='RdYlGn_r', s=5, alpha=0.4)
cbar = plt.colorbar(sc, ax=axes[1])
cbar.set_label('Anomaly Risk', color='#8b949e')
cbar.ax.yaxis.set_tick_params(color='#8b949e')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e')
axes[1].set_xlabel('Engine RPM'); axes[1].set_ylabel('Coolant Temp (°C)')
axes[1].set_title('Anomaly Risk Map — RPM vs Coolant Temp')
save_fig("adv_anomaly_distribution.png")

# ── GRAPH A2 — Feature Contribution to Anomaly ──
fig, ax = plt.subplots(figsize=(9, 5))
corr_vals = [abs(np.corrcoef(df[f].values, anom_risk)[0,1]) for f in FEATURES]
idx_sort  = np.argsort(corr_vals)[::-1]
ax.barh([FEATURES[i] for i in idx_sort],
        [corr_vals[i] for i in idx_sort],
        color=[PAL[i] for i in range(len(FEATURES))], edgecolor='none', alpha=0.85)
ax.set_xlabel('Correlation with Anomaly Risk')
ax.set_title('Feature Contribution to Anomaly Score')
ax.axvline(0.1, color=WARNING, linestyle='--', lw=1, label='Significance threshold')
ax.legend(framealpha=0)
save_fig("adv_anomaly_features.png")


# ═══════════════════════════════════════════════════
# PART 2 — RUL PREDICTOR
# ═══════════════════════════════════════════════════
print("\n[2] Training Remaining Useful Life (RUL) Predictor")

# ── Engineer RUL labels ──
# Healthy parameter ranges (from domain knowledge)
RANGES = {
    'Engine rpm':       (700, 2000),
    'Lub oil pressure': (2.0, 6.0),
    'Fuel pressure':    (8.0, 22.0),
    'Coolant pressure': (1.0, 4.0),
    'lub oil temp':     (60, 100),
    'Coolant temp':     (70, 95),
}

def compute_health_index(row):
    """0=critical, 1=perfect. Uses distance from healthy range."""
    penalties = []
    for feat, (lo, hi) in RANGES.items():
        val = row[feat]
        midpoint = (lo + hi) / 2
        half_range = (hi - lo) / 2
        dist = max(0, abs(val - midpoint) - half_range)
        penalty = min(1.0, dist / (half_range * 2))
        penalties.append(penalty)
    avg_penalty = np.mean(penalties)
    return max(0.0, 1.0 - avg_penalty)

df['health_index'] = df.apply(compute_health_index, axis=1)

# RUL in km: healthy engine ~ 200,000 km life
# Use deterministic formula with small noise so model is predictive
MAX_KM = 200_000
np.random.seed(42)
# Per-feature deviation scores
feat_penalties = np.zeros(len(df))
for feat, (lo, hi) in RANGES.items():
    vals = df[feat].values
    mid  = (lo + hi) / 2
    half = (hi - lo) / 2
    dist = np.maximum(0, np.abs(vals - mid) - half)
    # normalize by half-range
    feat_penalties += np.minimum(1.0, dist / (half + 1e-9))
feat_penalties /= len(RANGES)

# Non-linear RUL formula
df['rul_km'] = (
    MAX_KM * (1 - feat_penalties) ** 1.5
    * (1 - df[TARGET] * 0.40)
    + np.random.normal(0, MAX_KM * 0.005, len(df))   # tiny noise
).clip(0, MAX_KM)

# RUL in days (assume 50 km/day avg usage)
df['rul_days'] = (df['rul_km'] / 50).clip(0, MAX_KM / 50)

X_rul = df[FEATURES]
y_km  = df['rul_km']
y_day = df['rul_days']

X_tr, X_te, yk_tr, yk_te = train_test_split(X_rul, y_km, test_size=0.2, random_state=42)
_, _, yd_tr, yd_te        = train_test_split(X_rul, y_day, test_size=0.2, random_state=42)

# Train km model
rul_km_model = RandomForestRegressor(n_estimators=200, max_depth=18,
                                      random_state=42, n_jobs=-1)
rul_km_model.fit(X_tr, yk_tr)
pred_km = rul_km_model.predict(X_te)
mae_km  = mean_absolute_error(yk_te, pred_km)
r2_km   = r2_score(yk_te, pred_km)
print(f"  RUL (km)  → MAE={mae_km:.0f} km  R²={r2_km:.4f}")

# Train day model
rul_day_model = RandomForestRegressor(n_estimators=200, max_depth=18,
                                       random_state=42, n_jobs=-1)
rul_day_model.fit(X_tr, yd_tr)
pred_day = rul_day_model.predict(X_te)
mae_day  = mean_absolute_error(yd_te, pred_day)
r2_day   = r2_score(yd_te, pred_day)
print(f"  RUL (day) → MAE={mae_day:.0f} days  R²={r2_day:.4f}")

joblib.dump(rul_km_model,  os.path.join(MODEL_DIR, "rul_km_model.pkl"))
joblib.dump(rul_day_model, os.path.join(MODEL_DIR, "rul_day_model.pkl"))
print("  ✓  RUL models saved")

# ── GRAPH B1 — RUL Actual vs Predicted ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(yk_te/1000, pred_km/1000, alpha=0.25, s=8, color=ACCENT, edgecolors='none')
mn, mx = 0, MAX_KM/1000
axes[0].plot([mn,mx],[mn,mx],'--', color=WARNING, lw=1.5, label='Perfect')
axes[0].set_xlabel('Actual RUL (×1000 km)'); axes[0].set_ylabel('Predicted RUL (×1000 km)')
axes[0].set_title(f'RUL Prediction — km  |  R²={r2_km:.4f}  MAE={mae_km/1000:.1f}k km')
axes[0].legend(framealpha=0)

axes[1].scatter(yd_te, pred_day, alpha=0.25, s=8, color=PURPLE, edgecolors='none')
mn2, mx2 = 0, MAX_KM/50
axes[1].plot([mn2,mx2],[mn2,mx2],'--', color=WARNING, lw=1.5, label='Perfect')
axes[1].set_xlabel('Actual RUL (days)'); axes[1].set_ylabel('Predicted RUL (days)')
axes[1].set_title(f'RUL Prediction — Days  |  R²={r2_day:.4f}  MAE={mae_day:.0f} days')
axes[1].legend(framealpha=0)
save_fig("adv_rul_actual_predicted.png")

# ── GRAPH B2 — RUL Distribution by Condition ──
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df[df[TARGET]==0]['rul_km']/1000, bins=60, color=SUCCESS, alpha=0.75, label='Normal Engine', density=True)
ax.hist(df[df[TARGET]==1]['rul_km']/1000, bins=60, color=DANGER,  alpha=0.75, label='Faulty Engine', density=True)
ax.set_xlabel('Remaining Useful Life (×1000 km)')
ax.set_ylabel('Density'); ax.set_title('RUL Distribution by Engine Condition')
ax.legend(framealpha=0)
ax.axvline(df['rul_km'].mean()/1000, color=WARNING, lw=1.5, linestyle='--',
           label=f'Mean RUL: {df["rul_km"].mean()/1000:.0f}k km')
save_fig("adv_rul_distribution.png")

# ── GRAPH B3 — Feature Importance for RUL ──
imp_rul = rul_km_model.feature_importances_
idx_rul = np.argsort(imp_rul)[::-1]
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(len(FEATURES)),
       [imp_rul[i] for i in idx_rul],
       color=[PAL[i] for i in range(len(FEATURES))],
       edgecolor='none', alpha=0.9)
ax.set_xticks(range(len(FEATURES)))
ax.set_xticklabels([FEATURES[i] for i in idx_rul], rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Importance Score')
ax.set_title('Feature Importance — RUL Prediction')
for i, v in enumerate([imp_rul[j] for j in idx_rul]):
    ax.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=8, color='#8b949e')
save_fig("adv_rul_feature_importance.png")

# ── GRAPH B4 — Health Index vs RUL ──
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(df['health_index'], df['rul_km']/1000,
                c=df[TARGET], cmap='RdYlGn_r', s=5, alpha=0.3)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Fault (1) / Normal (0)', color='#8b949e')
cbar.ax.yaxis.set_tick_params(color='#8b949e')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e')
ax.set_xlabel('Health Index (0=Critical, 1=Perfect)')
ax.set_ylabel('RUL (×1000 km)')
ax.set_title('Health Index vs Remaining Useful Life')
save_fig("adv_health_vs_rul.png")


# ═══════════════════════════════════════════════════
# PART 3 — RISK SCORE CALIBRATOR
# ═══════════════════════════════════════════════════
print("\n[3] Building Risk Score Calibrator")

# Load existing engine model
engine_model  = joblib.load(os.path.join(MODEL_DIR, "engine_model.pkl"))
engine_scaler = joblib.load(os.path.join(MODEL_DIR, "engine_scaler.pkl"))

X_full_sc = engine_scaler.transform(df[FEATURES])
if hasattr(engine_model, 'predict_proba'):
    fault_probs = engine_model.predict_proba(X_full_sc)[:, 1]
else:
    fault_probs = engine_model.predict(X_full_sc).astype(float)

# Composite risk = weighted combination
# fault_prob 50% + anomaly_risk 30% + health_deficit 20%
df['fault_prob']    = fault_probs
df['anomaly_risk']  = anom_risk
df['health_deficit'] = 1 - df['health_index']

df['risk_score'] = (
    df['fault_prob']     * 50 +
    df['anomaly_risk']   * 30 +
    df['health_deficit'] * 20
).clip(0, 100)

# Save risk score weights
risk_weights = {'fault_prob': 0.50, 'anomaly_risk': 0.30, 'health_deficit': 0.20}
joblib.dump(risk_weights, os.path.join(MODEL_DIR, "risk_weights.pkl"))
print(f"  Risk score range: {df['risk_score'].min():.1f} – {df['risk_score'].max():.1f}")
print(f"  Mean risk score:  {df['risk_score'].mean():.1f}")

# ── GRAPH C1 — Risk Score Distribution ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df[df[TARGET]==0]['risk_score'], bins=60, color=SUCCESS, alpha=0.75,
             label='Normal Engine', density=True)
axes[0].hist(df[df[TARGET]==1]['risk_score'], bins=60, color=DANGER,  alpha=0.75,
             label='Faulty Engine', density=True)
axes[0].set_xlabel('Composite Risk Score (0–100)')
axes[0].set_ylabel('Density'); axes[0].set_title('Risk Score Distribution by Engine Condition')
axes[0].legend(framealpha=0)
axes[0].axvline(50, color=WARNING, lw=1.5, linestyle='--', label='Risk threshold=50')

# Pie of risk levels
risk_bins = pd.cut(df['risk_score'],
                   bins=[0, 25, 50, 75, 100],
                   labels=['Low\n(0-25)', 'Moderate\n(25-50)', 'High\n(50-75)', 'Critical\n(75-100)'])
counts = risk_bins.value_counts().sort_index()
wedge_colors = [SUCCESS, WARNING, CAUTION, DANGER]
axes[1].pie(counts.values, labels=counts.index,
            colors=wedge_colors, autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2},
            textprops={'color': '#e6edf3', 'fontsize': 9})
axes[1].set_title('Distribution of Risk Score Levels')
save_fig("adv_risk_distribution.png")

# ── GRAPH C2 — Risk Score vs RUL Heatmap ──
fig, ax = plt.subplots(figsize=(9, 6))
h = ax.hist2d(df['risk_score'], df['rul_km']/1000, bins=40, cmap='plasma')
plt.colorbar(h[3], ax=ax, label='Count')
ax.set_xlabel('Risk Score (0–100)')
ax.set_ylabel('RUL (×1000 km)')
ax.set_title('Risk Score vs Remaining Useful Life — 2D Density')
save_fig("adv_risk_vs_rul.png")


# ═══════════════════════════════════════════════════
# PART 4 — WHAT-IF SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════
print("\n[4] Generating What-If Sensitivity Analysis")

# For each feature, vary it across its range while holding others at mean
X_mean = df[FEATURES].mean().values

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, feat in enumerate(FEATURES):
    lo, hi = df[feat].quantile(0.01), df[feat].quantile(0.99)
    sweep  = np.linspace(lo, hi, 80)
    preds  = []
    for val in sweep:
        row = X_mean.copy()
        row[i] = val
        row_df = pd.DataFrame([{f: row[j] for j, f in enumerate(FEATURES)}])
        sc = engine_scaler.transform(row_df)
        if hasattr(engine_model, 'predict_proba'):
            p = engine_model.predict_proba(sc)[0][1]
        else:
            p = float(engine_model.predict(sc)[0])
        preds.append(p * 100)

    color = PAL[i % len(PAL)]
    axes[i].plot(sweep, preds, color=color, lw=2)
    axes[i].fill_between(sweep, preds, alpha=0.15, color=color)
    axes[i].axhline(50, color=WARNING, lw=1, linestyle='--', alpha=0.7)
    axes[i].set_xlabel(feat, fontsize=9)
    axes[i].set_ylabel('Fault Prob (%)')
    axes[i].set_title(f'Sensitivity — {feat}', fontsize=10)
    axes[i].set_ylim(0, 100)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('What-If Sensitivity Analysis — Engine Fault Probability',
             fontsize=13, color='#58a6ff', y=1.01)
plt.tight_layout()
save_fig("adv_whatif_sensitivity.png")

print("\n" + "═"*60)
print("  ADVANCED TRAINING COMPLETE")
print("═"*60)
new_graphs = [g for g in os.listdir(GRAPH_DIR) if g.startswith('adv_')]
print(f"  New graphs generated: {len(new_graphs)}")
for g in sorted(new_graphs):
    print(f"    → {g}")
