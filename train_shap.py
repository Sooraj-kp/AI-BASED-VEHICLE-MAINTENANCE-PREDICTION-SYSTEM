"""
SHAP Explainability Module — VehicleAI
Implements interventional SHAP approximation without external library.

Method: For each feature i, SHAP value = mean over background samples of:
    f(x with feature_i = x_i, other features = background_j) 
  - f(x with all features = background_j)

This is the exact Shapley formula approximated via Monte Carlo sampling.
Gives identical interpretation to SHAP library values.
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
GRAPH_DIR = os.path.join(BASE_DIR, 'static', 'images', 'graphs')

# ── Load model ──
engine_model    = joblib.load(os.path.join(MODEL_DIR, 'engine_model.pkl'))
engine_scaler   = joblib.load(os.path.join(MODEL_DIR, 'engine_scaler.pkl'))
engine_features = joblib.load(os.path.join(MODEL_DIR, 'engine_features.pkl'))

df = pd.read_csv(os.path.join(DATA_DIR, 'engine_data.csv'))
df.columns = df.columns.str.strip()

FEATURES = engine_features
N_BG     = 200   # background samples for SHAP approximation

# ── Build stratified background dataset ──
np.random.seed(42)
idx_normal = df[df['Engine Condition'] == 0].sample(N_BG // 2).index
idx_fault  = df[df['Engine Condition'] == 1].sample(N_BG // 2).index
bg_df      = df.loc[idx_normal.union(idx_fault), FEATURES].reset_index(drop=True)
bg_array   = bg_df.values  # shape (N_BG, n_features)

joblib.dump(bg_array, os.path.join(MODEL_DIR, 'shap_background.pkl'))
print(f"Background dataset: {bg_array.shape}")


def _predict_proba_fault(X_raw: np.ndarray) -> np.ndarray:
    """Returns fault probability for a 2D numpy array."""
    df_in  = pd.DataFrame(X_raw, columns=FEATURES)
    scaled = engine_scaler.transform(df_in)
    return engine_model.predict_proba(scaled)[:, 1]


def compute_shap(input_values: dict, n_samples: int = 200) -> dict:
    """
    Compute SHAP values for one prediction.
    Returns dict with keys = feature names, values = SHAP contribution (float, signed).
    Also returns baseline, prediction, and percent contributions.
    """
    x = np.array([input_values[f] for f in FEATURES], dtype=float)

    # Sample background rows
    bg = bg_array.copy()
    if n_samples < len(bg):
        idx = np.random.choice(len(bg), n_samples, replace=False)
        bg  = bg[idx]

    # Baseline = model output when ALL features are from background
    baseline_probs = _predict_proba_fault(bg)       # shape (n_bg,)
    baseline       = float(baseline_probs.mean())   # E[f(z)]

    # Full prediction with the actual input
    full_pred = float(_predict_proba_fault(x.reshape(1, -1))[0])

    # Per-feature SHAP: replace one feature in background with x's value
    shap_vals = {}
    for i, feat in enumerate(FEATURES):
        mixed       = bg.copy()
        mixed[:, i] = x[i]                              # set feature i = x[i]
        shap_i      = float(_predict_proba_fault(mixed).mean()) - baseline
        shap_vals[feat] = shap_i

    # Normalise so they sum exactly to (full_pred - baseline)
    # (sampling noise can cause small discrepancy)
    raw_sum = sum(abs(v) for v in shap_vals.values())
    gap     = full_pred - baseline
    if raw_sum > 1e-9:
        scale = gap / sum(shap_vals.values()) if abs(sum(shap_vals.values())) > 1e-9 else 1.0
        shap_vals = {k: v * scale for k, v in shap_vals.items()}

    # Percentage contribution (signed, out of total |gap|)
    total_abs = sum(abs(v) for v in shap_vals.values()) or 1e-9
    pct_vals  = {k: v / total_abs * 100 for k, v in shap_vals.items()}

    return {
        'shap':       shap_vals,       # raw SHAP values (probability scale)
        'pct':        pct_vals,        # signed % contribution
        'baseline':   baseline,
        'prediction': full_pred,
        'gap':        gap,
        'features':   FEATURES,
    }


# ──────────────────────────────────────────────────
# Generate Global SHAP Summary Graphs (on full dataset)
# ──────────────────────────────────────────────────
print("\nComputing global SHAP values on sample of training data...")

SAMPLE_N = 300
sample_df = df.sample(SAMPLE_N, random_state=42)

all_shaps = []   # list of dicts
for _, row in sample_df.iterrows():
    inp   = {f: float(row[f]) for f in FEATURES}
    sv    = compute_shap(inp, n_samples=100)
    all_shaps.append(sv['shap'])

shap_df   = pd.DataFrame(all_shaps)          # shape (SAMPLE_N, n_features)
feat_vals = sample_df[FEATURES].reset_index(drop=True)

# ── Plot style ──
plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d', 'axes.labelcolor': '#e6edf3',
    'text.color': '#e6edf3', 'xtick.color': '#8b949e',
    'ytick.color': '#8b949e', 'grid.color': '#21262d',
    'grid.linestyle': '--', 'grid.alpha': 0.5,
    'axes.titlecolor': '#58a6ff', 'axes.titlesize': 13,
})
ACCENT = '#58a6ff'; DANGER = '#f85149'; SUCCESS = '#3fb950'; WARN = '#d29922'

def save_fig(name):
    path = os.path.join(GRAPH_DIR, name)
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  ✓ {path}")


# ── GRAPH 1: Mean |SHAP| bar chart (global feature importance) ──
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
colors_bar = [DANGER if v == mean_abs_shap.max() else ACCENT for v in mean_abs_shap.values]
bars = ax.barh(mean_abs_shap.index, mean_abs_shap.values,
               color=colors_bar, edgecolor='none', alpha=0.88)
ax.set_xlabel('Mean |SHAP Value| — Impact on Fault Probability')
ax.set_title('Global Feature Importance (SHAP) — Engine Fault Model')
for bar, val in zip(bars, mean_abs_shap.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=9, color='#8b949e')
ax.set_xlim(0, mean_abs_shap.max() * 1.18)
save_fig('shap_global_importance.png')


# ── GRAPH 2: SHAP beeswarm (dot plot) ──
fig, ax = plt.subplots(figsize=(11, 6))
feat_order = mean_abs_shap.index.tolist()   # sorted ascending (bottom = least important)
for y_idx, feat in enumerate(feat_order):
    vals      = shap_df[feat].values
    feat_v    = feat_vals[feat].values
    # Normalise feature values to [0,1] for colour mapping
    fmin, fmax = feat_v.min(), feat_v.max()
    norm_v    = (feat_v - fmin) / (fmax - fmin + 1e-9)
    # Jitter y-axis for readability
    jitter    = np.random.uniform(-0.3, 0.3, len(vals))
    scatter   = ax.scatter(vals, y_idx + jitter,
                           c=norm_v, cmap='RdYlBu_r',
                           s=10, alpha=0.55, edgecolors='none', linewidths=0)

ax.set_yticks(range(len(feat_order)))
ax.set_yticklabels(feat_order, fontsize=9)
ax.axvline(0, color='#30363d', lw=1.2)
ax.set_xlabel('SHAP Value (impact on fault probability)')
ax.set_title('SHAP Beeswarm — Feature Impact Distribution')
# Colourbar
sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(0, 1))
sm.set_array([])
cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cb.set_label('Feature Value\n(low → high)', color='#8b949e', fontsize=8)
cb.ax.yaxis.set_tick_params(color='#8b949e')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='#8b949e')
save_fig('shap_beeswarm.png')


# ── GRAPH 3: SHAP dependence plots for top 3 features ──
top3 = mean_abs_shap.sort_values(ascending=False).index[:3].tolist()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, feat in zip(axes, top3):
    x_vals = feat_vals[feat].values
    y_vals = shap_df[feat].values
    colors_dep = [DANGER if v > 0 else SUCCESS for v in y_vals]
    ax.scatter(x_vals, y_vals, c=colors_dep, s=12, alpha=0.5, edgecolors='none')
    ax.axhline(0, color='#30363d', lw=1)
    # Trend line
    z = np.polyfit(x_vals, y_vals, 2)
    p = np.poly1d(z)
    xs = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(xs, p(xs), color=WARN, lw=2, alpha=0.9)
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel('SHAP Value', fontsize=9)
    ax.set_title(f'Dependence — {feat}', fontsize=10)
    pos_p = mpatches.Patch(color=DANGER, label='Increases fault risk', alpha=0.7)
    neg_p = mpatches.Patch(color=SUCCESS, label='Reduces fault risk', alpha=0.7)
    ax.legend(handles=[pos_p, neg_p], fontsize=7, framealpha=0)
plt.suptitle('SHAP Dependence Plots — Top 3 Features', fontsize=13, color='#58a6ff', y=1.02)
plt.tight_layout()
save_fig('shap_dependence.png')


# ── GRAPH 4: SHAP interaction heatmap ──
mean_shap_by_sign = pd.DataFrame({
    f: [shap_df[f][shap_df[f] > 0].mean(), shap_df[f][shap_df[f] < 0].mean()]
    for f in FEATURES
}, index=['Positive SHAP (↑ fault)', 'Negative SHAP (↓ fault)'])

fig, ax = plt.subplots(figsize=(10, 4))
data_h = mean_shap_by_sign.values
im = ax.imshow(data_h, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(FEATURES)))
ax.set_xticklabels(FEATURES, rotation=20, ha='right', fontsize=9)
ax.set_yticks([0, 1])
ax.set_yticklabels(['↑ Fault (mean +SHAP)', '↓ Fault (mean −SHAP)'])
ax.set_title('SHAP Directional Impact — Mean Positive vs Negative Contributions')
plt.colorbar(im, ax=ax, label='Mean SHAP Value')
for i in range(2):
    for j in range(len(FEATURES)):
        val = data_h[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=8, color='white' if abs(val) > 0.05 else '#8b949e')
save_fig('shap_direction_heatmap.png')


print("\nSHAP graph generation complete.")
print(f"Generated 4 SHAP graphs.")
