

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score,
    precision_recall_curve, average_precision_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ─────────────────────── PATHS ───────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
GRAPH_DIR = os.path.join(BASE_DIR, "static", "images", "graphs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# ─────────────────────── STYLE ───────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#e6edf3',
    'text.color':       '#e6edf3',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'grid.color':       '#21262d',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
    'axes.titlecolor':  '#58a6ff',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})

ACCENT    = '#58a6ff'
SUCCESS   = '#3fb950'
DANGER    = '#f85149'
WARNING   = '#d29922'
PURPLE    = '#bc8cff'
PALETTE   = [ACCENT, SUCCESS, DANGER, WARNING, PURPLE, '#79c0ff', '#ffa657']

def save_fig(name):
    path = os.path.join(GRAPH_DIR, name)
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  ✓  Saved → {path}")
    return path

# ═══════════════════════════════════════════════════════════
#  PART 1 — ENGINE CONDITION MODEL
# ═══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  PART 1 — ENGINE CONDITION CLASSIFICATION")
print("═"*60)

df_engine = pd.read_csv(os.path.join(DATA_DIR, "engine_data.csv"))
df_engine.columns = df_engine.columns.str.strip()
print(f"  Dataset: {df_engine.shape[0]} rows × {df_engine.shape[1]} cols")
print(f"  Class balance:\n{df_engine['Engine Condition'].value_counts()}")

FEATURES_ENG = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure',
                 'Coolant pressure', 'lub oil temp', 'Coolant temp']
TARGET_ENG   = 'Engine Condition'

X_e = df_engine[FEATURES_ENG]
y_e = df_engine[TARGET_ENG]

scaler_e = StandardScaler()
X_e_scaled = scaler_e.fit_transform(X_e)

X_tr_e, X_te_e, y_tr_e, y_te_e = train_test_split(
    X_e_scaled, y_e, test_size=0.2, random_state=42, stratify=y_e)

# ── Train multiple classifiers ──
classifiers = {
    'Random Forest':       RandomForestClassifier(n_estimators=150, max_depth=12,
                                                   random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                       random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
}

results_e = {}
for name, clf in classifiers.items():
    clf.fit(X_tr_e, y_tr_e)
    acc_tr = accuracy_score(y_tr_e, clf.predict(X_tr_e))
    acc_te = accuracy_score(y_te_e, clf.predict(X_te_e))
    cv     = cross_val_score(clf, X_e_scaled, y_e, cv=5, scoring='accuracy')
    results_e[name] = {'train_acc': acc_tr, 'test_acc': acc_te,
                       'cv_mean': cv.mean(), 'cv_std': cv.std(), 'model': clf}
    print(f"  {name:25s} → Test Acc: {acc_te:.4f}  CV: {cv.mean():.4f}±{cv.std():.4f}")

best_name_e = max(results_e, key=lambda k: results_e[k]['cv_mean'])
best_clf_e  = results_e[best_name_e]['model']
print(f"\n  Best model: {best_name_e}")

joblib.dump(best_clf_e, os.path.join(MODEL_DIR, "engine_model.pkl"))
joblib.dump(scaler_e,   os.path.join(MODEL_DIR, "engine_scaler.pkl"))
joblib.dump(FEATURES_ENG, os.path.join(MODEL_DIR, "engine_features.pkl"))
print("  ✓  Engine model saved")

# ── GRAPH 1 — Model Comparison Bar Chart ──
fig, ax = plt.subplots(figsize=(10, 5))
names  = list(results_e.keys())
tr_acc = [results_e[n]['train_acc'] for n in names]
te_acc = [results_e[n]['test_acc']  for n in names]
x      = np.arange(len(names))
w      = 0.35
bars1 = ax.bar(x - w/2, tr_acc, w, label='Train Accuracy', color=ACCENT,  alpha=0.85, edgecolor='none')
bars2 = ax.bar(x + w/2, te_acc, w, label='Test Accuracy',  color=SUCCESS, alpha=0.85, edgecolor='none')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0.7, 1.02); ax.set_ylabel('Accuracy')
ax.set_title('Engine Condition — Model Comparison')
ax.legend(framealpha=0)
ax.axhline(1.0, color='#30363d', lw=0.8)
for bar in bars1: ax.text(bar.get_x() + bar.get_width()/2,
                           bar.get_height() + 0.003,
                           f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#8b949e')
for bar in bars2: ax.text(bar.get_x() + bar.get_width()/2,
                           bar.get_height() + 0.003,
                           f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#8b949e')
save_fig("eng_model_comparison.png")

# ── GRAPH 2 — Confusion Matrix ──
y_pred_e = best_clf_e.predict(X_te_e)
cm_e = confusion_matrix(y_te_e, y_pred_e)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_e, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Good (0)', 'Fault (1)'],
            yticklabels=['Good (0)', 'Fault (1)'],
            linewidths=0.5, linecolor='#30363d', ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title(f'Confusion Matrix — {best_name_e}')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
save_fig("eng_confusion_matrix.png")

# ── GRAPH 3 — ROC Curve ──
fig, ax = plt.subplots(figsize=(7, 5))
colors_roc = [ACCENT, SUCCESS, DANGER, WARNING]
for (name, res), col in zip(results_e.items(), colors_roc):
    clf = res['model']
    if hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba(X_te_e)[:, 1]
    else:
        proba = clf.decision_function(X_te_e)
    fpr, tpr, _ = roc_curve(y_te_e, proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, lw=1.8, label=f'{name} (AUC={roc_auc:.3f})')
ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Engine Condition Classifiers')
ax.legend(framealpha=0, fontsize=9)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
save_fig("eng_roc_curves.png")

# ── GRAPH 4 — Feature Importance ──
if hasattr(best_clf_e, 'feature_importances_'):
    imp = best_clf_e.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(FEATURES_ENG)),
                  [imp[i] for i in idx],
                  color=[PALETTE[i % len(PALETTE)] for i in range(len(FEATURES_ENG))],
                  edgecolor='none', alpha=0.9)
    ax.set_xticks(range(len(FEATURES_ENG)))
    ax.set_xticklabels([FEATURES_ENG[i] for i in idx], rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Importance Score')
    ax.set_title('Feature Importance — Engine Condition')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#8b949e')
    save_fig("eng_feature_importance.png")

# ── GRAPH 5 — Learning Curve ──
train_sizes, train_scores, val_scores = learning_curve(
    best_clf_e, X_e_scaled, y_e, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy', n_jobs=-1)

fig, ax = plt.subplots(figsize=(8, 5))
tr_m  = train_scores.mean(axis=1); tr_s  = train_scores.std(axis=1)
val_m = val_scores.mean(axis=1);   val_s = val_scores.std(axis=1)
ax.fill_between(train_sizes, tr_m - tr_s,  tr_m + tr_s,  alpha=0.15, color=ACCENT)
ax.fill_between(train_sizes, val_m - val_s, val_m + val_s, alpha=0.15, color=SUCCESS)
ax.plot(train_sizes, tr_m,  'o-', color=ACCENT,   lw=2,   label='Training Score')
ax.plot(train_sizes, val_m, 's-', color=SUCCESS,  lw=2,   label='CV Score')
ax.set_xlabel('Training Samples'); ax.set_ylabel('Accuracy')
ax.set_title('Learning Curve — Engine Condition Model')
ax.legend(framealpha=0)
save_fig("eng_learning_curve.png")

# ── GRAPH 6 — Cross-Validation Scores ──
fig, ax = plt.subplots(figsize=(9, 5))
cv_data = {n: cross_val_score(results_e[n]['model'], X_e_scaled, y_e, cv=5) for n in names}
positions = range(len(names))
parts = ax.violinplot([cv_data[n] for n in names], positions=positions,
                       showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(PALETTE[i % len(PALETTE)])
    pc.set_alpha(0.7)
parts['cmeans'].set_color(WARNING)
parts['cmedians'].set_color('#ffffff')
ax.set_xticks(positions)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('CV Accuracy'); ax.set_title('Cross-Validation Score Distribution')
save_fig("eng_cv_scores.png")

# ── GRAPH 7 — Precision-Recall Curve ──
fig, ax = plt.subplots(figsize=(7, 5))
for (name, res), col in zip(results_e.items(), colors_roc):
    clf = res['model']
    if hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba(X_te_e)[:, 1]
        prec, rec, _ = precision_recall_curve(y_te_e, proba)
        ap = average_precision_score(y_te_e, proba)
        ax.plot(rec, prec, color=col, lw=1.8, label=f'{name} (AP={ap:.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves — Engine Condition')
ax.legend(framealpha=0, fontsize=9)
save_fig("eng_precision_recall.png")

# ── GRAPH 8 — Class Distribution ──
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
counts = df_engine['Engine Condition'].value_counts()
labels = ['Normal (0)', 'Fault (1)']
wedge_colors = [SUCCESS, DANGER]
axes[0].pie(counts.values, labels=labels, colors=wedge_colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2},
            textprops={'color': '#e6edf3'})
axes[0].set_title('Engine Condition Distribution')

for feat, col in zip(FEATURES_ENG[:3], [ACCENT, SUCCESS, DANGER]):
    axes[1].hist(df_engine[df_engine['Engine Condition']==0][feat],
                 bins=40, alpha=0.5, color=SUCCESS, density=True, label=f'{feat} (Normal)')
    axes[1].hist(df_engine[df_engine['Engine Condition']==1][feat],
                 bins=40, alpha=0.5, color=DANGER, density=True, label=f'{feat} (Fault)')
axes[1].set_xlabel('Value'); axes[1].set_ylabel('Density')
axes[1].set_title('Feature Distribution by Condition')
axes[1].legend(fontsize=7, framealpha=0, ncol=2)
save_fig("eng_distributions.png")

print(f"\n  Engine model accuracy: {results_e[best_name_e]['test_acc']*100:.2f}%")

# ═══════════════════════════════════════════════════════════
#  PART 2 — SERVICE RECORDS MODEL
# ═══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  PART 2 — SERVICE COST & MAINTENANCE PREDICTION")
print("═"*60)

df_svc = pd.read_csv(os.path.join(DATA_DIR, "service_records.csv"))
df_svc.columns = df_svc.columns.str.strip()
print(f"  Dataset: {df_svc.shape[0]} rows × {df_svc.shape[1]} cols")

MAINT_ITEMS = ['oil_filter', 'engine_oil', 'washer_plug_drain',
               'dust_and_pollen_filter', 'whell_alignment_and_balancing',
               'air_clean_filter', 'fuel_filter', 'spark_plug',
               'brake_fluid', 'brake_and_clutch_oil', 'transmission_fluid',
               'brake_pads', 'clutch', 'coolant']

le_brand  = LabelEncoder()
le_model  = LabelEncoder()
le_engine = LabelEncoder()
le_region = LabelEncoder()

df_svc['brand_enc']  = le_brand.fit_transform(df_svc['brand'])
df_svc['model_enc']  = le_model.fit_transform(df_svc['model'])
df_svc['engine_enc'] = le_engine.fit_transform(df_svc['engine_type'])
df_svc['region_enc'] = le_region.fit_transform(df_svc['region'])

FEATURES_SVC = ['brand_enc', 'model_enc', 'engine_enc', 'region_enc',
                 'make_year', 'mileage', 'mileage_range']

X_s = df_svc[FEATURES_SVC]
y_cost = df_svc['cost']

X_tr_s, X_te_s, y_tr_cost, y_te_cost = train_test_split(
    X_s, y_cost, test_size=0.2, random_state=42)

# ── Cost Regression ──
reg = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
reg.fit(X_tr_s, y_tr_cost)
y_pred_cost = reg.predict(X_te_s)

mae  = mean_absolute_error(y_te_cost, y_pred_cost)
mse  = mean_squared_error(y_te_cost, y_pred_cost)
rmse = np.sqrt(mse)
r2   = r2_score(y_te_cost, y_pred_cost)
print(f"  Cost Regressor → MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.4f}")

joblib.dump(reg,       os.path.join(MODEL_DIR, "cost_model.pkl"))
joblib.dump(le_brand,  os.path.join(MODEL_DIR, "le_brand.pkl"))
joblib.dump(le_model,  os.path.join(MODEL_DIR, "le_model.pkl"))
joblib.dump(le_engine, os.path.join(MODEL_DIR, "le_engine.pkl"))
joblib.dump(le_region, os.path.join(MODEL_DIR, "le_region.pkl"))
joblib.dump(FEATURES_SVC, os.path.join(MODEL_DIR, "service_features.pkl"))

# ── Multi-label maintenance classifiers ──
item_models = {}
item_scores = {}
for item in MAINT_ITEMS:
    clf_i = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_i.fit(X_tr_s, df_svc.loc[X_tr_s.index, item])
    acc = accuracy_score(df_svc.loc[X_te_s.index, item], clf_i.predict(X_te_s))
    item_models[item] = clf_i
    item_scores[item] = acc
    print(f"  {item:35s} accuracy: {acc:.4f}")

joblib.dump(item_models, os.path.join(MODEL_DIR, "item_models.pkl"))
print("  ✓  Service models saved")

# ── GRAPH 9 — Actual vs Predicted Cost ──
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_te_cost, y_pred_cost, alpha=0.45, s=18, color=ACCENT, edgecolors='none')
mn = min(y_te_cost.min(), y_pred_cost.min())
mx = max(y_te_cost.max(), y_pred_cost.max())
ax.plot([mn, mx], [mn, mx], '--', color=WARNING, lw=1.5, label='Perfect Prediction')
ax.set_xlabel('Actual Cost (₹)'); ax.set_ylabel('Predicted Cost (₹)')
ax.set_title(f'Actual vs Predicted Service Cost  |  R²={r2:.4f}  MAE=₹{mae:.0f}')
ax.legend(framealpha=0)
save_fig("svc_actual_vs_predicted.png")

# ── GRAPH 10 — Residual Plot ──
residuals = y_te_cost.values - y_pred_cost
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(y_pred_cost, residuals, alpha=0.45, s=15, color=PURPLE, edgecolors='none')
axes[0].axhline(0, color=WARNING, lw=1.5, linestyle='--')
axes[0].set_xlabel('Predicted Cost'); axes[0].set_ylabel('Residuals')
axes[0].set_title('Residual Plot')
axes[1].hist(residuals, bins=40, color=ACCENT, alpha=0.8, edgecolor='none')
axes[1].axvline(0, color=WARNING, lw=1.5, linestyle='--')
axes[1].set_xlabel('Residual Value'); axes[1].set_ylabel('Frequency')
axes[1].set_title('Residual Distribution')
save_fig("svc_residuals.png")

# ── GRAPH 11 — Maintenance Item Accuracy ──
fig, ax = plt.subplots(figsize=(12, 5))
items  = list(item_scores.keys())
scores = [item_scores[i] for i in items]
colors = [SUCCESS if s >= 0.80 else WARNING if s >= 0.65 else DANGER for s in scores]
bars = ax.barh(items, scores, color=colors, edgecolor='none', alpha=0.85)
ax.axvline(0.8, color=WARNING, linestyle='--', lw=1.2, label='80% threshold')
ax.set_xlabel('Accuracy'); ax.set_title('Maintenance Item Classification Accuracy')
ax.legend(framealpha=0)
for bar, s in zip(bars, scores):
    ax.text(s + 0.003, bar.get_y() + bar.get_height()/2,
            f'{s:.3f}', va='center', fontsize=8, color='#8b949e')
ax.set_xlim(0, 1.08)
save_fig("svc_item_accuracy.png")

# ── GRAPH 12 — Feature Importance (Cost) ──
imp_s = reg.feature_importances_
idx_s = np.argsort(imp_s)[::-1]
feat_labels = ['Brand', 'Model', 'Engine Type', 'Region', 'Make Year', 'Mileage', 'Mileage Range']
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(len(feat_labels)),
       [imp_s[i] for i in idx_s],
       color=[PALETTE[i] for i in range(len(feat_labels))], edgecolor='none', alpha=0.9)
ax.set_xticks(range(len(feat_labels)))
ax.set_xticklabels([feat_labels[i] for i in idx_s], rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Importance'); ax.set_title('Feature Importance — Service Cost Prediction')
save_fig("svc_feature_importance.png")

# ── GRAPH 13 — Cost Distribution ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df_svc['cost'], bins=50, color=ACCENT, alpha=0.8, edgecolor='none')
axes[0].axvline(df_svc['cost'].mean(), color=WARNING, lw=1.5, linestyle='--',
                label=f'Mean ₹{df_svc["cost"].mean():.0f}')
axes[0].set_xlabel('Cost (₹)'); axes[0].set_ylabel('Frequency')
axes[0].set_title('Service Cost Distribution'); axes[0].legend(framealpha=0)

brand_cost = df_svc.groupby('brand')['cost'].mean().sort_values(ascending=True)
colors_bc  = [PALETTE[i % len(PALETTE)] for i in range(len(brand_cost))]
axes[1].barh(brand_cost.index, brand_cost.values, color=colors_bc, edgecolor='none', alpha=0.9)
axes[1].set_xlabel('Average Cost (₹)'); axes[1].set_title('Average Service Cost by Brand')
save_fig("svc_cost_distribution.png")

# ── GRAPH 14 — Maintenance Item Frequency ──
fig, ax = plt.subplots(figsize=(12, 5))
freq = {item: df_svc[item].sum() for item in MAINT_ITEMS}
freq_sorted = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
colors_freq = [PALETTE[i % len(PALETTE)] for i in range(len(freq_sorted))]
ax.bar(freq_sorted.keys(), freq_sorted.values(), color=colors_freq, edgecolor='none', alpha=0.9)
ax.set_xticklabels(freq_sorted.keys(), rotation=35, ha='right', fontsize=8)
ax.set_ylabel('Count'); ax.set_title('Maintenance Item Replacement Frequency')
save_fig("svc_item_frequency.png")

# ── GRAPH 15 — Mileage vs Cost Scatter ──
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(df_svc['mileage'], df_svc['cost'],
                     c=df_svc['make_year'], cmap='viridis',
                     alpha=0.5, s=20, edgecolors='none')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Make Year', color='#8b949e')
cbar.ax.yaxis.set_tick_params(color='#8b949e')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e')
ax.set_xlabel('Mileage (km)'); ax.set_ylabel('Service Cost (₹)')
ax.set_title('Mileage vs Service Cost (colored by Make Year)')
save_fig("svc_mileage_vs_cost.png")

# ── GRAPH 16 — Model Metrics Summary ──
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
# Engine classifier summary
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
from sklearn.metrics import precision_score, recall_score, f1_score
p = precision_score(y_te_e, y_pred_e, average='weighted')
r_score = recall_score(y_te_e, y_pred_e, average='weighted')
f1 = f1_score(y_te_e, y_pred_e, average='weighted')
acc_final = accuracy_score(y_te_e, y_pred_e)
values_e = [acc_final, p, r_score, f1]
bars_e = axes[0].bar(metrics_names, values_e,
                      color=[ACCENT, SUCCESS, WARNING, PURPLE], edgecolor='none', alpha=0.85)
axes[0].set_ylim(0, 1.1); axes[0].set_title('Engine Model — Performance Metrics')
for bar, v in zip(bars_e, values_e):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{v:.3f}', ha='center', fontsize=10, color='#e6edf3', fontweight='bold')

# Cost regressor summary
reg_metrics = ['MAE', 'RMSE', 'R² × 1000']
reg_values  = [mae, rmse, r2 * 1000]
colors_rm   = [DANGER, WARNING, SUCCESS]
bars_r = axes[1].bar(reg_metrics, reg_values, color=colors_rm, edgecolor='none', alpha=0.85)
axes[1].set_title('Cost Prediction — Performance Metrics')
for bar, v, lbl in zip(bars_r, reg_values, ['₹'+str(int(mae)), '₹'+str(int(rmse)), f'{r2:.4f}']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(reg_values)*0.01,
                 lbl, ha='center', fontsize=9, color='#e6edf3')
save_fig("summary_metrics.png")

print("\n" + "═"*60)
print("  ALL TRAINING COMPLETE — GRAPHS SAVED")
print("═"*60)
all_graphs = [f for f in os.listdir(GRAPH_DIR) if f.endswith('.png')]
print(f"  Total graphs generated: {len(all_graphs)}")
for g in sorted(all_graphs):
    print(f"    → {g}")
print()
