
import os
import numpy as np
import pandas as pd
import joblib

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

engine_model    = joblib.load(os.path.join(MODEL_DIR, 'engine_model.pkl'))
engine_scaler   = joblib.load(os.path.join(MODEL_DIR, 'engine_scaler.pkl'))
engine_features = joblib.load(os.path.join(MODEL_DIR, 'engine_features.pkl'))
bg_array        = joblib.load(os.path.join(MODEL_DIR, 'shap_background.pkl'))

FEATURES = engine_features

FEATURE_LABELS = {
    'Engine rpm':       'Engine RPM',
    'Lub oil pressure': 'Lub Oil Pressure',
    'Fuel pressure':    'Fuel Pressure',
    'Coolant pressure': 'Coolant Pressure',
    'lub oil temp':     'Lub Oil Temp',
    'Coolant temp':     'Coolant Temp',
}
FEATURE_UNITS = {
    'Engine rpm': 'RPM', 'Lub oil pressure': 'bar',
    'Fuel pressure': 'bar', 'Coolant pressure': 'bar',
    'lub oil temp': '°C', 'Coolant temp': '°C',
}
HEALTHY_RANGES = {
    'Engine rpm':       (700, 2000),
    'Lub oil pressure': (2.0, 6.0),
    'Fuel pressure':    (8.0, 22.0),
    'Coolant pressure': (1.0, 4.0),
    'lub oil temp':     (60, 100),
    'Coolant temp':     (70, 95),
}


def _predict_fault_proba(X_raw: np.ndarray) -> np.ndarray:
    df_in  = pd.DataFrame(X_raw, columns=FEATURES)
    scaled = engine_scaler.transform(df_in)
    return engine_model.predict_proba(scaled)[:, 1]


def explain(input_values: dict, n_samples: int = 150) -> dict:
    """
    Compute SHAP values using interventional sampling.
    Returns a rich dict ready to be serialised to JSON.
    """
    x  = np.array([input_values[f] for f in FEATURES], dtype=float)
    bg = bg_array.copy()
    if n_samples < len(bg):
        idx = np.random.choice(len(bg), n_samples, replace=False)
        bg  = bg[idx]

    # Baseline = average model output over background
    baseline  = float(_predict_fault_proba(bg).mean())
    full_pred = float(_predict_fault_proba(x.reshape(1, -1))[0])
    gap       = full_pred - baseline

    # Per-feature SHAP
    raw_shap = {}
    for i, feat in enumerate(FEATURES):
        mixed        = bg.copy()
        mixed[:, i]  = x[i]
        raw_shap[feat] = float(_predict_fault_proba(mixed).mean()) - baseline

    # Rescale so values sum exactly to gap (remove Monte Carlo noise)
    shap_sum = sum(raw_shap.values())
    if abs(shap_sum) > 1e-9:
        scale = gap / shap_sum
        shap_vals = {k: v * scale for k, v in raw_shap.items()}
    else:
        shap_vals = raw_shap

    # Sort by absolute value descending
    sorted_feats = sorted(FEATURES, key=lambda f: abs(shap_vals[f]), reverse=True)

    # Build rich feature list
    total_abs = sum(abs(v) for v in shap_vals.values()) or 1e-9
    features_out = []
    cumulative   = baseline
    for feat in sorted_feats:
        sv       = shap_vals[feat]
        pct      = sv / total_abs * 100
        lo, hi   = HEALTHY_RANGES[feat]
        val      = input_values[feat]
        in_range = (lo <= val <= hi)
        cumulative += sv
        features_out.append({
            'key':        feat,
            'label':      FEATURE_LABELS.get(feat, feat),
            'unit':       FEATURE_UNITS.get(feat, ''),
            'value':      round(float(val), 2),
            'shap':       round(sv, 6),
            'shap_pct':   round(pct, 1),
            'cumulative': round(cumulative, 4),
            'direction':  'increase' if sv > 0 else 'decrease',
            'in_range':   in_range,
            'healthy_lo': lo,
            'healthy_hi': hi,
            'bar_pct':    round(min(100, abs(pct)), 1),
        })

    # Natural language summary
    top = features_out[0]
    second = features_out[1]
    dir1 = 'increases' if top['shap'] > 0 else 'decreases'
    dir2 = 'increases' if second['shap'] > 0 else 'decreases'
    summary = (
        f"{top['label']} is the biggest driver: it {dir1} fault probability by "
        f"{abs(top['shap_pct']):.1f}%. {second['label']} {dir2} it by "
        f"{abs(second['shap_pct']):.1f}%. "
        f"Overall the model predicts {full_pred*100:.1f}% fault probability "
        f"vs. a typical baseline of {baseline*100:.1f}%."
    )

    # Verdict
    if gap > 0.2:
        verdict = "High-risk configuration — multiple parameters are pushing toward fault."
    elif gap > 0.05:
        verdict = "Elevated risk — some parameters are abnormal."
    elif gap < -0.1:
        verdict = "Healthy configuration — parameters are actively reducing fault probability."
    else:
        verdict = "Near-baseline prediction — parameters are close to typical engine conditions."

    return {
        'success':    True,
        'baseline':   round(baseline * 100, 2),
        'prediction': round(full_pred * 100, 2),
        'gap':        round(gap * 100, 2),
        'features':   features_out,
        'summary':    summary,
        'verdict':    verdict,
        'n_bg':       len(bg),
    }
