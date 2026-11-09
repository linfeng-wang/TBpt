"""Revision 2, Reviewer 2: temporal leakage sensitivity, calibration, decision curves, clinical baseline, DeLong tests.

Uses the same development / held-out split as r1_heldout_benchmark.py.
"""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import SEED, STEPWISE, load
from r1_heldout_benchmark import XGB_GRID, run, xgb

warnings.filterwarnings('ignore')
OUT = 'results'
CULTURE = ['1 to 19', '100 to 200', '20 to 100', 'MOTT', 'More than 200', 'Negative', 'Nonspecific microflora',
           'Not_Reported-culture', 'Not done-culture', 'Positive', 'Study in progress', 'Unknown_result-culture']
STATUS = ['Adverse event', 'Continuation of treatment', 'Drug(s) no longer available', 'New drugs available',
          'Not_Reported-treatment_status', 'Patient stopped treatment', 'Terminated from study',
          'Treatment ineffective due to additional resistance']
XRAY_CODING = ['overall_percent_of_abnormal_volume', 'pleural_effusion_percent_of_hemithorax_involved']


# ---------- DeLong test (Sun & Xu fast implementation) ----------
def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1; i = k
    out = np.empty(n); out[j] = t; return out


def delong(y, p1, p2):
    y = np.asarray(y); pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum(); aucs, v10, v01 = [], [], []
    for p in (p1, p2):
        p = np.asarray(p); tx, ty = _midrank(p[pos]), _midrank(p[neg]); tz = _midrank(np.concatenate([p[pos], p[neg]]))
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n); aucs.append(auc)
        v10.append((tz[:m] - tx) / n); v01.append(1 - (tz[m:] - ty) / m)
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], 2 * norm.sf(abs(z))


# ---------- calibration and decision curves ----------
def calibration(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    intercept = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return {'intercept': float(intercept), 'slope': float(slope)}


def net_benefit(y_unfav, risk, thresholds):
    n = len(y_unfav); out = []
    for t in thresholds:
        pos = risk >= t
        tp = np.sum(pos & (y_unfav == 1)); fp = np.sum(pos & (y_unfav == 0))
        out.append({'threshold': t, 'model': tp / n - fp / n * t / (1 - t),
                    'treat_all': y_unfav.mean() - (1 - y_unfav.mean()) * t / (1 - t)})
    return out


def main():
    X, y = load()
    Xd, Xt, yd, yt = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    allc = STEPWISE['All features']; res = {}
    # 1. leakage / coding sensitivity (tuned XGBoost, held-out)
    sets = {
        'All features excl. period span': [c for c in allc if c != 'period_span'],
        'All features excl. period span and treatment status': [c for c in allc if c != 'period_span' and c not in STATUS],
        'All features excl. period span, treatment status and culture': [c for c in allc if c != 'period_span' and c not in STATUS + CULTURE],
        'Model 2 excl. culture': [c for c in STEPWISE['Model 2'] if c not in CULTURE],
        'Model 3 excl. culture': [c for c in STEPWISE['Model 3'] if c not in CULTURE],
        'All features excl. two X-ray ordinal features': [c for c in allc if c not in XRAY_CODING],
    }
    probs = {'y_true': yt.to_numpy()}
    rows = []
    for name, cols in sets.items():
        r, p = run(name, xgb(), XGB_GRID, Xd[cols], yd, Xt[cols], yt); rows.append(r); probs[name] = p
    res['sensitivity'] = rows
    # 2. clinical baseline: established indicators (drug resistance, nutritional status, treatment response)
    base = ['type_of_resistance', 'bmi', 'Negative', 'HIV', 'age_of_onset', 'gender']
    est = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=5000))])
    r, p = run('Clinical baseline (logistic regression)', est, {'clf__C': [0.01, 0.1, 1, 10]}, Xd[base], yd, Xt[base], yt)
    res['baseline'] = r; probs['Clinical baseline'] = p
    # 3. DeLong comparisons, calibration and decision curves on held-out probabilities
    sp = pd.read_csv(f'{OUT}/r1_stepwise_test_probs.csv'); bp = pd.read_csv(f'{OUT}/r1_benchmark_test_probs.csv')
    assert np.array_equal(sp.y_true, yt.to_numpy()) and np.array_equal(bp.y_true, yt.to_numpy())
    full = sp['stepwise|All features'].to_numpy()
    de = []
    for c in bp.columns[1:]:
        m = c.split('|')[1]
        if m != 'XGBoost':
            a1, a2, pv = delong(yt, full, bp[c]); de.append({'comparison': f'XGBoost vs {m}', 'auc1': a1, 'auc2': a2, 'p': pv})
    steps = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'All features']
    for a, b in zip(steps[:-1], steps[1:]):
        a1, a2, pv = delong(yt, sp[f'stepwise|{b}'], sp[f'stepwise|{a}']); de.append({'comparison': f'{b} vs {a}', 'auc1': a1, 'auc2': a2, 'p': pv})
    for m in ['Model 2', 'Model 3', 'All features']:
        a1, a2, pv = delong(yt, sp[f'stepwise|{m}'], probs['Clinical baseline']); de.append({'comparison': f'{m} vs clinical baseline', 'auc1': a1, 'auc2': a2, 'p': pv})
    for k in sets:
        a1, a2, pv = delong(yt, full, probs[k]); de.append({'comparison': f'All features vs {k}', 'auc1': a1, 'auc2': a2, 'p': pv})
    res['delong'] = de
    cal = {s: calibration(yt.to_numpy(), sp[f'stepwise|{s}'].to_numpy()) for s in steps}
    cal['Clinical baseline'] = calibration(yt.to_numpy(), probs['Clinical baseline'])
    res['calibration'] = cal
    thr = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    yu = 1 - yt.to_numpy()
    res['decision_curve'] = {s: net_benefit(yu, 1 - sp[f'stepwise|{s}'].to_numpy(), thr) for s in steps}
    res['decision_curve']['Clinical baseline'] = net_benefit(yu, 1 - probs['Clinical baseline'], thr)
    with open(f'{OUT}/r2_analyses.json', 'w') as f:
        json.dump(res, f, indent=1, default=float)
    pd.DataFrame(probs).to_csv(f'{OUT}/r2_test_probs.csv', index=False)
    for d in de:
        print(f"{d['comparison']:70s} {d['auc1']:.3f} vs {d['auc2']:.3f}  p={d['p']:.3g}")
    for k, v in cal.items():
        print(f"calibration {k:20s} intercept {v['intercept']:.3f} slope {v['slope']:.3f}")
    for k, v in res['decision_curve'].items():
        print('NB', k, [(d['threshold'], round(d['model'], 3), round(d['treat_all'], 3)) for d in v])


if __name__ == '__main__':
    main()
