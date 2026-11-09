"""Revision 2, Reviewer 1 (comments 2 and 3): held-out evaluation, splines, imputation, balancing, thresholds.

Design
- Stratified 80/20 split (seed 42) into a development set and an untouched held-out test set.
- Hyperparameters are tuned by grid search with 5-fold stratified CV (AUC) on the development set only.
- Imputation / scaling / spline expansion are fitted inside each pipeline (per fold), so nothing leaks
  from validation folds or the test set.
- The tuned model is refitted on the full development set and evaluated once on the held-out test set,
  at the default 0.5 threshold and at the Youden-optimal threshold chosen on development-set
  out-of-fold predictions. 95% CIs on the test set by 1,000 stratified bootstrap resamples.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from common import CONTINUOUS, SEED, STEPWISE, load

warnings.filterwarnings('ignore')
OUT = 'results'
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)
NJ = 24


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {'Accuracy': accuracy_score(y, pred), 'Sensitivity': recall_score(y, pred, pos_label=1),
            'Specificity': recall_score(y, pred, pos_label=0), 'Precision': precision_score(y, pred),
            'F1': f1_score(y, pred), 'AUC-ROC': roc_auc_score(y, p), 'Brier': brier_score_loss(y, p)}


def bootstrap_ci(y, p, thr, n=1000):
    rng = np.random.default_rng(SEED)
    y, p = np.asarray(y), np.asarray(p)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    draws = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        draws.append(metrics(y[idx], p[idx], thr))
    d = pd.DataFrame(draws)
    return {k: (d[k].quantile(0.025), d[k].quantile(0.975)) for k in d}


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def imputer(strategy, cols):
    """Mode imputation for all columns ('mode'), or median for continuous + mode for the rest ('median')."""
    if strategy == 'mode':
        return SimpleImputer(strategy='most_frequent')
    cont = [c for c in CONTINUOUS if c in cols]
    return ColumnTransformer([('med', SimpleImputer(strategy='median'), cont)],
                             remainder=SimpleImputer(strategy='most_frequent'))


def spline_lr(cols, strategy='mode'):
    cont = [c for c in CONTINUOUS if c in cols]
    other = [c for c in cols if c not in cont]
    imp = SimpleImputer(strategy='median' if strategy == 'median' else 'most_frequent')
    pre = ColumnTransformer([
        ('spl', Pipeline([('imp', imp), ('sc', StandardScaler()),
                          ('spl', SplineTransformer(n_knots=5, degree=3))]), cont),
        ('lin', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('sc', StandardScaler())]), other)])
    return Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=5000))]), {'clf__C': [0.01, 0.1, 1, 10]}


def xgb(spw=None):
    return XGBClassifier(subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss',
                         tree_method='hist', random_state=SEED, n_jobs=4, scale_pos_weight=spw)


XGB_GRID = {'max_depth': [3, 6, 9], 'learning_rate': [0.05, 0.1], 'n_estimators': [300, 500]}


def models(cols, strategy='mode'):
    def std(clf):
        return Pipeline([('imp', imputer(strategy, cols)), ('sc', StandardScaler()), ('clf', clf)])
    return {
        'Logistic regression': (std(LogisticRegression(max_iter=5000)), {'clf__C': [0.01, 0.1, 1, 10]}),
        'Logistic regression (splines)': spline_lr(cols, strategy),
        'Naive Bayes': (std(GaussianNB()), {'clf__var_smoothing': [1e-9, 1e-6, 1e-3]}),
        'SVM': (std(SVC(probability=True, random_state=SEED)), {'clf__C': [0.1, 1, 10]}),
        'k-NN': (std(KNeighborsClassifier()), {'clf__n_neighbors': [5, 15, 31]}),
        'MLP': (std(MLPClassifier(max_iter=1000, early_stopping=True, random_state=SEED)),
                {'clf__hidden_layer_sizes': [(100,), (100, 50, 30)], 'clf__alpha': [1e-4, 1e-2]}),
        'CatBoost': (CatBoostClassifier(verbose=0, random_seed=SEED, thread_count=4),
                     {'depth': [4, 6], 'iterations': [500, 1000], 'learning_rate': [0.05, 0.1]}),
        'LightGBM': (LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=4),
                     {'num_leaves': [15, 31], 'n_estimators': [300, 500], 'learning_rate': [0.05, 0.1]}),
        'XGBoost': (xgb(), XGB_GRID),
    }


def run(name, est, grid, Xd, yd, Xt, yt):
    gs = GridSearchCV(est, grid, scoring='roc_auc', cv=CV, n_jobs=NJ, refit=True).fit(Xd, yd)
    oof = cross_val_predict(gs.best_estimator_, Xd, yd, cv=CV, method='predict_proba', n_jobs=5)[:, 1]
    thr = youden(yd, oof)
    p = gs.best_estimator_.predict_proba(Xt)[:, 1]
    row = {'model': name, 'n_features': Xd.shape[1], 'best_params': gs.best_params_,
           'youden_threshold_dev': thr, 'dev_cv': metrics(yd, oof),
           'test_0.5': metrics(yt, p), 'test_0.5_ci': bootstrap_ci(yt, p, 0.5),
           'test_youden': metrics(yt, p, thr)}
    print(f"{name:32s} devCV AUC {row['dev_cv']['AUC-ROC']:.4f} | test AUC {row['test_0.5']['AUC-ROC']:.4f} "
          f"acc {row['test_0.5']['Accuracy']:.4f} sens {row['test_0.5']['Sensitivity']:.4f} "
          f"spec {row['test_0.5']['Specificity']:.4f} | thr {thr:.3f}", flush=True)
    return row, p


def main(part):
    X, y = load()
    Xd, Xt, yd, yt = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f'development n={len(yd)} (cured {yd.sum()}), held-out test n={len(yt)} (cured {yt.sum()})', flush=True)
    res, probs = [], {'y_true': yt.to_numpy()}
    if part in ('stepwise', 'all'):
        for m, cols in STEPWISE.items():
            r, p = run(m, xgb(), XGB_GRID, Xd[cols], yd, Xt[cols], yt)
            r['analysis'] = 'stepwise'; res.append(r); probs[f'stepwise|{m}'] = p
    if part in ('benchmark', 'all'):
        cols = STEPWISE['All features']
        for m, (est, grid) in models(cols).items():
            r, p = run(m, est, grid, Xd[cols], yd, Xt[cols], yt)
            r['analysis'] = 'benchmark'; res.append(r); probs[f'benchmark|{m}'] = p
    if part in ('sensitivity', 'all'):
        cols = STEPWISE['All features']
        for m in ['Logistic regression', 'Logistic regression (splines)', 'MLP', 'SVM']:
            est, grid = models(cols, 'median')[m]
            r, _ = run(f'{m} [median imputation]', est, grid, Xd[cols], yd, Xt[cols], yt)
            r['analysis'] = 'imputation_sensitivity'; res.append(r)
        spw = (yd == 0).sum() / (yd == 1).sum()  # up-weights the minority (unfavourable) class
        r, _ = run('XGBoost [class-weighted]', xgb(spw), XGB_GRID, Xd[cols], yd, Xt[cols], yt)
        r['analysis'] = 'balancing_sensitivity'; res.append(r)
    with open(f'{OUT}/r1_{part}.json', 'w') as f:
        json.dump(res, f, indent=1, default=float)
    pd.DataFrame(probs).to_csv(f'{OUT}/r1_{part}_test_probs.csv', index=False)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')
