"""Revision 2: Model 1 without culture categories, and DeLong tests of each culture-excluded model vs its own model."""
import json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from common import SEED, STEPWISE, load
from r1_heldout_benchmark import XGB_GRID, run, xgb
from r2_analyses import CULTURE, delong
warnings.filterwarnings('ignore')
X, y = load()
Xd, Xt, yd, yt = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
cols = [c for c in STEPWISE['Model 1'] if c not in CULTURE]
r, p = run('Model 1 excl. culture', xgb(), XGB_GRID, Xd[cols], yd, Xt[cols], yt)
sp = pd.read_csv('results/r1_stepwise_test_probs.csv'); r2p = pd.read_csv('results/r2_test_probs.csv')
out = {'model1_excl_culture': r, 'delong_own': {}}
for m, q in [('Model 1', p), ('Model 2', r2p['Model 2 excl. culture']), ('Model 3', r2p['Model 3 excl. culture'])]:
    out['delong_own'][m] = delong(yt, sp[f'stepwise|{m}'], q)
    print(m, 'vs excl. culture', [round(v, 4) for v in out['delong_own'][m]])
json.dump(out, open('results/r2_extra.json', 'w'), indent=1, default=float)
