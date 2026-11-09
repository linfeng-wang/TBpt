"""Revision 2 (R1 comment 2, R2 comment 7): crude and adjusted odds ratios computed consistently.

Crude OR: univariable logistic regression per feature. Adjusted OR: one multivariable logistic regression
including all S9 features jointly. Complete cases only. Outcome 1 = cured.
"""
import sys
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import fisher_exact
from common import load

S9 = ['Pretomanid', 'Negative', 'Clofazimine', 'Patient stopped treatment', 'Bedaquiline',
      'Treatment ineffective due to additional resistance', 'Adverse event', 'Continuation of treatment', 'HIV',
      'p-aminosalicylic acid', 'gender', 'Capreomycin', 'pleural_effusion_percent_of_hemithorax_involved',
      'Ethionamide', 'Patient alcohol abuse', 'Amikacin', 'Amoxicillin-clavulanate', '100 to 200', 'lineage2',
      'More than 200', 'Delamanid', 'type_of_resistance', 'regimen_count', 'mediumcavities', 'collapse',
      'isanynoncalcifiednoduleexist', 'infiltrate_lowgroundglassdensity', 'aremultiplenoduleexists',
      'number_of_daily_contacts', 'bmi', 'age_of_onset']
X, y = load()
d = pd.concat([X[S9], y.rename('y')], axis=1).dropna()
print('complete cases:', len(d), 'cured', int(d.y.sum()))
rows = []
full = sm.Logit(d.y, sm.add_constant(d[S9])).fit(disp=0, maxiter=500)
ci = np.exp(full.conf_int())
for f in S9:
    uni = sm.Logit(d.y, sm.add_constant(d[[f]])).fit(disp=0, maxiter=500)
    binary = set(d[f].unique()) <= {0, 1}
    fisher = fisher_exact(pd.crosstab(d[f], d.y).values)[0] if binary else np.nan
    rows.append({'feature': f, 'binary': binary, 'crude_OR_logit': np.exp(uni.params[f]), 'crude_OR_fisher': fisher,
                 'aOR': np.exp(full.params[f]), 'aOR_lo': ci.loc[f, 0], 'aOR_hi': ci.loc[f, 1], 'p': full.pvalues[f]})
r = pd.DataFrame(rows)
r.to_csv('results/r1_odds_ratios.csv', index=False)
pd.set_option('display.width', 250); print(r.round(3).to_string())
