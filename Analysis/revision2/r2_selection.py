"""Revision 2 (R2 comment 4): cohort selection counts and included vs excluded comparison."""
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
RAW = '/mnt/storageG1/lwang/Projects/TBpt/Clinical/TB_Portals_Published_Clinical_Data_August_2023/'
raw = pd.read_csv(RAW + 'TB_Portals_Patient_Cases_August_2023.csv', low_memory=False)
na1 = pd.read_csv('/mnt/storageG1/lwang/Projects/TBpt/Analysis/clinical_lung_na1.csv')
coh = na1.loc[na1[na1.outcome != 'Completed'].drop(columns=['condition_id']).drop_duplicates().index]
inc = raw.condition_id.isin(coh.condition_id)
print('raw', len(raw)); print(raw.outcome.value_counts(dropna=False).to_dict())
elig = raw.outcome.isin(['Cured', 'Died', 'Failure', 'Palliative Care'])
print('eligible outcome', int(elig.sum()), '| included', int(inc.sum()), '| eligible not included', int((elig & ~inc).sum()))
raw['HIV'] = raw.comorbidity.fillna('').str.contains('HIV')
raw['MDRplus'] = raw.type_of_resistance.isin(['MDR non XDR', 'Pre-XDR', 'XDR'])
raw['male'] = raw.gender == 'Male'
rows = []
for v in ['male', 'HIV', 'MDRplus']:
    t = pd.crosstab(inc, raw[v]); p = chi2_contingency(t)[1]
    rows.append((v, f"{raw.loc[inc, v].sum()} ({raw.loc[inc, v].mean()*100:.1f}%)", f"{raw.loc[~inc, v].sum()} ({raw.loc[~inc, v].mean()*100:.1f}%)", p))
for v in ['age_of_onset', 'bmi']:
    a, b = raw.loc[inc, v].dropna(), raw.loc[~inc, v].dropna()
    rows.append((v, f"{a.median():.1f} ({a.quantile(.25):.1f}-{a.quantile(.75):.1f})", f"{b.median():.1f} ({b.quantile(.25):.1f}-{b.quantile(.75):.1f})", mannwhitneyu(a, b)[1]))
t = pd.crosstab(inc, raw.country); rows.append(('country', '', '', chi2_contingency(t)[1]))
for c in ['Ukraine', 'Georgia', 'Moldova', 'Belarus']:
    rows.append((c, f"{(raw.country[inc]==c).sum()} ({(raw.country[inc]==c).mean()*100:.1f}%)", f"{(raw.country[~inc]==c).sum()} ({(raw.country[~inc]==c).mean()*100:.1f}%)", np.nan))
r = pd.DataFrame(rows, columns=['variable', 'included', 'excluded', 'p']); print(r.to_string())
print('n included', int(inc.sum()), 'n excluded', int((~inc).sum()))
r.to_csv('results/r2_selection.csv', index=False)
