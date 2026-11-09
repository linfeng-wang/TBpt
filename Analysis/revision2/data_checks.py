"""Revision 2: data checks for reviewer queries (HIV coding, period, resistance breakdown, culture, selection)."""
import numpy as np, pandas as pd
from common import load
RAW = '/mnt/storageG1/lwang/Projects/TBpt/Clinical/TB_Portals_Published_Clinical_Data_August_2023/'
raw = pd.read_csv(RAW + 'TB_Portals_Patient_Cases_August_2023.csv', low_memory=False)
na1 = pd.read_csv('/mnt/storageG1/lwang/Projects/TBpt/Analysis/clinical_lung_na1.csv')
coh = na1[na1.outcome != 'Completed'].drop(columns=['condition_id']).drop_duplicates()
ids = na1.loc[coh.index, 'condition_id']
c = raw[raw.condition_id.isin(ids)].drop_duplicates('condition_id').set_index('condition_id')
print('raw cases', len(raw), '| na1 rows', len(na1), '| cohort rows', len(coh), '| cohort ids matched', len(c))
print('\n--- outcome in raw release'); print(raw.outcome.value_counts(dropna=False))
print('\n--- HIV: comorbidity flag in cohort'); print(coh['HIV'].value_counts(dropna=False))
print(pd.crosstab(coh['HIV'], coh['outcome'], margins=True))
print('\n--- registration dates'); d = pd.to_datetime(c.registration_date, errors='coerce'); print(d.min(), d.max(), d.dt.year.value_counts().sort_index().to_dict())
ps = pd.to_datetime(c.period_start, errors='coerce'); print('period_start', ps.min(), ps.max())
print('\n--- type_of_resistance (raw)'); print(c.type_of_resistance.value_counts())
mono = c[c.type_of_resistance == 'Mono DR']
def res(row, drug):
    for t in ['bactec', 'le', 'hain', 'lpaother', 'genexpert']:
        v = row.get(f'{t}_{drug}')
        if isinstance(v, str) and 'R' in v: return True
    return False
inh = mono.apply(lambda r: res(r, 'isoniazid'), axis=1); rif = mono.apply(lambda r: res(r, 'rifampicin'), axis=1)
print('Mono DR n', len(mono), '| INH-R', int(inh.sum()), '| RIF-R', int(rif.sum()), '| neither recorded', int((~inh & ~rif).sum()))
print('\n--- culture field (raw, cohort)'); print(c.culture.value_counts().head(8)); print(c.culturetype.value_counts().head())
print('\n--- lung localisation'); print(c.lung_localization.value_counts(dropna=False))
print('\n--- missing in Table 1 variables (cohort)')
for v in ['education', 'employment', 'bmi', 'age_of_onset', 'gender', 'type_of_resistance']:
    print(v, int(coh[v].isna().sum()), f'{coh[v].isna().mean()*100:.1f}%')
print('\n--- period_span (days) by outcome'); print(coh.groupby('outcome')['period_span'].describe())
mdr = c.type_of_resistance.isin(['MDR non XDR', 'Pre-XDR', 'XDR'])
cur = c.outcome == 'Cured'
print('MDR+ cured with period_span <= 183 d:', int((c.period_span[mdr & cur] <= 183).sum()), 'of', int((mdr & cur).sum()))
bpal = c.regimen_drug.fillna('').str.contains('Pretomanid') & c.regimen_drug.fillna('').str.contains('Bedaquiline') & c.regimen_drug.fillna('').str.contains('Linezolid')
print('BPaL(M)-containing regimens:', int(bpal.sum()), '| pretomanid any:', int(c.regimen_drug.fillna('').str.contains('Pretomanid').sum()))
pr = c[c.regimen_drug.fillna('').str.contains('Pretomanid')]
print('pretomanid period_start range', pd.to_datetime(pr.period_start, errors='coerce').min(), pd.to_datetime(pr.period_start, errors='coerce').max())
print('\n--- country'); print(c.country.value_counts().to_dict())
