"""Shared data preparation for revision-2 analyses.

Reproduces the preprocessing in multiclass_prediction.ipynb (cells 3 and 12) so that
all revision analyses use exactly the same cohort (n = 8,094) and feature levels.
"""
from itertools import chain

import numpy as np
import pandas as pd

DATA = '/mnt/storageG1/lwang/Projects/TBpt/Analysis/clinical_lung_na1.csv'
SEED = 42

FEATURE_LEVELS = {
    'pre-diagnosis': ['education', 'gender', 'employment', 'Current smoker', 'Documented MDR contact', 'Ex prisoner',
                      'Homeless', 'Immigrants', 'Not_Reported-social_risk', 'Patient alcohol abuse',
                      'Patient alcohol abuse not available/unknown', 'Patient drug abuse',
                      'Patient smoking data not available/unknown', 'TB care worker', 'Worked abroad',
                      'internal migrants', 'refugees', 'age_of_onset', 'bmi', 'number_of_daily_contacts',
                      '1 to 19', '100 to 200', '20 to 100', 'MOTT', 'More than 200',
                      ' TNF-α antagonists', 'Anemia', 'COVID-19', 'Diabetes', 'HIV', 'Hepatic diseases',
                      'Hepatitis B', 'Hepatitis C', 'Not specified', 'Not_reported-Comorbidity', 'Others',
                      'Pneumoconiosis', 'Post-COVID-19', 'Psychiatric illness', 'Renal disease',
                      'Systemically administered glucocorticoids', 'cytostatics'],
    'post-diagnosis': ['type_of_resistance', 'A15.0', 'A15.1', 'A15.2', 'A15.3', 'A15.4', 'A15.5', 'A15.6', 'A15.7',
                       'A15.8', 'A15.9', 'A16', 'A16.0', 'A16.2', 'A16.3', 'A16.5', 'A16.8', 'A17.0', 'A18.0',
                       'A18.1', 'A18.2', 'A18.3', 'A18.4', 'A18.7', 'A18.8', 'A19.0', 'A19.1', 'A19.8', 'A19.9',
                       'Negative', 'Nonspecific microflora', 'Not_Reported-culture', 'Not done-culture', 'Positive',
                       'Study in progress', 'Unknown_result-culture', 'genexpert_test', 'bactec_isoniazid',
                       'bactec_rifampicin', 'bactec_ethambutol', 'La1', 'La2', 'lineage1', 'lineage1-lineage2',
                       'lineage2', 'lineage2-lineage4', 'lineage3', 'lineage3-lineage4', 'lineage4',
                       'other_non_tb_abnormalities'],
    'post-scan': ['hugenodules', 'largenodules', 'mediumdensitystabalizedfibroticnodules', 'collapse',
                  'overall_percent_of_abnormal_volume', 'pleural_effusion_percent_of_hemithorax_involved',
                  'ispleuraleffusionbilateral', 'are_mediastinal_lymphnodes_present', 'smallcavities',
                  'mediumcavities', 'largecavities', 'isanylargecavitybelongtoamultisextantcavity',
                  'canmultiplecavitiesbeseen', 'infiltrate_lowgroundglassdensity', 'infiltrate_mediumdensity',
                  'infiltrate_highdensity', 'smallnodules', 'mediumnodules',
                  'isanycalcifiedorpartiallycalcifiednoduleexist', 'isanynoncalcifiednoduleexist',
                  'isanyclusterednoduleexists', 'aremultiplenoduleexists', 'lowgroundglassdensityactivefreshnodules',
                  'highdensitycalcifiedtypicallysequella', 'timika_score', 'lung_localization'],
    'Treatment-period': ['Amikacin', 'Aminoglycosides - injectible agents', 'Amoxicillin-clavulanate',
                         'Antiretroviral therapy', 'Bedaquiline', 'Capreomycin', 'Clarithromycin', 'Clofazimine',
                         'Cotrimoxazol preventive', 'Cycloserine', 'Delamanid', 'Ethambutol', 'Ethionamide',
                         'Fluoroquinolones', 'Gatifloxacin', 'Gatifloxacin 0.5', 'Imipenem-cilastatin', 'Isoniazid',
                         'Kanamycin', 'Levofloxacin', 'Linezolid', 'Moxifloxacin', 'Mycobutin',
                         'Not_Reported-regimen_drug', 'Ofloxacin', 'Pretomanid', 'Prothionamide', 'Pyrazinamide',
                         'Rifampicin', 'Streptomycin', 'Terizidone', 'p-aminosalicylic acid', 'regimen_count',
                         'period_span', 'Adverse event', 'Continuation of treatment', 'Drug(s) no longer available',
                         'New drugs available', 'Not_Reported-treatment_status', 'Patient stopped treatment',
                         'Terminated from study', 'Treatment ineffective due to additional resistance'],
}
L = FEATURE_LEVELS
STEPWISE = {
    'Model 1': L['pre-diagnosis'],
    'Model 2': L['pre-diagnosis'] + L['post-diagnosis'],
    'Model 3': L['pre-diagnosis'] + L['post-diagnosis'] + L['post-scan'],
    'Model 4': L['pre-diagnosis'] + L['post-diagnosis'] + L['Treatment-period'],
    'All features': L['pre-diagnosis'] + L['post-diagnosis'] + L['Treatment-period'] + L['post-scan'],
}
ALL_COLS = list(chain.from_iterable(FEATURE_LEVELS.values()))
# Continuous / count variables (non-binary) used for spline terms and median imputation
CONTINUOUS = ['age_of_onset', 'bmi', 'number_of_daily_contacts', 'timika_score', 'regimen_count', 'period_span']


def load():
    """Return X (all feature columns, NaN kept) and y (1 = cured, 0 = died/failure/palliative care)."""
    df = pd.read_csv(DATA).drop(columns=['condition_id', 'country'])
    df['education'] = (df['education'] - 1).replace({-1: np.nan})
    df = df[df['outcome'] != 'Completed'].drop_duplicates()
    y = df['outcome'].map({'Cured': 1, 'Died': 0, 'Failure': 0, 'Palliative Care': 0}).astype(int)
    X = df[ALL_COLS].astype(float)
    return X.reset_index(drop=True), y.reset_index(drop=True)
