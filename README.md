# Stepwise Prediction of Tuberculosis Treatment Outcomes with XGBoost

**Author**: Linfeng Wang  
**Affiliation**: London School of Hygiene & Tropical Medicine  
**Publication**: A multi-stage machine learning framework for stepwise prediction of tuberculosis treatment outcomes  
**Dataset**: [TB Portals](https://tbportals.niaid.nih.gov)

**Streamlit Dashboard**: [TB Portals](https://8vl3xgriqww4uuvcxp9iqo.streamlit.app/)

---

## Overview

This repository accompanies a machine learning study that develops a stepwise predictive model for tuberculosis (TB) treatment outcomes using gradient-boosted decision trees (XGBoost). The model integrates clinical, microbiological, imaging, and treatment features, reflecting the stages of real-world TB care. The approach supports dynamic, data-driven decision-making even with missing information.

Key features:
- Data from patients across 13 countries via [TB Portals](https://tbportals.niaid.nih.gov)
- Processed features after cleaning, encoding, and imputation
- Progressive model architecture (4 stages): Pre-treatment → Microbiology → Imaging → Treatment
- Handles missing data using XGBoost's native capabilities
- Validated against alternative ML models (SVM, MLP, logistic regression, catboost etc.)

---

## Repository Structure 
_Original data are not included, as permission for access is needed from TB portal_

- `environment.yml`: conda environment file
- `Appl`: streamlit dashboard
  - `example_input.csv/xslx`: example input, add your own data in, in this format
  - `feature_list.pkl`: list of features for input
- `Analysis/`: Data analysis and modelling.
  - `Figures/`: plots and tables
  - `Figures/`: save model model weights for the full model
  - `data_processing.ipynb`: data cleaning and engineering to generate clinical_lung_na1.csv
  - `multiclass_prediction.ipynb`: Modelling and associated calculations.
  - `KS_statistics.ipynb`: KS calculation
  - `clinical_lung_na1.csv`: Processed data.

## Environment Setup

Install all dependencies with Conda:

```bash
conda env create -f environment.yml
conda activate tb_mic

Environment includes:
- Catboost
- XGBoost
- scikit-learn
- SHAP
- NumPy, pandas, SciPy
- seaborn, matplotlib
- JupyterLab
```



## Citation

Please cite the accompanying manuscript if you use this repository in your research.

Wang, L., Campino, S., Clark, T.G. and Phelan, J.E. (2025a). A multi-stage machine learning framework for stepwise prediction of tuberculosis treatment outcomes: Integrating gradient boosted decision trees and feature-level analysis for clinical decision support. Research Square, (Preprint). doi:https://doi.org/10.21203/rs.3.rs-7558046/v1.

