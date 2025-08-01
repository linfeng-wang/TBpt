# Stepwise Prediction of Tuberculosis Treatment Outcomes with XGBoost

**Author**: Linfeng Wang  
**Affiliation**: London School of Hygiene & Tropical Medicine  
**Paper**: A multi-stage machine learning framework for stepwise prediction of tuberculosis treatment outcomes  
**Dataset**: [TB Portals](https://tbportals.niaid.nih.gov)

---

## Overview

This repository accompanies a machine learning study that develops a stepwise predictive model for tuberculosis (TB) treatment outcomes using gradient-boosted decision trees (XGBoost). The model integrates clinical, microbiological, imaging, and treatment features, reflecting the stages of real-world TB care. The approach supports dynamic, data-driven decision-making even with missing information.

Key features:
- Data from patients across 13 countries via [TB Portals](https://tbportals.niaid.nih.gov)
- Processed features after cleaning, encoding, and imputation
- Progressive model architecture (4 stages): Pre-treatment → Microbiology → Imaging → Treatment
- Handles missing data using XGBoost's native capabilities
- Validated against alternative ML models (SVM, MLP, logistic regression, etc.)

---

## Repository Structure
- `Clinical/`: Contains data and data dictionary for clinical features.
- `Cenomic/`: Contains data and data dictionary for genomic features.
- `Analysis/`: Data analysis and modelling.
  - `EDA.ipynb`: Initial data exploration.
  - `data_explore1.ipynb`: Data generation and preprocessing.
  - `multiclass_prediction2.ipynb`: Modelling and associated calculations.
  - `clinical_lung_na1.csv`: Processed data.