# Differential Microbiome Taxa Identification Using Machine Learning
A reproducible machine-learning framework for identifying discriminatory gut microbiome genera associated with early-onset colorectal cancer (EoCRC) and distinguishing EoCRC from non-colorectal cancer controls (NC) and late-onset colorectal cancer (LoCRC).

## Overview
This repository implements a multi-class microbiome machine-learning pipeline designed to identify and evaluate microbial genera that discriminate among:

## Group	Description
1. NC	Non-colorectal cancer controls
2. EoCRC	Early-onset colorectal cancer
3. LoCRC	Late-onset colorectal cancer

## The pipeline integrates:

Microbial abundance preprocessing
Centered log-ratio (CLR) transformation
Predefined microbial genus panels
Supervised machine learning
Nested stratified cross-validation
Hyperparameter optimization
Out-of-fold (OOF) prediction
Multi-class ROC/AUC analysis
Cross-model comparison
Identification of candidate discriminatory microbial signatures
Three complementary machine-learning approaches are evaluated:

## Random Forest (RF) · Support Vector Machine (SVM) · XGBoost

The primary objective is not simply to determine which machine-learning algorithm performs best, but to identify microbial signals that remain discriminatory and robust across different modeling approaches.

## Study Objective
### Primary objective
To determine whether gut microbiome genera can simultaneously discriminate among:

NC → EoCRC → LoCRC

This multi-class framework extends conventional pairwise microbiome comparisons by evaluating the ability of microbial features to distinguish EoCRC from both non-cancer controls and LoCRC within a single classification framework.

### Secondary objective
To compare RF, SVM, and XGBoost under a common analytical framework and determine whether candidate microbial signatures demonstrate consistent discriminatory performance across machine-learning methods.

### requirements.

Processed data may be provided separately where appropriate.

Interpretation and Limitations
This repository represents a computational microbial biomarker-discovery framework.

Cross-validated model performance should not be interpreted as evidence of clinical diagnostic utility without independent validation.

Potential sources of heterogeneity include:

*Study population*
Geographic and demographic differences
Sample collection procedures
DNA extraction methods
Sequencing platforms
Bioinformatic processing
Taxonomic classification methods
Batch effects
Disease and treatment characteristics
Accordingly, candidate microbial signatures identified through this pipeline should ideally be evaluated in an independent external cohort.

## Current Analysis
The current implementation evaluates predefined microbial genus panels using nested cross-validation and compares:

