Differential Microbiome Taxa Identification Using Machine Learning
A reproducible machine-learning framework for identifying discriminatory gut microbiome genera associated with early-onset colorectal cancer (EoCRC) and distinguishing EoCRC from non-colorectal cancer controls (NC) and late-onset colorectal cancer (LoCRC).

Overview
This repository implements a multi-class microbiome machine-learning pipeline designed to identify and evaluate microbial genera that discriminate among:

Group	Description
NC	Non-colorectal cancer controls
EoCRC	Early-onset colorectal cancer
LoCRC	Late-onset colorectal cancer

The pipeline integrates:

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

Random Forest (RF) · Support Vector Machine (SVM) · XGBoost

The primary objective is not simply to determine which machine-learning algorithm performs best, but to identify microbial signals that remain discriminatory and robust across different modeling approaches.

Study Objective
Primary objective
To determine whether gut microbiome genera can simultaneously discriminate among:

NC → EoCRC → LoCRC

This multi-class framework extends conventional pairwise microbiome comparisons by evaluating the ability of microbial features to distinguish EoCRC from both non-cancer controls and LoCRC within a single classification framework.

Secondary objective
To compare RF, SVM, and XGBoost under a common analytical framework and determine whether candidate microbial signatures demonstrate consistent discriminatory performance across machine-learning methods.

Analytical Workflow
                 Microbiome abundance data
                           │
                           ▼
                   Data preprocessing
                           │
                           ▼
                  Taxa / genus selection
                           │
                           ▼
                  CLR transformation
                           │
                           ▼
              Stratified outer CV (10-fold)
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         │
       Outer training set        Outer test set
              │                         │
              ▼                         │
      Inner CV (5-fold)                │
              │                         │
              ▼                         │
      Hyperparameter tuning             │
              │                         │
              ▼                         │
      ┌───────┼────────┐                │
      │       │        │                │
      ▼       ▼        ▼                │
     RF      SVM    XGBoost             │
      │       │        │                │
      └───────┼────────┘                │
              │                         │
              ▼                         │
       Optimized model                  │
              │                         │
              └────────────┬────────────┘
                           ▼
                 Held-out outer fold
                           │
                           ▼
                Out-of-fold predictions
                           │
                           ▼
                  Multi-class ROC/AUC
                           │
                           ▼
                 Model performance
                    comparison
                           │
                           ▼
             Discriminatory taxa/signatures

Data Processing
Microbiome abundance data are organized with samples as rows and microbial genera as columns.

The classification variable is stored as:

Patient_Status

with three classes:

NC
EoCRC
LoCRC

CLR Transformation
Microbiome abundance data are compositional. The pipeline therefore applies a centered log-ratio (CLR) transformation prior to machine-learning analysis.

A small pseudocount is added to avoid undefined logarithms for zero-abundance features:

X_pseudo = X + 1e-6

X_clr = (
    np.log(X_pseudo)
    - np.log(X_pseudo).mean(axis=1).values.reshape(-1, 1)
)

The resulting CLR-transformed data are used as input to the machine-learning models.

Note: The choice of pseudocount should be documented and kept consistent across analyses.

Microbial Feature Panels
The pipeline supports predefined microbial genus panels. This allows candidate microbial signatures to be evaluated consistently across RF, SVM, and XGBoost.

Example:

panel_B = [
    "Peptostreptococcus",
    "Parvimonas",
    "Clostridium",
    "Ruthenibacterium",
    "Gemella",
    "Faecalibacterium",
    "Anaerotignum",
    "Megamonas",
    "Mediterraneibacter"
]

Only genera present in the input abundance matrix are retained:

panel_B = [g for g in panel_B if g in X.columns]

Feature panels should be defined independently of the held-out test data whenever possible to minimize the risk of information leakage.

Machine-Learning Models
Random Forest
Random Forest (RF) is implemented as a nonlinear ensemble tree-based classifier.

The model is evaluated using nested cross-validation, with hyperparameters optimized exclusively within the inner training folds.

RF can additionally provide feature-importance measures for investigating the contribution of individual microbial genera.

Support Vector Machine
Support Vector Machine (SVM) is used as a complementary classification approach capable of modeling nonlinear decision boundaries in microbiome feature space.

Hyperparameters are optimized within the inner cross-validation loop, while final performance is estimated using the independent outer test folds.

XGBoost
Extreme Gradient Boosting (XGBoost) is implemented as a gradient-boosting classifier capable of modeling nonlinear relationships and interactions among microbial genera.

For the three-class classification problem, probability predictions are generated using:

XGBClassifier(
    objective="multi:softprob",
    num_class=len(classes)
)

Hyperparameters are optimized exclusively within the inner cross-validation procedure.

Nested Cross-Validation
To minimize optimistic bias arising from hyperparameter optimization, model evaluation uses nested stratified cross-validation.

Outer cross-validation
The outer loop uses 10-fold stratified cross-validation:

outer_cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

Each outer test fold remains independent of model selection and hyperparameter tuning.

Inner cross-validation
Within each outer training set, five-fold stratified cross-validation is used for hyperparameter optimization:

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)

The procedure can be summarized as:

Outer training data
        │
        ▼
   Inner CV
        │
        ▼
Hyperparameter selection
        │
        ▼
Optimized model
        │
        ▼
Outer test fold

This separation ensures that the outer test observations are not used during hyperparameter selection.

Hyperparameter Optimization
Hyperparameters are optimized using GridSearchCV within the inner cross-validation loop.

For example, the XGBoost implementation evaluates combinations of:

Parameter	Example search space
n_estimators	100, 200, 300
max_depth	2, 3, 5
learning_rate	0.01, 0.05, 0.1
subsample	0.7, 0.9, 1.0
colsample_bytree	0.7, 0.9, 1.0
min_child_weight	1, 3, 5
gamma	0, 0.1
reg_lambda	1, 5, 10

The optimal parameter set from each outer fold is retained to assess parameter stability across folds.

Performance Evaluation
Because the classification problem contains three classes, model performance is evaluated using one-vs-rest (OvR) ROC-AUC.

Primary metric
Macro-average AUC

Macro AUC gives equal weight to each class:

NC
EoCRC
LoCRC

Additional metrics
The pipeline records:

Mean outer-fold Macro AUC
Standard deviation across outer folds
95% confidence interval
Pooled OOF Macro AUC
Pooled OOF weighted AUC
Class-specific AUC
Class-specific ROC curves
Out-of-fold predictions
Fold-specific optimized hyperparameters
Out-of-Fold Predictions
Predictions generated for each held-out outer fold are stored as out-of-fold (OOF) predictions.

Each sample therefore receives a prediction from a model that was not trained on that sample.

These OOF predictions are subsequently used to calculate pooled ROC curves and AUC values.

Fold 1 ──► predict held-out samples
Fold 2 ──► predict held-out samples
Fold 3 ──► predict held-out samples
   ...
Fold 10 ─► predict held-out samples
              │
              ▼
       Combined OOF predictions
              │
              ▼
        Pooled ROC / AUC

Model Comparison
RF, SVM, and XGBoost are evaluated using the same overall analytical framework.

Models are compared according to:

Evaluation criterion	Purpose
Mean outer-fold Macro AUC	Overall classification performance
SD across folds	Performance variability
95% CI	Uncertainty around estimated performance
Pooled OOF Macro AUC	Overall OOF discrimination
Class-specific AUC	Performance for individual classes
Feature/signature stability	Reproducibility of microbial signals

The primary goal is not simply to identify the highest-performing classifier.

Instead, the analysis asks whether microbial genera demonstrate consistent discriminatory signals across independent machine-learning approaches.

Identification of Discriminatory Microbial Signatures
A central objective of the pipeline is to identify microbial genera that contribute to discrimination among NC, EoCRC, and LoCRC.

Candidate taxa can be prioritized based on their consistency across:

RF
SVM
XGBoost
Cross-validation folds
Feature panels
Clinically relevant class comparisons
Particular emphasis is placed on microbial genera that:

Discriminate EoCRC from NC.
Discriminate EoCRC from LoCRC.
Contribute to multi-class discrimination.
Demonstrate stable model contribution across folds.
Are consistently identified across multiple machine-learning approaches.
The resulting taxa can be considered candidate discriminatory microbial signatures for subsequent biological interpretation and independent validation.

Pairwise vs Multi-Class Classification
A major motivation for this framework is to move beyond conventional pairwise microbiome comparisons.

Traditional analyses may evaluate:

NC       vs       EoCRC

NC       vs       LoCRC

EoCRC    vs       LoCRC

The present framework additionally evaluates the combined three-class problem:

                    ┌── NC
                    │
Microbiome ─────────┼── EoCRC
                    │
                    └── LoCRC

This allows the pipeline to assess whether microbial features can distinguish EoCRC from both control and later-onset disease states simultaneously.

Repository Structure
CRC-Microbiome-Biomarker-ML/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── README.md
│   └── processed/
│
├── panels/
│   ├── panel_A.txt
│   ├── panel_B.txt
│   └── ...
│
├── scripts/
│   ├── random_forest/
│   │   ├── RF_pipeline.py
│   │   └── ...
│   │
│   ├── svm/
│   │   ├── SVM_pipeline.py
│   │   └── ...
│   │
│   └── xgboost/
│       ├── XGB_pipeline.py
│       └── ...
│
├── results/
│   ├── random_forest/
│   ├── svm/
│   └── xgboost/
│
├── figures/
│   ├── roc/
│   ├── feature_importance/
│   └── model_comparison/
│
└── docs/
    └── parameter_tuning/

Reproducibility
To facilitate reproducibility, analyses should document:

Python version
Package versions
Random seeds
Feature panels
CLR transformation parameters
Cross-validation strategy
Hyperparameter grids
Model parameters
Performance metrics
Random seeds are explicitly specified in the cross-validation procedures.

The computational environment can be reproduced using:

pip install -r requirements.txt

Data Availability
Raw sequencing data and restricted clinical metadata are not included in this repository unless their distribution is permitted under the applicable data-use and ethical requirements.

Processed data may be provided separately where appropriate.

Interpretation and Limitations
This repository represents a computational microbial biomarker-discovery framework.

Cross-validated model performance should not be interpreted as evidence of clinical diagnostic utility without independent validation.

Potential sources of heterogeneity include:

Study population
Geographic and demographic differences
Sample collection procedures
DNA extraction methods
Sequencing platforms
Bioinformatic processing
Taxonomic classification methods
Batch effects
Disease and treatment characteristics
Accordingly, candidate microbial signatures identified through this pipeline should ideally be evaluated in an independent external cohort.

Current Analysis
The current implementation evaluates predefined microbial genus panels using nested cross-validation and compares:

Random Forest
      │
      ├──────────────┐
      │              │
     SVM          XGBoost
      │              │
      └──────┬───────┘
             ▼
      Model comparison
             │
             ▼
    Microbial signature
       identification

The principal classification problem is:

NC ──────┐
         │
EoCRC ───┼──► Multi-class classifier ──► ROC/AUC
         │
LoCRC ───┘

Scientific Rationale
Early-onset colorectal cancer represents an increasingly important clinical problem, and microbiome-based approaches may provide complementary information for disease discrimination.

Although numerous studies have reported CRC-associated microbial taxa, the robustness and generalizability of reported microbial signatures may be affected by substantial biological and technical heterogeneity across cohorts and analytical workflows.

Furthermore, microbiome studies have traditionally emphasized pairwise comparisons between disease groups. Such approaches may not fully capture microbial patterns capable of simultaneously distinguishing controls, EoCRC, and LoCRC.

This pipeline therefore combines:

compositional microbiome analysis + multi-class classification + nested validation + cross-model comparison

to identify microbial genera with reproducible discriminatory potential.

Future Development
Planned or potential extensions include:

Automated feature-selection procedures
Model-specific feature importance
SHAP-based model interpretation
Stability selection across resampling iterations
Pairwise and multi-class performance comparison
Differential abundance integration
External cohort validation
Batch-effect assessment
Calibration analysis
Decision-curve analysis
Integrated microbial signature scoring
Cross-cohort reproducibility analysis
Citation
If this repository or analytical framework contributes to a publication, please cite:

[Manuscript / preprint citation to be added]

License
This project is currently under:

[License to be added]

Contact
For questions regarding the analytical framework or repository, please open an issue or contact the corresponding project author.

