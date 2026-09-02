```python
import pandas as pd

# Load your microbiome abundance table
abund = pd.read_csv("data/Chemo_Gut_ML_clean.csv", index_col=0)  # samples x taxa

# Load significant taxa list

sig_taxa = pd.read_csv("data/sig_taxa.csv")['x'].tolist()
# print(sig_taxa)
# abund
```


```python
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ---------------------------
# 1. Prepare data
# ---------------------------


X = abund[sig_taxa]
X = abund.drop(columns=['Patient_Status'])
y = abund['Patient_Status']
target_genera = [
    "Peptostreptococcus",
    "Clostridium",
    "Parvimonas",
    "Solibacillus",
    "Gemella",
    "Limosilactobacillus",
    "Novisyntrophococcus"
]

selected_genera = [g for g in target_genera if g in X.columns]
X = X[selected_genera].copy()

# CLR transformation
X_pseudo = X + 1e-6
X_clr = np.log(X_pseudo) - np.log(X_pseudo).mean(axis=1).values.reshape(-1, 1)

# Prevalence filtering (10%)
prevalence = (X > 0).sum(axis=0) / X.shape[0]
X_filtered = X_clr.loc[:, prevalence > 0.1]

print("Filtered shape:", X_filtered.shape)

# ---------------------------
# 2. Define pipeline
# ---------------------------
pipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# ---------------------------
# 3. Your tuning grid (adapted)
# ---------------------------
params_grid = {
    'feature_selection__k': [1, 2, 3, 4, 5, 6],
    'rf__n_estimators': [130, 160, 190],
    'rf__criterion': ['gini'],
    'rf__max_depth': [35, 55],
    'rf__min_samples_split': [0.001, 0.005],
    'rf__min_samples_leaf': [0.001, 0.005],
    'rf__max_features': ['log2']
}

# params_grid = {
#     'feature_selection__k': [3, 5, 7, 9],   # MUST be ≤ 9    
#     'rf__n_estimators': [100, 200],    
#     'rf__criterion': ['gini'],    
#     'rf__max_depth': [5, 10, None],   # shallower trees now better    
#     'rf__min_samples_split': [2, 5],    
#     'rf__min_samples_leaf': [1, 2],    
#     'rf__max_features': ['sqrt', 'log2', None]
# }

# Count combinations
num_combinations = np.prod([len(v) for v in params_grid.values()])
print("Number of combinations =", num_combinations)

# ---------------------------
# 4. Cross-validation setup
# ---------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------
# 5. Grid search
# ---------------------------
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=params_grid,
    cv=cv,
    scoring='roc_auc_ovr',  # multiclass AUC
    n_jobs=-1,
    verbose=0,
    return_train_score=True   # <--- important

)

# ---------------------------
# 6. Fit model
# ---------------------------
grid_search.fit(X_filtered, y)

# ---------------------------
# 7. Results
# ---------------------------
print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest CV AUC:")
print(grid_search.best_score_)
```

    Filtered shape: (597, 6)
    Number of combinations = 144
    
    Best parameters:
    {'feature_selection__k': 6, 'rf__criterion': 'gini', 'rf__max_depth': 35, 'rf__max_features': 'log2', 'rf__min_samples_leaf': 0.005, 'rf__min_samples_split': 0.001, 'rf__n_estimators': 190}
    
    Best CV AUC:
    0.6906473359091694



```python
X = abund[sig_taxa]
X = abund.drop(columns=['Patient_Status'])
y = abund['Patient_Status']
target_genera = [
    "Peptostreptococcus",
    "Clostridium",
    "Parvimonas",
    "Solibacillus",
    "Gemella",
    "Limosilactobacillus",
    "Novisyntrophococcus"
]

selected_genera = [g for g in target_genera if g in X.columns]
X_selected = X[selected_genera].copy()

print(X_selected.shape)
print(X_selected.columns)
y.value_counts()
```

    (597, 7)
    Index(['Peptostreptococcus', 'Clostridium', 'Parvimonas', 'Solibacillus',
           'Gemella', 'Limosilactobacillus', 'Novisyntrophococcus'],
          dtype='object')





    Patient_Status
    Control    237
    LoCRC      232
    EoCRC      128
    Name: count, dtype: int64




```python
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# 1. Prepare data
# ---------------------------

X = abund.drop(columns=['Patient_Status'])
y = abund['Patient_Status']

target_genera = [
    "Gemella",
    "Limosilactobacillus",
    "Ruthenibacterium",
    "Faecalibacterium",
    "Megamonas",
    "Clostridium",
    "Peptostreptococcus",
    "Parvimonas",
    # "Solibacillus",
    "Solobacterium",
    "Anaerotignum",
    "Mediterraneibacter",
#     "Lactobacillus",
#     "Porphyromonas",
#     "Fusobacterium",
#     "Longicatena"
 ]

# ---------------------------
# 2. Check which taxa exist
# ---------------------------

selected_genera = [
    g for g in target_genera
    if g in X.columns
]

missing_genera = [
    g for g in target_genera
    if g not in X.columns
]

print("Selected genera:")
print(selected_genera)

print("\nMissing genera:")
print(missing_genera)

# ---------------------------
# 3. Subset features
# ---------------------------

X = X[selected_genera].copy()

print("\nInitial shape:", X.shape)

# ---------------------------
# 4. Prevalence filtering
# ---------------------------

prevalence = (X > 0).mean(axis=0)

prevalence_table = pd.DataFrame({
    "Genus": prevalence.index,
    "Prevalence": prevalence.values,
    "Percent": prevalence.values * 100
})

print("\nPrevalence:")
print(prevalence_table.sort_values("Percent", ascending=False))

keep = prevalence > 0.10

X_filtered_raw = X.loc[:, keep]

print("\nFeatures retained after 10% prevalence filtering:")
print(X_filtered_raw.columns.tolist())

print("\nFiltered shape:", X_filtered_raw.shape)

# ---------------------------
# 5. CLR transformation
# ---------------------------

X_pseudo = X_filtered_raw + 1e-6

X_clr = (
    np.log(X_pseudo)
    - np.log(X_pseudo).mean(axis=1).values.reshape(-1, 1)
)

# ---------------------------
# 6. Random Forest pipeline
# ---------------------------

pipeline = Pipeline([
    (
        'rf',
        RandomForestClassifier(
            random_state=42,
            class_weight='balanced'
        )
    )
])

# ---------------------------
# 7. RF tuning grid
# ---------------------------

params_grid = {
    'rf__n_estimators': [130, 160, 190],
    'rf__criterion': ['gini'],
    'rf__max_depth': [35, 55],
    'rf__min_samples_split': [0.001, 0.005],
    'rf__min_samples_leaf': [0.001, 0.005],
    'rf__max_features': ['sqrt', 'log2', None]
}

# Number of combinations

num_combinations = np.prod(
    [len(v) for v in params_grid.values()]
)

print("\nNumber of combinations:", num_combinations)

# ---------------------------
# 8. Cross-validation
# ---------------------------

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

# ---------------------------
# 9. Grid search
# ---------------------------

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=params_grid,
    cv=cv,
    scoring='roc_auc_ovr',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

# ---------------------------
# 10. Fit
# ---------------------------

grid_search.fit(X_clr, y)

# ---------------------------
# 11. Results
# ---------------------------

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest CV AUC:")
print(grid_search.best_score_)
```

    Selected genera:
    ['Gemella', 'Limosilactobacillus', 'Ruthenibacterium', 'Faecalibacterium', 'Megamonas', 'Clostridium', 'Peptostreptococcus', 'Parvimonas', 'Solobacterium', 'Anaerotignum', 'Mediterraneibacter']
    
    Missing genera:
    []
    
    Initial shape: (597, 11)
    
    Prevalence:
                      Genus  Prevalence    Percent
    10   Mediterraneibacter    0.948074  94.807370
    3      Faecalibacterium    0.899497  89.949749
    9          Anaerotignum    0.695142  69.514238
    5           Clostridium    0.658291  65.829146
    2      Ruthenibacterium    0.654941  65.494137
    0               Gemella    0.557789  55.778894
    6    Peptostreptococcus    0.390285  39.028476
    7            Parvimonas    0.343384  34.338358
    4             Megamonas    0.299832  29.983250
    1   Limosilactobacillus    0.232831  23.283082
    8         Solobacterium    0.201005  20.100503
    
    Features retained after 10% prevalence filtering:
    ['Gemella', 'Limosilactobacillus', 'Ruthenibacterium', 'Faecalibacterium', 'Megamonas', 'Clostridium', 'Peptostreptococcus', 'Parvimonas', 'Solobacterium', 'Anaerotignum', 'Mediterraneibacter']
    
    Filtered shape: (597, 11)
    
    Number of combinations: 72
    Fitting 10 folds for each of 72 candidates, totalling 720 fits
    
    Best parameters:
    {'rf__criterion': 'gini', 'rf__max_depth': 35, 'rf__max_features': None, 'rf__min_samples_leaf': 0.001, 'rf__min_samples_split': 0.005, 'rf__n_estimators': 190}
    
    Best CV AUC:
    0.7016760434308807



```python

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# 1. Prepare data
# ---------------------------

X = abund.drop(columns=['Patient_Status'])
y = abund['Patient_Status']

target_genera = [
    "Peptostreptococcus",
    "Parvimonas",
    "Clostridium",
    "Limosilactobacillus",
    "Ruthenibacterium",
    "Gemella"
 ]

# ---------------------------
# 2. Check which taxa exist
# ---------------------------

selected_genera = [
    g for g in target_genera
    if g in X.columns
]

missing_genera = [
    g for g in target_genera
    if g not in X.columns
]

print("Selected genera:")
print(selected_genera)

print("\nMissing genera:")
print(missing_genera)

# ---------------------------
# 3. Subset features
# ---------------------------

X = X[selected_genera].copy()

print("\nInitial shape:", X.shape)

# ---------------------------
# 4. Prevalence filtering
# ---------------------------

prevalence = (X > 0).mean(axis=0)

prevalence_table = pd.DataFrame({
    "Genus": prevalence.index,
    "Prevalence": prevalence.values,
    "Percent": prevalence.values * 100
})

print("\nPrevalence:")
print(prevalence_table.sort_values("Percent", ascending=False))

keep = prevalence > 0.10

X_filtered_raw = X.loc[:, keep]

print("\nFeatures retained after 10% prevalence filtering:")
print(X_filtered_raw.columns.tolist())

print("\nFiltered shape:", X_filtered_raw.shape)

# ---------------------------
# 5. CLR transformation
# ---------------------------

X_pseudo = X_filtered_raw + 1e-6

X_clr = (
    np.log(X_pseudo)
    - np.log(X_pseudo).mean(axis=1).values.reshape(-1, 1)
)

# ---------------------------
# 6. Random Forest pipeline
# ---------------------------

pipeline = Pipeline([
    (
        'rf',
        RandomForestClassifier(
            random_state=42,
            class_weight='balanced'
        )
    )
])

# ---------------------------
# 7. RF tuning grid
# ---------------------------

params_grid = {
    'rf__n_estimators': [130, 160, 190],
    'rf__criterion': ['gini'],
    'rf__max_depth': [35, 55],
    'rf__min_samples_split': [0.001, 0.005],
    'rf__min_samples_leaf': [0.001, 0.005],
    'rf__max_features': ['sqrt', 'log2', None]
}

# Number of combinations

num_combinations = np.prod(
    [len(v) for v in params_grid.values()]
)

print("\nNumber of combinations:", num_combinations)

# ---------------------------
# 8. Cross-validation
# ---------------------------

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

# ---------------------------
# 9. Grid search
# ---------------------------

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=params_grid,
    cv=cv,
    scoring='roc_auc_ovr',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

# ---------------------------
# 10. Fit
# ---------------------------

grid_search.fit(X_clr, y)

# ---------------------------
# 11. Results
# ---------------------------

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest CV AUC:")
print(grid_search.best_score_)
```

    Selected genera:
    ['Peptostreptococcus', 'Parvimonas', 'Clostridium', 'Limosilactobacillus', 'Ruthenibacterium', 'Gemella']
    
    Missing genera:
    []
    
    Initial shape: (597, 6)
    
    Prevalence:
                     Genus  Prevalence    Percent
    2          Clostridium    0.658291  65.829146
    4     Ruthenibacterium    0.654941  65.494137
    5              Gemella    0.557789  55.778894
    0   Peptostreptococcus    0.390285  39.028476
    1           Parvimonas    0.343384  34.338358
    3  Limosilactobacillus    0.232831  23.283082
    
    Features retained after 10% prevalence filtering:
    ['Peptostreptococcus', 'Parvimonas', 'Clostridium', 'Limosilactobacillus', 'Ruthenibacterium', 'Gemella']
    
    Filtered shape: (597, 6)
    
    Number of combinations: 72
    Fitting 10 folds for each of 72 candidates, totalling 720 fits
    
    Best parameters:
    {'rf__criterion': 'gini', 'rf__max_depth': 35, 'rf__max_features': None, 'rf__min_samples_leaf': 0.005, 'rf__min_samples_split': 0.001, 'rf__n_estimators': 130}
    
    Best CV AUC:
    0.6668236325471621



```python
##### Category B


import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# 1. Prepare data
# ---------------------------

X = abund.drop(columns=['Patient_Status'])
y = abund['Patient_Status']

target_genera = [
    "Peptostreptococcus",
    "Parvimonas",
    "Clostridium",
    "Limosilactobacillus",
    "Ruthenibacterium",
    "Gemella",
    "Faecalibacterium",
    "Anaerotignum",
    "Megamonas"
 ]

# ---------------------------
# 2. Check which taxa exist
# ---------------------------

selected_genera = [
    g for g in target_genera
    if g in X.columns
]

missing_genera = [
    g for g in target_genera
    if g not in X.columns
]

print("Selected genera:")
print(selected_genera)

print("\nMissing genera:")
print(missing_genera)

# ---------------------------
# 3. Subset features
# ---------------------------

X = X[selected_genera].copy()

print("\nInitial shape:", X.shape)

# ---------------------------
# 4. Prevalence filtering
# ---------------------------

prevalence = (X > 0).mean(axis=0)

prevalence_table = pd.DataFrame({
    "Genus": prevalence.index,
    "Prevalence": prevalence.values,
    "Percent": prevalence.values * 100
})

print("\nPrevalence:")
print(prevalence_table.sort_values("Percent", ascending=False))

keep = prevalence > 0.10

X_filtered_raw = X.loc[:, keep]

print("\nFeatures retained after 10% prevalence filtering:")
print(X_filtered_raw.columns.tolist())

print("\nFiltered shape:", X_filtered_raw.shape)

# ---------------------------
# 5. CLR transformation
# ---------------------------

X_pseudo = X_filtered_raw + 1e-6

X_clr = (
    np.log(X_pseudo)
    - np.log(X_pseudo).mean(axis=1).values.reshape(-1, 1)
)

# ---------------------------
# 6. Random Forest pipeline
# ---------------------------

pipeline = Pipeline([
    (
        'rf',
        RandomForestClassifier(
            random_state=42,
            class_weight='balanced'
        )
    )
])

# ---------------------------
# 7. RF tuning grid
# ---------------------------

params_grid = {
    'rf__n_estimators': [130, 160, 190],
    'rf__criterion': ['gini'],
    'rf__max_depth': [35, 55],
    'rf__min_samples_split': [0.001, 0.005],
    'rf__min_samples_leaf': [0.001, 0.005],
    'rf__max_features': ['sqrt', 'log2', None]
}

# Number of combinations

num_combinations = np.prod(
    [len(v) for v in params_grid.values()]
)

print("\nNumber of combinations:", num_combinations)

# ---------------------------
# 8. Cross-validation
# ---------------------------

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

# ---------------------------
# 9. Grid search
# ---------------------------

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=params_grid,
    cv=cv,
    scoring='roc_auc_ovr',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

# ---------------------------
# 10. Fit
# ---------------------------

grid_search.fit(X_clr, y)

# ---------------------------
# 11. Results
# ---------------------------

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest CV AUC:")
print(grid_search.best_score_)
```

    Selected genera:
    ['Peptostreptococcus', 'Parvimonas', 'Clostridium', 'Limosilactobacillus', 'Ruthenibacterium', 'Gemella', 'Faecalibacterium', 'Anaerotignum', 'Megamonas']
    
    Missing genera:
    []
    
    Initial shape: (597, 9)
    
    Prevalence:
                     Genus  Prevalence    Percent
    6     Faecalibacterium    0.899497  89.949749
    7         Anaerotignum    0.695142  69.514238
    2          Clostridium    0.658291  65.829146
    4     Ruthenibacterium    0.654941  65.494137
    5              Gemella    0.557789  55.778894
    0   Peptostreptococcus    0.390285  39.028476
    1           Parvimonas    0.343384  34.338358
    8            Megamonas    0.299832  29.983250
    3  Limosilactobacillus    0.232831  23.283082
    
    Features retained after 10% prevalence filtering:
    ['Peptostreptococcus', 'Parvimonas', 'Clostridium', 'Limosilactobacillus', 'Ruthenibacterium', 'Gemella', 'Faecalibacterium', 'Anaerotignum', 'Megamonas']
    
    Filtered shape: (597, 9)
    
    Number of combinations: 72
    Fitting 10 folds for each of 72 candidates, totalling 720 fits
    
    Best parameters:
    {'rf__criterion': 'gini', 'rf__max_depth': 35, 'rf__max_features': 'sqrt', 'rf__min_samples_leaf': 0.005, 'rf__min_samples_split': 0.001, 'rf__n_estimators': 160}
    
    Best CV AUC:
    0.7145442162242771



```python
panel_B = [
    "Peptostreptococcus",
    "Parvimonas",
    "Clostridium",
    "Limosilactobacillus",
    "Ruthenibacterium",
    "Gemella",
    "Faecalibacterium",
    "Anaerotignum",
    "Megamonas"
]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score

# ============================================================
# 1. Panel B
# ============================================================

panel_B = [
    "Peptostreptococcus",
    "Parvimonas",
    "Clostridium",
    "Limosilactobacillus",
    "Ruthenibacterium",
    "Gemella",
    "Faecalibacterium",
    "Anaerotignum",
    "Megamonas"
]

X = abund.drop(columns=["Patient_Status"])
y = abund["Patient_Status"]

# Keep only taxa actually present
panel_B = [g for g in panel_B if g in X.columns]

X = X[panel_B].copy()

print("Panel B features:")
print(panel_B)

# ============================================================
# 2. CLR transformation
# ============================================================

X_pseudo = X + 1e-6

X_clr = (
    np.log(X_pseudo)
    - np.log(X_pseudo).mean(axis=1).values.reshape(-1, 1)
)

# ============================================================
# 3. Encode classes
# ============================================================

classes = np.unique(y)

print("\nClasses:")
print(classes)

y_bin = label_binarize(y, classes=classes)

# ============================================================
# 4. Out-of-fold predictions
# ============================================================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

oof_proba = np.zeros(
    (X_clr.shape[0], len(classes))
)

for train_idx, test_idx in cv.split(X_clr, y):

    X_train = X_clr.iloc[train_idx]
    X_test = X_clr.iloc[test_idx]

    y_train = y.iloc[train_idx]

    rf = RandomForestClassifier(
        n_estimators=160,
        criterion="gini",
        max_depth=35,
        max_features="sqrt",
        min_samples_leaf=0.005,
        min_samples_split=0.001,
        class_weight="balanced",
        random_state=42
    )

    rf.fit(X_train, y_train)

    oof_proba[test_idx] = rf.predict_proba(X_test)

# ============================================================
# 5. Overall multiclass AUC
# ============================================================

macro_auc = roc_auc_score(
    y_bin,
    oof_proba,
    multi_class="ovr",
    average="macro"
)

weighted_auc = roc_auc_score(
    y_bin,
    oof_proba,
    multi_class="ovr",
    average="weighted"
)

print("\nOut-of-fold Macro AUC:", macro_auc)
print("Out-of-fold Weighted AUC:", weighted_auc)

# ============================================================
# 6. Plot ROC curves
# ============================================================

plt.figure(figsize=(8, 7))

for i, cls in enumerate(classes):

    fpr, tpr, _ = roc_curve(
        y_bin[:, i],
        oof_proba[:, i]
    )

    class_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{cls} (AUC = {class_auc:.3f})"
    )

# Random classifier
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curves for Panel B Microbial Signature"
)

plt.legend(
    loc="lower right",
    frameon=False
)

plt.tight_layout()
plt.show()
```

    Panel B features:
    ['Peptostreptococcus', 'Parvimonas', 'Clostridium', 'Limosilactobacillus', 'Ruthenibacterium', 'Gemella', 'Faecalibacterium', 'Anaerotignum', 'Megamonas']
    
    Classes:
    ['Control' 'EoCRC' 'LoCRC']
    
    Out-of-fold Macro AUC: 0.7114739800656217
    Out-of-fold Weighted AUC: 0.7212779625046181



    
![png](params_tunning_RF_files/params_tunning_RF_6_1.png)
    



```python

```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score


# ============================================================
# 1. Panel B
# ============================================================

panel_B = [
    "Peptostreptococcus",
    "Parvimonas",
    "Clostridium",
    "Limosintrichobacillus",
    "Ruthenibacterium",
    "Gemella",
    "Faecalibacterium",
    "Anaerotignum",
    "Megamonas"
]

X = abund.drop(columns=["Patient_Status"])
y = abund["Patient_Status"].copy()

# Keep only taxa actually present
panel_B = [g for g in panel_B if g in X.columns]

X = X[panel_B].copy()

print("Panel B features:")
print(panel_B)

print("\nNumber of features:", X.shape[1])
print("Number of samples:", X.shape[0])


# ============================================================
# 2. CLR transformation
# ============================================================

X_pseudo = X + 1e-6

X_clr = (
    np.log(X_pseudo)
    - np.log(X_pseudo).mean(axis=1).values.reshape(-1, 1)
)


# ============================================================
# 3. Encode classes
# ============================================================

classes = np.unique(y)

print("\nClasses:")
print(classes)

y_bin = label_binarize(y, classes=classes)


# ============================================================
# 4. OUTER CV
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


# ============================================================
# 5. INNER CV
# ============================================================

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)


# ============================================================
# 6. Hyperparameter grid
# ============================================================

param_grid = {
    "n_estimators": [130, 160, 190],
    "criterion": ["gini"],
    "max_depth": [10, 20, 35, None],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [0.001, 0.005],
    "min_samples_leaf": [0.001, 0.005]
}


# ============================================================
# 7. Storage for OOF predictions
# ============================================================

oof_proba = np.zeros(
    (X_clr.shape[0], len(classes))
)

best_params_each_fold = []

outer_fold_auc = []


# ============================================================
# 8. NESTED CROSS-VALIDATION
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X_clr, y),
    start=1
):

    print("\n" + "=" * 60)
    print(f"OUTER FOLD {fold}/10")
    print("=" * 60)

    # --------------------------------------------------------
    # Outer training/test split
    # --------------------------------------------------------

    X_train = X_clr.iloc[train_idx]
    X_test = X_clr.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


    # --------------------------------------------------------
    # Inner GridSearch
    # --------------------------------------------------------

    inner_grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="roc_auc_ovr",
        n_jobs=-1,
        refit=True,
        verbose=0
    )


    # --------------------------------------------------------
    # Tune ONLY using outer training data
    # --------------------------------------------------------

    inner_grid.fit(X_train, y_train)


    # --------------------------------------------------------
    # Best parameters from inner CV
    # --------------------------------------------------------

    best_params = inner_grid.best_params_

    best_params_each_fold.append(best_params)

    print("Best inner parameters:")
    print(best_params)

    print(
        "Best inner CV AUC:",
        inner_grid.best_score_
    )


    # --------------------------------------------------------
    # Predict OUTER test fold
    # --------------------------------------------------------

    fold_proba = inner_grid.predict_proba(X_test)


    # --------------------------------------------------------
    # Store genuine OOF predictions
    # --------------------------------------------------------

    oof_proba[test_idx] = fold_proba


    # --------------------------------------------------------
    # Outer-fold AUC
    # --------------------------------------------------------

    y_test_bin = label_binarize(
        y_test,
        classes=classes
    )

    fold_auc = roc_auc_score(
        y_test_bin,
        fold_proba,
        multi_class="ovr",
        average="macro"
    )

    outer_fold_auc.append(fold_auc)

    print(
        "Outer-fold Macro AUC:",
        round(fold_auc, 4)
    )


# ============================================================
# 9. Overall nested OOF AUC
# ============================================================

macro_auc = roc_auc_score(
    y_bin,
    oof_proba,
    multi_class="ovr",
    average="macro"
)

weighted_auc = roc_auc_score(
    y_bin,
    oof_proba,
    multi_class="ovr",
    average="weighted"
)


print("\n" + "=" * 60)
print("FINAL NESTED OOF RESULTS")
print("=" * 60)

print(
    "Nested OOF Macro AUC:",
    round(macro_auc, 4)
)

print(
    "Nested OOF Weighted AUC:",
    round(weighted_auc, 4)
)

print(
    "\nMean outer-fold Macro AUC:",
    round(np.mean(outer_fold_auc), 4)
)

print(
    "SD outer-fold Macro AUC:",
    round(np.std(outer_fold_auc, ddof=1), 4)
)


# ============================================================
# 10. Outer-fold AUC table
# ============================================================

auc_df = pd.DataFrame({
    "Outer_Fold": np.arange(1, 11),
    "Macro_AUC": outer_fold_auc
})

print("\nOuter-fold AUC:")
print(auc_df)


# ============================================================
# 11. Best parameters across outer folds
# ============================================================

params_df = pd.DataFrame(best_params_each_fold)

print("\nBest parameters selected in each outer fold:")
print(params_df)


# ============================================================
# 12. ROC curves
# ============================================================

plt.figure(figsize=(8, 7))

for i, cls in enumerate(classes):

    fpr, tpr, _ = roc_curve(
        y_bin[:, i],
        oof_proba[:, i]
    )

    class_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{cls} (AUC = {class_auc:.3f})"
    )


# Random classifier
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    color="gray"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "Nested OOF ROC Curves – Panel B Random Forest"
)

plt.legend(
    loc="lower right",
    frameon=False
)

plt.tight_layout()
plt.show()
```

    Panel B features:
    ['Peptostreptococcus', 'Parvimonas', 'Clostridium', 'Ruthenibacterium', 'Gemella', 'Faecalibacterium', 'Anaerotignum', 'Megamonas']
    
    Number of features: 8
    Number of samples: 597
    
    Classes:
    ['Control' 'EoCRC' 'LoCRC']
    
    ============================================================
    OUTER FOLD 1/10
    ============================================================
    Best inner parameters:
    {'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 0.005, 'min_samples_split': 0.001, 'n_estimators': 160}
    Best inner CV AUC: 0.7052248502581423
    Outer-fold Macro AUC: 0.6722
    
    ============================================================
    OUTER FOLD 2/10
    ============================================================
    Best inner parameters:
    {'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 0.005, 'min_samples_split': 0.001, 'n_estimators': 130}
    Best inner CV AUC: 0.7027176989925573
    Outer-fold Macro AUC: 0.6203
    
    ============================================================
    OUTER FOLD 3/10
    ============================================================
    Best inner parameters:
    {'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 0.005, 'min_samples_split': 0.001, 'n_estimators': 190}
    Best inner CV AUC: 0.6807200515384677
    Outer-fold Macro AUC: 0.7002
    
    ============================================================
    OUTER FOLD 4/10
    ============================================================
    Best inner parameters:
    {'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 0.005, 'min_samples_split': 0.001, 'n_estimators': 190}
    Best inner CV AUC: 0.6873504626944256
    Outer-fold Macro AUC: 0.6448
    
    ============================================================
    OUTER FOLD 5/10
    ============================================================
    Best inner parameters:
    {'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 0.005, 'min_samples_split': 0.001, 'n_estimators': 160}
    Best inner CV AUC: 0.6935557440256767
    Outer-fold Macro AUC: 0.6973
    
    ============================================================
    OUTER FOLD 6/10
    ============================================================
    Best inner parameters:
    {'criterion': 'gini', 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 0.005, 'min_samples_split': 0.001, 'n_estimators': 190}
    Best inner CV AUC: 0.7013097506015689
    Outer-fold Macro AUC: 0.6399
    
    ============================================================
    OUTER FOLD 7/10
    ============================================================
    Best inner parameters:
    {'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 0.001, 'min_samples_split': 0.005, 'n_estimators': 160}
    Best inner CV AUC: 0.6810719642247711
    Outer-fold Macro AUC: 0.7172
    
    ============================================================
    OUTER FOLD 8/10
    ============================================================
    Best inner parameters:
    {'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 0.005, 'min_samples_split': 0.001, 'n_estimators': 190}
    Best inner CV AUC: 0.7157437451312202
    Outer-fold Macro AUC: 0.6784
    
    ============================================================
    OUTER FOLD 9/10
    ============================================================
    Best inner parameters:
    {'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 0.005, 'min_samples_split': 0.001, 'n_estimators': 190}
    Best inner CV AUC: 0.6731042384050083
    Outer-fold Macro AUC: 0.7774
    
    ============================================================
    OUTER FOLD 10/10
    ============================================================
    Best inner parameters:
    {'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 0.005, 'min_samples_split': 0.001, 'n_estimators': 190}
    Best inner CV AUC: 0.6840871398536743
    Outer-fold Macro AUC: 0.7345
    
    ============================================================
    FINAL NESTED OOF RESULTS
    ============================================================
    Nested OOF Macro AUC: 0.6858
    Nested OOF Weighted AUC: 0.6961
    
    Mean outer-fold Macro AUC: 0.6882
    SD outer-fold Macro AUC: 0.0476
    
    Outer-fold AUC:
       Outer_Fold  Macro_AUC
    0           1   0.672204
    1           2   0.620278
    2           3   0.700164
    3           4   0.644753
    4           5   0.697334
    5           6   0.639853
    6           7   0.717207
    7           8   0.678372
    8           9   0.777375
    9          10   0.734516
    
    Best parameters selected in each outer fold:
      criterion  max_depth max_features  min_samples_leaf  min_samples_split  \
    0      gini         10         log2             0.005              0.001   
    1      gini         10         sqrt             0.005              0.001   
    2      gini         10         log2             0.005              0.001   
    3      gini         10         sqrt             0.005              0.001   
    4      gini         10         sqrt             0.005              0.001   
    5      gini         20         sqrt             0.005              0.001   
    6      gini         10         log2             0.001              0.005   
    7      gini         10         log2             0.005              0.001   
    8      gini         10         sqrt             0.005              0.001   
    9      gini         10         sqrt             0.005              0.001   
    
       n_estimators  
    0           160  
    1           130  
    2           190  
    3           190  
    4           160  
    5           190  
    6           160  
    7           190  
    8           190  
    9           190  



    
![png](params_tunning_RF_files/params_tunning_RF_8_1.png)
    



```python
from scipy.stats import t

mean_auc = np.mean(outer_fold_auc)
sd_auc = np.std(outer_fold_auc, ddof=1)

n_folds = len(outer_fold_auc)

# Standard error
se_auc = sd_auc / np.sqrt(n_folds)

# 95% CI using t-distribution
t_critical = t.ppf(0.975, df=n_folds - 1)

ci_lower = mean_auc - t_critical * se_auc
ci_upper = mean_auc + t_critical * se_auc

print("\nNested CV results:")
print(f"Mean Macro AUC: {mean_auc:.3f}")
print(f"SD: {sd_auc:.3f}")
print(f"95% CI: {ci_lower:.3f} – {ci_upper:.3f}")
```

    
    Nested CV results:
    Mean Macro AUC: 0.688
    SD: 0.048
    95% CI: 0.654 – 0.722



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve, auc


# ============================================================
# 1. Panel B
# ============================================================

panel_B = [
    "Peptostreptococcus",
    "Parvimonas",
    "Clostridium",
    "Limosilactobacillus",
    "Ruthenibacterium",
    "Gemella",
    "Faecalibacterium",
    "Anaerotignum",
    "Megamonas"
]

X = abund.drop(columns=["Patient_Status"])
y = abund["Patient_Status"].copy()

# Keep only taxa actually present
panel_B = [g for g in panel_B if g in X.columns]

X = X[panel_B].copy()

print("Panel B features:")
print(panel_B)

print("\nSamples:", X.shape[0])
print("Features:", X.shape[1])


# ============================================================
# 2. CLR transformation
# ============================================================

X_pseudo = X + 1e-6

X_clr = (
    np.log(X_pseudo)
    - np.log(X_pseudo).mean(axis=1).values.reshape(-1, 1)
)


# ============================================================
# 3. Classes
# ============================================================

classes = np.unique(y)

print("\nClasses:")
print(classes)

y_bin = label_binarize(
    y,
    classes=classes
)


# ============================================================
# 4. Nested CV
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)


# ============================================================
# 5. SVM pipeline
# ============================================================

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        probability=True,
        class_weight="balanced",
        random_state=42
    ))
])


# ============================================================
# 6. SVM parameter grid
# ============================================================

param_grid = {
    "svm__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "svm__gamma": [
        "scale",
        "auto",
        0.001,
        0.01,
        0.1,
        1
    ],

    "svm__kernel": [
        "rbf"
    ]
}


# ============================================================
# 7. OOF predictions
# ============================================================

oof_proba = np.zeros(
    (X_clr.shape[0], len(classes))
)

best_params_each_fold = []

outer_fold_auc = []


# ============================================================
# 8. Nested CV
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X_clr, y),
    start=1
):

    print("\n" + "=" * 60)
    print(f"OUTER FOLD {fold}/10")
    print("=" * 60)

    # --------------------------------------------------------
    # Outer training/test data
    # --------------------------------------------------------

    X_train = X_clr.iloc[train_idx]
    X_test = X_clr.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]


    # --------------------------------------------------------
    # Inner hyperparameter tuning
    # --------------------------------------------------------

    grid = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=param_grid,
        scoring="roc_auc_ovr",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )

    grid.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Best parameters
    # --------------------------------------------------------

    print("Best parameters:")
    print(grid.best_params_)

    print(
        "Best inner CV AUC:",
        round(grid.best_score_, 4)
    )

    best_params_each_fold.append(
        grid.best_params_
    )


    # --------------------------------------------------------
    # Predict outer test fold
    # --------------------------------------------------------

    fold_proba = grid.predict_proba(
        X_test
    )

    oof_proba[test_idx] = fold_proba


    # --------------------------------------------------------
    # Outer-fold AUC
    # --------------------------------------------------------

    y_test_bin = label_binarize(
        y_test,
        classes=classes
    )

    fold_auc = roc_auc_score(
        y_test_bin,
        fold_proba,
        multi_class="ovr",
        average="macro"
    )

    outer_fold_auc.append(
        fold_auc
    )

    print(
        "Outer Macro AUC:",
        round(fold_auc, 4)
    )


# ============================================================
# 9. Mean outer-fold AUC
# ============================================================

mean_auc = np.mean(
    outer_fold_auc
)

sd_auc = np.std(
    outer_fold_auc,
    ddof=1
)

print("\n" + "=" * 60)
print("NESTED SVM RESULTS")
print("=" * 60)

print(
    f"Mean outer-fold Macro AUC: {mean_auc:.3f}"
)

print(
    f"SD: {sd_auc:.3f}"
)


# ============================================================
# 10. 95% CI for mean outer-fold AUC
# ============================================================

from scipy.stats import t

n_folds = len(
    outer_fold_auc
)

se_auc = (
    sd_auc /
    np.sqrt(n_folds)
)

t_critical = t.ppf(
    0.975,
    df=n_folds - 1
)

ci_lower = (
    mean_auc -
    t_critical * se_auc
)

ci_upper = (
    mean_auc +
    t_critical * se_auc
)

print(
    f"95% CI: {ci_lower:.3f} – {ci_upper:.3f}"
)


# ============================================================
# 11. Pooled OOF Macro AUC
# ============================================================

pooled_macro_auc = roc_auc_score(
    y_bin,
    oof_proba,
    multi_class="ovr",
    average="macro"
)

pooled_weighted_auc = roc_auc_score(
    y_bin,
    oof_proba,
    multi_class="ovr",
    average="weighted"
)

print(
    f"\nPooled OOF Macro AUC: "
    f"{pooled_macro_auc:.3f}"
)

print(
    f"Pooled OOF Weighted AUC: "
    f"{pooled_weighted_auc:.3f}"
)


# ============================================================
# 12. Outer fold results
# ============================================================

auc_df = pd.DataFrame({
    "Outer_Fold": np.arange(
        1,
        len(outer_fold_auc) + 1
    ),
    "Macro_AUC": outer_fold_auc
})

print("\nOuter-fold AUC:")
print(auc_df)


# ============================================================
# 13. ROC curves
# ============================================================

plt.figure(
    figsize=(8, 7)
)

for i, cls in enumerate(classes):

    fpr, tpr, _ = roc_curve(
        y_bin[:, i],
        oof_proba[:, i]
    )

    class_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{cls} (AUC = {class_auc:.3f})"
    )


# Random classifier
plt.plot(
    [0, 1],
    [0, 1],
    "--",
    color="gray",
    linewidth=1
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Nested OOF ROC Curves – Panel B SVM"
)

plt.legend(
    loc="lower right",
    frameon=False
)

plt.tight_layout()

plt.show()
```

    Panel B features:
    ['Peptostreptococcus', 'Parvimonas', 'Clostridium', 'Limosilactobacillus', 'Ruthenibacterium', 'Gemella', 'Faecalibacterium', 'Anaerotignum', 'Megamonas']
    
    Samples: 597
    Features: 9
    
    Classes:
    ['Control' 'EoCRC' 'LoCRC']
    
    ============================================================
    OUTER FOLD 1/10
    ============================================================
    Best parameters:
    {'svm__C': 100, 'svm__gamma': 0.01, 'svm__kernel': 'rbf'}
    Best inner CV AUC: 0.7274
    Outer Macro AUC: 0.6688
    
    ============================================================
    OUTER FOLD 2/10
    ============================================================
    Best parameters:
    {'svm__C': 100, 'svm__gamma': 0.01, 'svm__kernel': 'rbf'}
    Best inner CV AUC: 0.73
    Outer Macro AUC: 0.6622
    
    ============================================================
    OUTER FOLD 3/10
    ============================================================
    Best parameters:
    {'svm__C': 100, 'svm__gamma': 0.01, 'svm__kernel': 'rbf'}
    Best inner CV AUC: 0.7108
    Outer Macro AUC: 0.7189
    
    ============================================================
    OUTER FOLD 4/10
    ============================================================
    Best parameters:
    {'svm__C': 10, 'svm__gamma': 0.01, 'svm__kernel': 'rbf'}
    Best inner CV AUC: 0.709
    Outer Macro AUC: 0.6953
    
    ============================================================
    OUTER FOLD 5/10
    ============================================================
    Best parameters:
    {'svm__C': 100, 'svm__gamma': 0.01, 'svm__kernel': 'rbf'}
    Best inner CV AUC: 0.7086
    Outer Macro AUC: 0.7429
    
    ============================================================
    OUTER FOLD 6/10
    ============================================================
    Best parameters:
    {'svm__C': 1, 'svm__gamma': 0.1, 'svm__kernel': 'rbf'}
    Best inner CV AUC: 0.7025
    Outer Macro AUC: 0.7506
    
    ============================================================
    OUTER FOLD 7/10
    ============================================================
    Best parameters:
    {'svm__C': 100, 'svm__gamma': 0.01, 'svm__kernel': 'rbf'}
    Best inner CV AUC: 0.7173
    Outer Macro AUC: 0.7479
    
    ============================================================
    OUTER FOLD 8/10
    ============================================================
    Best parameters:
    {'svm__C': 1, 'svm__gamma': 0.1, 'svm__kernel': 'rbf'}
    Best inner CV AUC: 0.738
    Outer Macro AUC: 0.6484
    
    ============================================================
    OUTER FOLD 9/10
    ============================================================
    Best parameters:
    {'svm__C': 10, 'svm__gamma': 0.01, 'svm__kernel': 'rbf'}
    Best inner CV AUC: 0.7098
    Outer Macro AUC: 0.7426
    
    ============================================================
    OUTER FOLD 10/10
    ============================================================
    Best parameters:
    {'svm__C': 1, 'svm__gamma': 0.1, 'svm__kernel': 'rbf'}
    Best inner CV AUC: 0.7118
    Outer Macro AUC: 0.7446
    
    ============================================================
    NESTED SVM RESULTS
    ============================================================
    Mean outer-fold Macro AUC: 0.712
    SD: 0.040
    95% CI: 0.684 – 0.741
    
    Pooled OOF Macro AUC: 0.711
    Pooled OOF Weighted AUC: 0.718
    
    Outer-fold AUC:
       Outer_Fold  Macro_AUC
    0           1   0.668775
    1           2   0.662226
    2           3   0.718897
    3           4   0.695271
    4           5   0.742893
    5           6   0.750579
    6           7   0.747878
    7           8   0.648396
    8           9   0.742630
    9          10   0.744581



    
![png](params_tunning_RF_files/params_tunning_RF_10_1.png)
    



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc

from xgboost import XGBClassifier
from scipy.stats import t


# ============================================================
# 1. Panel B
# ============================================================

panel_B = [
    "Peptostreptococcus",
    "Parvimonas",
    "Clostridium",
    "Limosilactobacillus",
    "Ruthenibacterium",
    "Gemella",
    "Faecalibacterium",
    "Anaerotignum",
    "Megamonas"
]

X = abund.drop(columns=["Patient_Status"])
y = abund["Patient_Status"].copy()

# Keep only taxa actually present
panel_B = [g for g in panel_B if g in X.columns]

X = X[panel_B].copy()

print("Panel B features:")
print(panel_B)

print("\nNumber of samples:", X.shape[0])
print("Number of features:", X.shape[1])


# ============================================================
# 2. CLR transformation
# ============================================================

X_pseudo = X + 1e-6

X_clr = (
    np.log(X_pseudo)
    - np.log(X_pseudo).mean(axis=1).values.reshape(-1, 1)
)


# ============================================================
# 3. Classes
# ============================================================

classes = np.unique(y)

print("\nClasses:")
print(classes)

y_bin = label_binarize(
    y,
    classes=classes
)


# ============================================================
# 4. Encode y for XGBoost
# ============================================================

# XGBoost multiclass requires integer class labels
class_to_int = {
    cls: i for i, cls in enumerate(classes)
}

y_encoded = y.map(class_to_int).astype(int)


# ============================================================
# 5. Nested CV
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)


# ============================================================
# 6. XGBoost model
# ============================================================

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(classes),

    eval_metric="mlogloss",

    random_state=42,
    n_jobs=-1,

    # Prevent excessive complexity
    verbosity=0
)


# ============================================================
# 7. Conservative parameter grid
# ============================================================

param_grid = {

    "n_estimators": [
        100,
        200,
        300
    ],

    "max_depth": [
        2,
        3,
        5
    ],

    "learning_rate": [
        0.01,
        0.05,
        0.1
    ],

    "subsample": [
        0.7,
        0.9,
        1.0
    ],

    "colsample_bytree": [
        0.7,
        0.9,
        1.0
    ],

    "min_child_weight": [
        1,
        3,
        5
    ],

    "gamma": [
        0,
        0.1
    ],

    "reg_lambda": [
        1,
        5,
        10
    ]
}


# ============================================================
# 8. OOF storage
# ============================================================

oof_proba = np.zeros(
    (X_clr.shape[0], len(classes))
)

outer_fold_auc = []

best_params_each_fold = []


# ============================================================
# 9. Nested CV
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X_clr, y_encoded),
    start=1
):

    print("\n" + "=" * 60)
    print(f"OUTER FOLD {fold}/10")
    print("=" * 60)


    # --------------------------------------------------------
    # Outer training/test
    # --------------------------------------------------------

    X_train = X_clr.iloc[train_idx]
    X_test = X_clr.iloc[test_idx]

    y_train = y_encoded.iloc[train_idx]
    y_test = y_encoded.iloc[test_idx]


    # --------------------------------------------------------
    # Inner GridSearch
    # --------------------------------------------------------

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,

        scoring="roc_auc_ovr",

        cv=inner_cv,

        n_jobs=-1,

        refit=True,

        verbose=0
    )


    # --------------------------------------------------------
    # Tune ONLY using outer training data
    # --------------------------------------------------------

    grid.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Best parameters
    # --------------------------------------------------------

    print("\nBest parameters:")
    print(grid.best_params_)

    print(
        "Best inner CV AUC:",
        round(grid.best_score_, 4)
    )

    best_params_each_fold.append(
        grid.best_params_
    )


    # --------------------------------------------------------
    # Predict outer test fold
    # --------------------------------------------------------

    fold_proba = grid.predict_proba(
        X_test
    )

    oof_proba[test_idx] = fold_proba


    # --------------------------------------------------------
    # Outer fold AUC
    # --------------------------------------------------------

    y_test_bin = label_binarize(
        y_test,
        classes=np.arange(len(classes))
    )

    fold_auc = roc_auc_score(
        y_test_bin,
        fold_proba,
        multi_class="ovr",
        average="macro"
    )

    outer_fold_auc.append(
        fold_auc
    )

    print(
        "Outer Macro AUC:",
        round(fold_auc, 4)
    )


# ============================================================
# 10. Mean outer-fold AUC
# ============================================================

mean_auc = np.mean(
    outer_fold_auc
)

sd_auc = np.std(
    outer_fold_auc,
    ddof=1
)

print("\n" + "=" * 60)
print("NESTED XGBOOST RESULTS")
print("=" * 60)

print(
    f"Mean outer-fold Macro AUC: {mean_auc:.3f}"
)

print(
    f"SD: {sd_auc:.3f}"
)


# ============================================================
# 11. 95% CI
# ============================================================

n_folds = len(
    outer_fold_auc
)

se_auc = (
    sd_auc /
    np.sqrt(n_folds)
)

t_critical = t.ppf(
    0.975,
    df=n_folds - 1
)

ci_lower = (
    mean_auc -
    t_critical * se_auc
)

ci_upper = (
    mean_auc +
    t_critical * se_auc
)

print(
    f"95% CI: {ci_lower:.3f} – {ci_upper:.3f}"
)


# ============================================================
# 12. Pooled OOF AUC
# ============================================================

pooled_macro_auc = roc_auc_score(
    y_bin,
    oof_proba,
    multi_class="ovr",
    average="macro"
)

pooled_weighted_auc = roc_auc_score(
    y_bin,
    oof_proba,
    multi_class="ovr",
    average="weighted"
)

print(
    f"\nPooled OOF Macro AUC: "
    f"{pooled_macro_auc:.3f}"
)

print(
    f"Pooled OOF Weighted AUC: "
    f"{pooled_weighted_auc:.3f}"
)


# ============================================================
# 13. Outer fold results
# ============================================================

auc_df = pd.DataFrame({
    "Outer_Fold": np.arange(
        1,
        len(outer_fold_auc) + 1
    ),

    "Macro_AUC": outer_fold_auc
})

print("\nOuter-fold AUC:")
print(auc_df)


# ============================================================
# 14. Best parameters across folds
# ============================================================

params_df = pd.DataFrame(
    best_params_each_fold
)

print("\nBest parameters across outer folds:")
print(params_df)


# ============================================================
# 15. ROC curves
# ============================================================

plt.figure(
    figsize=(8, 7)
)

for i, cls in enumerate(classes):

    fpr, tpr, _ = roc_curve(
        y_bin[:, i],
        oof_proba[:, i]
    )

    class_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{cls} (AUC = {class_auc:.3f})"
    )


# Random classifier
plt.plot(
    [0, 1],
    [0, 1],
    "--",
    color="gray",
    linewidth=1
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Nested OOF ROC Curves – Panel B XGBoost"
)

plt.legend(
    loc="lower right",
    frameon=False
)

plt.tight_layout()

plt.show()
```

    Panel B features:
    ['Peptostreptococcus', 'Parvimonas', 'Clostridium', 'Limosilactobacillus', 'Ruthenibacterium', 'Gemella', 'Faecalibacterium', 'Anaerotignum', 'Megamonas']
    
    Number of samples: 597
    Number of features: 9
    
    Classes:
    ['Control' 'EoCRC' 'LoCRC']
    
    ============================================================
    OUTER FOLD 1/10
    ============================================================
    
    Best parameters:
    {'colsample_bytree': 0.9, 'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 300, 'reg_lambda': 10, 'subsample': 1.0}
    Best inner CV AUC: 0.729
    Outer Macro AUC: 0.6241
    
    ============================================================
    OUTER FOLD 2/10
    ============================================================
    
    Best parameters:
    {'colsample_bytree': 0.9, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_weight': 5, 'n_estimators': 100, 'reg_lambda': 5, 'subsample': 0.9}
    Best inner CV AUC: 0.7153
    Outer Macro AUC: 0.6412
    
    ============================================================
    OUTER FOLD 3/10
    ============================================================
    
    Best parameters:
    {'colsample_bytree': 0.7, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 5, 'n_estimators': 100, 'reg_lambda': 10, 'subsample': 1.0}
    Best inner CV AUC: 0.7118
    Outer Macro AUC: 0.7224
    
    ============================================================
    OUTER FOLD 4/10
    ============================================================
    
    Best parameters:
    {'colsample_bytree': 0.9, 'gamma': 0, 'learning_rate': 0.01, 'max_depth': 2, 'min_child_weight': 5, 'n_estimators': 300, 'reg_lambda': 5, 'subsample': 1.0}
    Best inner CV AUC: 0.7253
    Outer Macro AUC: 0.6523
    
    ============================================================
    OUTER FOLD 5/10
    ============================================================
    
    Best parameters:
    {'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'reg_lambda': 5, 'subsample': 1.0}
    Best inner CV AUC: 0.7193
    Outer Macro AUC: 0.6815
    
    ============================================================
    OUTER FOLD 6/10
    ============================================================
    
    Best parameters:
    {'colsample_bytree': 0.9, 'gamma': 0.1, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 200, 'reg_lambda': 1, 'subsample': 0.9}
    Best inner CV AUC: 0.7229
    Outer Macro AUC: 0.7018
    
    ============================================================
    OUTER FOLD 7/10
    ============================================================
    
    Best parameters:
    {'colsample_bytree': 1.0, 'gamma': 0.1, 'learning_rate': 0.01, 'max_depth': 5, 'min_child_weight': 5, 'n_estimators': 100, 'reg_lambda': 10, 'subsample': 0.7}
    Best inner CV AUC: 0.7014
    Outer Macro AUC: 0.7969
    
    ============================================================
    OUTER FOLD 8/10
    ============================================================
    
    Best parameters:
    {'colsample_bytree': 0.7, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 100, 'reg_lambda': 10, 'subsample': 1.0}
    Best inner CV AUC: 0.736
    Outer Macro AUC: 0.7015
    
    ============================================================
    OUTER FOLD 9/10
    ============================================================
    
    Best parameters:
    {'colsample_bytree': 0.9, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_weight': 3, 'n_estimators': 100, 'reg_lambda': 1, 'subsample': 0.7}
    Best inner CV AUC: 0.6933
    Outer Macro AUC: 0.7702
    
    ============================================================
    OUTER FOLD 10/10
    ============================================================
    
    Best parameters:
    {'colsample_bytree': 0.9, 'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 100, 'reg_lambda': 5, 'subsample': 1.0}
    Best inner CV AUC: 0.7194
    Outer Macro AUC: 0.7559
    
    ============================================================
    NESTED XGBOOST RESULTS
    ============================================================
    Mean outer-fold Macro AUC: 0.705
    SD: 0.057
    95% CI: 0.664 – 0.746
    
    Pooled OOF Macro AUC: 0.692
    Pooled OOF Weighted AUC: 0.698
    
    Outer-fold AUC:
       Outer_Fold  Macro_AUC
    0           1   0.624064
    1           2   0.641186
    2           3   0.722443
    3           4   0.652298
    4           5   0.681492
    5           6   0.701775
    6           7   0.796875
    7           8   0.701505
    8           9   0.770222
    9          10   0.755868
    
    Best parameters across outer folds:
       colsample_bytree  gamma  learning_rate  max_depth  min_child_weight  \
    0               0.9    0.1           0.05          3                 5   
    1               0.9    0.0           0.05          2                 5   
    2               0.7    0.1           0.10          5                 5   
    3               0.9    0.0           0.01          2                 5   
    4               1.0    0.0           0.10          3                 1   
    5               0.9    0.1           0.01          3                 3   
    6               1.0    0.1           0.01          5                 5   
    7               0.7    0.0           0.10          3                 3   
    8               0.9    0.0           0.05          2                 3   
    9               0.9    0.1           0.05          3                 5   
    
       n_estimators  reg_lambda  subsample  
    0           300          10        1.0  
    1           100           5        0.9  
    2           100          10        1.0  
    3           300           5        1.0  
    4           100           5        1.0  
    5           200           1        0.9  
    6           100          10        0.7  
    7           100          10        1.0  
    8           100           1        0.7  
    9           100           5        1.0  



    
![png](params_tunning_RF_files/params_tunning_RF_11_1.png)
    



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc

from scipy.stats import t


# ============================================================
# 1. CONCORDANT GENERA
# ============================================================

concordant_genera = [

    # --------------------------------------------------------
    # Control
    # --------------------------------------------------------
    "Bilophila",
    "Clostridium",
    "Faecalibacillus",
    "Lachnospira",
    "Lactobacillus",
    "Megamonas",
    "Parasutterella",
    "Romboutsia",
    "Wujia",

    # --------------------------------------------------------
    # EoCRC
    # --------------------------------------------------------
    "Anaerotignum",
    "Butyribacter",
    "Campylobacter",
    "Eisenbergiella",
    "Faecalibacterium",
    "Fusicatenibacter",
    "Ligilactobacillus",
    "Marseillibacter",
    "Novisyntrophococcus",
    "Prevotella",
    "Roseburia",
    "Solibacillus",
    "Vescimonas",

    # --------------------------------------------------------
    # LoCRC
    # --------------------------------------------------------
    "Alistipes",
    "Anaerococcus",
    "Coprobacillus",
    "Enterococcus",
    "Filifactor",
    "Fusobacterium",
    "Gemella",
    "Hungatella",
    "Limosilactobacillus",
    "Morganella",
    "Odoribacter",
    "Paraeggerthella",
    "Parvimonas",
    "Peptostreptococcus",
    "Porphyromonas",
    "Ruthenibacterium",
    "Solobacterium"
]


# ============================================================
# 2. CHECK GENERA AGAINST DATASET
# ============================================================

X_all = abund.drop(
    columns=["Patient_Status"]
)

y = abund[
    "Patient_Status"
].copy()


present_genera = [
    g for g in concordant_genera
    if g in X_all.columns
]

missing_genera = [
    g for g in concordant_genera
    if g not in X_all.columns
]


print("=" * 70)
print("CONCORDANT GENERA")
print("=" * 70)

print(
    "Requested genera:",
    len(concordant_genera)
)

print(
    "Present in dataset:",
    len(present_genera)
)

print(
    "Missing from dataset:",
    len(missing_genera)
)

if missing_genera:
    print("\nMissing genera:")
    print(missing_genera)


# ============================================================
# 3. BUILD X
# ============================================================

X = X_all[
    present_genera
].copy()


print("\nFinal feature matrix:")
print(
    "Samples:",
    X.shape[0]
)

print(
    "Genera:",
    X.shape[1]
)


# ============================================================
# 4. CHECK ZERO / CONSTANT FEATURES
# ============================================================

zero_features = [
    col for col in X.columns
    if (X[col] == 0).all()
]

constant_features = [
    col for col in X.columns
    if X[col].nunique() <= 1
]

print("\nZero-abundance genera:")
print(zero_features)

print("\nConstant genera:")
print(constant_features)


# Remove completely absent / constant genera
remove_features = list(
    set(zero_features + constant_features)
)

if remove_features:

    X = X.drop(
        columns=remove_features
    )

    print(
        "\nRemoved constant features:",
        remove_features
    )


# ============================================================
# 5. CLR TRANSFORMATION
# ============================================================

X_pseudo = X + 1e-6

X_clr = (
    np.log(X_pseudo)
    -
    np.log(X_pseudo).mean(
        axis=1
    ).values.reshape(-1, 1)
)


print(
    "\nCLR matrix shape:",
    X_clr.shape
)


# ============================================================
# 6. CLASS INFORMATION
# ============================================================

classes = np.unique(y)

print("\nClasses:")
print(classes)

print("\nClass counts:")
print(
    y.value_counts()
)


# Binarized labels for multiclass ROC
y_bin = label_binarize(
    y,
    classes=classes
)


# ============================================================
# 7. OUTER CV
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


# ============================================================
# 8. INNER CV
# ============================================================

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)


# ============================================================
# 9. RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 10. HYPERPARAMETER GRID
# ============================================================

param_grid = {

    "n_estimators": [
        100,
        160,
        250
    ],

    "criterion": [
        "gini",
        "entropy"
    ],

    "max_depth": [
        5,
        10,
        20,
        None
    ],

    "max_features": [
        "sqrt",
        "log2"
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        5
    ]
}


# ============================================================
# 11. STORAGE FOR OOF PREDICTIONS
# ============================================================

oof_proba = np.zeros(
    (
        X_clr.shape[0],
        len(classes)
    )
)


outer_fold_auc = []

best_params_each_fold = []


# ============================================================
# 12. NESTED CROSS-VALIDATION
# ============================================================

for fold, (
    train_idx,
    test_idx
) in enumerate(
    outer_cv.split(
        X_clr,
        y
    ),
    start=1
):

    print("\n")
    print("=" * 70)
    print(
        f"OUTER FOLD {fold}/10"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Outer training set
    # --------------------------------------------------------

    X_train = X_clr.iloc[
        train_idx
    ]

    X_test = X_clr.iloc[
        test_idx
    ]

    y_train = y.iloc[
        train_idx
    ]

    y_test = y.iloc[
        test_idx
    ]


    # --------------------------------------------------------
    # Inner grid search
    # --------------------------------------------------------

    grid = GridSearchCV(

        estimator=rf,

        param_grid=param_grid,

        scoring="roc_auc_ovr",

        cv=inner_cv,

        n_jobs=-1,

        refit=True,

        verbose=0

    )


    # --------------------------------------------------------
    # Fit ONLY on outer training data
    # --------------------------------------------------------

    grid.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Best parameters
    # --------------------------------------------------------

    print(
        "\nBest parameters:"
    )

    print(
        grid.best_params_
    )

    print(
        "\nBest inner CV AUC:",
        round(
            grid.best_score_,
            4
        )
    )


    best_params_each_fold.append(
        grid.best_params_
    )


    # --------------------------------------------------------
    # Predict outer test fold
    # --------------------------------------------------------

    fold_proba = grid.predict_proba(
        X_test
    )


    # Store genuine OOF predictions
    oof_proba[
        test_idx
    ] = fold_proba


    # --------------------------------------------------------
    # Outer fold AUC
    # --------------------------------------------------------

    y_test_bin = label_binarize(
        y_test,
        classes=classes
    )


    fold_auc = roc_auc_score(

        y_test_bin,

        fold_proba,

        multi_class="ovr",

        average="macro"

    )


    outer_fold_auc.append(
        fold_auc
    )


    print(
        "\nOuter-fold Macro AUC:",
        round(
            fold_auc,
            4
        )
    )


# ============================================================
# 13. MEAN OUTER-FOLD AUC
# ============================================================

mean_auc = np.mean(
    outer_fold_auc
)

sd_auc = np.std(
    outer_fold_auc,
    ddof=1
)


print("\n")
print("=" * 70)
print("NESTED RANDOM FOREST RESULTS")
print("=" * 70)


print(
    f"Mean outer-fold Macro AUC: "
    f"{mean_auc:.3f}"
)

print(
    f"SD: "
    f"{sd_auc:.3f}"
)


# ============================================================
# 14. 95% CI FOR MEAN OUTER-FOLD AUC
# ============================================================

n_folds = len(
    outer_fold_auc
)

se_auc = (
    sd_auc /
    np.sqrt(n_folds)
)


t_critical = t.ppf(
    0.975,
    df=n_folds - 1
)


ci_lower = (
    mean_auc -
    t_critical *
    se_auc
)

ci_upper = (
    mean_auc +
    t_critical *
    se_auc
)


print(
    f"95% CI: "
    f"{ci_lower:.3f} – "
    f"{ci_upper:.3f}"
)


# ============================================================
# 15. POOLED OOF MACRO AUC
# ============================================================

pooled_macro_auc = roc_auc_score(

    y_bin,

    oof_proba,

    multi_class="ovr",

    average="macro"

)


pooled_weighted_auc = roc_auc_score(

    y_bin,

    oof_proba,

    multi_class="ovr",

    average="weighted"

)


print(
    "\nPooled OOF Macro AUC:",
    round(
        pooled_macro_auc,
        3
    )
)

print(
    "Pooled OOF Weighted AUC:",
    round(
        pooled_weighted_auc,
        3
    )
)


# ============================================================
# 16. OUTER-FOLD AUC TABLE
# ============================================================

auc_df = pd.DataFrame({

    "Outer_Fold":
        np.arange(
            1,
            len(
                outer_fold_auc
            ) + 1
        ),

    "Macro_AUC":
        outer_fold_auc

})


print(
    "\nOuter-fold AUC:"
)

print(
    auc_df
)


# ============================================================
# 17. CLASS-SPECIFIC OOF AUC
# ============================================================

print(
    "\nClass-specific OOF AUC:"
)

class_auc_results = []


for i, cls in enumerate(
    classes
):

    class_auc = roc_auc_score(

        y_bin[:, i],

        oof_proba[:, i]

    )

    class_auc_results.append({

        "Class": cls,

        "AUC": class_auc

    })

    print(
        f"{cls}: "
        f"{class_auc:.3f}"
    )


class_auc_df = pd.DataFrame(
    class_auc_results
)


# ============================================================
# 18. BEST PARAMETERS ACROSS OUTER FOLDS
# ============================================================

params_df = pd.DataFrame(
    best_params_each_fold
)

print(
    "\nBest parameters selected in each outer fold:"
)

print(
    params_df
)


# ============================================================
# 19. PARAMETER FREQUENCY
# ============================================================

print(
    "\nMost frequently selected parameters:"
)

for col in params_df.columns:

    print(
        f"\n{col}:"
    )

    print(
        params_df[col]
        .value_counts()
    )


# ============================================================
# 20. ROC CURVES
# ============================================================

plt.figure(
    figsize=(8, 7)
)


for i, cls in enumerate(
    classes
):

    fpr, tpr, _ = roc_curve(

        y_bin[:, i],

        oof_proba[:, i]

    )


    class_auc = auc(
        fpr,
        tpr
    )


    plt.plot(

        fpr,

        tpr,

        linewidth=2,

        label=(
            f"{cls} "
            f"(AUC = {class_auc:.3f})"
        )

    )


# Random classifier
plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

    color="gray",

    linewidth=1

)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Nested OOF ROC Curves – "
    "Concordant LEfSe + ANCOM-BC2 Genera"
)

plt.legend(
    loc="lower right",
    frameon=False
)

plt.tight_layout()

plt.show()
```

    ======================================================================
    CONCORDANT GENERA
    ======================================================================
    Requested genera: 39
    Present in dataset: 39
    Missing from dataset: 0
    
    Final feature matrix:
    Samples: 597
    Genera: 39
    
    Zero-abundance genera:
    []
    
    Constant genera:
    []
    
    CLR matrix shape: (597, 39)
    
    Classes:
    ['Control' 'EoCRC' 'LoCRC']
    
    Class counts:
    Patient_Status
    Control    237
    LoCRC      232
    EoCRC      128
    Name: count, dtype: int64
    
    
    ======================================================================
    OUTER FOLD 1/10
    ======================================================================
    
    Best parameters:
    {'criterion': 'entropy', 'max_depth': 5, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 250}
    
    Best inner CV AUC: 0.7154
    
    Outer-fold Macro AUC: 0.7292
    
    
    ======================================================================
    OUTER FOLD 2/10
    ======================================================================
    
    Best parameters:
    {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 250}
    
    Best inner CV AUC: 0.7234
    
    Outer-fold Macro AUC: 0.6167
    
    
    ======================================================================
    OUTER FOLD 3/10
    ======================================================================
    
    Best parameters:
    {'criterion': 'entropy', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 250}
    
    Best inner CV AUC: 0.7119
    
    Outer-fold Macro AUC: 0.6683
    
    
    ======================================================================
    OUTER FOLD 4/10
    ======================================================================
    
    Best parameters:
    {'criterion': 'gini', 'max_depth': 5, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 250}
    
    Best inner CV AUC: 0.7058
    
    Outer-fold Macro AUC: 0.674
    
    
    ======================================================================
    OUTER FOLD 5/10
    ======================================================================
    
    Best parameters:
    {'criterion': 'gini', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 250}
    
    Best inner CV AUC: 0.709
    
    Outer-fold Macro AUC: 0.7338
    
    
    ======================================================================
    OUTER FOLD 6/10
    ======================================================================
    
    Best parameters:
    {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 160}
    
    Best inner CV AUC: 0.7161
    
    Outer-fold Macro AUC: 0.6732
    
    
    ======================================================================
    OUTER FOLD 7/10
    ======================================================================
    
    Best parameters:
    {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 160}
    
    Best inner CV AUC: 0.7111
    
    Outer-fold Macro AUC: 0.7251
    
    
    ======================================================================
    OUTER FOLD 8/10
    ======================================================================
    
    Best parameters:
    {'criterion': 'entropy', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 160}
    
    Best inner CV AUC: 0.7131
    
    Outer-fold Macro AUC: 0.7718
    
    
    ======================================================================
    OUTER FOLD 9/10
    ======================================================================
    
    Best parameters:
    {'criterion': 'entropy', 'max_depth': 5, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 160}
    
    Best inner CV AUC: 0.71
    
    Outer-fold Macro AUC: 0.7421
    
    
    ======================================================================
    OUTER FOLD 10/10
    ======================================================================
    
    Best parameters:
    {'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 160}
    
    Best inner CV AUC: 0.7046
    
    Outer-fold Macro AUC: 0.7145
    
    
    ======================================================================
    NESTED RANDOM FOREST RESULTS
    ======================================================================
    Mean outer-fold Macro AUC: 0.705
    SD: 0.046
    95% CI: 0.672 – 0.738
    
    Pooled OOF Macro AUC: 0.696
    Pooled OOF Weighted AUC: 0.709
    
    Outer-fold AUC:
       Outer_Fold  Macro_AUC
    0           1   0.729186
    1           2   0.616729
    2           3   0.668340
    3           4   0.674019
    4           5   0.733791
    5           6   0.673225
    6           7   0.725116
    7           8   0.771832
    8           9   0.742103
    9          10   0.714480
    
    Class-specific OOF AUC:
    Control: 0.720
    EoCRC: 0.625
    LoCRC: 0.743
    
    Best parameters selected in each outer fold:
      criterion  max_depth max_features  min_samples_leaf  min_samples_split  \
    0   entropy          5         log2                 1                  2   
    1   entropy         10         log2                 1                  5   
    2   entropy          5         sqrt                 1                  2   
    3      gini          5         log2                 1                  2   
    4      gini          5         sqrt                 2                 10   
    5   entropy         10         sqrt                 1                 10   
    6   entropy         10         log2                 1                  5   
    7   entropy          5         sqrt                 2                 10   
    8   entropy          5         log2                 2                  5   
    9      gini         10         log2                 2                  5   
    
       n_estimators  
    0           250  
    1           250  
    2           250  
    3           250  
    4           250  
    5           160  
    6           160  
    7           160  
    8           160  
    9           160  
    
    Most frequently selected parameters:
    
    criterion:
    criterion
    entropy    7
    gini       3
    Name: count, dtype: int64
    
    max_depth:
    max_depth
    5     6
    10    4
    Name: count, dtype: int64
    
    max_features:
    max_features
    log2    6
    sqrt    4
    Name: count, dtype: int64
    
    min_samples_leaf:
    min_samples_leaf
    1    6
    2    4
    Name: count, dtype: int64
    
    min_samples_split:
    min_samples_split
    5     4
    2     3
    10    3
    Name: count, dtype: int64
    
    n_estimators:
    n_estimators
    250    5
    160    5
    Name: count, dtype: int64



    
![png](params_tunning_RF_files/params_tunning_RF_12_1.png)
    



```python
for i, cls in enumerate(classes):

    class_auc = roc_auc_score(
        y_bin[:, i],
        oof_proba[:, i]
    )

    print(f"{cls}: AUC = {class_auc:.3f}")
```

    Control: AUC = 0.737
    EoCRC: AUC = 0.656
    LoCRC: AUC = 0.741



```python
##### Category C


import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# 1. Prepare data
# ---------------------------

X = abund.drop(columns=['Patient_Status'])
y = abund['Patient_Status']

target_genera = [
    "Peptostreptococcus",
    "Parvimonas",
    "Clostridium",
    "Limosilactobacillus",
    "Ruthenibacterium",
    "Gemella",
    "Faecalibacterium",
    "Anaerotignum",
    "Megamonas",
    "Solibacillus",
    "Longicatena",
    "Fusobacterium"
 ]

# ---------------------------
# 2. Check which taxa exist
# ---------------------------

selected_genera = [
    g for g in target_genera
    if g in X.columns
]

missing_genera = [
    g for g in target_genera
    if g not in X.columns
]

print("Selected genera:")
print(selected_genera)

print("\nMissing genera:")
print(missing_genera)

# ---------------------------
# 3. Subset features
# ---------------------------

X = X[selected_genera].copy()

print("\nInitial shape:", X.shape)

# ---------------------------
# 4. Prevalence filtering
# ---------------------------

prevalence = (X > 0).mean(axis=0)

prevalence_table = pd.DataFrame({
    "Genus": prevalence.index,
    "Prevalence": prevalence.values,
    "Percent": prevalence.values * 100
})

print("\nPrevalence:")
print(prevalence_table.sort_values("Percent", ascending=False))

keep = prevalence > 0.10

X_filtered_raw = X.loc[:, keep]

print("\nFeatures retained after 10% prevalence filtering:")
print(X_filtered_raw.columns.tolist())

print("\nFiltered shape:", X_filtered_raw.shape)

# ---------------------------
# 5. CLR transformation
# ---------------------------

X_pseudo = X_filtered_raw + 1e-6

X_clr = (
    np.log(X_pseudo)
    - np.log(X_pseudo).mean(axis=1).values.reshape(-1, 1)
)

# ---------------------------
# 6. Random Forest pipeline
# ---------------------------

pipeline = Pipeline([
    (
        'rf',
        RandomForestClassifier(
            random_state=42,
            class_weight='balanced'
        )
    )
])

# ---------------------------
# 7. RF tuning grid
# ---------------------------

params_grid = {
    'rf__n_estimators': [130, 160, 190],
    'rf__criterion': ['gini'],
    'rf__max_depth': [35, 55],
    'rf__min_samples_split': [0.001, 0.005],
    'rf__min_samples_leaf': [0.001, 0.005],
    'rf__max_features': ['sqrt', 'log2', None]
}

# Number of combinations

num_combinations = np.prod(
    [len(v) for v in params_grid.values()]
)

print("\nNumber of combinations:", num_combinations)

# ---------------------------
# 8. Cross-validation
# ---------------------------

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

# ---------------------------
# 9. Grid search
# ---------------------------

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=params_grid,
    cv=cv,
    scoring='roc_auc_ovr',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

# ---------------------------
# 10. Fit
# ---------------------------

grid_search.fit(X_clr, y)

# ---------------------------
# 11. Results
# ---------------------------

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest CV AUC:")
print(grid_search.best_score_)
```

    Selected genera:
    ['Peptostreptococcus', 'Parvimonas', 'Clostridium', 'Limosilactobacillus', 'Ruthenibacterium', 'Gemella', 'Faecalibacterium', 'Anaerotignum', 'Megamonas', 'Solibacillus', 'Longicatena', 'Fusobacterium']
    
    Missing genera:
    []
    
    Initial shape: (597, 12)
    
    Prevalence:
                      Genus  Prevalence    Percent
    6      Faecalibacterium    0.899497  89.949749
    10          Longicatena    0.720268  72.026801
    7          Anaerotignum    0.695142  69.514238
    2           Clostridium    0.658291  65.829146
    4      Ruthenibacterium    0.654941  65.494137
    5               Gemella    0.557789  55.778894
    11        Fusobacterium    0.522613  52.261307
    0    Peptostreptococcus    0.390285  39.028476
    1            Parvimonas    0.343384  34.338358
    8             Megamonas    0.299832  29.983250
    3   Limosilactobacillus    0.232831  23.283082
    9          Solibacillus    0.060302   6.030151
    
    Features retained after 10% prevalence filtering:
    ['Peptostreptococcus', 'Parvimonas', 'Clostridium', 'Limosilactobacillus', 'Ruthenibacterium', 'Gemella', 'Faecalibacterium', 'Anaerotignum', 'Megamonas', 'Longicatena', 'Fusobacterium']
    
    Filtered shape: (597, 11)
    
    Number of combinations: 72
    Fitting 10 folds for each of 72 candidates, totalling 720 fits
    
    Best parameters:
    {'rf__criterion': 'gini', 'rf__max_depth': 35, 'rf__max_features': 'sqrt', 'rf__min_samples_leaf': 0.001, 'rf__min_samples_split': 0.005, 'rf__n_estimators': 190}
    
    Best CV AUC:
    0.7051579331041827



```python
# Convert GridSearchCV results to DataFrame
df_cv_results = pd.DataFrame(grid_search.cv_results_)

# Keep only relevant columns
df_cv_results = df_cv_results[[
    'rank_test_score',
    'mean_test_score',
    'mean_train_score',          # available only if return_train_score=True in GridSearchCV
    'param_rf__n_estimators',
    'param_rf__min_samples_split',
    'param_rf__min_samples_leaf',
    'param_rf__max_features',
    'param_rf__max_depth',
    'param_rf__criterion',
    'param_feature_selection__k'
]]

# Sort by rank of test score
df_cv_results.sort_values('rank_test_score', inplace=True)

# Display
df_cv_results.reset_index(drop=True, inplace=True)
df_cv_results.iloc[:, :4]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank_test_score</th>
      <th>mean_test_score</th>
      <th>mean_train_score</th>
      <th>param_rf__n_estimators</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.696319</td>
      <td>0.999785</td>
      <td>130</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.696319</td>
      <td>0.999785</td>
      <td>130</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.696319</td>
      <td>0.999785</td>
      <td>130</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.696319</td>
      <td>0.999785</td>
      <td>130</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.695725</td>
      <td>0.999855</td>
      <td>190</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>67</th>
      <td>65</td>
      <td>0.681654</td>
      <td>0.999869</td>
      <td>130</td>
    </tr>
    <tr>
      <th>68</th>
      <td>69</td>
      <td>0.681211</td>
      <td>0.999996</td>
      <td>130</td>
    </tr>
    <tr>
      <th>69</th>
      <td>69</td>
      <td>0.681211</td>
      <td>0.999996</td>
      <td>130</td>
    </tr>
    <tr>
      <th>70</th>
      <td>71</td>
      <td>0.680527</td>
      <td>0.999996</td>
      <td>130</td>
    </tr>
    <tr>
      <th>71</th>
      <td>71</td>
      <td>0.680527</td>
      <td>0.999996</td>
      <td>130</td>
    </tr>
  </tbody>
</table>
<p>72 rows × 4 columns</p>
</div>




```python
# Define output directory (using your 'data' folder)
output_dir = 'data/'

# Save the cross-validation results
df_cv_results.to_csv(output_dir + 'grid_search_results.csv', index=False)

print(f"Grid search results saved to {output_dir}grid_search_results.csv")
```

    Grid search results saved to data/grid_search_results.csv



```python
# Access the best parameters from your grid search
grid_search.best_params_
```




    {'feature_selection__k': 100,
     'rf__criterion': 'gini',
     'rf__max_depth': 35,
     'rf__max_features': 'log2',
     'rf__min_samples_leaf': 0.005,
     'rf__min_samples_split': 0.001,
     'rf__n_estimators': 130}




```python
best_auc = grid_search.best_score_
print("Best cross-validated AUC:", best_auc)
```

    Best cross-validated AUC: 0.696318998263557



```python
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. Prepare data (same as for tuning)
# -----------------------------
X = abund.drop(columns=['Patient_Status'])
y = abund['Patient_Status']

# CLR transform
X_pseudo = X + 1e-6
X_clr = np.log(X_pseudo) - np.log(X_pseudo).mean(axis=1).values.reshape(-1,1)

# Prevalence filtering (10%)
prevalence = (X > 0).sum(axis=0) / X.shape[0]
X_filtered = X_clr.loc[:, prevalence > 0.1]

# -----------------------------
# 2. Create final Random Forest with tuned parameters
# -----------------------------
model_rf_fin = RandomForestClassifier(
    class_weight='balanced',
    criterion='gini',
    max_depth=55,
    max_features='log2',
    min_samples_leaf=0.005,     # fraction of samples
    min_samples_split=0.005,    # fraction of samples
    n_estimators=190,
    random_state=42
)

# -----------------------------
# 3. Fit the model on full filtered dataset
# -----------------------------
model_rf_fin.fit(X_filtered, y)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.estimator-table summary {
    padding: .5rem;
    font-family: monospace;
    cursor: pointer;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover {
    background-color: #e0e0e0;
}

.estimator-table table td {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left;
}

.user-set td.value pre {
    color:rgb(255, 94, 0) !important;
    background-color: transparent !important;
}

.default td {
    color: black;
    text-align: left;
}

.user-set td i,
.default td i {
    color: black;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}
</style><body><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(class_weight=&#x27;balanced&#x27;, max_depth=55,
                       max_features=&#x27;log2&#x27;, min_samples_leaf=0.005,
                       min_samples_split=0.005, n_estimators=190,
                       random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_estimators',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">n_estimators&nbsp;</td>
            <td class="value">190</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('criterion',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">criterion&nbsp;</td>
            <td class="value">&#x27;gini&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_depth',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_depth&nbsp;</td>
            <td class="value">55</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_samples_split',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_samples_split&nbsp;</td>
            <td class="value">0.005</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_samples_leaf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_samples_leaf&nbsp;</td>
            <td class="value">0.005</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_weight_fraction_leaf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_weight_fraction_leaf&nbsp;</td>
            <td class="value">0.0</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_features&nbsp;</td>
            <td class="value">&#x27;log2&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_leaf_nodes',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_leaf_nodes&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_impurity_decrease',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_impurity_decrease&nbsp;</td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('bootstrap',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">bootstrap&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('oob_score',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">oob_score&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_jobs',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">n_jobs&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('random_state',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">random_state&nbsp;</td>
            <td class="value">42</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">verbose&nbsp;</td>
            <td class="value">0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('warm_start',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">warm_start&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('class_weight',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">class_weight&nbsp;</td>
            <td class="value">&#x27;balanced&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('ccp_alpha',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">ccp_alpha&nbsp;</td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_samples',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_samples&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('monotonic_cst',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">monotonic_cst&nbsp;</td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div><script>function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.fa-regular.fa-copy').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const paramName = element.parentElement.nextElementSibling.textContent.trim();
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});
</script></body>




```python
from sklearn import metrics

# Predict probabilities
y_proba = model_rf_fin.predict_proba(X_filtered)

# Compute multiclass AUC (one-vs-rest)
auc_score = metrics.roc_auc_score(y, y_proba, multi_class='ovr')

print('AUC Score = {:.4f}'.format(auc_score))
```

    AUC Score = 0.9999



```python
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import numpy as np

# -----------------------------
# 1. Define classes and binarize labels
# -----------------------------
classes = ['Control', 'EoCRC', 'LoCRC']
y_bin = label_binarize(y, classes=classes)  # shape = (n_samples, n_classes)

# -----------------------------
# 2. Cross-validation setup
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store ROC curves for averaging
mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []

plt.figure(figsize=(7,7))

# -----------------------------
# 3. Run 5-fold CV ROC
# -----------------------------
for train_idx, test_idx in cv.split(X_filtered, y):
    # Fit on training fold
    model_rf_fin.fit(X_filtered.iloc[train_idx], y.iloc[train_idx])
    
    # Predict probabilities on test fold
    y_score = model_rf_fin.predict_proba(X_filtered.iloc[test_idx])
    
    # Compute ROC for each class
    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[test_idx, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Interpolate TPR for mean curve
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold ROC {class_label} (AUC = {roc_auc:.2f})')

# -----------------------------
# 4. Plot chance line
# -----------------------------
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2, alpha=0.8, label='Chance')

# -----------------------------
# 5. Mean ROC across folds
# -----------------------------
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue', lw=3, label=f'Mean ROC (AUC = {mean_auc:.2f})')

# -----------------------------
# 6. Formatting
# -----------------------------
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Multiclass ROC Curve (One-vs-Rest)', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

# -----------------------------
# 7. Save figure
# -----------------------------
plt.savefig('plots/ROC_final_model.png', dpi=300)
plt.show()
```


    
![png](params_tunning_RF_files/params_tunning_RF_21_0.png)
    



```python
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import numpy as np

# -----------------------------
# 1. Define classes
# -----------------------------
classes = ['Control', 'EoCRC', 'LoCRC']
y_bin = label_binarize(y, classes=classes)

# -----------------------------
# 2. Cross-validation
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Common FPR axis
mean_fpr = np.linspace(0, 1, 100)

# Store per-class results
tprs_per_class = {cls: [] for cls in classes}
aucs_per_class = {cls: [] for cls in classes}

plt.figure(figsize=(7,7))

# -----------------------------
# 3. CV loop
# -----------------------------
for train_idx, test_idx in cv.split(X_filtered, y):
    
    model_rf_fin.fit(X_filtered.iloc[train_idx], y.iloc[train_idx])
    y_score = model_rf_fin.predict_proba(X_filtered.iloc[test_idx])
    
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[test_idx, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Store AUC
        aucs_per_class[cls].append(roc_auc)
        
        # Interpolate TPR
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        
        tprs_per_class[cls].append(interp_tpr)

# -----------------------------
# 4. Plot mean ROC per class
# -----------------------------
colors = ['blue', 'green', 'red']

for cls, color in zip(classes, colors):
    mean_tpr = np.mean(tprs_per_class[cls], axis=0)
    mean_auc = np.mean(aucs_per_class[cls])
    
    plt.plot(mean_fpr, mean_tpr,
             color=color,
             lw=3,
             label=f'{cls} (AUC = {mean_auc:.2f})')

# -----------------------------
# 5. Chance line
# -----------------------------
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2, label='Chance')

# -----------------------------
# 6. Formatting
# -----------------------------
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Multiclass ROC Curve (One-vs-Rest, Averaged per Class)', fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()

# -----------------------------
# 7. Save
# -----------------------------
plt.savefig('plots/ROC_per_class_mean.png', dpi=300)
plt.show()
```


    
![png](params_tunning_RF_files/params_tunning_RF_22_0.png)
    



```python
df_var_imp = (
    pd.DataFrame({
        'Variable': X_filtered.columns,
        'Importance': model_rf_fin.feature_importances_
    })
    .sort_values(by='Importance', ascending=False)
    .reset_index(drop=True)
)

type(X_filtered)
```




    pandas.core.frame.DataFrame




```python
import matplotlib.pyplot as plt

ax = df_var_imp.head(15) \
    .sort_values(by='Importance') \
    .plot(
        x='Variable',
        y='Importance',
        kind='barh',
        figsize=(15, 5),
        legend=False
    )

ax.set_title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()
print(df_var_imp.head(20))
```


    
![png](params_tunning_RF_files/params_tunning_RF_24_0.png)
    


                     Variable  Importance
    0      Peptostreptococcus    0.021853
    1              Parvimonas    0.021619
    2                 Gemella    0.017526
    3     Novisyntrophococcus    0.013589
    4             Longicatena    0.013034
    5      Mediterraneibacter    0.012990
    6               Bilophila    0.012973
    7              Romboutsia    0.012788
    8             Clostridium    0.012175
    9     Limosilactobacillus    0.012167
    10          Fusobacterium    0.011979
    11                Blautia    0.011949
    12                  Wujia    0.011940
    13       Faecalibacterium    0.011655
    14       Ruthenibacterium    0.011521
    15           Anaerotignum    0.011440
    16         Parasutterella    0.011338
    17              Roseburia    0.010846
    18          Lactobacillus    0.010824
    19  Phascolarctobacterium    0.010585



```python
# top_features = [
#     "Peptostreptococcus", "Parvimonas", "Gemella", "Novisyntrophococcus",
#     "Longicatena", "Mediterraneibacter", "Bilophila", "Romboutsia",
#     "Clostridium", "Limosilactobacillus", "Fusobacterium", "Blautia",
#     "Wujia", "Faecalibacterium", "Ruthenibacterium", "Anaerotignum",
#     "Parasutterella", "Roseburia", "Lactobacillus", "Phascolarctobacterium"
# ]

# X_selected = X_filtered[top_features]


# from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score
# import numpy as np

# final_model = Pipeline([
#     ('feature_selection', SelectKBest(score_func=f_classif, k=100)),
#     ('rf', RandomForestClassifier(
#         class_weight='balanced',
#         criterion='gini',
#         max_depth=35,
#         max_features='log2',
#         min_samples_leaf=0.005,
#         min_samples_split=0.001,
#         n_estimators=130,
#         random_state=42
#     ))
# ])

# # Fit
# final_model.fit(X_filtered, y)

# # Evaluate
# scores = cross_val_score(
#     final_model,
#     X_filtered,
#     y,
#     cv=5,
#     scoring='roc_auc'
# )

# print("Mean AUC:", np.mean(scores))
```


```python
y_binary = y.replace({
    'Control': 0,
    'EoCRC': 1,
    'LoCRC': 1
})

scores = cross_val_score(
    final_model,
    X_filtered,
    y_binary,
    cv=10,
    scoring='roc_auc'
)

scoring='roc_auc_ovr'

scores = cross_val_score(
    final_model,
    X_filtered,
    y,
    cv=5,
    scoring='roc_auc_ovr'
)

print("Mean AUC:", np.mean(scores))
```

    /tmp/ipykernel_87422/1712892597.py:1: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      y_binary = y.replace({


    Mean AUC: 0.6537940525766001



```python
# ============================================================
# BORUTA + RANDOM FOREST
# Three-class CRC classification:
# Control vs EoCRC vs LoCRC
#
# 10-fold outer CV + inner CV hyperparameter tuning
# Boruta performed ONLY within each outer training fold
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

from sklearn.preprocessing import label_binarize

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    log_loss,
    classification_report
)

# ============================================================
# 1. DATA
# ============================================================

# X = microbial abundance matrix
# y = class labels

# Ensure matching indices
X = X.copy()

y = pd.Series(
    y,
    index=X.index,
    name="Patient_Status"
)

# ------------------------------------------------------------
# Explicit class order
# ------------------------------------------------------------

classes = np.array([
    "Control",
    "EoCRC",
    "LoCRC"
])

# Make sure the observed classes are exactly as expected
missing_classes = set(classes) - set(y.unique())

if missing_classes:
    raise ValueError(
        f"Missing expected classes: {missing_classes}"
    )

print("Number of samples:", X.shape[0])
print("Number of genera:", X.shape[1])

print("\nClass distribution:")
print(y.value_counts().reindex(classes))


# ============================================================
# 2. CROSS-VALIDATION
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=123
)

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)


# ============================================================
# 3. RANDOM FOREST HYPERPARAMETER GRID
# ============================================================

param_grid = {

    "n_estimators": [
        100,
        200,
        300
    ],

    "max_depth": [
        None,
        10,
        20,
        30
    ],

    "max_features": [
        "sqrt",
        "log2"
    ],

    "min_samples_leaf": [
        1,
        2,
        5
    ],

    "min_samples_split": [
        2,
        5
    ]
}


# ============================================================
# 4. STORAGE OBJECTS
# ============================================================

# OOF probabilities
oof_proba = np.zeros(
    (
        X.shape[0],
        len(classes)
    )
)

# OOF predicted class
oof_pred = np.empty(
    X.shape[0],
    dtype=object
)

# True labels
oof_true = y.values.copy()

# Feature selection frequency
feature_counter = Counter()

# Store features selected in every fold
fold_features = {}

# Store best parameters
fold_best_params = {}

# Store number of Boruta-selected features
fold_n_features = []

# Store fold-specific AUC
fold_auc = []


# ============================================================
# 5. OUTER CROSS-VALIDATION
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X, y),
    start=1
):

    print("\n" + "=" * 70)
    print(f"OUTER FOLD {fold}")
    print("=" * 70)

    # --------------------------------------------------------
    # Split data
    # --------------------------------------------------------

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    print(
        f"Training samples: {len(train_idx)} | "
        f"Test samples: {len(test_idx)}"
    )


    # ========================================================
    # 6. BORUTA FEATURE SELECTION
    # ========================================================

    rf_boruta = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        max_depth=None,
        max_features="sqrt",
        n_jobs=-1,
        random_state=123
    )

    boruta = BorutaPy(
        estimator=rf_boruta,

        n_estimators="auto",

        # Two-sided statistical test
        two_step=True,

        # Conservative significance level
        alpha=0.05,

        # More iterations improve stability
        max_iter=100,

        random_state=123,

        verbose=0
    )

    print("\nRunning Boruta...")

    boruta.fit(
        X_train.values,
        y_train.values
    )


    # --------------------------------------------------------
    # Confirmed features
    # --------------------------------------------------------

    selected_features = X_train.columns[
        boruta.support_
    ].tolist()


    # --------------------------------------------------------
    # Tentative features
    # --------------------------------------------------------

    tentative_features = X_train.columns[
        boruta.support_weak_
    ].tolist()


    print(
        f"Confirmed Boruta features: "
        f"{len(selected_features)}"
    )

    print(
        f"Tentative Boruta features: "
        f"{len(tentative_features)}"
    )


    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(selected_features) == 0:

        print(
            "WARNING: No confirmed Boruta features "
            "in this fold."
        )

        continue


    # ========================================================
    # 7. RECORD FEATURE STABILITY
    # ========================================================

    feature_counter.update(
        selected_features
    )

    fold_features[
        f"Fold_{fold}"
    ] = selected_features

    fold_n_features.append(
        len(selected_features)
    )


    # ========================================================
    # 8. INNER CV RANDOM FOREST TUNING
    # ========================================================

    rf_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=123,
        n_jobs=-1
    )

    grid = GridSearchCV(

        estimator=rf_model,

        param_grid=param_grid,

        cv=inner_cv,

        # IMPORTANT:
        # Three-class ROC-AUC
        scoring="roc_auc_ovr",

        n_jobs=-1,

        refit=True,

        verbose=0
    )


    print("\nOptimizing Random Forest...")

    grid.fit(
        X_train[selected_features],
        y_train
    )


    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_rf = grid.best_estimator_

    fold_best_params[
        f"Fold_{fold}"
    ] = grid.best_params_


    print(
        "\nBest inner-CV AUC:",
        round(
            grid.best_score_,
            4
        )
    )

    print(
        "Selected features:",
        len(selected_features)
    )

    print(
        "Best parameters:",
        grid.best_params_
    )


    # ========================================================
    # 9. OUTER TEST-SET PROBABILITIES
    # ========================================================

    # IMPORTANT:
    # Keep ALL THREE probabilities
    #
    # [Control probability,
    #  EoCRC probability,
    #  LoCRC probability]

    prob_test = best_rf.predict_proba(
        X_test[selected_features]
    )


    # --------------------------------------------------------
    # Align probability columns with explicit class order
    # --------------------------------------------------------

    model_classes = best_rf.classes_

    aligned_prob = np.zeros(
        (
            len(test_idx),
            len(classes)
        )
    )

    for j, cls in enumerate(model_classes):

        class_position = np.where(
            classes == cls
        )[0][0]

        aligned_prob[:, class_position] = (
            prob_test[:, j]
        )


    # --------------------------------------------------------
    # Store OOF probabilities
    # --------------------------------------------------------

    oof_proba[test_idx, :] = aligned_prob


    # --------------------------------------------------------
    # Predicted class
    # --------------------------------------------------------

    oof_pred[test_idx] = classes[
        np.argmax(
            aligned_prob,
            axis=1
        )
    ]


    # ========================================================
    # 10. FOLD-SPECIFIC AUC
    # ========================================================

    y_test_bin = label_binarize(
        y_test,
        classes=classes
    )

    fold_macro_auc = roc_auc_score(
        y_test_bin,
        aligned_prob,
        multi_class="ovr",
        average="macro"
    )

    fold_auc.append(
        fold_macro_auc
    )

    print(
        "Outer-fold Macro AUC:",
        round(
            fold_macro_auc,
            4
        )
    )


# ============================================================
# 11. FEATURE STABILITY TABLE
# ============================================================

feature_freq = pd.DataFrame(
    feature_counter.items(),
    columns=[
        "Feature",
        "Selection_Count"
    ]
)

feature_freq[
    "Selection_Frequency"
] = (
    feature_freq["Selection_Count"]
    / outer_cv.n_splits
)

feature_freq[
    "Selection_Percentage"
] = (
    feature_freq["Selection_Frequency"]
    * 100
)

feature_freq = feature_freq.sort_values(
    "Selection_Frequency",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# 12. STABLE FEATURE PANELS
# ============================================================

stable_90 = feature_freq[
    feature_freq[
        "Selection_Percentage"
    ] >= 90
].copy()

stable_80 = feature_freq[
    feature_freq[
        "Selection_Percentage"
    ] >= 80
].copy()

stable_70 = feature_freq[
    feature_freq[
        "Selection_Percentage"
    ] >= 70
].copy()


print("\n" + "=" * 70)
print("BORUTA FEATURE STABILITY")
print("=" * 70)

print(
    "\nFeatures selected in ≥90% of folds:",
    len(stable_90)
)

print(
    "\nFeatures selected in ≥80% of folds:",
    len(stable_80)
)

print(
    "\nFeatures selected in ≥70% of folds:",
    len(stable_70)
)


print("\nTop stable genera:")

print(
    feature_freq.head(30).to_string(
        index=False
    )
)


# ============================================================
# 13. THREE-CLASS OOF PERFORMANCE
# ============================================================

y_bin = label_binarize(
    oof_true,
    classes=classes
)


# ------------------------------------------------------------
# Macro AUC
# ------------------------------------------------------------

macro_auc = roc_auc_score(
    y_bin,
    oof_proba,
    multi_class="ovr",
    average="macro"
)


# ------------------------------------------------------------
# Weighted AUC
# ------------------------------------------------------------

weighted_auc = roc_auc_score(
    y_bin,
    oof_proba,
    multi_class="ovr",
    average="weighted"
)


# ------------------------------------------------------------
# Per-class AUC
# ------------------------------------------------------------

class_auc = {}

for i, cls in enumerate(classes):

    class_auc[cls] = roc_auc_score(
        y_bin[:, i],
        oof_proba[:, i]
    )


# ============================================================
# 14. OTHER PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    oof_true,
    oof_pred
)

balanced_accuracy = balanced_accuracy_score(
    oof_true,
    oof_pred
)

logloss = log_loss(
    oof_true,
    oof_proba,
    labels=classes
)


# ============================================================
# 15. MULTICLASS BRIER SCORE
# ============================================================

# Multiclass extension:
# mean squared error across one-hot outcomes

brier_score = np.mean(
    np.sum(
        (
            oof_proba - y_bin
        ) ** 2,
        axis=1
    )
)


# ============================================================
# 16. FINAL PERFORMANCE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL OUT-OF-FOLD PERFORMANCE")
print("=" * 70)

print(
    f"\nControl AUC : "
    f"{class_auc['Control']:.3f}"
)

print(
    f"EoCRC AUC   : "
    f"{class_auc['EoCRC']:.3f}"
)

print(
    f"LoCRC AUC   : "
    f"{class_auc['LoCRC']:.3f}"
)

print(
    f"\nMacro AUC   : "
    f"{macro_auc:.3f}"
)

print(
    f"Weighted AUC: "
    f"{weighted_auc:.3f}"
)

print(
    f"Accuracy    : "
    f"{accuracy:.3f}"
)

print(
    f"Balanced Acc: "
    f"{balanced_accuracy:.3f}"
)

print(
    f"Log loss    : "
    f"{logloss:.3f}"
)

print(
    f"Brier score : "
    f"{brier_score:.3f}"
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    oof_true,
    oof_pred,
    labels=classes
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual {c}"
        for c in classes
    ],
    columns=[
        f"Predicted {c}"
        for c in classes
    ]
)

print("\nConfusion Matrix:")
print(cm_df)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        oof_true,
        oof_pred,
        labels=classes,
        target_names=classes,
        digits=3
    )
)


# ============================================================
# 19. OOF PROBABILITY TABLE
# ============================================================

oof_results = pd.DataFrame(
    {
        "Observed": oof_true,

        "Predicted": oof_pred,

        "P_Control":
            oof_proba[:, 0],

        "P_EoCRC":
            oof_proba[:, 1],

        "P_LoCRC":
            oof_proba[:, 2]
    },
    index=X.index
)

print("\nOOF probability table:")
print(
    oof_results.head()
)


# ============================================================
# 20. SAVE RESULTS
# ============================================================

feature_freq.to_csv(
    "Boruta_feature_stability_3class.csv",
    index=False
)

oof_results.to_csv(
    "RF_OOF_probabilities_3class.csv"
)

cm_df.to_csv(
    "RF_OOF_confusion_matrix_3class.csv"
)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[2], line 45
         26 from sklearn.metrics import (
         27     roc_auc_score,
         28     roc_curve,
       (...)
         34     classification_report
         35 )
         37 # ============================================================
         38 # 1. DATA
         39 # ============================================================
       (...)
         43 
         44 # Ensure matching indices
    ---> 45 X = X.copy()
         47 y = pd.Series(
         48     y,
         49     index=X.index,
         50     name="Patient_Status"
         51 )
         53 # ------------------------------------------------------------
         54 # Explicit class order
         55 # ------------------------------------------------------------


    NameError: name 'X' is not defined

