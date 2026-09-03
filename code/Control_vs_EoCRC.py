import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_validate
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    average_precision_score
)

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/Chemo_Gut_ML_clean.csv")


df.head()
metadata = df.iloc[:, :2]
abundance = df.iloc[:, 2:]

prevalence = (abundance > 0).mean(axis=0)
keep = prevalence >= 0.05
abundance_filtered = abundance.loc[:, keep]
print(abundance_filtered.shape)
#############################
## Adding a pseudo-count of 1 to the abundance matrix
###############################

abundance_filtered += 1e-6

from skbio.stats.composition import clr

clr_matrix = clr(abundance_filtered)

clr_df = pd.DataFrame(
    clr_matrix,
    columns=abundance_filtered.columns,
    index=abundance_filtered.index
)

############################
## Build dataset for ML
############################

data = pd.concat(
    [metadata, clr_df],
    axis=1
)

### Control_vs_EoCRC
ce = data[
    data.Patient_Status.isin(
        ["Control","EoCRC"]
    )
]

### Control_vs_LoCRC
cl = data[
    data.Patient_Status.isin(
        ["Control","LoCRC"]
    )
]
### EoCRC_vs_LoCRC
el = data[
    data.Patient_Status.isin(
        ["LoCRC", "EoCRC"]
    )
]

### Create X and y
X = ce.iloc[:,2:]
y = ce["Patient_Status"]

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# pd.DataFrame({
#     "Patient_Status": ce["Patient_Status"],
#     "y": y
# }).head(20)

# classes, counts = np.unique(y, return_counts=True)
# for c, n in zip(classes, counts):
#     print(f"Class {c}: {n}")

### Nested cross-validation with hyperparameter tuning
# Outer CV for model evaluation

outer_cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=123
)

## Inner CV for hyperparameter tuning

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)

### Bruta feature selection with Random Forest and GridSearchCV
from boruta import BorutaPy

rf = RandomForestClassifier(
    n_jobs=-1,
    class_weight="balanced",
    random_state=123
)

boruta = BorutaPy(
    estimator=rf,
    n_estimators="auto",
    random_state=123
)

boruta.fit(
    X.values,
    y
)

selected = X.columns[
    boruta.support_
]

selected

### Hyperparameter tuning with GridSearchCV

param_grid = {

    "n_estimators":[200,500,1000],

    "max_depth":[
        None,
        5,
        10,
        20
    ],

    "min_samples_leaf":[
        1,
        2,
        4
    ]
}

grid = GridSearchCV(

    RandomForestClassifier(),

    param_grid,

    cv=inner_cv,

    scoring="roc_auc",

    n_jobs=-1
)

grid.fit(
    X[selected],
    y
)

best_rf = grid.best_estimator_
best_rf

prob = grid.predict_proba(
    X[selected]
)[:,1]

pred = best_rf.predict(
    X[selected]
)

auc = roc_auc_score(
    y,
    prob
)

print(auc)

##### plot ROC curve
fpr,tpr,_ = roc_curve(
    y,
    prob
)

plt.plot(
    fpr,
    tpr
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

##### Precision-recall curve
precision,recall,_ = precision_recall_curve(
    y,
    prob
)

### Confusion matrix
cm = confusion_matrix(
    y,
    pred
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

#### Shap
import shap

explainer = shap.TreeExplainer(
    best_rf
)

shap_values = explainer.shap_values(
    X[selected]
)

#### SHAP summary plot

shap.summary_plot(
    shap_values,
    X[selected]
)

#############################
## publication grade
########################
y = pd.Series(
    y,
    index=X.index
)

all_y_test = []
all_y_prob = []
from collections import Counter

feature_counter = Counter()

for train_idx, test_idx in outer_cv.split(X, y):
    X_train = X.iloc[train_idx]
    X_test  = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test  = y.iloc[test_idx]
    rf = RandomForestClassifier(
        n_jobs=-1,
        class_weight="balanced",
        random_state=123
    )
    boruta = BorutaPy(
        estimator=rf,
        n_estimators="auto",
        random_state=123
    )
    boruta.fit(
        X_train.values,
        y_train.values
    )
    selected_features = X_train.columns[
        boruta.support_
    ]
    feature_counter.update(selected_features)
    grid = GridSearchCV(
        RandomForestClassifier(
            class_weight="balanced",
            random_state=123
        ),
        param_grid,
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    grid.fit(
        X_train[selected_features],
        y_train
    )
    prob_test = grid.predict_proba(
        X_test[selected_features]
    )[:,1]
    all_y_test.extend(
        y_test.tolist()
    )
    all_y_prob.extend(
        prob_test.tolist()
    )
    
feature_freq = (
    pd.DataFrame(
        feature_counter.items(),
        columns=["Feature", "Frequency"]
    )
    .sort_values(
        "Frequency",
        ascending=False
    )
)

print(feature_freq)
#####################################
## Lolliplot of feature importance
#####################################
import matplotlib.pyplot as plt
import seaborn as sns

# Sort from lowest to highest so the strongest biomarkers appear at the top
plot_df = feature_freq.sort_values("Frequency", ascending=True)

# Publication style
sns.set_style("white")

fig, ax = plt.subplots(
    figsize=(7, 5.5),
    dpi=600
)

# Horizontal lines
ax.hlines(
    y=plot_df["Feature"],
    xmin=0,
    xmax=plot_df["Frequency"],
    color="lightgray",
    linewidth=2
)

# Points
ax.scatter(
    plot_df["Frequency"],
    plot_df["Feature"],
    s=90,
    color="#2A9D8F",
    edgecolor="black",
    linewidth=0.6,
    zorder=3
)

# Frequency labels
for i, freq in enumerate(plot_df["Frequency"]):
    ax.text(
        freq + 0.15,
        i,
        str(freq),
        va="center",
        fontsize=10
    )

# Stability threshold (optional)
ax.axvline(
    x=9,
    color="#D55E00",
    linestyle="--",
    linewidth=1.5,
    label="≥90% folds"
)

# Formatting
ax.set_xlabel(
    "Boruta selection frequency (10-fold nested CV)",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylabel(
    "Microbial genus",
    fontsize=12,
    fontweight="bold"
)

ax.set_xlim(0, 10.8)
ax.set_xticks(range(0, 11))

ax.tick_params(
    axis="both",
    labelsize=11
)

ax.legend(
    frameon=False,
    fontsize=10,
    loc="lower right"
)

sns.despine()

plt.tight_layout()

plt.savefig(
    "Boruta_feature_stability_lollipop.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

###### Final ROC curve

import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap_auc_ci(
    y_true,
    y_prob,
    n_bootstraps=2000,
    random_state=123
):
    rng = np.random.RandomState(random_state)

    bootstrapped_scores = []

    for i in range(n_bootstraps):

        indices = rng.randint(
            0,
            len(y_true),
            len(y_true)
        )

        if len(np.unique(y_true[indices])) < 2:
            continue

        score = roc_auc_score(
            y_true[indices],
            y_prob[indices]
        )

        bootstrapped_scores.append(score)

    sorted_scores = np.sort(
        bootstrapped_scores
    )

    lower = np.percentile(
        sorted_scores,
        2.5
    )

    upper = np.percentile(
        sorted_scores,
        97.5
    )

    return lower, upper
    
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(
    all_y_test,
    all_y_prob
)

print(f"AUC = {auc:.3f}")    

auc_lower, auc_upper = bootstrap_auc_ci(
    np.array(all_y_test),
    np.array(all_y_prob)
)

print(f"95% CI = {auc_lower:.3f}–{auc_upper:.3f}")

###########################
## Final plot
###########################

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# ROC coordinates
fpr, tpr, _ = roc_curve(all_y_test, all_y_prob)

plt.figure(figsize=(6,6), dpi=600)

# ROC curve
plt.plot(
    fpr,
    tpr,
    color="#0072B2",
    linewidth=3,
    label=f"AUC = {auc:.3f} (95% CI {auc_lower:.3f}–{auc_upper:.3f})"
)

# Chance line
plt.plot(
    [0,1],
    [0,1],
    "--",
    color="gray",
    linewidth=1.5
)

plt.xlim(0,1)
plt.ylim(0,1.02)

plt.xlabel(
    "False Positive Rate",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "True Positive Rate",
    fontsize=14,
    fontweight="bold"
)

plt.title(
    "Control vs EoCRC",
    fontsize=15,
    fontweight="bold"
)

plt.legend(
    loc="lower right",
    frameon=False,
    fontsize=12
)

ax = plt.gca()

# Publication style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(
    axis="both",
    labelsize=12,
    width=1.2
)

plt.tight_layout()

plt.savefig(
    "ROC_Control_vs_EoCRC_Publication.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
