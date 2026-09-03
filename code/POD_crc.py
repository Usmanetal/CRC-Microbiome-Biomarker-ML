import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator


def generate_POD(dataset, positive_class):
    
    # Features and labels
    X = dataset.iloc[:, 2:]
    y = dataset["Patient_Status"]

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Identify which encoded class is the disease class
    positive_label = encoder.transform([positive_class])[0]


    outer_cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=123
    )


    pod = np.zeros(len(y_encoded))


    for train_idx, test_idx in outer_cv.split(X, y_encoded):

        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]

        y_train = y_encoded[train_idx]


        model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=123
        )


        model.fit(
            X_train,
            y_train
        )


        # Probability of positive class
        class_index = list(model.classes_).index(
            positive_label
        )


        pod[test_idx] = model.predict_proba(
            X_test
        )[:, class_index]


    result = pd.DataFrame({

        "Sample_ID": dataset["Patient_ID"].values,

        "Patient_Status": dataset["Patient_Status"].values,

        "POD": pod

    })

    return result

#### Control_vs_EoCRC POD Table
pod_table_ce = generate_POD(
    ce,
    positive_class="EoCRC"
)

pod_table_ce = pod_table_ce.rename(
    columns={"POD":"POD_EoCRC"}
)

#### Control_vs_LoCRC POD Table
pod_table_cl = generate_POD(
    cl,
    positive_class="LoCRC"
)

pod_table_cl = pod_table_cl.rename(
    columns={"POD":"POD_LoCRC"}
)

### EoCRC_vs_LoCRC POD Table

pod_table_el = generate_POD(
    el,
    positive_class="LoCRC"
)

pod_table_el = pod_table_el.rename(
    columns={"POD":"POD_LoCRC_vs_EoCRC"}
)

###########################
##
def add_POD_boxplot(
    ax,
    data,
    pod_column,
    group_order,
    palette,
    y_label,
    title
):

    # Boxplot
    sns.boxplot(
        data=data,
        x="Patient_Status",
        y=pod_column,
        order=group_order,
        palette=palette,
        width=0.55,
        linewidth=1.5,
        showfliers=True,
        ax=ax
    )

    # Remove ONLY whisker caps
    for line in ax.lines:
        x = line.get_xdata()
        y = line.get_ydata()

        if len(x) == 2 and y[0] == y[1]:
            # Identify short horizontal lines
            if abs(x[1] - x[0]) < 0.3:
                line.set_visible(False)
    # Individual observations
    sns.stripplot(
        data=data,
        x="Patient_Status",
        y=pod_column,
        order=group_order,
        color="black",
        size=2.5,
        alpha=0.4,
        jitter=True,
        ax=ax
    )

    # Pairwise comparison
    pairs = [
        (group_order[0], group_order[1])
    ]

    annotator = Annotator(
        ax,
        pairs,
        data=data,
        x="Patient_Status",
        y=pod_column,
        order=group_order
    )

    annotator.configure(
        test="Mann-Whitney",
        text_format="star",
        loc="outside",
        verbose=0
    )

    annotator.apply_and_annotate()

    # Titles and labels
    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=30
    )

    ax.set_xlabel("")

    ax.set_ylabel(
        y_label,
        fontsize=11
    )

    sns.despine(ax=ax)
#### combining plots for Control_vs_EoCRC, Control_vs_LoCRC, and EoCRC_vs_LoCRC
fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(15,5),
    dpi=600
)


# Panel 1: Control vs EoCRC
add_POD_boxplot(
    ax=axes[0],
    data=pod_table_ce,
    pod_column="POD_EoCRC",
    group_order=["Control","EoCRC"],
    palette={
        "Control":"lightgray",
        "EoCRC":"#F39C12"
    },
    y_label="Probability of EoCRC",
    title="A. Control vs EoCRC"
)


# Panel 2: Control vs LoCRC
add_POD_boxplot(
    ax=axes[1],
    data=pod_table_cl,
    pod_column="POD_LoCRC",
    group_order=["Control","LoCRC"],
    palette={
        "Control":"lightgray",
        "LoCRC":"#00A087"
    },
    y_label="Probability of LoCRC",
    title="B. Control vs LoCRC"
)


# Panel 3: EoCRC vs LoCRC
add_POD_boxplot(
    ax=axes[2],
    data=pod_table_el,
    pod_column="POD_LoCRC_vs_EoCRC",
    group_order=["EoCRC","LoCRC"],
    palette={
        "EoCRC":"#F39C12",
        "LoCRC":"#00A087"
    },
    y_label="Probability of LoCRC",
    title="C. EoCRC vs LoCRC"
)


plt.tight_layout()
plt.subplots_adjust(top=0.90)

plt.savefig(
    "CRC_POD_boxplots_3panel.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


