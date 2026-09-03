import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_stability(
    feature_freq,
    title,
    outfile,
    threshold=9,
    color="#2A9D8F"
):
    """
    Publication-quality lollipop plot of Boruta feature selection frequency.
    """

    plot_df = feature_freq.sort_values(
        "Frequency",
        ascending=True
    )

    sns.set_theme(style="white")

    fig, ax = plt.subplots(
        figsize=(7, 5.8),
        dpi=600
    )

    # Lollipop stems
    ax.hlines(
        y=plot_df["Feature"],
        xmin=0,
        xmax=plot_df["Frequency"],
        color="lightgray",
        linewidth=2
    )

    # Lollipop heads
    ax.scatter(
        plot_df["Frequency"],
        plot_df["Feature"],
        s=90,
        color=color,
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

    # Stability threshold
    ax.axvline(
        x=threshold,
        color="#D55E00",
            linestyle="--",
            linewidth=1.5,
            label = f"≥{threshold * 10:.0f}% folds"
    )

    # # Percentage label above the line
    # ax.text(
    #     threshold + 0.05,
    #     0.02,
    #     f"{threshold*10:.0f}%",
    #     transform=ax.get_xaxis_transform(),
    #     ha="left",
    #     va="bottom",
    #     fontsize=10,
    #     color="#D55E00",
    #     fontweight="bold"
    # )

    ax.set_xlabel(
        "Boruta selection frequency",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Microbial genus",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlim(0, 10.8)
    ax.set_xticks(range(0, 11))

    ax.tick_params(
        axis="both",
        labelsize=11
    )

    sns.despine()

    plt.tight_layout()

    plt.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()
    
plot_feature_stability(
    feature_freq=feature_freq,
    title="Feature stability: Control vs EoCRC",
    outfile="Boruta_Control_vs_EoCRC_lollipop.png",
    threshold=7,
    color="#F39C12"
)    