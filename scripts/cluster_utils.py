import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from IPython.display import display


def find_best_k(df_X, k_range=range(2, 11), random_state=42):
    """
    Find the best number of clusters (k) for KMeans using inertia and silhouette scores.
    Args:
        df_X (DataFrame): Feature matrix for clustering.
        k_range (range): The range of k values to try.
        random_state (int): Random seed for reproducibility.
    Returns:
        best_k (int): The value of k with the highest silhouette score.
    """

    # Initialise the silhouette score and inertia (within-cluster sum of squares, for reference)
    silhouettes = []
    inertias = []

    # Loop through each k value, fit KMeans and record silhouette scores and inertias
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(df_X)

        silhouettes.append(silhouette_score(df_X, labels))
        inertias.append(kmeans.inertia_)

    # Find the best K based on the highest silhouette score
    best_k = k_range[np.argmax(silhouettes)]

    # Plot silhouette scores and inertias with dual y-axis
    fig, ax1 = plt.subplots(figsize=(6, 3))

    color1 = "tab:blue"
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Silhouette Score", color=color1)
    ax1.plot(k_range, silhouettes, "s--", color=color1, label="Silhouette Score")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:green"
    ax2.set_ylabel("Inertia", color=color2)
    ax2.plot(k_range, inertias, "o-", color=color2, label="Inertia")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title(f"Best k based on silhouette score: {best_k}")
    fig.tight_layout()
    plt.show()

    return best_k


def evaluate_clusters(df_X, labels):
    """
    Print key clustering evaluation metrics.
    Args:
        df_X (Dataframe): Feature matrix.
        labels (array-like): Cluster labels for each sample.
    """
    print("Cluster Evaluation Metrics:")
    print(f" - Silhouette Score:       {silhouette_score(df_X, labels):.3f}")
    print(f" - Calinski-Harabasz Score: {calinski_harabasz_score(df_X, labels):.3f}")
    print(f" - Davies-Bouldin Score:    {davies_bouldin_score(df_X, labels):.3f}\n")


def plot_cluster_radar(df_X, labels):
    """Generate radar charts showing the median, mean, and std of each nutrition feature by cluster."""

    df = df_X.copy()
    df["cluster"] = labels
    feature_cols = df.columns.drop("cluster")

    stats_mean = df.groupby("cluster")[feature_cols].mean()
    stats_median = df.groupby("cluster")[feature_cols].median()
    stats_std = df.groupby("cluster")[feature_cols].std()

    # Radar chart settings
    categories = feature_cols.tolist()
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
    fig.suptitle("Cluster Feature Summaries", fontsize=16, fontweight="bold")

    # Helper function to plot radar for a statistic dataframe
    def plot_radar(ax, df_stats, title):
        for cluster in df_stats.index:
            values = df_stats.loc[cluster].tolist()
            values += values[:1]
            ax.plot(angles, values, label=f"Cluster {cluster}")
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_title(title, size=14, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=9)

    # Plot each chart
    plot_radar(axes[0], stats_median, "Median per Cluster")
    plot_radar(axes[1], stats_mean, "Mean per Cluster")
    plot_radar(axes[2], stats_std, "Standard Deviation per Cluster")

    plt.tight_layout()
    plt.show()


def list_cluster_examples(df_X, labels, df_text, n=5, random_state=42):
    """Print a few example items (e.g. food names) from each cluster.
    Args:
        df_X (DataFrame): Feature matrix.
        labels (array-like): Cluster labels.
        df_text (DataFrame): Text reference DataFrame with original item names.
        n (int): Number of samples to show per cluster.
    """
    df_out = df_text.copy()
    df_out["cluster"] = labels
    df_merged = pd.concat([df_out, df_X], axis=1)

    for cluster in sorted(df_out["cluster"].unique()):
        print(f"\nCluster {cluster}:")
        sample = df_merged[df_merged["cluster"] == cluster].sample(
            n=min(n, len(df_merged[df_merged["cluster"] == cluster])),
            random_state=random_state,
        )
        display(
            sample.style.format(precision=3).set_table_styles(
                [
                    {"selector": "th", "props": [("max-width", "100px")]},
                    {
                        "selector": "td",
                        "props": [("max-width", "100px"), ("white-space", "pre-wrap")],
                    },
                ]
            )
        )


def clustering_evaluation_pipeline(df_X, df_text, labels, preview_n=10, random_state=42):
    """Run a clustering pipeline: visualise, and list examples.
    Args:
        df_X (DataFrame): Feature matrix.
        df_text (DataFrame): Item text info (e.g. food names).
        labels (array-like): Cluster labels for each sample.
        preview_n (int): Number of samples to preview per cluster.
        random_state (int): Random seed for reproducibility.
    Returns:
        labels (array-like): Cluster assignments.
    """
    evaluate_clusters(df_X, labels)
    plot_cluster_radar(df_X, labels)
    if df_text is not None:
        print("\nSample foods in each cluster:")
        list_cluster_examples(df_X, labels, df_text, n=preview_n)
