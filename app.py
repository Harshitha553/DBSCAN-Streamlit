import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="DBSCAN Clustering",
    layout="wide"
)

st.title("DBSCAN Clustering Visualization")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select numerical columns
    numerical_cols = [
        col for col in df.select_dtypes(
            include=['int64', 'float64']
        ).columns
        if col != 'CustomerID'
    ]

    x_col = st.selectbox(
        "Select Feature 1",
        numerical_cols,
        index=0
    )

    y_col = st.selectbox(
        "Select Feature 2",
        numerical_cols,
        index=1 if len(numerical_cols) > 1 else 0
    )

    if x_col == y_col:
        st.error("Please select different features.")
        st.stop()

    eps = st.slider(
        "Epsilon (eps)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1
    )

    min_samples = st.slider(
        "Minimum Samples",
        min_value=2,
        max_value=20,
        value=5
    )

    X = df[[x_col, y_col]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = model.fit_predict(X_scaled)

    df["Cluster"] = labels

    st.subheader("Clustered Data")
    st.dataframe(df.head())

    # ====================================
    # VISUALIZATIONS
    # ====================================

    col1, col2 = st.columns(2)

    # Scatter Plot

    with col1:

        st.subheader("DBSCAN Cluster Visualization")

        fig, ax = plt.subplots(figsize=(4, 3))

        scatter = ax.scatter(
            df[x_col],
            df[y_col],
            c=df["Cluster"],
            s=50
        )

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title("DBSCAN Clusters")

        plt.colorbar(scatter)

        plt.tight_layout()

        st.pyplot(fig)

    # Cluster Distribution

    with col2:

        st.subheader("Cluster Distribution")

        cluster_count = df["Cluster"].value_counts()

        fig2, ax2 = plt.subplots(figsize=(4, 3))

        ax2.bar(
            cluster_count.index.astype(str),
            cluster_count.values
        )

        ax2.set_xlabel("Cluster")
        ax2.set_ylabel("Count")
        ax2.set_title("Cluster Counts")

        plt.tight_layout()

        st.pyplot(fig2)

    # ====================================
    # OUTLIER DETECTION
    # ====================================

    st.subheader("Outlier Detection")

    noise_count = (labels == -1).sum()

    total_points = len(df)

    outlier_percentage = round(
        (noise_count / total_points) * 100,
        2
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            "Noise Points",
            noise_count
        )

    with metric_col2:
        st.metric(
            "Total Records",
            total_points
        )

    with metric_col3:
        st.metric(
            "Outlier %",
            f"{outlier_percentage}%"
        )

    # ====================================
    # CLUSTER SUMMARY
    # ====================================

    st.subheader("Cluster Summary")

    summary_df = pd.DataFrame({
        "Cluster": cluster_count.index,
        "Count": cluster_count.values
    })

    st.dataframe(summary_df)
