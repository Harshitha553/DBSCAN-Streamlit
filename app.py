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

    numerical_cols = [
        col for col in df.select_dtypes(
            include=['int64','float64']
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
        st.error(
            "Please select different features."
        )
        st.stop()

    eps = st.slider(
        "Epsilon (eps)",
        0.1,
        2.0,
        0.5,
        0.1
    )

    min_samples = st.slider(
        "Minimum Samples",
        2,
        20,
        5
    )

    X = df[[x_col, y_col]]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = model.fit_predict(X_scaled)

    df['Cluster'] = labels

    st.subheader("Clustered Data")
    st.dataframe(df.head())

    # Scatter Plot

    st.subheader("DBSCAN Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8,5))

    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        c=df['Cluster']
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("DBSCAN Clusters")

    st.pyplot(fig)

    # Cluster Distribution

    st.subheader("Cluster Distribution")

    cluster_count = df['Cluster'].value_counts()

    fig2, ax2 = plt.subplots()

    ax2.bar(
        cluster_count.index.astype(str),
        cluster_count.values
    )

    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Count")
    ax2.set_title("Cluster Counts")

    st.pyplot(fig2)

    # Noise Points

    noise_count = (labels == -1).sum()

    st.subheader("Outlier Detection")

    st.metric(
        "Noise Points Found",
        noise_count
    )