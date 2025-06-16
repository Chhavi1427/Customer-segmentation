import streamlit as st
import io
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

st.title('Customer Segmentation App')

uploaded_file = st.file_uploader("Drag and Drop or Select Files", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        decoded = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        decoded = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format")
        st.stop()

    st.write("Preview of uploaded data:")
    st.dataframe(decoded.head())

    num_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10, value=3)

    def perform_clustering(data, n_clusters):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters_result = kmeans.fit_predict(scaled_data)
        return clusters_result

    # Use only numeric columns for clustering
    features = decoded.select_dtypes(include=[np.number])

    if features.empty:
        st.warning("No numeric features available for clustering.")
        st.stop()

    clusters = perform_clustering(features, num_clusters)

    # Add cluster label to original dataframe
    decoded['Cluster'] = clusters

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(features)

    st.plotly_chart(
        px.scatter(
            x=reduced_data[:, 0],
            y=reduced_data[:, 1],
            color=clusters.astype(str),  # convert to str for coloring
            title='Customer Segmentation',
            labels={'color': 'Cluster', 'x': 'PCA 1', 'y': 'PCA 2'},
        ).update_layout(margin=dict(l=0, r=0, b=0, t=30))
    )

    st.write("Clustered Data:")
    st.dataframe(decoded.head())
