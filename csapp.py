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
    content = uploaded_file.read()
    content_type = uploaded_file.type

    if 'csv' in content_type:
        decoded = pd.read_csv(io.StringIO(content.decode('utf-8')))
    elif 'excel' in content_type:
        decoded = pd.read_excel(io.BytesIO(content))
    else:
        raise ValueError("Unsupported file format")

    num_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10, value=3)

    def perform_clustering(data, n_clusters):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters_result = kmeans.fit_predict(scaled_data)
        return clusters_result

    features = decoded.select_dtypes(include=[np.number])
    clusters = perform_clustering(features, num_clusters)

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(features)

    st.plotly_chart(
        px.scatter(
            x=reduced_data[:, 0],
            y=reduced_data[:, 1],
            color=clusters,
            title='Customer Segmentation',
            labels={'color': 'Cluster'},
        ).update_layout(margin=dict(l=0, r=0, b=0, t=0))
    )
