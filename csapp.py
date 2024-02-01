import dash
import io
import base64
from dash import html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dash.exceptions import PreventUpdate


app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1(
        'Customer Segmentation App',
        style={'color': '#FFFFFF', 'background-color': '#148F77', 'text-align': 'center', 'padding': 10}
    ),

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        multiple=False,
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    ),

    html.Div([
        html.Label([
            "Select the number of clusters",
            dcc.Slider(
                id='num_clusters',
                min=2,
                max=10,
                value=3,
                marks={i: str(i) for i in range(2, 11)},
            ),
        ]),
        html.Br([]),
        dcc.Graph(id='cluster-result'),

    ], style={'width': '80%', 'margin': 'auto', 'padding': 10, 'margin-top': 20}),
])


def perform_clustering(data, n_clusters):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    return clusters


@app.callback(
    Output('cluster-result', 'figure'),
    Input('num_clusters', 'value'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_clusters(num_clusters, contents):
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents.split(',')

    if 'csv' in content_type:
        decoded = pd.read_csv(io.StringIO(base64.b64decode(content_string).decode('utf-8')))
    elif 'excel' in content_type:
        decoded = pd.read_excel(io.BytesIO(base64.b64decode(content_string)))
    else:
        raise ValueError("Unsupported file format")
    features = decoded.select_dtypes(include=[np.number])
    clusters = perform_clustering(features, num_clusters)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(features)
    fig = px.scatter(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        color=clusters,
        title='Customer Segmentation',
        labels={'color': 'Cluster'},
    )

    return fig
if __name__ == '__main__':
    app.run_server(debug=True)
