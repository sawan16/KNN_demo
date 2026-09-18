import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="KNN Visualizer",
    page_icon="📍",
    layout="wide"
)

st.title("📍 K-Nearest Neighbors (KNN) Visualizer")
st.markdown(
    "Select a dataset, choose up to **3 features**, enter one test point, "
    "and visualize its **K nearest neighbors**."
)


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

@st.cache_data
def load_builtin_dataset(name):

    if name == "Iris":
        data = load_iris()
    elif name == "Wine":
        data = load_wine()
    elif name == "Breast Cancer":
        data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="Target")

    # Add human-readable target names if available
    if hasattr(data, "target_names"):
        y = y.map(dict(enumerate(data.target_names)))

    df = X.copy()
    df["Target"] = y

    return df


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("1. Dataset")

dataset_source = st.sidebar.radio(
    "Choose dataset",
    ["Built-in dataset", "Upload CSV"]
)

if dataset_source == "Built-in dataset":

    dataset_name = st.sidebar.selectbox(
        "Select dataset",
        ["Iris", "Wine", "Breast Cancer"]
    )

    df = load_builtin_dataset(dataset_name)

else:

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info("👈 Upload a CSV file from the sidebar.")
        st.stop()

    df = pd.read_csv(uploaded_file)


# ---------------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------------

st.subheader("Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Missing values", int(df.isna().sum().sum()))

st.dataframe(df.head(10), use_container_width=True)


# ---------------------------------------------------------
# TARGET SELECTION
# ---------------------------------------------------------

st.sidebar.header("2. Columns")

target_column = st.sidebar.selectbox(
    "Target column",
    df.columns,
    index=len(df.columns) - 1
)

feature_candidates = [
    c for c in df.columns
    if c != target_column and pd.api.types.is_numeric_dtype(df[c])
]

if len(feature_candidates) == 0:
    st.error("No numeric feature columns found.")
    st.stop()


# ---------------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------------

selected_features = st.sidebar.multiselect(
    "Select 1–3 features",
    feature_candidates,
    default=feature_candidates[:min(2, len(feature_candidates))],
    max_selections=3
)

if len(selected_features) == 0:
    st.warning("Please select at least one feature.")
    st.stop()


# ---------------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------------

work_df = df[selected_features + [target_column]].dropna().copy()

X = work_df[selected_features].astype(float).values
y = work_df[target_column].values


# ---------------------------------------------------------
# SCALING
# ---------------------------------------------------------

st.sidebar.header("3. KNN Settings")

scaling = st.sidebar.checkbox(
    "Standardize features",
    value=False,
    help="KNN is distance-based, so scaling can significantly affect neighbors."
)

if scaling:

    scaler = StandardScaler()
    X_model = scaler.fit_transform(X)

else:

    scaler = None
    X_model = X.copy()


# ---------------------------------------------------------
# K VALUE
# ---------------------------------------------------------

max_k = min(20, len(X_model))

k = st.sidebar.slider(
    "K (number of neighbors)",
    min_value=1,
    max_value=max_k,
    value=min(5, max_k)
)


# ---------------------------------------------------------
# TEST POINT
# ---------------------------------------------------------

st.sidebar.header("4. Test Point")

test_values = []

for feature in selected_features:

    min_value = float(work_df[feature].min())
    max_value = float(work_df[feature].max())

    # Give a small margin around the observed range
    margin = (max_value - min_value) * 0.05

    if margin == 0:
        margin = 1

    value = st.sidebar.number_input(
        feature,
        min_value=min_value - margin,
        max_value=max_value + margin,
        value=float(work_df[feature].mean())
    )

    test_values.append(value)


test_point = np.array(test_values).reshape(1, -1)


# ---------------------------------------------------------
# TRANSFORM TEST POINT
# ---------------------------------------------------------

if scaler is not None:
    test_model = scaler.transform(test_point)
else:
    test_model = test_point


# ---------------------------------------------------------
# KNN MODEL
# ---------------------------------------------------------

knn = KNeighborsClassifier(
    n_neighbors=k
)

knn.fit(X_model, y)


prediction = knn.predict(test_model)[0]

distances, indices = knn.kneighbors(
    test_model,
    n_neighbors=k
)

neighbor_indices = indices[0]
neighbor_distances = distances[0]


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

st.subheader("KNN Result")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("K", k)

with col2:
    st.metric("Prediction", str(prediction))

with col3:
    st.metric(
        "Features used",
        len(selected_features)
    )


# ---------------------------------------------------------
# NEIGHBOR TABLE
# ---------------------------------------------------------

st.subheader(f"Top {k} Nearest Neighbors")

neighbor_df = work_df.iloc[neighbor_indices].copy()

neighbor_df.insert(
    0,
    "Neighbor Rank",
    range(1, k + 1)
)

neighbor_df.insert(
    1,
    "Distance",
    neighbor_distances
)

st.dataframe(
    neighbor_df,
    use_container_width=True
)


# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------

st.subheader("Visualization")

if len(selected_features) == 1:

    feature = selected_features[0]

    plot_df = pd.DataFrame({
        feature: X[:, 0],
        "Target": y
    })

    fig = px.scatter(
        plot_df,
        x=feature,
        y=[0] * len(plot_df),
        color="Target",
        hover_data=[feature],
        title=f"{feature} — KNN Neighbors"
    )

    # All training points
    fig.update_traces(
        marker=dict(size=9)
    )

    # Highlight neighbors
    neighbor_x = X[neighbor_indices, 0]

    fig.add_trace(
        go.Scatter(
            x=neighbor_x,
            y=[0] * len(neighbor_x),
            mode="markers",
            name="K Nearest Neighbors",
            marker=dict(
                size=16,
                symbol="circle-open",
                line=dict(width=3)
            )
        )
    )

    # Test point
    fig.add_trace(
        go.Scatter(
            x=[test_point[0, 0]],
            y=[0],
            mode="markers",
            name="Test Point",
            marker=dict(
                size=20,
                symbol="x",
                line=dict(width=3)
            )
        )
    )

    fig.update_yaxes(
        visible=False,
        showticklabels=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


elif len(selected_features) == 2:

    f1, f2 = selected_features

    plot_df = work_df.copy()

    plot_df["Point Type"] = "Training Data"

    plot_df.loc[
        plot_df.index.isin(work_df.iloc[neighbor_indices].index),
        "Point Type"
    ] = "K Nearest Neighbors"

    fig = px.scatter(
        plot_df,
        x=f1,
        y=f2,
        color=target_column,
        symbol="Point Type",
        hover_data=selected_features,
        title=f"{f1} vs {f2}"
    )

    # Test point
    fig.add_trace(
        go.Scatter(
            x=[test_point[0, 0]],
            y=[test_point[0, 1]],
            mode="markers",
            name="Test Point",
            marker=dict(
                size=18,
                symbol="x"
            )
        )
    )

    # Draw lines from test point to neighbors
    for idx in neighbor_indices:

        fig.add_trace(
            go.Scatter(
                x=[
                    test_point[0, 0],
                    X[idx, 0]
                ],
                y=[
                    test_point[0, 1],
                    X[idx, 1]
                ],
                mode="lines",
                showlegend=False,
                line=dict(
                    dash="dot",
                    width=1
                )
            )
        )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


elif len(selected_features) == 3:

    f1, f2, f3 = selected_features

    plot_df = work_df.copy()

    plot_df["Point Type"] = "Training Data"

    neighbor_indices_original = work_df.iloc[
        neighbor_indices
    ].index

    plot_df.loc[
        plot_df.index.isin(neighbor_indices_original),
        "Point Type"
    ] = "K Nearest Neighbors"

    fig = px.scatter_3d(
        plot_df,
        x=f1,
        y=f2,
        z=f3,
        color=target_column,
        symbol="Point Type",
        hover_data=selected_features,
        title=f"3D KNN Visualization"
    )

    # Test point
    fig.add_trace(
        go.Scatter3d(
            x=[test_point[0, 0]],
            y=[test_point[0, 1]],
            z=[test_point[0, 2]],
            mode="markers",
            name="Test Point",
            marker=dict(
                size=10,
                symbol="x"
            )
        )
    )

    # Lines from test point to neighbors
    for idx in neighbor_indices:

        fig.add_trace(
            go.Scatter3d(
                x=[
                    test_point[0, 0],
                    X[idx, 0]
                ],
                y=[
                    test_point[0, 1],
                    X[idx, 1]
                ],
                z=[
                    test_point[0, 2],
                    X[idx, 2]
                ],
                mode="lines",
                showlegend=False,
                line=dict(
                    dash="dot",
                    width=3
                )
            )
        )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ---------------------------------------------------------
# EXPLANATION
# ---------------------------------------------------------

with st.expander("📖 How KNN made this prediction"):

    st.write(
        f"The test point was compared with every training instance "
        f"using Euclidean distance."
    )

    st.write(
        f"The **{k} closest instances** were selected as its nearest neighbors."
    )

    st.write(
        "The predicted class is determined by majority voting among these neighbors."
    )

    if scaling:
        st.info(
            "Feature standardization is ON. Distances were calculated "
            "after transforming all selected features to a comparable scale."
        )
    else:
        st.info(
            "Feature standardization is OFF. Features are using their "
            "original scales, so large-scale features can have greater "
            "influence on Euclidean distance."
        )