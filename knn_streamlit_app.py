"""
KNN Interactive Demo -- Streamlit App
--------------------------------------
Run with:  streamlit run knn_streamlit_app.py

Features:
- Dataset loader (built-in sklearn datasets OR upload your own CSV)
- Feature selection (choose 1-3 features to use & plot)
- Feature scaling toggle (None / Min-Max / Standardize) -- see how it
  changes which neighbors are "nearest"
- Single test-point input (via sidebar number inputs)
- 1D / 2D / 3D plot (Plotly) with the K nearest neighbors highlighted
  and the test point marked as a star
- KNN distance + majority-vote prediction computed with NumPy only
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

st.set_page_config(page_title="KNN Interactive Demo", layout="wide")
st.title("K-Nearest Neighbors \u2014 Interactive Demo")
st.caption("Load a dataset, pick features, place a test point, and see which neighbors KNN actually picks.")


# =====================================================================
# 1. DATASET LOADER
# =====================================================================

@st.cache_data
def load_builtin(name):
    loaders = {"Iris": load_iris, "Wine": load_wine, "Breast Cancer": load_breast_cancer}
    d = loaders[name](as_frame=True)
    df = d.data.copy()
    class_names = dict(enumerate(d.target_names))
    df["label"] = d.target.map(class_names)
    feature_cols = list(d.feature_names)
    return df, feature_cols


st.sidebar.header("1. Dataset")
source = st.sidebar.radio("Data source", ["Built-in dataset", "Upload CSV"])

if source == "Built-in dataset":
    dataset_name = st.sidebar.selectbox("Choose dataset", ["Iris", "Wine", "Breast Cancer"])
    df, feature_cols = load_builtin(dataset_name)
    label_col = "label"
else:
    uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.info("\u2b06\ufe0f Upload a CSV in the sidebar, or switch to a built-in dataset, to get started.")
        st.stop()
    df = pd.read_csv(uploaded)
    label_col = st.sidebar.selectbox("Which column is the class label?", df.columns)
    feature_cols = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        st.error("No numeric feature columns found (besides the label). Please upload a different CSV.")
        st.stop()

with st.expander("Dataset preview", expanded=False):
    st.dataframe(df.head(10))
    st.write(f"**{len(df)} rows** \u00b7 **{len(feature_cols)} numeric feature(s)** available \u00b7 "
             f"**{df[label_col].nunique()} classes**: {', '.join(map(str, df[label_col].unique()))}")


# =====================================================================
# 2. FEATURE SELECTION
# =====================================================================

st.sidebar.header("2. Feature selection")
selected_features = st.sidebar.multiselect(
    "Choose 1 to 3 features (used for both KNN and the plot)",
    feature_cols,
    default=feature_cols[: min(2, len(feature_cols))],
)

if len(selected_features) == 0:
    st.warning("Select at least 1 feature in the sidebar to continue.")
    st.stop()
if len(selected_features) > 3:
    st.sidebar.warning("Only the first 3 selected features will be used (plots go up to 3D).")
    selected_features = selected_features[:3]

n_feat = len(selected_features)
X_raw = df[selected_features].to_numpy(dtype=float)


# =====================================================================
# 3. FEATURE SCALING
# =====================================================================

st.sidebar.header("3. Feature scaling")
scaling = st.sidebar.radio("Scaling method", ["None", "Min-Max (0\u20131)", "Standardize (z-score)"], index=0)


def fit_scaler(X, method):
    if method == "None":
        return X, None
    if method == "Min-Max (0\u20131)":
        mn, mx = X.min(axis=0), X.max(axis=0)
        rng = np.where(mx - mn == 0, 1.0, mx - mn)
        return (X - mn) / rng, ("minmax", mn, rng)
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma, ("zscore", mu, sigma)


def apply_scaler(x, params):
    if params is None:
        return x.copy()
    kind = params[0]
    if kind == "minmax":
        _, mn, rng = params
        return (x - mn) / rng
    _, mu, sigma = params
    return (x - mu) / sigma


X_scaled, scale_params = fit_scaler(X_raw, scaling)


# =====================================================================
# 4. K SELECTION
# =====================================================================

st.sidebar.header("4. Number of neighbors")
max_k = max(1, min(30, len(df) - 1))
k = st.sidebar.slider("K", 1, max_k, min(5, max_k))


# =====================================================================
# 5. TEST POINT INPUT (single point)
# =====================================================================

st.sidebar.header("5. Test point")
st.sidebar.caption("Enter the coordinates of the single point you want to classify.")
test_vals = []
for feat in selected_features:
    col_min, col_max = float(df[feat].min()), float(df[feat].max())
    default_val = float(df[feat].mean())
    step = (col_max - col_min) / 100 if col_max > col_min else 1.0
    val = st.sidebar.number_input(
        f"{feat}",
        value=round(default_val, 4),
        step=round(step, 4) if step > 0 else 1.0,
        format="%.4f",
    )
    test_vals.append(val)

test_point_raw = np.array(test_vals, dtype=float)
test_point_scaled = apply_scaler(test_point_raw.reshape(1, -1), scale_params).ravel()


# =====================================================================
# 6. KNN COMPUTATION -- NumPy only
# =====================================================================

distances = np.sqrt(np.sum((X_scaled - test_point_scaled) ** 2, axis=1))
order = np.argsort(distances)
topk_idx = order[:k]

neighbor_labels = df[label_col].to_numpy()[topk_idx]
vote_labels, vote_counts = np.unique(neighbor_labels, return_counts=True)
predicted_label = vote_labels[np.argmax(vote_counts)]

col_pred, col_table = st.columns([1, 2])

with col_pred:
    st.subheader("Prediction")
    st.markdown(f"## \U0001F3AF **{predicted_label}**")
    st.write(f"Majority vote among the **{k} nearest neighbors** "
             f"(scaling: *{scaling}*, features: {', '.join(selected_features)}).")
    vote_df = pd.DataFrame({"class": vote_labels, "votes (out of k)": vote_counts})
    st.dataframe(vote_df.sort_values("votes (out of k)", ascending=False), hide_index=True)

with col_table:
    st.subheader(f"Top {k} nearest neighbors")
    result_table = df.iloc[topk_idx][selected_features + [label_col]].copy()
    result_table.insert(0, "rank", range(1, len(result_table) + 1))
    result_table["distance"] = np.round(distances[topk_idx], 4)
    st.dataframe(result_table.reset_index(drop=True), hide_index=True)


# =====================================================================
# 7. PLOT (1D / 2D / 3D) -- Plotly, K nearest neighbors highlighted
# =====================================================================

st.subheader("Visualization")
use_scaled_plot = st.checkbox(
    "Plot using SCALED coordinates (fairer visual sense of 'nearest' when scaling is on)",
    value=(scaling != "None"),
)

if use_scaled_plot:
    plot_coords, tp_coords, suffix = X_scaled, test_point_scaled, " (scaled)"
else:
    plot_coords, tp_coords, suffix = X_raw, test_point_raw, ""

is_neighbor = np.zeros(len(df), dtype=bool)
is_neighbor[topk_idx] = True
class_labels = df[label_col].to_numpy()
palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]
class_color = {cls: palette[i % len(palette)] for i, cls in enumerate(pd.unique(class_labels))}

fig = go.Figure()

# --- one trace per class (background points) ---
for cls in pd.unique(class_labels):
    mask = class_labels == cls
    if n_feat == 1:
        x, y = plot_coords[mask, 0], np.zeros(mask.sum())
        z = None
    elif n_feat == 2:
        x, y = plot_coords[mask, 0], plot_coords[mask, 1]
        z = None
    else:
        x, y, z = plot_coords[mask, 0], plot_coords[mask, 1], plot_coords[mask, 2]

    marker = dict(size=8, color=class_color[cls], opacity=0.75, line=dict(width=0.5, color="white"))
    if z is None:
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=f"class = {cls}", marker=marker))
    else:
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers", name=f"class = {cls}", marker=marker))

# --- ring highlight around the K nearest neighbors ---
if n_feat == 1:
    nx, ny = plot_coords[topk_idx, 0], np.zeros(len(topk_idx))
    nz = None
elif n_feat == 2:
    nx, ny = plot_coords[topk_idx, 0], plot_coords[topk_idx, 1]
    nz = None
else:
    nx, ny, nz = plot_coords[topk_idx, 0], plot_coords[topk_idx, 1], plot_coords[topk_idx, 2]

ring_marker = dict(size=16, color="rgba(0,0,0,0)", line=dict(width=3, color="black"))
if nz is None:
    fig.add_trace(go.Scatter(x=nx, y=ny, mode="markers", name=f"{k} nearest neighbors",
                              marker=ring_marker))
else:
    ring_marker3d = dict(size=8, color="rgba(0,0,0,0)", line=dict(width=4, color="black"))
    fig.add_trace(go.Scatter3d(x=nx, y=ny, z=nz, mode="markers", name=f"{k} nearest neighbors",
                                marker=ring_marker3d))

# --- the test / query point ---
if n_feat == 1:
    qx, qy = [tp_coords[0]], [0]
    star = go.Scatter(x=qx, y=qy, mode="markers", name="query (test point)",
                       marker=dict(symbol="star", size=20, color="red", line=dict(width=1, color="black")))
elif n_feat == 2:
    qx, qy = [tp_coords[0]], [tp_coords[1]]
    star = go.Scatter(x=qx, y=qy, mode="markers", name="query (test point)",
                       marker=dict(symbol="star", size=20, color="red", line=dict(width=1, color="black")))
else:
    qx, qy, qz = [tp_coords[0]], [tp_coords[1]], [tp_coords[2]]
    star = go.Scatter3d(x=qx, y=qy, z=qz, mode="markers", name="query (test point)",
                         marker=dict(symbol="diamond", size=8, color="red", line=dict(width=1, color="black")))
fig.add_trace(star)

axis_titles = [f + suffix for f in selected_features]
if n_feat == 1:
    fig.update_layout(xaxis_title=axis_titles[0], yaxis=dict(visible=False, range=[-1, 1]))
elif n_feat == 2:
    fig.update_layout(xaxis_title=axis_titles[0], yaxis_title=axis_titles[1])
else:
    fig.update_layout(scene=dict(xaxis_title=axis_titles[0], yaxis_title=axis_titles[1], zaxis_title=axis_titles[2]))

fig.update_layout(height=650, legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Black rings / outlined markers = the K nearest neighbors. Red star/diamond = your test point. "
    "Toggle 'scaled coordinates' above to see how the picture (and the neighbors) can change with scaling."
)
