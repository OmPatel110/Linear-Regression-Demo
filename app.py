# app.py
"""
Interactive Streamlit demo: Simple Linear Regression vs Polynomial Regression

Features implemented:
- generate synthetic data (linear, poly, sin, custom)
- control: n_points, noise, train/test split,
- fit analytic solution (sklearn) and optional gradient descent demo
- polynomial degree slider (1..12)
- regularization: None / Ridge / Lasso
- toggle residuals, overlay train/test predictions
- compare up to 3 degrees simultaneously
- metrics (R2, MAE, MSE, RMSE) for train and test
- prediction input to compare models
- bias-variance sweep plot across degrees
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
import io
import math
import base64
from typing import Tuple, Dict, Any, List
from streamlit_plotly_events import plotly_events
import kaleido

if "fitted_models" not in st.session_state:
    st.session_state["fitted_models"] = {}

if "clicked_trace_idx" not in st.session_state:
    st.session_state["clicked_trace_idx"] = None

if "highlight_time" not in st.session_state:
    st.session_state["highlight_time"] = None

# If you use other session keys anywhere, initialize them here too:
if "some_other_key" not in st.session_state:
    st.session_state["some_other_key"] = None

# Optional libs for GIF export
try:
    import imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

st.set_page_config(page_title="Linear vs Polynomial Regression", layout="wide")
st.title("Linear vs Polynomial Regression - Interactive Demo")
# -----------------------
# Utilities & caching
# -----------------------
@st.cache_data
def generate_data(func_name: str, n: int, noise: float, x_min: float, x_max: float, custom_expr: str = None):
    """
    Create a synthetic dataset where:
    - X is evenly spaced (linspace) across [x_min, x_max] so the dataset shape is deterministic
    - y_true is computed from chosen function
    - gaussian noise (with no fixed seed) is added according to `noise`
    - result is shifted so that all x and y are non-negative (>= 0)
    """
    # Deterministic X
    X = np.linspace(x_min, x_max, n)

    # Compute true underlying function
    if func_name == "linear":
        y_true = 1.5 * X + 0.5
    elif func_name == "poly3":
        y_true = 0.2 * X**3 - 0.5 * X**2 + 1.2 * X - 0.3
    elif func_name == "custom":
        allowed = {"np": np, "x": X}
        try:
            y_true = eval(custom_expr, {"__builtins__": {}}, allowed)
        except Exception:
            y_true = np.zeros_like(X)
    else:
        y_true = np.zeros_like(X)

    # Add noise (random each generation) - not controlled by user seed
    noise_term = np.random.normal(loc=0.0, scale=noise, size=X.shape)
    y = y_true + noise_term

    # Ensure non-negative x and y by shifting upward if negative values exist
    if X.min() < 0:
        X = X - X.min()  # shift so min(X) == 0
    if y.min() < 0:
        y = y - y.min()  # shift so min(y) == 0
        y_true = y_true - y_true.min()  # shift true values similarly for consistency

    df = pd.DataFrame({"x": X, "y": y, "y_true": y_true})
    return df.sort_values("x").reset_index(drop=True)


def metrics_dict(y_true, y_pred) -> Dict[str, float]:
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return {"R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}

def fit_polynomial_sklearn(X: np.ndarray, y: np.ndarray, degree: int, reg: str="none", alpha: float=1.0):
    """
    Returns fitted model and preprocessing objects.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X.reshape(-1,1))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    if reg == "none":
        model = LinearRegression()
    elif reg == "ridge":
        model = Ridge(alpha=alpha)
    elif reg == "lasso":
        model = Lasso(alpha=alpha, max_iter=5000)
    else:
        model = LinearRegression()
    model.fit(X_scaled, y)
    return {"model": model, "poly": poly, "scaler": scaler}

def fit_polynomial_gd(X_train: np.ndarray, y_train: np.ndarray, degree: int,
                      reg: str = "none", alpha: float = 1.0,
                      lr: float = 0.01, epochs: int = 500, batch_size: int = None) -> Dict[str, Any]:
    """
    Fit polynomial regression with simple gradient descent on scaled polynomial features.
    Returns a bundle with same structure as fit_polynomial_sklearn: {"model","poly","scaler","losses"}
    Note: no advanced optimizers; educational/demo only.
    """
    # Prepare features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_train.reshape(-1, 1))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    m, n_feats = X_scaled.shape

    # Initialize weights and bias
    w = np.zeros(n_feats, dtype=float)
    b = 0.0

    losses = []

    # Simple GD loop (supports mini-batch)
    for epoch in range(int(epochs)):
        if batch_size is None:
            # full-batch
            y_pred = X_scaled.dot(w) + b
            err = y_pred - y_train
            # apply L2 regularization in gradient if requested
            grad_w = (2.0 / m) * (X_scaled.T.dot(err))
            if reg == "ridge":
                grad_w += 2.0 * alpha * w / m
            # Lasso is non-differentiable; skipping prox step for demo (could approximate)
            grad_b = (2.0 / m) * err.sum()
            w = w - lr * grad_w
            b = b - lr * grad_b
            loss = (err ** 2).mean()
        else:
            # mini-batch GD
            indices = np.random.permutation(m)
            for start in range(0, m, batch_size):
                idx = indices[start:start + batch_size]
                Xb = X_scaled[idx]
                yb = y_train[idx]
                ypb = Xb.dot(w) + b
                errb = ypb - yb
                grad_w = (2.0 / max(1, len(idx))) * Xb.T.dot(errb)
                if reg == "ridge":
                    grad_w += 2.0 * alpha * w / m
                grad_b = (2.0 / max(1, len(idx))) * errb.sum()
                w = w - lr * grad_w
                b = b - lr * grad_b
            y_pred = X_scaled.dot(w) + b
            loss = ((y_pred - y_train) ** 2).mean()

        losses.append(float(loss))

    # Wrap as sklearn-like model
    class _GDWrapper:
        def __init__(self, w, b):
            self.coef_ = np.array(w)
            self.intercept_ = float(b)
        def predict(self, Xs):
            # Xs expected to be scaled polynomial features
            return Xs.dot(self.coef_) + self.intercept_

    model = _GDWrapper(w, b)
    return {"model": model, "poly": poly, "scaler": scaler, "losses": losses}

def predict_with_model(model_bundle, X: np.ndarray):
    poly = model_bundle["poly"]
    scaler = model_bundle["scaler"]
    model = model_bundle["model"]
    X_poly = poly.transform(X.reshape(-1,1))
    X_scaled = scaler.transform(X_poly)
    y_pred = model.predict(X_scaled)
    return y_pred

# -----------------------
# Sidebar: Controls
# -----------------------
st.sidebar.header("Data generation")
n_points = st.sidebar.slider("Number of points", min_value=10, max_value=500, value=80, step=5)
x_min, x_max = st.sidebar.number_input("X range min", value=-3.0), st.sidebar.number_input("X range max", value=3.0)
func = st.sidebar.selectbox("True function", ["linear", "polynomial", "custom"], index=2, help="Choose ground-truth function for demonstration.")
custom_expr = ""
if func == "custom":
    custom_expr = st.sidebar.text_input("Custom expression (use 'np' and 'x')", value="0.5*x**3 - 2*x + 1")
noise = st.sidebar.slider("Noise (std dev)", 0.0, 3.0, 0.5, step=0.1)
# seed = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
train_size = st.sidebar.slider("Train size (%)", 10, 90, 70)

st.sidebar.markdown("---")
st.sidebar.header("Model & Visualization")
degree = st.sidebar.slider("Polynomial degree (1 = linear)", 1, 6, 3)
compare_degrees = st.sidebar.multiselect("Compare degrees (max 3)", options=list(range(1,7)), default=[1, degree][:3])
if len(compare_degrees) > 3:
    compare_degrees = compare_degrees[:3]
reg = st.sidebar.selectbox("Regularization", ["none", "ridge", "lasso"])
alpha = st.sidebar.slider("Regularization alpha", 0.0, 10.0, 1.0)
fit_method = st.sidebar.radio("Fit method", ["Analytic (sklearn)", "Gradient Descent (demo)"])
show_residuals = st.sidebar.checkbox("Show residuals", value=True)
overlay_train_test = st.sidebar.checkbox("Overlay train/test predictions", value=True)
st.sidebar.markdown("---")
st.sidebar.header("Gradient Descent (if selected)")
gd_lr = st.sidebar.number_input("GD learning rate", value=0.01, format="%.5f")
gd_epochs = st.sidebar.slider("GD epochs", 10, 2000, 200)
gd_batch = st.sidebar.selectbox("GD batch size", [None, 1, 8, 16, 32], index=3)

# --------------------------------
# View Controls
# --------------------------------
st.sidebar.markdown("---")
st.sidebar.header("View Controls")

# Toggle for full curve or focused view
show_full_curve = st.sidebar.checkbox(
    "Show full curve (may autoscale)",
    value=False,
    help="Turn ON to view the full polynomial, including extreme values outside the data range."
)

# Zoom slider (used only when show_full_curve is OFF)
zoom_factor = st.sidebar.slider(
    "Y-axis zoom × data range",
    min_value=1.0, max_value=10.0, value=2.0, step=0.5,
    help="Adjust vertical zoom around data region. Smaller = tighter focus."
)

st.sidebar.markdown("---")
st.sidebar.header("Export & Sharing")
download_csv = st.sidebar.button("Download dataset CSV")
# gif_record = st.sidebar.button("Record degree-sweep GIF (1..degree)")

# -----------------------
# Generate data & split
# -----------------------
df = generate_data(func, n_points, noise, x_min, x_max, custom_expr=custom_expr)
X = df["x"].values
y = df["y"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size/100.0, random_state=42)

# Display quick dataset preview
with st.expander("Dataset preview & basic stats", expanded=False):
    st.write(df.head(10))
    st.write(df.describe())

# -----------------------
# Fit models for selected degree(s)
# -----------------------
# -----------------------
# Fit models for selected degree(s) - supports Analytic (sklearn) and Gradient Descent
# Always include degree=1 (linear) and the main selected degree
# -----------------------
required_degrees = sorted(set(list(compare_degrees) + [1, degree]))

# We'll build/overwrite fitted_models dict in-place and persist to session_state
fitted_models = st.session_state.get("fitted_models", {})

# Ensure we fit every required degree according to the selected fit_method
for d in required_degrees:
    # If model already exists in session_state and hyperparams haven't changed,
    # you might skip refitting. For simplicity we refit each run to reflect current reg/alpha.
    if fit_method == "Analytic (sklearn)":
        # analytic sklearn fit
        fitted_models[d] = fit_polynomial_sklearn(X_train, y_train, d, reg, alpha)
    else:
        # Gradient Descent selected - use GD fitter
        # Use the global GD controls from sidebar (gd_lr, gd_epochs, gd_batch)
        fitted_models[d] = fit_polynomial_gd(
            X_train, y_train, d,
            reg=reg, alpha=alpha,
            lr=float(gd_lr), epochs=int(gd_epochs),
            batch_size=(None if gd_batch is None else int(gd_batch))
        )

# Persist into session_state so Streamlit retains this across reruns
st.session_state["fitted_models"] = fitted_models


# Optionally fit gradient-descent for the displayed main degree
gd_bundle = None
if fit_method == "Gradient Descent (demo)":
    # Simple batch gradient descent on scaled polynomial features (educational only)
    # Implementation: create poly features, scale, then run gradient descent
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_train.reshape(-1,1))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    # initialize weights
    w = np.zeros(X_scaled.shape[1])
    b = 0.0
    losses = []
    m = X_scaled.shape[0]
    lr = float(gd_lr)
    for epoch in range(gd_epochs):
        if gd_batch is None:
            # full batch
            y_pred = X_scaled.dot(w) + b
            err = y_pred - y_train
            grad_w = (2.0/m) * X_scaled.T.dot(err)
            grad_b = (2.0/m) * err.sum()
            w = w - lr * grad_w
            b = b - lr * grad_b
            loss = (err**2).mean()
        else:
            # mini-batch GD
            indices = np.random.permutation(m)
            for start in range(0,m,gd_batch):
                idx = indices[start:start+gd_batch]
                Xb = X_scaled[idx]
                yb = y_train[idx]
                ypb = Xb.dot(w) + b
                errb = ypb - yb
                grad_w = (2.0/len(idx)) * Xb.T.dot(errb)
                grad_b = (2.0/len(idx)) * errb.sum()
                w = w - lr * grad_w
                b = b - lr * grad_b
            y_pred = X_scaled.dot(w) + b
            loss = ((y_pred - y_train)**2).mean()
        losses.append(loss)
    # Create a fake sklearn-like wrapper for predictions
    class GDModel:
        def __init__(self, w, b):
            self.coef_ = w
            self.intercept_ = b
        def predict(self, Xs):
            return Xs.dot(self.coef_) + self.intercept_
    gd_bundle = {"model": GDModel(w, b), "poly": poly, "scaler": scaler, "losses": losses}

    fitted_models[degree] = gd_bundle          # replace sklearn model for this degree
    st.session_state["fitted_models"] = fitted_models

    # Optional info marker
    # st.info(f"Gradient Descent model is active for degree {degree}. Metrics & predictions now use GD results.")

# -----------------------
# Prepare plotly figure
# -----------------------
# Create a dense X for smooth curves
# plot x over the actual data range (prevents huge extrapolation far outside data)
x_plot = np.linspace(X.min(), X.max(), 400)

# If we have fitted models in session_state, use them (keeps previously fitted degrees)
if "fitted_models" in st.session_state:
    # Update local fitted_models to whatever is stored (this avoids accidental overwrites)
    stored = st.session_state.get("fitted_models", {})
    # Merge stored models with local required ones, preferring recent local fits
    # (so we keep previously fitted degrees that might not be in compare_degrees)
    fitted_models = {**stored, **locals().get("fitted_models", {})}
    st.session_state["fitted_models"] = fitted_models  # re-store merged
else:
    # nothing in session_state yet (first run) - we expect fitted_models was created above
    st.session_state["fitted_models"] = locals().get("fitted_models", {})
    fitted_models = st.session_state["fitted_models"]

fig = go.Figure()
# Scatter: train / test points
fig.add_trace(go.Scatter(x=X_train, y=y_train, mode="markers", name="Train", marker=dict(symbol="circle", size=7)))
fig.add_trace(go.Scatter(x=X_test, y=y_test, mode="markers", name="Test", marker=dict(symbol="x", size=8)))

# Add model curves for each compared degree
# color_seq = px.colors.qualitative.Dark24
# for i, d in enumerate(compare_degrees):
#     bundle = fitted_models[d]
#     y_plot = predict_with_model(bundle, x_plot)
#     name = f"Degree {d}"
#     style = dict(width=3)
#     fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode="lines", name=name, line=style, marker=dict(color=color_seq[i % len(color_seq)])))
# --- robust plotting of model curves (avoid blow-ups for high-degree polynomials) ---

# color_seq = px.colors.qualitative.Dark24

# # compute robust clipping bounds from observed y (train+test)
# y_all = np.concatenate([y_train, y_test])
# y_median = np.median(y_all)
# y_std = np.std(y_all)
# data_range = max(y_all.max() - y_all.min(), 1e-6)
# clip_margin = max(3 * y_std, 0.1 * data_range)
# y_min_clip = y_median - clip_margin
# y_max_clip = y_median + clip_margin

# for i, d in enumerate(compare_degrees):
#     bundle = fitted_models.get(d)
#     if bundle is None:
#         continue

#     # Get model predictions
#     y_plot = predict_with_model(bundle, x_plot)

#     # Optional: warn user if curve goes out of bounds
#     if np.any(np.abs(y_plot) > (y_median + 10 * clip_margin)):
#         st.warning(f"Degree {d} produces extreme predictions (extrapolation). Plot clipped for readability.")

#     # Clip only for visualization (model predictions remain unchanged)
#     y_plot_display = np.clip(y_plot, y_min_clip, y_max_clip)

#     fig.add_trace(go.Scatter(
#         x=x_plot,
#         y=y_plot_display,
#         mode="lines",
#         name=f"Degree {d}",
#         line=dict(width=3),
#         marker=dict(color=color_seq[i % len(color_seq)]),
#         hoverinfo="x+y"
#     ))
# --- Plot model curves with focus/zoom controls ---
color_seq = px.colors.qualitative.Dark24

# Compute robust clipping bounds from observed data
y_all = np.concatenate([y_train, y_test])
y_median = np.median(y_all)
y_std = np.std(y_all)
data_range = max(y_all.max() - y_all.min(), 1e-6)
clip_margin = zoom_factor * max(3 * y_std, 0.1 * data_range)
y_min_clip = y_median - clip_margin
y_max_clip = y_median + clip_margin

for i, d in enumerate(compare_degrees):
    bundle = fitted_models.get(d)
    if bundle is None:
        continue

    y_plot = predict_with_model(bundle, x_plot)

    # Warn if extreme extrapolation
    if np.any(np.abs(y_plot) > y_max_clip * 5):
        st.info(f"Degree {d} shows large extrapolation outside data range.")

    # Choose what to display based on toggle
    y_plot_display = y_plot if show_full_curve else np.clip(y_plot, y_min_clip, y_max_clip)

    fig.add_trace(go.Scatter(
        x=x_plot,
        y=y_plot_display,
        mode="lines",
        name=f"Degree {d}",
        line=dict(width=3),
        marker=dict(color=color_seq[i % len(color_seq)]),
        hoverinfo="x+y"
    ))

# Adjust y-axis range depending on toggle
if not show_full_curve:
    fig.update_yaxes(range=[y_min_clip, y_max_clip])
else:
    fig.update_yaxes(autorange=True)

# If gradient descent is active for main degree, show its curve too (distinct style)
# if fit_method == "Gradient Descent (demo)" and gd_bundle is not None:
#     y_gd = predict_with_model(gd_bundle, x_plot) if hasattr(gd_bundle["model"], "predict") else gd_bundle["model"].predict(gd_bundle["scaler"].transform(gd_bundle["poly"].transform(x_plot.reshape(-1,1))))
#     fig.add_trace(go.Scatter(x=x_plot, y=y_gd, mode="lines", name=f"GD degree {degree}", line=dict(dash="dash", width=2, color="black")))

# Training/test predictions overlay for all compared degrees
if overlay_train_test:
    for i, d in enumerate(compare_degrees):
        bundle = fitted_models.get(d)
        if bundle is None:
            continue

        y_train_pred = predict_with_model(bundle, X_train)
        y_test_pred = predict_with_model(bundle, X_test)

        fig.add_trace(go.Scatter(
            x=X_train, y=y_train_pred,
            mode="markers",
            name=f"Train preds (degree {d})",
            marker=dict(symbol="triangle-up", size=7, opacity=0.8,
                        color=color_seq[i % len(color_seq)])
        ))
        fig.add_trace(go.Scatter(
            x=X_test, y=y_test_pred,
            mode="markers",
            name=f"Test preds (degree {d})",
            marker=dict(symbol="diamond", size=7, opacity=0.8,
                        color=color_seq[i % len(color_seq)])
        ))

# Residuals lines
if show_residuals:
    main_bundle = fitted_models.get(degree)
    if main_bundle is not None:
        y_train_pred = predict_with_model(main_bundle, X_train)
        y_test_pred = predict_with_model(main_bundle, X_test)
        # draw line segments from each point to predicted point
        for xi, yi, ypi in zip(X_train, y_train, y_train_pred):
            fig.add_trace(go.Scatter(x=[xi, xi], y=[yi, ypi], mode="lines", line=dict(color="rgba(0,0,0,0.15)"), showlegend=False))
        for xi, yi, ypi in zip(X_test, y_test, y_test_pred):
            fig.add_trace(go.Scatter(x=[xi, xi], y=[yi, ypi], mode="lines", line=dict(color="rgba(255,0,0,0.12)"), showlegend=False))

fig.update_layout(title=f"Data and model curves (degree(s): {compare_degrees})",
                  xaxis_title="x", yaxis_title="y", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))


# -----------------------
# Side-by-side: metrics, prediction, and plots
# -----------------------
# --- Display chart first ---
st.plotly_chart(fig, use_container_width=True)

# --- Then metrics below ---
st.header("Model metrics (train vs test)")
metrics_table = []
for d in compare_degrees:
    bundle = fitted_models[d]
    y_train_pred = predict_with_model(bundle, X_train)
    y_test_pred = predict_with_model(bundle, X_test)
    m_train = metrics_dict(y_train, y_train_pred)
    m_test = metrics_dict(y_test, y_test_pred)
    metrics_table.append({"degree": d, "split": "train", **m_train})
    metrics_table.append({"degree": d, "split": "test", **m_test})
metrics_df = pd.DataFrame(metrics_table)

# format only numeric columns (fix previous error)
num_cols = ["R2", "MAE", "MSE", "RMSE"]
st.dataframe(metrics_df.style.format({c: "{:.4f}" for c in num_cols}), use_container_width=True)
# Custom CSS to shrink metric font size
st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Quick adaptive metric cards for each selected degree (1..3 shown side-by-side)
st.subheader("Quick metrics - selected degrees")
# Ensure there is at least one degree displayed
display_degrees = compare_degrees if len(compare_degrees) > 0 else [degree]

# Limit to max 3 columns (you already restrict compare_degrees to 3 elsewhere)
num_cols = min(len(display_degrees), 3)
cols = st.columns(num_cols)

for i, d in enumerate(display_degrees[:num_cols]):
    with cols[i]:
        st.markdown(f"### Degree {d}")
        bundle = fitted_models.get(d)
        if bundle is None:
            st.warning(f"Model for degree {d} not fitted.")
            continue

        # predictions and metrics
        y_tr_pred = predict_with_model(bundle, X_train)
        y_te_pred = predict_with_model(bundle, X_test)
        m_tr = metrics_dict(y_train, y_tr_pred)
        m_te = metrics_dict(y_test, y_te_pred)

        # Display as stacked metric pairs
        st.metric(label="Train R²", value=f"{m_tr['R2']:.4f}")
        st.metric(label="Train RMSE", value=f"{m_tr['RMSE']:.4f}")
        st.markdown("")  # small spacer
        st.metric(label="Test R²", value=f"{m_te['R2']:.4f}")
        st.metric(label="Test RMSE", value=f"{m_te['RMSE']:.4f}")

st.markdown("---")
st.subheader("Predict & compare")

predict_x = st.number_input(
    "Enter x value to predict",
    value=float((X.min() + X.max()) / 2.0),
    help="Enter an x-value to see predictions for selected polynomial degrees."
)

if st.button("Predict now"):
    if not compare_degrees:
        st.info("No polynomial degrees selected in 'Compare degrees'. Please select at least one degree.")
    else:
        st.markdown("### Predictions for selected degrees")

        predictions = {}  # store results for optional comparison

        # Loop through only selected degrees
        for d in compare_degrees:
            bundle = fitted_models.get(d)
            if bundle is None:
                bundle = fit_polynomial_sklearn(X_train, y_train, d, reg, alpha)
                fitted_models[d] = bundle
                st.session_state["fitted_models"] = fitted_models

            y_pred = predict_with_model(bundle, np.array([predict_x]))[0]
            predictions[d] = y_pred

            # ✅ show prediction in green box (same style as before)
            st.success(f"Polynomial (deg {d}) prediction: {y_pred:.4f}")

        # Optional: show pairwise differences if more than one degree is selected
        if len(predictions) > 1:
            st.markdown("---")
            st.markdown("#### Pairwise differences between selected degrees")
            degrees = list(predictions.keys())
            base_deg = degrees[0]
            base_val = predictions[base_deg]
            for d in degrees[1:]:
                diff = predictions[d] - base_val
                st.info(f"Difference (deg {d} − deg {base_deg}): {diff:.4f}")

st.markdown("---")
st.subheader("Bias–Variance sweep")

# Choose max degree to evaluate (keep small for speed)
max_d = 12
degrees_range = list(range(1, max_d + 1))

# Compute train/test MSE for degrees 1..max_d
train_errors = []
test_errors = []
for d in degrees_range:
    b = fit_polynomial_sklearn(X_train, y_train, d, reg, alpha)
    trp = predict_with_model(b, X_train)
    tep = predict_with_model(b, X_test)
    train_errors.append(mean_squared_error(y_train, trp))
    test_errors.append(mean_squared_error(y_test, tep))

# Find optimal degree (min test error)
opt_idx = int(np.argmin(test_errors))
opt_degree = degrees_range[opt_idx]
opt_test_mse = test_errors[opt_idx]

# Build the figure
sweep_fig = go.Figure()

sweep_fig.add_trace(go.Scatter(
    x=degrees_range, y=train_errors,
    mode="lines+markers", name="Train MSE",
    line=dict(color="rgba(31,119,180,0.9)", width=3),
    marker=dict(size=8)
))
sweep_fig.add_trace(go.Scatter(
    x=degrees_range, y=test_errors,
    mode="lines+markers", name="Test MSE",
    line=dict(color="rgba(174,199,232,0.95)", width=3),
    marker=dict(size=8)
))

# Shade underfitting (left) and overfitting (right) regions
sweep_fig.add_vrect(
    x0=degrees_range[0] - 0.5, x1=opt_degree - 0.5,
    fillcolor="rgba(255,200,200,0.12)", line_width=0,
    annotation_text="Underfitting\n(high bias)", annotation_position="top left"
)
sweep_fig.add_vrect(
    x0=opt_degree + 0.5, x1=degrees_range[-1] + 0.5,
    fillcolor="rgba(200,220,255,0.12)", line_width=0,
    annotation_text="Overfitting\n(high variance)", annotation_position="top right"
)

# Highlight the optimal degree (sweet spot)
sweep_fig.add_vline(x=opt_degree, line=dict(color="green", dash="dash", width=2),
                    annotation_text=f"Sweet spot (deg {opt_degree})", annotation_position="top")

# Mark current slider degree (updates live because Streamlit reruns on slider change)
current_deg = int(degree)  # uses your existing slider variable
sweep_fig.add_vline(x=current_deg, line=dict(color="black", dash="dot", width=2),
                    annotation_text=f"Current: deg {current_deg}", annotation_position="bottom right")

# Axis labels and layout
sweep_fig.update_layout(
    title="Train vs Test MSE by degree (bias–variance sweep)",
    xaxis_title="Degree",
    yaxis_title="MSE",
    xaxis=dict(tickmode="array", tickvals=degrees_range),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=60, b=40, l=60, r=40),
    hovermode="x unified"
)

# Add an annotation that explains what's happening at the selected degree
# Compute train/test at selected degree for inline info
selected_idx = degrees_range.index(current_deg) if current_deg in degrees_range else None
if selected_idx is not None:
    sel_train = train_errors[selected_idx]
    sel_test = test_errors[selected_idx]
    sweep_fig.add_annotation(
        x=current_deg, y=max(sel_train, sel_test),
        text=f"Train MSE: {sel_train:.3f}<br>Test MSE: {sel_test:.3f}",
        showarrow=True, arrowhead=2, ax=40, ay=-40, bgcolor="white"
    )

# Display the plot
st.plotly_chart(sweep_fig, use_container_width=True)

# Short textual explanation below the chart
st.markdown("""
**Interpretation:**  
- Left shaded area = *underfitting* (high bias).  
- Right shaded area = *overfitting* (high variance).  
- The green dashed line marks the *sweet spot* (degree with minimum Test MSE).  
Move the **Polynomial degree** slider above to see how the current choice compares to the sweet spot.
""")


# -----------------------
# Export / Download handlers
# -----------------------
# CSV download
if download_csv:
    csv = df.to_csv(index=False).encode()
    st.download_button(label="Download dataset (.csv)", data=csv, file_name="regression_demo_dataset.csv", mime="text/csv")

# # -----------------------
# Help & explanation text
# -----------------------
st.markdown("---")
st.header("Why this demo?")
st.markdown("""
- **Degree** controls model complexity. Degree 1 = linear (low variance, may underfit).  
- **Higher degrees** increase flexibility - they can reduce training error but often increase test error when noise is present (**overfitting**).  
- **Regularization** (Ridge/Lasso) penalizes large coefficients and helps reduce overfitting.  
- Use the Bias-Variance sweep to observe the typical U-shaped test error curve as degree increases.
""")

st.markdown("**Try this:** set noise high (e.g., 1.0), n_points small (e.g., 20), then increase degree → watch how training error drops but test error increases (overfitting). Toggle Ridge to see improvements.")
# Simple personal footer 
st.markdown("---")
linkedin_url = "https://www.linkedin.com/in/om-patel-tech/"  
footer_html = f'''
<div style="text-align:center; font-size:14px; padding:8px 0; color:#444;">
  Made by <strong>Om Patel</strong> - 
  <a href="{linkedin_url}" target="_blank" style="color:#0A66C2; text-decoration:none; font-weight:600;">
    LinkedIn
  </a>
</div>
'''
st.markdown(footer_html, unsafe_allow_html=True)
