from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Assignment 4: Advertising Budget & Sales ML Data App",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "advertising.csv"
CHANNELS = ["TV", "Radio", "Newspaper"]


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and validate the Kaggle Advertising Dataset snapshot."""
    data = pd.read_csv(DATA_PATH)
    required = CHANNELS + ["Sales"]
    missing_columns = sorted(set(required) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if data[required].isna().any().any():
        raise ValueError("The dataset contains missing values in required columns.")
    return data


df = load_data()
df["Total Spend"] = df[CHANNELS].sum(axis=1)

correlations = df[CHANNELS + ["Sales"]].corr(numeric_only=True)["Sales"].drop("Sales")
strongest_channel = correlations.abs().idxmax()
strongest_correlation = correlations[strongest_channel]

# ---------------------------------------------------------------------------
# Sidebar: Model Configuration & Interactive Budget Simulator
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Model & Data Split Settings")

selected_features = st.sidebar.multiselect(
    "Select predictor channels (X)",
    options=CHANNELS,
    default=CHANNELS,
    help="Choose which advertising channels to include when training the regression models.",
)
if not selected_features:
    st.sidebar.warning("Please select at least one advertising channel.")
    st.stop()

test_size = st.sidebar.slider(
    "Test set proportion",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05,
)
random_state = st.sidebar.number_input(
    "Random seed (random_state)",
    min_value=0,
    max_value=9999,
    value=42,
    step=1,
)

st.sidebar.divider()
st.sidebar.subheader("📉 SGD Regressor Hyperparameters")
sgd_lr_schedule = st.sidebar.selectbox(
    "Learning rate schedule",
    options=["invscaling", "constant", "adaptive"],
    index=0,
)
sgd_eta0 = st.sidebar.select_slider(
    "Initial learning rate (eta0)",
    options=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
    value=0.01,
)
sgd_max_iter = st.sidebar.slider(
    "Max iterations (max_iter)",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
)
sgd_penalty = st.sidebar.selectbox(
    "Regularization penalty",
    options=["l2", "l1", "elasticnet", None],
    index=0,
)
sgd_alpha = st.sidebar.select_slider(
    "Regularization strength (alpha)",
    options=[0.0001, 0.001, 0.01, 0.1, 1.0],
    value=0.0001,
)

st.sidebar.divider()
st.sidebar.subheader("🌲 Random Forest Hyperparameters")
rf_n_estimators = st.sidebar.slider(
    "Number of trees (n_estimators)",
    min_value=10,
    max_value=500,
    value=100,
    step=10,
)
rf_max_depth = st.sidebar.slider(
    "Maximum tree depth (max_depth)",
    min_value=2,
    max_value=20,
    value=8,
    step=1,
)
rf_min_samples_split = st.sidebar.slider(
    "Min samples to split (min_samples_split)",
    min_value=2,
    max_value=20,
    value=2,
    step=1,
)

# ---------------------------------------------------------------------------
# Train Models: Linear Regression, SGD Regressor, Random Forest Regressor
# ---------------------------------------------------------------------------
X = df[selected_features]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state)
)

# 1. Ordinary Least Squares Linear Regression (with StandardScaler for fair coefficient comparison)
lr_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]
)
lr_raw = LinearRegression()
lr_pipeline.fit(X_train, y_train)
lr_raw.fit(X_train, y_train)

# 2. Stochastic Gradient Descent (SGD) Regressor (scaled features required for gradient convergence)
sgd_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            SGDRegressor(
                learning_rate=sgd_lr_schedule,
                eta0=sgd_eta0,
                max_iter=sgd_max_iter,
                penalty=sgd_penalty,
                alpha=sgd_alpha,
                random_state=int(random_state),
            ),
        ),
    ]
)
sgd_pipeline.fit(X_train, y_train)

# 3. Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=rf_n_estimators,
    max_depth=rf_max_depth,
    min_samples_split=rf_min_samples_split,
    random_state=int(random_state),
)
rf_model.fit(X_train, y_train)

models = {
    "Linear Regression (OLS)": lr_pipeline,
    "SGD Regressor": sgd_pipeline,
    "Random Forest": rf_model,
}

metrics_rows = []
predictions_test = {}
for name, model in models.items():
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    predictions_test[name] = y_test_pred
    metrics_rows.append(
        {
            "Model": name,
            "Train R²": r2_score(y_train, y_train_pred),
            "Test R²": r2_score(y_test, y_test_pred),
            "Test RMSE": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "Test MAE": mean_absolute_error(y_test, y_test_pred),
        }
    )

metrics_df = pd.DataFrame(metrics_rows)
best_model_row = metrics_df.sort_values("Test R²", ascending=False).iloc[0]

# ---------------------------------------------------------------------------
# Main Page Header & KPI Cards
# ---------------------------------------------------------------------------
st.title("Advertising Budget & Sales Dashboard | Assignment 4")
st.caption(
    "Exploratory channel analysis and predictive modeling comparing Linear Regression (OLS), "
    "Stochastic Gradient Descent (SGD), and Random Forest on the Kaggle Advertising Dataset."
)

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("Average sales", f"{df['Sales'].mean():.2f}")
kpi_2.metric("Average total ad spend", f"{df['Total Spend'].mean():.2f}")
kpi_3.metric(
    "Strongest historical channel",
    strongest_channel,
    f"r = {strongest_correlation:.2f}",
)
kpi_4.metric(
    "Best model (Test R²)",
    str(best_model_row["Model"]),
    f"R² = {best_model_row['Test R²']:.3f}",
)

st.divider()

tab_eda, tab_models, tab_importance, tab_simulator = st.tabs(
    [
        "📊 1. Exploratory Dashboard (Team 12)",
        "🤖 2. Model Comparison (LR vs. SGD vs. RF)",
        "🔍 3. Coefficients & Feature Importance",
        "🎯 4. Interactive Budget & Sales Simulator",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1: Original Team 12 Exploratory Dashboard
# ---------------------------------------------------------------------------
with tab_eda:
    st.info(
        "Decision focus: use the historical relationships in this snapshot to decide which channel deserves a closer budget review. "
        "Correlation is descriptive and does not prove that advertising caused the sales result."
    )
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Channel comparison")
        correlation_data = (
            correlations.rename("Correlation")
            .rename_axis("Channel")
            .reset_index()
            .sort_values("Correlation", ascending=True)
        )
        correlation_chart = px.bar(
            correlation_data,
            x="Correlation",
            y="Channel",
            orientation="h",
            text=correlation_data["Correlation"].map(lambda value: f"{value:.2f}"),
            color="Correlation",
            color_continuous_scale="Blues",
            range_x=[0, 1],
        )
        correlation_chart.update_layout(
            height=360,
            coloraxis_showscale=False,
            xaxis_title="Correlation with sales",
            yaxis_title="",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        correlation_chart.update_traces(textposition="outside")
        st.plotly_chart(correlation_chart, use_container_width=True)

    with right:
        st.subheader("Selected channel and sales")
        selected_channel = st.selectbox(
            "Choose an advertising channel",
            CHANNELS,
            index=0,
            help="The marketing manager can inspect one channel at a time before reviewing its budget.",
        )

        x = df[selected_channel].to_numpy()
        y_vals = df["Sales"].to_numpy()
        slope, intercept = np.polyfit(x, y_vals, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = slope * line_x + intercept

        scatter_chart = go.Figure()
        scatter_chart.add_trace(
            go.Scatter(
                x=x,
                y=y_vals,
                mode="markers",
                name="Market observations",
                marker=dict(color="#2563eb", size=8, opacity=0.72),
            )
        )
        scatter_chart.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name=f"Linear trend (slope={slope:.3f})",
                line=dict(color="#dc2626", width=3),
            )
        )
        scatter_chart.update_layout(
            height=360,
            xaxis_title=f"{selected_channel} advertising spend ($K)",
            yaxis_title="Sales",
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(scatter_chart, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Model Comparison (Linear Regression, SGD, Random Forest)
# ---------------------------------------------------------------------------
with tab_models:
    st.subheader("Model Performance Summary (Train vs. Test Split)")
    st.dataframe(
        metrics_df.style.format(
            {
                "Train R²": "{:.4f}",
                "Test R²": "{:.4f}",
                "Test RMSE": "{:.4f}",
                "Test MAE": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    col_pred, col_res = st.columns(2)

    with col_pred:
        st.subheader("Actual vs. Predicted Sales (Test Set)")
        fig_pred = go.Figure()
        min_y, max_y = float(y_test.min()), float(y_test.max())
        fig_pred.add_trace(
            go.Scatter(
                x=[min_y, max_y],
                y=[min_y, max_y],
                mode="lines",
                name="Ideal 1:1 Fit",
                line=dict(color="#64748b", dash="dash", width=2),
            )
        )
        palette = {
            "Linear Regression (OLS)": "#2563eb",
            "SGD Regressor": "#f59e0b",
            "Random Forest": "#10b981",
        }
        for name, y_pred in predictions_test.items():
            fig_pred.add_trace(
                go.Scatter(
                    x=y_test,
                    y=y_pred,
                    mode="markers",
                    name=name,
                    marker=dict(color=palette[name], size=8, opacity=0.8),
                )
            )
        fig_pred.update_layout(
            height=380,
            xaxis_title="Actual Sales (Test Set)",
            yaxis_title="Predicted Sales",
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    with col_res:
        st.subheader("Test Set RMSE & MAE Comparison")
        melted_err = metrics_df.melt(
            id_vars="Model",
            value_vars=["Test RMSE", "Test MAE"],
            var_name="Metric",
            value_name="Error",
        )
        fig_err = px.bar(
            melted_err,
            x="Model",
            y="Error",
            color="Metric",
            barmode="group",
            text=melted_err["Error"].map(lambda v: f"{v:.3f}"),
            color_discrete_sequence=["#ef4444", "#3b82f6"],
        )
        fig_err.update_layout(
            height=380,
            xaxis_title="",
            yaxis_title="Error (Sales Units)",
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_err.update_traces(textposition="outside")
        st.plotly_chart(fig_err, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Coefficients & Feature Importance
# ---------------------------------------------------------------------------
with tab_importance:
    col_coef, col_rf = st.columns(2)

    with col_coef:
        st.subheader("Linear Regression vs. SGD (Standardized Weights)")
        lr_std_coefs = lr_pipeline.named_steps["model"].coef_
        sgd_std_coefs = sgd_pipeline.named_steps["model"].coef_
        coef_df = pd.DataFrame(
            {
                "Channel": selected_features,
                "Linear Regression (Std Coef)": lr_std_coefs,
                "SGD Regressor (Std Coef)": sgd_std_coefs,
                "OLS Raw Slope (ΔSales per $1K)": lr_raw.coef_,
            }
        )
        coef_melted = coef_df.melt(
            id_vars="Channel",
            value_vars=["Linear Regression (Std Coef)", "SGD Regressor (Std Coef)"],
            var_name="Model",
            value_name="Standardized Coefficient",
        )
        fig_coef = px.bar(
            coef_melted,
            x="Channel",
            y="Standardized Coefficient",
            color="Model",
            barmode="group",
            text=coef_melted["Standardized Coefficient"].map(lambda v: f"{v:.3f}"),
            color_discrete_sequence=["#2563eb", "#f59e0b"],
        )
        fig_coef.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_coef.update_traces(textposition="outside")
        st.plotly_chart(fig_coef, use_container_width=True)
        st.caption(
            f"Unscaled OLS Equation: Sales = {lr_raw.intercept_:.3f} + "
            + " + ".join(
                f"({coef:.4f} × {feat})"
                for feat, coef in zip(selected_features, lr_raw.coef_)
            )
        )

    with col_rf:
        st.subheader("Random Forest Feature Importance")
        rf_imp_df = pd.DataFrame(
            {
                "Channel": selected_features,
                "Importance": rf_model.feature_importances_,
            }
        ).sort_values("Importance", ascending=True)
        fig_rf = px.bar(
            rf_imp_df,
            x="Importance",
            y="Channel",
            orientation="h",
            text=rf_imp_df["Importance"].map(lambda v: f"{v:.1%}"),
            color="Importance",
            color_continuous_scale="Emrld",
            range_x=[0, 1],
        )
        fig_rf.update_layout(
            height=360,
            coloraxis_showscale=False,
            xaxis_title="Relative Importance (Gini / MSE Reduction)",
            yaxis_title="",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        fig_rf.update_traces(textposition="outside")
        st.plotly_chart(fig_rf, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Interactive Budget & Sales Simulator
# ---------------------------------------------------------------------------
with tab_simulator:
    st.subheader("Simulate Proposed Channel Budgets")
    sim_cols = st.columns(len(selected_features))
    input_budget = {}
    for idx, channel in enumerate(selected_features):
        with sim_cols[idx]:
            input_budget[channel] = st.slider(
                f"{channel} Budget ($K)",
                min_value=float(df[channel].min()),
                max_value=float(df[channel].max()),
                value=float(df[channel].median()),
                step=1.0,
            )

    input_df = pd.DataFrame([input_budget])
    pred_lr = float(lr_pipeline.predict(input_df)[0])
    pred_sgd = float(sgd_pipeline.predict(input_df)[0])
    pred_rf = float(rf_model.predict(input_df)[0])

    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.metric("Linear Regression Forecast", f"{pred_lr:.2f} units")
    p_col2.metric(
        "SGD Regressor Forecast",
        f"{pred_sgd:.2f} units",
        f"{pred_sgd - pred_lr:+.2f} vs OLS",
    )
    p_col3.metric(
        "Random Forest Forecast",
        f"{pred_rf:.2f} units",
        f"{pred_rf - pred_lr:+.2f} vs OLS",
    )

st.caption(
    "Data snapshot covers 200 market observations with spending across TV, Radio, and Newspaper channels. "
    "Review channel allocations weekly alongside broader campaign metrics."
)
