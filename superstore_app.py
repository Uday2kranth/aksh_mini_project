from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Superstore Analytics", page_icon="🏪", layout="wide")

st.markdown(
    """
    <style>
    .main { padding: 0rem 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { padding: 1rem 2rem; font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = Path(__file__).parent / "superstore_data.csv"

# -----------------------------
# Data Loading + Validation
# -----------------------------
REQUIRED_COLUMNS = {
    "Ship Mode",
    "Segment",
    "Country",
    "City",
    "State",
    "Postal Code",
    "Region",
    "Category",
    "Sub-Category",
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
}

NUMERIC_COLUMNS = ["Sales", "Quantity", "Discount", "Profit"]
CATEGORICAL_COLUMNS = [
    "Ship Mode",
    "Segment",
    "Country",
    "City",
    "State",
    "Postal Code",
    "Region",
    "Category",
    "Sub-Category",
]


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        dtype={"Postal Code": "string"},
        encoding="utf-8",
        low_memory=False,
    )

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_cols))}")

    # Normalize column types
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("string")

    # Basic cleanup
    df = df.dropna(subset=NUMERIC_COLUMNS)
    df = df.drop_duplicates()

    # Add a safe discount bucket
    df["Discount_Range"] = pd.cut(
        df["Discount"],
        bins=[-0.001, 0, 0.1, 0.2, 0.3, 0.5, 1.0],
        labels=["0%", "0-10%", "10-20%", "20-30%", "30-50%", "50%+"],
        include_lowest=True,
    )

    return df


# -----------------------------
# ML Pipeline
# -----------------------------
MODEL_FEATURES = [
    "Sales",
    "Quantity",
    "Discount",
    "Ship Mode",
    "Segment",
    "Region",
    "Category",
    "Sub-Category",
]


@st.cache_resource
def train_model(data: pd.DataFrame):
    X = data[MODEL_FEATURES].copy()
    y = data["Profit"].copy()

    categorical_features = [
        "Ship Mode",
        "Segment",
        "Region",
        "Category",
        "Sub-Category",
    ]
    numeric_features = ["Sales", "Quantity", "Discount"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=18,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "R2": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
    }

    return pipeline, X_test, y_test, y_pred, metrics


# -----------------------------
# App Layout
# -----------------------------
st.title("🏪 Superstore Data Analytics Dashboard")
st.markdown("---")

try:
    df = load_data()

    tab_eda, tab_viz, tab_ml = st.tabs(
        ["📊 EDA", "📈 Visualizations", "🤖 ML Model"]
    )

    # -------------------------
    # EDA
    # -------------------------
    with tab_eda:
        st.header("Exploratory Data Analysis")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Sales", f"${df['Sales'].sum():,.2f}")
        with col3:
            st.metric("Total Profit", f"${df['Profit'].sum():,.2f}")
        with col4:
            profit_margin = (df["Profit"].sum() / df["Sales"].sum()) * 100
            st.metric("Avg Profit Margin", f"{profit_margin:.2f}%")

        st.markdown("---")

        st.subheader("📋 Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.info(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
            st.write(", ".join(df.columns.tolist()))

        st.markdown("---")

        st.subheader("🔍 Data Quality")
        col1, col2 = st.columns(2)
        with col1:
            dtypes_df = pd.DataFrame(
                {"Column": df.columns, "Data Type": df.dtypes.astype(str)}
            )
            st.dataframe(dtypes_df, use_container_width=True)
        with col2:
            missing_df = pd.DataFrame(
                {
                    "Column": df.columns,
                    "Missing Count": df.isnull().sum().values,
                    "Missing %": (
                        df.isnull().sum().values / len(df) * 100
                    ).round(2),
                }
            )
            st.dataframe(missing_df, use_container_width=True)

        st.markdown("---")

        st.subheader("📊 Numerical Summary")
        st.dataframe(df[NUMERIC_COLUMNS].describe().T, use_container_width=True)

        st.markdown("---")

        st.subheader("🧩 Categorical Summary")
        for col in CATEGORICAL_COLUMNS:
            with st.expander(col):
                value_counts = df[col].value_counts().head(15)
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Value": value_counts.index,
                            "Count": value_counts.values,
                            "Percentage": (
                                value_counts.values / len(df) * 100
                            ).round(2),
                        }
                    )
                )

        st.markdown("---")

        st.subheader("🔗 Correlation (Numeric Only)")
        corr = df[NUMERIC_COLUMNS].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Visualizations
    # -------------------------
    with tab_viz:
        st.header("Data Visualizations")

        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_region = st.multiselect(
                "Region", sorted(df["Region"].unique()), default=None
            )
        with col2:
            filter_category = st.multiselect(
                "Category", sorted(df["Category"].unique()), default=None
            )
        with col3:
            filter_segment = st.multiselect(
                "Segment", sorted(df["Segment"].unique()), default=None
            )

        filtered_df = df.copy()
        if filter_region:
            filtered_df = filtered_df[filtered_df["Region"].isin(filter_region)]
        if filter_category:
            filtered_df = filtered_df[filtered_df["Category"].isin(filter_category)]
        if filter_segment:
            filtered_df = filtered_df[filtered_df["Segment"].isin(filter_segment)]

        st.markdown("---")

        st.subheader("💰 Sales by Category")
        sales_by_category = (
            filtered_df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        )
        fig = px.bar(
            x=sales_by_category.index,
            y=sales_by_category.values,
            labels={"x": "Category", "y": "Sales"},
            title="Sales by Category",
            color=sales_by_category.values,
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.subheader("📈 Profit by Region")
        profit_by_region = (
            filtered_df.groupby("Region")["Profit"].sum().sort_values(ascending=False)
        )
        fig = px.bar(
            x=profit_by_region.index,
            y=profit_by_region.values,
            labels={"x": "Region", "y": "Profit"},
            title="Profit by Region",
            color=profit_by_region.values,
            color_continuous_scale="Greens",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.subheader("📦 Top 10 Sub-Categories by Profit")
        profit_by_subcat = (
            filtered_df.groupby("Sub-Category")["Profit"].sum().sort_values(ascending=False)
        )
        fig = px.bar(
            x=profit_by_subcat.head(10).values,
            y=profit_by_subcat.head(10).index,
            labels={"x": "Profit", "y": "Sub-Category"},
            title="Top 10 Sub-Categories by Profit",
            orientation="h",
            color=profit_by_subcat.head(10).values,
            color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.subheader("💳 Discount vs Profit")
        fig = px.scatter(
            filtered_df,
            x="Discount",
            y="Profit",
            color="Category",
            size="Sales",
            hover_data=["Sub-Category", "Region"],
            title="Discount vs Profit",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.subheader("🚚 Ship Mode Analysis")
        ship_mode = (
            filtered_df.groupby("Ship Mode")[["Sales", "Profit"]].sum().reset_index()
        )
        fig = px.bar(
            ship_mode,
            x="Ship Mode",
            y=["Sales", "Profit"],
            barmode="group",
            title="Sales & Profit by Ship Mode",
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Machine Learning
    # -------------------------
    with tab_ml:
        st.header("Machine Learning Model - Profit Prediction")
        st.write(
            "Model predicts **Profit** using Sales, Quantity, Discount and category/region features."
        )

        st.markdown("---")

        retrain = st.toggle("Retrain model", value=False)
        if retrain:
            st.cache_resource.clear()

        with st.spinner("Training model..."):
            model, X_test, y_test, y_pred, metrics = train_model(df)

        st.success("✅ Model trained successfully")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test R²", f"{metrics['R2']:.4f}")
        with col2:
            st.metric("Test RMSE", f"${metrics['RMSE']:.2f}")
        with col3:
            st.metric("Test MAE", f"${metrics['MAE']:.2f}")

        st.markdown("---")

        st.subheader("Predicted vs Actual")
        fig = px.scatter(
            x=y_test,
            y=y_pred,
            labels={"x": "Actual Profit", "y": "Predicted Profit"},
            title="Predicted vs Actual Profit",
        )
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="red", dash="dash"),
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.subheader("🔮 Predict Profit")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            input_sales = st.number_input("Sales ($)", min_value=0.0, value=100.0, step=10.0)
            input_quantity = st.number_input("Quantity", min_value=1, max_value=20, value=3)
        with col2:
            input_discount = st.slider("Discount", min_value=0.0, max_value=0.8, value=0.0, step=0.1)
            input_ship_mode = st.selectbox("Ship Mode", sorted(df["Ship Mode"].unique()))
        with col3:
            input_segment = st.selectbox("Segment", sorted(df["Segment"].unique()))
            input_region = st.selectbox("Region", sorted(df["Region"].unique()))
        with col4:
            input_category = st.selectbox("Category", sorted(df["Category"].unique()))
            input_subcategory = st.selectbox(
                "Sub-Category",
                sorted(df[df["Category"] == input_category]["Sub-Category"].unique()),
            )

        if st.button("Predict Profit", type="primary"):
            input_df = pd.DataFrame(
                {
                    "Sales": [input_sales],
                    "Quantity": [input_quantity],
                    "Discount": [input_discount],
                    "Ship Mode": [input_ship_mode],
                    "Segment": [input_segment],
                    "Region": [input_region],
                    "Category": [input_category],
                    "Sub-Category": [input_subcategory],
                }
            )

            predicted_profit = model.predict(input_df)[0]

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Profit", f"${predicted_profit:.2f}")
            with col2:
                profit_margin = (predicted_profit / input_sales * 100) if input_sales > 0 else 0
                st.metric("Profit Margin", f"{profit_margin:.2f}%")
            with col3:
                if predicted_profit > 0:
                    st.success("✅ Profitable Transaction")
                else:
                    st.error("❌ Loss-Making Transaction")

except FileNotFoundError:
    st.error("❌ Dataset not found. Please ensure superstore_data.csv is in the app folder.")
except Exception as e:
    st.error(f"❌ An error occurred: {e}")
    st.write("Please check your dataset and try again.")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit 🎈 | Superstore Analytics Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)
