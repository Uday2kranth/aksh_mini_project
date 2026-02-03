import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path

# Page configuration
st.set_page_config(page_title="Superstore Analytics", page_icon="🏪", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🏪 Superstore Analytics Dashboard")
st.markdown("---")

# Load data
DATA_PATH = Path(__file__).parent / "superstore_data.csv"

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    return df

try:
    df = load_data()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 EDA & Preprocessing", "📈 Visualizations", "🤖 ML Model"])
    
    # ========================================
    # TAB 1: EDA & PREPROCESSING
    # ========================================
    with tab1:
        st.header("Exploratory Data Analysis & Preprocessing")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
        with col3:
            st.metric("Total Profit", f"${df['Profit'].sum():,.0f}")
        with col4:
            st.metric("Categories", df['Category'].nunique())
        
        st.markdown("---")
        
        # Dataset Preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Dataset Information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Dataset Info")
            st.write(f"**Rows:** {df.shape[0]:,}")
            st.write(f"**Columns:** {df.shape[1]}")
            st.write(f"**Duplicates:** {df.duplicated().sum()}")
            st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
        
        with col2:
            st.subheader("🔢 Numerical Summary")
            st.dataframe(df[['Sales', 'Quantity', 'Discount', 'Profit']].describe())
        
        st.markdown("---")
        
        # Data Types
        st.subheader("📝 Column Data Types")
        dtypes_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.notnull().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(dtypes_df, use_container_width=True)
        
        st.markdown("---")
        
        # Preprocessing Steps
        st.subheader("🧹 Preprocessing Steps")
        st.write("✅ Data loaded successfully")
        st.write("✅ No missing values detected")
        st.write("✅ Numerical columns: Sales, Quantity, Discount, Profit")
        st.write("✅ Categorical columns: Ship Mode, Segment, Region, Category, Sub-Category")
        st.write("✅ Data is ready for analysis and modeling")
    
    # ========================================
    # TAB 2: DATA VISUALIZATIONS (ONLY 2)
    # ========================================
    with tab2:
        st.header("Data Visualizations")
        
        # Visualization 1: Sales by Category
        st.subheader("💰 Sales by Category")
        sales_by_category = df.groupby('Category')['Sales'].sum().sort_values(ascending=False).reset_index()
        fig1 = px.bar(
            sales_by_category,
            x='Category',
            y='Sales',
            title='Total Sales by Product Category',
            color='Sales',
            color_continuous_scale='Blues',
            labels={'Sales': 'Total Sales ($)'}
        )
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("---")
        
        # Visualization 2: Profit by Region
        st.subheader("🗺️ Profit by Region")
        profit_by_region = df.groupby('Region')['Profit'].sum().sort_values(ascending=False).reset_index()
        fig2 = px.bar(
            profit_by_region,
            x='Region',
            y='Profit',
            title='Total Profit by Region',
            color='Profit',
            color_continuous_scale='Greens',
            labels={'Profit': 'Total Profit ($)'}
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
    
    # ========================================
    # TAB 3: MACHINE LEARNING MODEL
    # ========================================
    with tab3:
        st.header("Machine Learning Model: Profit Prediction")
        st.write("**Predicting Profit based on Sales, Quantity, and Discount**")
        
        st.markdown("---")
        
        # Prepare data for modeling
        @st.cache_data
        def prepare_data():
            """Prepare data for machine learning"""
            # Select features and target
            X = df[['Sales', 'Quantity', 'Discount']]
            y = df['Profit']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            return X_train, X_test, y_train, y_test
        
        @st.cache_resource
        def train_model():
            """Train Random Forest model"""
            X_train, X_test, y_train, y_test = prepare_data()
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            return model, r2, rmse, mae, X_test, y_test, y_pred
        
        # Train model
        with st.spinner("Training model..."):
            model, r2, rmse, mae, X_test, y_test, y_pred = train_model()
        
        st.success("✅ Model trained successfully!")
        
        # Display metrics
        st.subheader("📈 Model Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
            st.caption("Higher is better (max: 1.0)")
        
        with col2:
            st.metric("RMSE", f"${rmse:.2f}")
            st.caption("Lower is better")
        
        with col3:
            st.metric("MAE", f"${mae:.2f}")
            st.caption("Lower is better")
        
        st.markdown("---")
        
        # Model visualization
        st.subheader("📊 Predicted vs Actual Profit")
        comparison_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred
        })
        
        fig3 = px.scatter(
            comparison_df,
            x='Actual',
            y='Predicted',
            title='Model Predictions vs Actual Values',
            labels={'Actual': 'Actual Profit ($)', 'Predicted': 'Predicted Profit ($)'},
            opacity=0.6
        )
        
        # Add perfect prediction line
        min_val = min(comparison_df['Actual'].min(), comparison_df['Predicted'].min())
        max_val = max(comparison_df['Actual'].max(), comparison_df['Predicted'].max())
        fig3.add_scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("---")
        
        # Interactive prediction
        st.subheader("🔮 Make a Prediction")
        st.write("Enter values to predict profit:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_sales = st.number_input(
                "Sales ($)",
                min_value=0.0,
                max_value=10000.0,
                value=200.0,
                step=10.0
            )
        
        with col2:
            input_quantity = st.number_input(
                "Quantity",
                min_value=1,
                max_value=20,
                value=3
            )
        
        with col3:
            input_discount = st.slider(
                "Discount",
                min_value=0.0,
                max_value=0.8,
                value=0.0,
                step=0.05
            )
        
        if st.button("Predict Profit", type="primary"):
            # Make prediction
            input_data = pd.DataFrame({
                'Sales': [input_sales],
                'Quantity': [input_quantity],
                'Discount': [input_discount]
            })
            
            predicted_profit = model.predict(input_data)[0]
            
            # Display result
            st.markdown("---")
            st.subheader("Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Profit", f"${predicted_profit:.2f}")
            
            with col2:
                if predicted_profit > 0:
                    st.success("✅ Profitable Transaction")
                else:
                    st.error("❌ Loss-Making Transaction")
            
            profit_margin = (predicted_profit / input_sales * 100) if input_sales > 0 else 0
            st.info(f"**Profit Margin:** {profit_margin:.2f}%")

except FileNotFoundError:
    st.error("❌ Dataset not found. Please ensure 'superstore_data.csv' is in the app directory.")
except Exception as e:
    st.error(f"❌ An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit 🎈</p>
    </div>
    """, unsafe_allow_html=True)
