import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Superstore Analytics", page_icon="🏪", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('superstore data.csv')
    return df

# Main title
st.title("🏪 Superstore Data Analytics Dashboard")
st.markdown("---")

# Load data
try:
    df = load_data()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 EDA", "📈 Data Visualization", "🤖 Machine Learning Model"])
    
    # TAB 1: EDA
    with tab1:
        st.header("Exploratory Data Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Sales", f"${df['Sales'].sum():,.2f}")
        with col3:
            st.metric("Total Profit", f"${df['Profit'].sum():,.2f}")
        with col4:
            st.metric("Avg Profit Margin", f"{(df['Profit'].sum()/df['Sales'].sum()*100):.2f}%")
        
        st.markdown("---")
        
        # Dataset Overview
        st.subheader("📋 Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**First 10 Rows:**")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.write("**Dataset Shape:**")
            st.info(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
            
            st.write("**Column Names:**")
            st.write(", ".join(df.columns.tolist()))
        
        st.markdown("---")
        
        # Data Types and Missing Values
        st.subheader("🔍 Data Quality Check")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types:**")
            dtypes_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Data Type': df.dtypes.values.astype(str)
            })
            st.dataframe(dtypes_df, use_container_width=True)
        
        with col2:
            st.write("**Missing Values:**")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
            
            if df.isnull().sum().sum() == 0:
                st.success("✅ No missing values found!")
        
        st.markdown("---")
        
        # Statistical Summary
        st.subheader("📊 Statistical Summary")
        
        tab_num, tab_cat = st.tabs(["Numerical Features", "Categorical Features"])
        
        with tab_num:
            st.write("**Numerical Columns Statistics:**")
            st.dataframe(df.describe().T, use_container_width=True)
        
        with tab_cat:
            st.write("**Categorical Columns Distribution:**")
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                with st.expander(f"📌 {col}"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        value_counts = df[col].value_counts()
                        st.dataframe(pd.DataFrame({
                            'Value': value_counts.index,
                            'Count': value_counts.values,
                            'Percentage': (value_counts.values / len(df) * 100).round(2)
                        }))
                    with col2:
                        fig = px.pie(values=value_counts.values, names=value_counts.index, 
                                   title=f"{col} Distribution")
                        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Correlation Analysis
        st.subheader("🔗 Correlation Analysis")
        numerical_df = df.select_dtypes(include=[np.number])
        
        # Create correlation matrix using plotly instead of matplotlib
        corr_matrix = numerical_df.corr()
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       aspect='auto',
                       color_continuous_scale='RdBu_r',
                       title='Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Data Visualization
    with tab2:
        st.header("Data Visualization")
        
        # Sales Analysis
        st.subheader("💰 Sales Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales by Category
            sales_by_category = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
            fig = px.bar(x=sales_by_category.index, y=sales_by_category.values,
                        labels={'x': 'Category', 'y': 'Sales'},
                        title='Sales by Category',
                        color=sales_by_category.values,
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sales by Region
            sales_by_region = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
            fig = px.pie(values=sales_by_region.values, names=sales_by_region.index,
                        title='Sales Distribution by Region')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Profit Analysis
        st.subheader("📈 Profit Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Profit by Category
            profit_by_category = df.groupby('Category')['Profit'].sum().sort_values(ascending=False)
            fig = px.bar(x=profit_by_category.index, y=profit_by_category.values,
                        labels={'x': 'Category', 'y': 'Profit'},
                        title='Profit by Category',
                        color=profit_by_category.values,
                        color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top 10 Sub-Categories by Profit
            profit_by_subcat = df.groupby('Sub-Category')['Profit'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(x=profit_by_subcat.values, y=profit_by_subcat.index,
                        labels={'x': 'Profit', 'y': 'Sub-Category'},
                        title='Top 10 Sub-Categories by Profit',
                        orientation='h',
                        color=profit_by_subcat.values,
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Segment and Shipping Analysis
        st.subheader("🚚 Segment & Shipping Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales by Segment
            segment_data = df.groupby('Segment').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Quantity': 'sum'
            }).reset_index()
            
            fig = go.Figure(data=[
                go.Bar(name='Sales', x=segment_data['Segment'], y=segment_data['Sales']),
                go.Bar(name='Profit', x=segment_data['Segment'], y=segment_data['Profit'])
            ])
            fig.update_layout(title='Sales & Profit by Segment', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Ship Mode Analysis
            ship_mode_data = df.groupby('Ship Mode').agg({
                'Sales': 'sum',
                'Profit': 'sum'
            }).reset_index()
            
            fig = px.bar(ship_mode_data, x='Ship Mode', y=['Sales', 'Profit'],
                        title='Sales & Profit by Ship Mode',
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Geographic Analysis
        st.subheader("🗺️ Geographic Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 States by Sales
            top_states = df.groupby('State')['Sales'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(x=top_states.index, y=top_states.values,
                        labels={'x': 'State', 'y': 'Sales'},
                        title='Top 10 States by Sales',
                        color=top_states.values,
                        color_continuous_scale='Viridis')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top 10 Cities by Profit
            top_cities = df.groupby('City')['Profit'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(x=top_cities.index, y=top_cities.values,
                        labels={'x': 'City', 'y': 'Profit'},
                        title='Top 10 Cities by Profit',
                        color=top_cities.values,
                        color_continuous_scale='RdYlGn')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Distribution Analysis
        st.subheader("📊 Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales Distribution
            fig = px.histogram(df, x='Sales', nbins=50,
                             title='Sales Distribution',
                             labels={'Sales': 'Sales Amount'})
            fig.add_vline(x=df['Sales'].mean(), line_dash="dash", 
                         annotation_text=f"Mean: ${df['Sales'].mean():.2f}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Profit Distribution
            fig = px.histogram(df, x='Profit', nbins=50,
                             title='Profit Distribution',
                             labels={'Profit': 'Profit Amount'},
                             color_discrete_sequence=['green'])
            fig.add_vline(x=df['Profit'].mean(), line_dash="dash",
                         annotation_text=f"Mean: ${df['Profit'].mean():.2f}")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Discount Analysis
        st.subheader("💳 Discount Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Discount vs Profit
            fig = px.scatter(df, x='Discount', y='Profit', 
                           title='Discount vs Profit',
                           color='Category',
                           size='Sales',
                           hover_data=['Sub-Category', 'Region'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average Profit by Discount Level
            df['Discount_Range'] = pd.cut(df['Discount'], bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0],
                                         labels=['0-10%', '10-20%', '20-30%', '30-50%', '50%+'])
            avg_profit_discount = df.groupby('Discount_Range')['Profit'].mean().reset_index()
            
            fig = px.bar(avg_profit_discount, x='Discount_Range', y='Profit',
                        title='Average Profit by Discount Range',
                        color='Profit',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Quantity Analysis
        st.subheader("📦 Quantity Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Quantity vs Sales
            fig = px.scatter(df, x='Quantity', y='Sales',
                           title='Quantity vs Sales',
                           color='Category')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Quantity Distribution by Category
            fig = px.box(df, x='Category', y='Quantity',
                        title='Quantity Distribution by Category',
                        color='Category')
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: Machine Learning Model
    with tab3:
        st.header("Machine Learning Model - Profit Prediction")
        st.write("This model predicts **Profit** based on various features like Sales, Quantity, Discount, Category, Region, etc.")
        
        st.markdown("---")
        
        # Model Training Section
        st.subheader("🎯 Model Training & Evaluation")
        
        # Prepare data for ML
        @st.cache_data
        def prepare_ml_data(dataframe):
            df_ml = dataframe.copy()
            
            # Encode categorical variables
            le_dict = {}
            categorical_cols = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category']
            
            for col in categorical_cols:
                le = LabelEncoder()
                df_ml[col + '_encoded'] = le.fit_transform(df_ml[col])
                le_dict[col] = le
            
            # Select features for model
            feature_cols = ['Sales', 'Quantity', 'Discount', 
                          'Ship Mode_encoded', 'Segment_encoded', 
                          'Region_encoded', 'Category_encoded', 'Sub-Category_encoded']
            
            X = df_ml[feature_cols]
            y = df_ml['Profit']
            
            return X, y, le_dict, feature_cols
        
        @st.cache_resource
        def train_model(X, y):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest model
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=20, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = {
                'R2': r2_score(y_train, y_pred_train),
                'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'MAE': mean_absolute_error(y_train, y_pred_train)
            }
            
            test_metrics = {
                'R2': r2_score(y_test, y_pred_test),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'MAE': mean_absolute_error(y_test, y_pred_test)
            }
            
            return model, X_train, X_test, y_train, y_test, y_pred_test, train_metrics, test_metrics
        
        # Prepare data and train model
        with st.spinner("Training the model... Please wait."):
            X, y, le_dict, feature_cols = prepare_ml_data(df)
            model, X_train, X_test, y_train, y_test, y_pred_test, train_metrics, test_metrics = train_model(X, y)
        
        st.success("✅ Model trained successfully!")
        
        # Display model performance
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Test R² Score", f"{test_metrics['R2']:.4f}")
            st.caption("Closer to 1 is better")
        
        with col2:
            st.metric("Test RMSE", f"${test_metrics['RMSE']:.2f}")
            st.caption("Lower is better")
        
        with col3:
            st.metric("Test MAE", f"${test_metrics['MAE']:.2f}")
            st.caption("Lower is better")
        
        st.markdown("---")
        
        # Model Details
        with st.expander("📋 View Detailed Model Metrics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Training Set Performance:**")
                for metric, value in train_metrics.items():
                    if metric == 'R2':
                        st.write(f"- {metric}: {value:.4f}")
                    else:
                        st.write(f"- {metric}: ${value:.2f}")
            
            with col2:
                st.write("**Test Set Performance:**")
                for metric, value in test_metrics.items():
                    if metric == 'R2':
                        st.write(f"- {metric}: {value:.4f}")
                    else:
                        st.write(f"- {metric}: ${value:.2f}")
        
        # Feature Importance
        st.subheader("🔍 Feature Importance")
        
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Clean feature names for display
        feature_importance['Feature_Display'] = feature_importance['Feature'].str.replace('_encoded', '')
        
        fig = px.bar(feature_importance, x='Importance', y='Feature_Display',
                    orientation='h',
                    title='Feature Importance in Profit Prediction',
                    labels={'Feature_Display': 'Feature'},
                    color='Importance',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Prediction vs Actual
        st.subheader("📊 Model Predictions vs Actual Values")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot
            fig = px.scatter(x=y_test, y=y_pred_test,
                           labels={'x': 'Actual Profit', 'y': 'Predicted Profit'},
                           title='Predicted vs Actual Profit')
            
            # Add diagonal line
            min_val = min(y_test.min(), y_pred_test.min())
            max_val = max(y_test.max(), y_pred_test.max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                   mode='lines', name='Perfect Prediction',
                                   line=dict(color='red', dash='dash')))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Residuals plot
            residuals = y_test - y_pred_test
            fig = px.scatter(x=y_pred_test, y=residuals,
                           labels={'x': 'Predicted Profit', 'y': 'Residuals'},
                           title='Residual Plot')
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Interactive Prediction Tool
        st.subheader("🎯 Make Your Own Predictions")
        st.write("Input values to predict profit:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            input_sales = st.number_input("Sales ($)", min_value=0.0, value=100.0, step=10.0)
            input_quantity = st.number_input("Quantity", min_value=1, max_value=20, value=3)
        
        with col2:
            input_discount = st.slider("Discount", min_value=0.0, max_value=0.8, value=0.0, step=0.1)
            input_ship_mode = st.selectbox("Ship Mode", df['Ship Mode'].unique())
        
        with col3:
            input_segment = st.selectbox("Segment", df['Segment'].unique())
            input_region = st.selectbox("Region", df['Region'].unique())
        
        with col4:
            input_category = st.selectbox("Category", df['Category'].unique())
            input_subcategory = st.selectbox("Sub-Category", 
                                           df[df['Category'] == input_category]['Sub-Category'].unique())
        
        if st.button("🔮 Predict Profit", type="primary"):
            # Prepare input data
            input_data = pd.DataFrame({
                'Sales': [input_sales],
                'Quantity': [input_quantity],
                'Discount': [input_discount],
                'Ship Mode_encoded': [le_dict['Ship Mode'].transform([input_ship_mode])[0]],
                'Segment_encoded': [le_dict['Segment'].transform([input_segment])[0]],
                'Region_encoded': [le_dict['Region'].transform([input_region])[0]],
                'Category_encoded': [le_dict['Category'].transform([input_category])[0]],
                'Sub-Category_encoded': [le_dict['Sub-Category'].transform([input_subcategory])[0]]
            })
            
            # Make prediction
            predicted_profit = model.predict(input_data)[0]
            
            # Display result
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
        
        st.markdown("---")
        
        # Model Information
        with st.expander("ℹ️ About the Model"):
            st.write("""
            **Model Type:** Random Forest Regressor
            
            **Features Used:**
            - Sales Amount
            - Quantity Ordered
            - Discount Applied
            - Shipping Mode
            - Customer Segment
            - Geographic Region
            - Product Category
            - Product Sub-Category
            
            **Target Variable:** Profit
            
            **Model Details:**
            - Number of Trees: 100
            - Train-Test Split: 80-20
            - Random State: 42
            
            The Random Forest algorithm creates multiple decision trees and combines their predictions 
            to provide more accurate and stable results. This model can help businesses predict profitability 
            and make informed decisions about pricing, discounts, and product focus.
            """)

except FileNotFoundError:
    st.error("❌ Error: 'superstore data.csv' file not found. Please ensure the file is in the same directory as this app.")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
    st.write("Please check your data file and try again.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit 🎈 | Superstore Analytics Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
