import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Page Configuration
st.set_page_config(
    page_title="House Price Prediction & Model Comparison",
    page_icon="🏠",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏠 Real Estate Price Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Comparing Machine Learning Models & Predicting Property Values</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Data & Cache Models
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('houses_fully_processed_v2.csv')
    return df

@st.cache_resource
def train_models(df):
    X = df.drop(columns=['Target_Price_Log', 'Original_Price'])
    y_log = df['Target_Price_Log']
    y_orig = df['Original_Price']
    
    X_train, X_test, y_train_log, y_test_log, y_train_orig, y_test_orig = train_test_split(
        X, y_log, y_orig, test_size=0.2, random_state=42
    )
    
    models = {
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train_log)
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        
        r2 = r2_score(y_test_orig, y_pred)
        mae = mean_absolute_error(y_test_orig, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
        
        results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
        predictions[name] = y_pred
        
    return models, results, X_train, X_test, y_test_orig, predictions

try:
    df = load_data()
    models, results, X_train, X_test, y_test_orig, predictions = train_models(df)
except Exception as e:
    st.error(f"Error loading dataset or training models: {e}")
    st.stop()

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/real-estate.png", width=80)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Select View", [
    "📊 Model Comparison Dashboard", 
    "⭐ Feature Importance", 
    "📈 Actual vs Predicted Plot",
    "🔮 Live House Price Predictor"
])

# ---------------------------------------------------------
# View 1: Model Comparison Dashboard
# ---------------------------------------------------------
if menu == "📊 Model Comparison Dashboard":
    st.subheader("🎯 Model Performance Metrics")
    
    res_df = pd.DataFrame(results).T.reset_index()
    res_df.columns = ['Model', 'R-Squared Score', 'MAE (RM)', 'RMSE (RM)']
    res_df = res_df.sort_values(by='R-Squared Score', ascending=False)
    
    # Highlight Best Model
    best_model = res_df.iloc[0]['Model']
    best_r2 = res_df.iloc[0]['R-Squared Score']
    
    st.success(f"🏆 **Best Performing Model:** **{best_model}** with an **R² Score of {best_r2:.4f}**")
    
    # Display Metric Cards
    cols = st.columns(4)
    for i, row in enumerate(res_df.itertuples()):
        with cols[i]:
            st.metric(
                label=f"🤖 {row.Model}",
                value=f"{row._2:.4f} R²",
                delta=f"MAE: RM {row._3:,.0f}"
            )
            
    st.markdown("---")
    
    # Comparison Plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### R-Squared Score Comparison")
        fig_r2 = px.bar(
            res_df, 
            x='Model', 
            y='R-Squared Score', 
            text_auto='.4f',
            color='R-Squared Score',
            color_continuous_scale='Blues',
            range_y=[0, 1.0]
        )
        fig_r2.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_r2, use_container_width=True)
        
    with col2:
        st.write("### Mean Absolute Error (MAE in RM)")
        fig_mae = px.bar(
            res_df, 
            x='Model', 
            y='MAE (RM)', 
            text_auto=',.0f',
            color='MAE (RM)',
            color_continuous_scale='Reds_r'
        )
        fig_mae.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_mae, use_container_width=True)
        
    st.subheader("📋 Detailed Comparison Table")
    st.dataframe(res_df.style.highlight_max(axis=0, subset=['R-Squared Score'], color='lightgreen')
                             .highlight_min(axis=0, subset=['MAE (RM)', 'RMSE (RM)'], color='lightgreen'),
                 use_container_width=True)

# ---------------------------------------------------------
# View 2: Feature Importance
# ---------------------------------------------------------
elif menu == "⭐ Feature Importance":
    st.subheader("⭐ Feature Importance Breakdown")
    st.write("Discover which features impact house prices the most in top tree models.")
    
    selected_tree_model = st.selectbox("Select Model", ["Gradient Boosting", "Random Forest"])
    model_obj = models[selected_tree_model]
    
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model_obj.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(15)
    
    fig_imp = px.bar(
        importances, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title=f"Top 15 Important Features ({selected_tree_model})",
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
    st.plotly_chart(fig_imp, use_container_width=True)

# ---------------------------------------------------------
# View 3: Actual vs Predicted
# ---------------------------------------------------------
elif menu == "📈 Actual vs Predicted Plot":
    st.subheader("📈 Actual vs Predicted Price Analysis")
    
    model_choice = st.selectbox("Select Model to Visualize", list(models.keys()), index=3)
    y_pred_vals = predictions[model_choice]
    
    plot_df = pd.DataFrame({
        'Actual Price (RM)': y_test_orig,
        'Predicted Price (RM)': y_pred_vals
    })
    
    fig_scatter = px.scatter(
        plot_df, 
        x='Actual Price (RM)', 
        y='Predicted Price (RM)',
        opacity=0.6,
        trendline="ols",
        title=f"Actual vs Predicted Prices ({model_choice})"
    )
    
    # Add ideal reference line y = x
    max_val = max(plot_df['Actual Price (RM)'].max(), plot_df['Predicted Price (RM)'].max())
    fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', name='Ideal Fit (y=x)', line=dict(dash='dash', color='red')))
    
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------
# View 4: Live House Price Predictor
# ---------------------------------------------------------
elif menu == "🔮 Live House Price Predictor":
    st.subheader("🔮 House Price Valuation Calculator")
    st.write("Input the property parameters below to predict the estimated market price.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        prop_size = st.number_input("Property Size (sq ft)", min_value=300, max_value=10000, value=1000, step=50)
        bedroom = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
        bathroom = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
        parking = st.number_input("Parking Lots", min_value=0, max_value=6, value=1)
        
    with col_b:
        # Extract available state options from feature names
        state_cols = [c.replace('State_Grouped_', '') for c in X_train.columns if c.startswith('State_Grouped_')]
        selected_state = st.selectbox("State", state_cols if state_cols else ["Selangor", "Kuala Lumpur", "Penang", "Melaka"])
        
        city_cols = [c.replace('City_Grouped_', '') for c in X_train.columns if c.startswith('City_Grouped_')]
        selected_city = st.selectbox("City", city_cols if city_cols else ["Kuala Lumpur", "Petaling Jaya", "Cheras"])
        
        prop_cols = [c.replace('Property Type_', '') for c in X_train.columns if c.startswith('Property Type_')]
        selected_type = st.selectbox("Property Type", prop_cols if prop_cols else ["Condominium", "Apartment", "Service Residence", "Flat"])

    if st.button("🚀 Estimate Price", type="primary"):
        # Construct input vector matching X_train
        input_data = pd.DataFrame(0, index=[0], columns=X_train.columns)
        
        input_data['Property Size'] = prop_size
        input_data['Bedroom'] = bedroom
        input_data['Bathroom'] = bathroom
        input_data['Parking Lot'] = parking
        
        if f"State_Grouped_{selected_state}" in input_data.columns:
            input_data[f"State_Grouped_{selected_state}"] = 1
            
        if f"City_Grouped_{selected_city}" in input_data.columns:
            input_data[f"City_Grouped_{selected_city}"] = 1
            
        if f"Property Type_{selected_type}" in input_data.columns:
            input_data[f"Property Type_{selected_type}"] = 1
            
        # Predict using Gradient Boosting
        gb_model = models['Gradient Boosting']
        pred_log = gb_model.predict(input_data)[0]
        estimated_price = np.expm1(pred_log)
        
        st.markdown("---")
        st.success(f"### 🏷️ Estimated Market Price: **RM {estimated_price:,.2f}**")
        st.info(f"Model used: **Gradient Boosting** (Accuracy: {results['Gradient Boosting']['R2']:.2%})")
