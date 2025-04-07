import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
import xgboost as xgb
from src.data_preprocessing import preprocess_data, remove_outliers, validate_data
from src.modeling import train_bg_nbd_gamma_gamma, train_xgboost_model
from src.utils import detect_columns, convert_to_standard_format, generate_customer_insights, calculate_percentiles, fix_dataframe_display
from config import FILE_READERS, MODEL_PARAMS, PLOT_CONFIG

# Page configuration
st.set_page_config(
    page_title="🛒 Customer Lifetime Value Prediction",
    layout="wide",
    page_icon="🛒"
)

# Title and introduction
st.title("🛒 Customer Lifetime Value Prediction")
st.markdown("Predict future customer value using BG/NBD and XGBoost models")

# =============================================
# SIDEBAR CONTROLS
# =============================================
with st.sidebar:
    st.header("📋 Data Requirements")
    with st.expander("Click to see required format"):
        st.markdown("""
        **Required columns:**
        - `customer_id`: Unique customer identifier
        - `transaction_date`: Date of purchase (YYYY-MM-DD)
        - `amount`: Transaction value (numeric)
        """)
        st.table(pd.DataFrame({
            "customer_id": ["CUST_001", "CUST_002"],
            "transaction_date": ["2023-01-15", "2023-01-20"],
            "amount": [49.99, 129.99]
        }))
    
    st.header("📤 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload transaction data",
        type=list(FILE_READERS.keys())
    )
    
    st.header("⚙️ Model Settings")
    model_choice = st.selectbox(
        "Select Model",
        ["BG/NBD + Gamma-Gamma", "XGBoost"]
    )
    
    if model_choice == "BG/NBD + Gamma-Gamma":
        st.markdown("**Model Parameters**")
        bg_penalizer = st.slider("BG/NBD Penalizer", 0.0, 1.0, 0.2, 0.01)
        gg_penalizer = st.slider("Gamma-Gamma Penalizer", 0.0, 1.0, 0.3, 0.01)
    
    if model_choice == "XGBoost":
        st.markdown("**XGBoost Parameters**")
        n_estimators = st.slider("Number of trees", 50, 500, MODEL_PARAMS['xgboost']['n_estimators'])
        max_depth = st.slider("Max depth", 3, 10, MODEL_PARAMS['xgboost']['max_depth'])
    
    st.header("📅 Prediction Period")
    months_to_predict = st.slider("Months to predict", 1, 36, 12)

# =============================================
# DATA PROCESSING
# =============================================
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        st.info("ℹ️ Using sample dataset. Upload your data in the sidebar.")
        return pd.read_csv("data/sample_data.csv")
    
    file_ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    try:
        df = FILE_READERS[file_ext](uploaded_file)
        column_map = detect_columns(df)
        
        if not all(k in column_map for k in ['customer_id', 'transaction_date', 'amount']):
            with st.expander("🔍 Manual Column Mapping", expanded=True):
                column_map = {
                    'customer_id': st.selectbox("Customer ID Column", df.columns),
                    'transaction_date': st.selectbox("Transaction Date Column", df.columns),
                    'amount': st.selectbox("Amount Column", 
                                         [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
                }
        
        df = convert_to_standard_format(df, column_map)
        return validate_data(df)
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.stop()

try:
    # Load and preprocess data
    raw_df = load_data(uploaded_file)
    raw_df = remove_outliers(raw_df)
    
    with st.spinner("🔍 Processing customer data..."):
        processed_df, summary_df = preprocess_data(raw_df)
    
    st.success(f"✅ Processed {len(processed_df)} customers from {len(raw_df)} transactions")
    
    # Show data summary
    with st.expander("📊 Data Overview", expanded=False):
        st.dataframe(fix_dataframe_display(summary_df), use_container_width=True)
    
    # =============================================
    # MODEL TRAINING
    # =============================================
    st.header("📈 Model Training")
    
    if model_choice == "BG/NBD + Gamma-Gamma":
        with st.spinner("🤖 Training probabilistic models..."):
            try:
                bgf, ggf = train_bg_nbd_gamma_gamma(
                    processed_df,
                    bg_penalizer=bg_penalizer,
                    gg_penalizer=gg_penalizer
                )
                
                # Predict CLV
                clv = ggf.customer_lifetime_value(
                    bgf,
                    processed_df['frequency'],
                    processed_df['recency'],
                    processed_df['T'],
                    processed_df['monetary_value'],
                    time=months_to_predict,
                    freq='D'
                )
                
                results_df = processed_df.copy()
                results_df['predicted_clv'] = clv.values
                st.success("✅ BG/NBD + Gamma-Gamma Model training complete!")
                
                # Show model parameters
                with st.expander("Model Parameters"):
                    st.write(f"BG/NBD Penalizer: {bg_penalizer}")
                    st.write(f"Gamma-Gamma Penalizer: {gg_penalizer}")
                    st.write(f"BG/NBD LL: {bgf._negative_log_likelihood_:.2f}")
                    st.write(f"Gamma-Gamma LL: {ggf._negative_log_likelihood_:.2f}")
                    
            except Exception as e:
                st.error(f"❌ Model failed to converge: {str(e)}")
                st.info("💡 Try increasing the penalizer coefficients in the sidebar")
                st.stop()
    
    elif model_choice == "XGBoost":
        with st.spinner("🤖 Training XGBoost model..."):
            try:
                xgb_model, xbg_features, xgb_predictions = train_xgboost_model(
                    processed_df, 
                    n_estimators=n_estimators, 
                    max_depth=max_depth
                )
                results_df = processed_df[['customer_id']].copy()
                results_df['predicted_clv'] = xgb_predictions
                #Merge back all the original features for display
                results_df=results_df.merge(
                    processed_df.drop(columns=['predicted_clv'],errors='ignore'),
                    on='customer_id'
                )
                st.success("✅ XGBoost Model training complete!")
                
                # Feature importance
                st.subheader("Feature Importance")
                fig, ax = plt.subplots(figsize=(10, 6))
                xgb.plot_importance(xgb_model, ax=ax, height=0.8)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Model training failed: {str(e)}")
                st.markdown("""
                            **Troubleshooting tips:***
                            1. Ensure your data contains numeric features.
                            2. Check for missing values.
                            3. Try different model parameters.""")
                st.stop()
    
    # Calculate percentiles
    results_df = calculate_percentiles(results_df)
    
    # =============================================
    # RESULTS DISPLAY
    # =============================================
    st.header("📊 Prediction Results")
    
    # Top customers table
    top_n = st.slider("Show top", 5, 50, 10)
    top_customers = results_df.sort_values('predicted_clv', ascending=False).head(top_n)
    st.dataframe(
        fix_dataframe_display(top_customers).style.format({
            'predicted_clv': '${:,.2f}',
            'monetary_value': '${:,.2f}',
            'frequency': '{:.1f}',
            'recency': '{:.1f}',
            'T': '{:.1f}',
            'spend_percentile': '{:.1f}%',
            'tenure_percentile': '{:.1f}%',
            'clv_percentile': '{:.1f}%'
        }),
        use_container_width=True
    )
    
    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CLV Distribution")
        fig, ax = plt.subplots()
        sns.histplot(results_df['predicted_clv'], bins=50, kde=True)
        st.pyplot(fig)
    
    with col2:
        st.subheader("CLV vs Frequency")
        fig, ax = plt.subplots()
        sns.scatterplot(data=results_df, x='frequency', y='predicted_clv', hue='monetary_value')
        st.pyplot(fig)
    
    # Customer insights
    st.header("🔍 Customer Insights")
    if 'customer_id' not in results_df.columns:
        st.warning("⚠️ Customer ID column not found in results")
    else:
        try:
            valid_customers = results_df['customer_id'].dropna().unique()
            if len(valid_customers) == 0:
                st.warning("⚠️ No valid customers found in results")
            else:
                selected_customer = st.selectbox(
                    "Select customer",
                    sorted(valid_customers, 
                          key=lambda x: results_df.loc[results_df['customer_id'] == x, 'predicted_clv'].values[0],
                          reverse=True)[:100]
                )
                
                customer_data = raw_df[raw_df['customer_id'] == selected_customer]
                customer_clv = results_df[results_df['customer_id'] == selected_customer].iloc[0]
                
                insights = generate_customer_insights(customer_data, customer_clv)
                
                cols = st.columns(4)
                cols[0].metric("Predicted CLV", f"${customer_clv['predicted_clv']:,.2f}",
                              f"Top {100 - customer_clv['clv_percentile']:.1f}%")
                cols[1].metric("Total Spend", f"${insights['total_spend']:,.2f}",
                              f"Top {100 - customer_clv['spend_percentile']:.1f}%")
                cols[2].metric("Avg. Purchase", f"${insights['avg_spend']:,.2f}")
                cols[3].metric("Tenure", f"{insights['tenure_years']:.1f} years",
                              f"Top {100 - customer_clv['tenure_percentile']:.1f}%")
                
                st.markdown(f"**First Purchase:** {insights['first_purchase']}")
                st.markdown(f"**Customer Value:** {insights['insights']}")
                
                st.subheader("🛒 Purchase History")
                st.dataframe(customer_data.sort_values('transaction_date', ascending=False))
        except Exception as e:
            st.error(f"❌ Error generating insights: {str(e)}")
    
    # Download results
    st.download_button(
        "💾 Download Predictions",
        results_df.to_csv(index=False),
        "clv_predictions.csv"
    )

except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")