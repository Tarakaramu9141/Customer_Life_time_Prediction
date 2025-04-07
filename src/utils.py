import pandas as pd
import numpy as np
from datetime import datetime

def detect_columns(df):
    """More robust column detection"""
    column_map = {}
    
    # Find customer ID column (case insensitive)
    id_cols = [col for col in df.columns if 'customer' in col.lower() or 'id' in col.lower()]
    if id_cols:
        column_map['customer_id'] = id_cols[0]
    
    # Find date column
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_cols:
        try:
            pd.to_datetime(df[col])
            column_map['transaction_date'] = col
            break
        except:
            continue
    
    # Find amount column
    amount_cols = [col for col in df.columns 
                  if ('amount' in col.lower() or 'price' in col.lower()) 
                  and pd.api.types.is_numeric_dtype(df[col])]
    if amount_cols:
        column_map['amount'] = amount_cols[0]
    
    return column_map

def convert_to_standard_format(df, column_map):
    """Convert to standard column names with validation"""
    required = ['customer_id', 'transaction_date', 'amount']
    missing = [col for col in required if col not in column_map]
    if missing:
        raise ValueError(f"Could not detect columns: {', '.join(missing)}")
    
    df = df.rename(columns={
        column_map['customer_id']: 'customer_id',
        column_map['transaction_date']: 'transaction_date',
        column_map['amount']: 'amount'
    })
    return df[['customer_id', 'transaction_date', 'amount']]

def generate_customer_insights(customer_df, clv_score):
    """Generate human-readable customer insights with error handling"""
    try:
        # Basic metrics
        total_spend = customer_df['amount'].sum()
        avg_spend = customer_df['amount'].mean()
        
        # Date handling
        first_purchase = pd.to_datetime(customer_df['transaction_date']).min()
        tenure_days = (pd.to_datetime('today') - first_purchase).days
        tenure_years = round(tenure_days / 365, 1)
        
        # Insights generation
        insights = []
        
        # Value tier
        if clv_score['clv_percentile'] >= 90:
            insights.append("🏆 Top-tier")
        elif clv_score['clv_percentile'] >= 75:
            insights.append("⭐ High-value")
        elif clv_score['clv_percentile'] <= 25:
            insights.append("🆓 Low-value")
        
        # Tenure
        if tenure_years > 2:
            insights.append("👴 Long-term")
        elif tenure_years > 1:
            insights.append("👵 Established")
        else:
            insights.append("🆕 New")
            
        # Spending
        if avg_spend > clv_score['monetary_value'] * 1.5:
            insights.append("💎 High spender")
        elif avg_spend < clv_score['monetary_value'] * 0.5:
            insights.append("💰 Budget")
            
        return {
            'total_spend': round(total_spend, 2),
            'avg_spend': round(avg_spend, 2),
            'tenure_years': tenure_years,
            'first_purchase': first_purchase.strftime('%Y-%m-%d'),
            'insights': " | ".join(insights) if insights else "🆗 Regular"
        }
        
    except Exception as e:
        return {
            'total_spend': 0,
            'avg_spend': 0,
            'tenure_years': 0,
            'first_purchase': "Unknown",
            'insights': f"❌ Error: {str(e)}"
        }

def calculate_percentiles(results_df):
    """Calculate percentile rankings with error handling"""
    try:
        results_df = results_df.copy()
        results_df['spend_percentile'] = results_df['monetary_value'].rank(pct=True) * 100
        results_df['tenure_percentile'] = results_df['T'].rank(pct=True) * 100
        results_df['clv_percentile'] = results_df['predicted_clv'].rank(pct=True) * 100
        return results_df
    except:
        return results_df

def fix_dataframe_display(df):
    """Ensure dataframe is Streamlit-compatible"""
    df = df.copy()
    for col in df.select_dtypes(include=['datetime']).columns:
        df[col] = df[col].astype(str)
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = df[col].astype(str)
        except:
            pass
    return df