import pandas as pd
import numpy as np
from lifetimes.utils import summary_data_from_transaction_data
from datetime import datetime

def validate_data(df):
    """
    Validate input data structure and content
    Returns cleaned DataFrame if valid, raises ValueError otherwise
    """
    # Check required columns
    required_cols = ['customer_id', 'transaction_date', 'amount']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # Convert and validate transaction dates
    try:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        if df['transaction_date'].isnull().any():
            raise ValueError("Invalid date values found")
    except Exception as e:
        raise ValueError(f"Date conversion failed: {str(e)}")

    # Validate amounts
    if not pd.api.types.is_numeric_dtype(df['amount']):
        raise ValueError("Amount column must be numeric")
    if df['amount'].min() <= 0:
        raise ValueError("All transaction amounts must be positive")

    return df

def remove_outliers(df, lower_quantile=0.01, upper_quantile=0.99):
    """
    Remove extreme values from transaction amounts
    Returns DataFrame with outliers removed
    """
    q_low = df['amount'].quantile(lower_quantile)
    q_hi = df['amount'].quantile(upper_quantile)
    return df[(df['amount'] > q_low) & (df['amount'] < q_hi)]

def preprocess_data(raw_df):
    """
    Convert raw transactions to RFM format with enhanced features
    Returns:
    - processed_df: DataFrame with RFM metrics and engineered features (includes customer_id)
    - summary_df: Descriptive statistics of the processed data
    """
    # Validate input data
    raw_df = validate_data(raw_df)
    
    # Create RFM summary
    rfm_df = summary_data_from_transaction_data(
        transactions=raw_df,
        customer_id_col='customer_id',
        datetime_col='transaction_date',
        monetary_value_col='amount',
        observation_period_end=pd.to_datetime('today')
    )
    
    # Preserve customer_id which gets lost in RFM calculation
    customer_ids = raw_df.groupby('customer_id').first().reset_index()[['customer_id']]
    processed_df = rfm_df.merge(customer_ids, left_index=True, right_on='customer_id')
    
    # Filter out invalid records
    processed_df = processed_df[
        (processed_df['monetary_value'] > 0) & 
        (processed_df['frequency'] > 0)
    ].copy()
    
    # Basic RFM features
    processed_df['avg_purchase_value'] = processed_df['monetary_value']
    processed_df['purchase_freq'] = processed_df['frequency'] / processed_df['T']
    
    # Enhanced features for XGBoost
    processed_df['recency_ratio'] = processed_df['recency'] / processed_df['T']
    processed_df['frequency_ratio'] = processed_df['frequency'] / processed_df['T']
    processed_df['log_monetary'] = np.log1p(processed_df['monetary_value'])
    processed_df['tenure'] = processed_df['T']
    processed_df['frequency_per_tenure'] = processed_df['frequency'] / processed_df['T']
    
    # Interaction features
    processed_df['value_freq_interaction'] = processed_df['monetary_value'] * processed_df['frequency']
    processed_df['recency_freq_interaction'] = processed_df['recency'] * processed_df['frequency']
    
    # Create summary stats (excluding customer_id)
    numeric_df = processed_df.select_dtypes(include=[np.number])
    summary_df = numeric_df.describe().transpose()
    
    return processed_df, summary_df

def prepare_xgboost_data(processed_df):
    """Prepare data specifically for XGBoost training"""
    # Ensure we have a copy
    xgb_df = processed_df.copy()
    
    # Convert potential boolean columns
    for col in xgb_df.select_dtypes(include=['bool']).columns:
        xgb_df[col] = xgb_df[col].astype(int)
    
    # Select numeric features only
    numeric_features = xgb_df.select_dtypes(include=['number']).columns
    xgb_df = xgb_df[numeric_features]
    
    # Validate required features
    required = ['frequency', 'recency', 'T', 'monetary_value']
    missing = [f for f in required if f not in xgb_df.columns]
    if missing:
        raise ValueError(f"Missing required XGBoost features: {', '.join(missing)}")
    
    return xgb_df