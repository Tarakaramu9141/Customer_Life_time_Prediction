from lifetimes import BetaGeoFitter, GammaGammaFitter
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings

def train_bg_nbd_gamma_gamma(data, bg_penalizer=0.2, gg_penalizer=0.3, max_iter=2000):
    """
    Train BG/NBD and Gamma-Gamma models with error handling
    """
    # Suppress warnings during fitting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # BG/NBD model
        bgf = BetaGeoFitter(penalizer_coef=bg_penalizer)
        try:
            bgf.fit(
                data['frequency'], 
                data['recency'], 
                data['T'],
                verbose=False,
                max_iter=max_iter
            )
        except Exception as e:
            raise ValueError(
                f"BG/NBD model failed to converge. "
                f"Try increasing penalizer (current: {bg_penalizer}) or removing outliers. "
                f"Error: {str(e)}"
            )
        
        # Gamma-Gamma model
        ggf = GammaGammaFitter(penalizer_coef=gg_penalizer)
        try:
            ggf.fit(
                data['frequency'], 
                data['monetary_value'],
                verbose=False,
                max_iter=max_iter
            )
        except Exception as e:
            raise ValueError(
                f"Gamma-Gamma model failed to converge. "
                f"Try increasing penalizer (current: {gg_penalizer}) or removing outliers. "
                f"Error: {str(e)}"
            )
    
    return bgf, ggf

def train_xgboost_model(data, n_estimators=100, max_depth=6):
    """Train XGBoost model with proper prediction handling"""
    try:
        # Create target variable
        y = data['monetary_value'] * data['frequency']
        
        # Prepare features - ensure numeric only
        X = data.select_dtypes(include=['number']).drop(columns=['customer_id'], errors='ignore')
        
        # Validate required features
        required_features = ['frequency', 'recency', 'T', 'monetary_value']
        missing = [f for f in required_features if f not in X.columns]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            objective='reg:squarederror',
            early_stopping_rounds=10
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # Make predictions on full dataset
        predictions = model.predict(X)
        
        return model, X, predictions
        
    except Exception as e:
        raise ValueError(f"XGBoost training failed: {str(e)}")