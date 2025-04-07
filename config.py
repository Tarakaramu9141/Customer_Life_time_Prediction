import pandas as pd
FILE_READERS = {
    'csv': lambda file: pd.read_csv(file),
    'xlsx': lambda file: pd.read_excel(file),
    'xls': lambda file: pd.read_excel(file),
    'json': lambda file: pd.read_json(file)
}

MODEL_PARAMS = {
    'bg_nbd': {
        'default_penalizer': 0.2,
        'max_iter': 2000
    },
    'gamma_gamma': {
        'default_penalizer': 0.3,
        'max_iter': 2000
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate':0.1,
        'early_stopping_rounds': 10,
        'eval_metric':'mae',
        'numeric_features':[
            'frequency',
            'recency',
            'T',
            'monetary_value',
            'avg_purchase_value',
            'purchase_freq',
            'recency_ratio',
            'frequency_ratio',
            'log_monetary'
        ]
    }
}

PLOT_CONFIG = {
    'clv_distribution': {
        'bins': 50,
        'color': 'skyblue'
    },
    'scatter': {
        'size': 100,
        'alpha': 0.6
    }
}