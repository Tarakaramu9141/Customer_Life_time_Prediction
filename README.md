# Customer Lifetime Value Prediction

This project predicts Customer Lifetime Value (CLV) using three different approaches:
1. Probabilistic models (BG/NBD + Gamma-Gamma) from the Lifetimes library
2. XGBoost machine learning model
3. Bayesian hierarchical model

## Live Demo:
You can view the live demo here : [https://customerlifetimeprediction-to.streamlit.app/]
## Models
1. BG/NBD + Gamma-Gamma
- BG/NBD model: Predicts transaction frequency
- Gamma-Gamma model: Predicts monetary value.
- Combined to estimate CLV
2. XGBoost
- Machine learning approach using RFM Features.
- Feature importance visualization
3. Bayesian Model
- Hierarchical Bayesian model.
- Provides uncertainty estimates.

## Key packages inclueded:
- Core: Streamlit (dashboard), Lifetimes (probabilistic models), XGBoost (ML).
- Data: Pandas, NumPy, scikit-learn.
- Visualization: Matplotlib, Seaborn, Plotly.
- File Support: PDF, Excel (OpenPyXL), Arrow (for efficient data handling).

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/clv-prediction.git
   cd clv-prediction
2. Create and activate a virtual environment
    python -m venv venv
    venv\scripts\activate
3. Install dependencies
    pip install -r requirments.txt

## Usage
1. Prepare your transaction data in csv format with columns
- customer_id:Unique customer identifier
- transaction_date: Date of Transaction(YYYY-MM-DD)
- amount : transaction amount
2. Run the streamlit app:
streamlit run app.py
3. In the app:
- Upload your data file
- Select a model type
- Adjust parameters as needed
- View Predictions and visualizations.

### Sample Data
This repository includes a sample dataset(data/sample_data.csv) with simulated transaction data.

