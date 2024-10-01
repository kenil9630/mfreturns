import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['nav_date'] = pd.to_datetime(df['nav_date'], format='%d-%b-%Y')
    return df

def create_features(df):
    df = df.copy()
    df['dayofweek'] = df['nav_date'].dt.dayofweek
    df['quarter'] = df['nav_date'].dt.quarter
    df['month'] = df['nav_date'].dt.month
    df['year'] = df['nav_date'].dt.year
    df['dayofyear'] = df['nav_date'].dt.dayofyear
    df['dayofmonth'] = df['nav_date'].dt.day
    df['weekofyear'] = df['nav_date'].dt.isocalendar().week
    
    # Create lagged features
    for lag in [1, 3, 7, 14, 30]:
        df[f'lag_{lag}'] = df['net_asset_value'].shift(lag)
    
    # Create rolling mean features
    for window in [7, 14, 30, 60]:
        df[f'rolling_mean_{window}'] = df['net_asset_value'].rolling(window=window).mean()
    
    # Create rolling standard deviation features
    for window in [7, 14, 30, 60]:
        df[f'rolling_std_{window}'] = df['net_asset_value'].rolling(window=window).std()
    
    # Create percentage change features
    for period in [1, 7, 30]:
        df[f'pct_change_{period}'] = df['net_asset_value'].pct_change(periods=period)
    
    return df

def prepare_data(df, scheme_code, train_end_date):
    scheme_data = df[df['scheme_code'] == scheme_code].sort_values('nav_date')
    scheme_data = create_features(scheme_data)
    
    # Convert categorical variables to numeric
    categorical_columns = scheme_data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col not in ['nav_date', 'scheme_code']:  # Exclude date and identifier columns
            scheme_data[col] = pd.Categorical(scheme_data[col]).codes
    
    scheme_data = scheme_data.dropna()  # Drop rows with NaN values after feature creation
    
    train_data = scheme_data[scheme_data['nav_date'].dt.date <= train_end_date]
    test_data = scheme_data[scheme_data['nav_date'].dt.date > train_end_date]
    
    feature_columns = [col for col in scheme_data.columns if col not in ['nav_date', 'scheme_code', 'scheme_name','net_asset_value']]
    
    X_train = train_data[feature_columns]
    y_train = train_data['net_asset_value']
    X_test = test_data[feature_columns]
    y_test = test_data['net_asset_value']
    
    return X_train, y_train, X_test, y_test, train_data, test_data

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)

def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R-squared': r2
    }

def plot_results(train_data, test_data, y_pred):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=('NAV Prediction', 'Prediction Error'))

    # NAV Prediction subplot
    fig.add_trace(go.Scatter(x=train_data['nav_date'], y=train_data['net_asset_value'],
                             mode='lines', name='Training Data', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_data['nav_date'], y=test_data['net_asset_value'],
                             mode='lines', name='Actual Test Data', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_data['nav_date'], y=y_pred,
                             mode='lines', name='Linear Regression Prediction', line=dict(color='red')), row=1, col=1)

    # Prediction Error subplot
    error = test_data['net_asset_value'] - y_pred
    fig.add_trace(go.Scatter(x=test_data['nav_date'], y=error,
                             mode='lines', name='Prediction Error', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

    fig.update_layout(height=800, title_text="Mutual Fund NAV Prediction - Linear Regression")
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=2, col=1)

    return fig

st.title('Advanced Mutual Fund NAV Prediction - Linear Regression')

# File upload or default path
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    default_path = r"C:\Users\Vinod1.Choudhary\Documents\Python Project\React app\portfolio optimization\119110 V3.csv"
    df = load_data(default_path)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = df
    st.session_state.selected_scheme = None
    st.session_state.train_end_date = None

# Select scheme
scheme_codes = st.session_state.df['scheme_code'].unique()
selected_scheme = st.selectbox('Select Scheme Code', scheme_codes, key='scheme_selector')

# Date range for training data
min_date = st.session_state.df['nav_date'].min().date()
max_date = st.session_state.df['nav_date'].max().date()
date_range = (max_date - min_date).days
train_end_date = st.slider(
    'Select Training Data End Date',
    min_value=min_date,
    max_value=max_date,
    value=min_date + pd.Timedelta(days=int(date_range * 0.8)),
    format="DD-MM-YYYY"
)

# Update session state
st.session_state.selected_scheme = selected_scheme
st.session_state.train_end_date = train_end_date

if st.button('Train Model and Predict'):
    # Prepare data
    X_train, y_train, X_test, y_test, train_data, test_data = prepare_data(st.session_state.df, st.session_state.selected_scheme, st.session_state.train_end_date)

    # Train model
    with st.spinner('Training Linear Regression...'):
        lr_model = train_linear_regression(X_train, y_train)

    # Predict
    y_pred = predict(lr_model, X_test)

    # Evaluate model
    metrics = evaluate_model(y_test, y_pred)

    # Display results
    st.subheader('Model Performance Metrics')
    st.write(metrics)

    # Plot results
    st.subheader('NAV Prediction and Error Analysis')
    fig = plot_results(train_data, test_data, y_pred)
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': np.abs(lr_model.coef_)
    }).sort_values('Importance', ascending=False)

    st.subheader('Feature Importance')
    st.bar_chart(feature_importance.set_index('Feature')['Importance'])

        # Create DataFrame with actual, predicted, difference, and accuracy
    results_df = pd.DataFrame({
        'Date': test_data['nav_date'],
        'Actual': y_test,
        'Predicted': y_pred,
        'Difference': y_test - y_pred,
        'Accuracy': (1 - np.abs((y_test - y_pred) / y_test)) * 100
    })

    st.subheader('Prediction Results')
    st.dataframe(results_df)

    # Option to download results as CSV
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="nav_prediction_results.csv",
        mime="text/csv",
    )

    st.success('Model training, prediction, and evaluation completed!')