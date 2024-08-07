import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set up the main structure of the app
st.set_page_config(page_title="Coffee Shop Sales Prediction", layout="wide")
st.title('Coffee Shop Sales Prediction')
st.write('This application predicts coffee shop sales based on various features. You can adjust the model settings and explore data insights interactively.')

# Sidebar for user inputs
st.sidebar.header('User Inputs')

# Media Display: Allow users to upload an image
uploaded_file = st.sidebar.file_uploader("Upload an image", type=['jpg', 'png'])
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('c:/Coffee/coffee_shop_sales.csv')
    data['total_sales'] = data['transaction_qty'] * data['unit_price']
    return data

data = load_data()
st.write(f"Dataset shape: {data.shape}")

# Display the first few rows of the dataset
if st.sidebar.checkbox('Show raw data'):
    st.subheader('Raw Data')
    st.write(data.head())

# User input for filtering data by product category
category_filter = st.sidebar.multiselect(
    'Filter by Product Category', 
    options=data['product_category'].unique(), 
    default=data['product_category'].unique()
)

filtered_data = data[data['product_category'].isin(category_filter)]
st.write(f"Filtered dataset shape: {filtered_data.shape}")

# Select features for prediction
features = ['transaction_qty', 'unit_price', 'store_id', 'product_category', 'product_type']
X = filtered_data[features]
y = filtered_data['total_sales']
st.write(f"Features shape: {X.shape}")
st.write(f"Target shape: {y.shape}")

# Ensure there is enough data
if X.shape[0] < 2:
    st.error("Not enough data to split into training and test sets.")
else:
    # Preprocessing: One-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['transaction_qty', 'unit_price', 'store_id']),
            ('cat', OneHotEncoder(), ['product_category', 'product_type'])
        ])

    # Create a pipeline with a preprocessor and a model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', LinearRegression())])

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    with st.spinner('Training the model...'):
        model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Display evaluation metrics
    st.write(f"### Model Evaluation Metrics")
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # Containers for organizing sections
    st.write("---")
    with st.container():
        st.header('Model Performance')
        st.write('This section displays the model performance metrics and plots. Adjust the filters and explore different data insights!')

        # Visualize actual vs predicted sales
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(y_test, predictions, alpha=0.7, label='Predictions')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Fit')
        ax1.set_xlabel('Actual Sales')
        ax1.set_ylabel('Predicted Sales')
        ax1.set_title('Actual vs Predicted Sales')
        ax1.legend()
        st.pyplot(fig1)

        # Histogram of errors
        errors = y_test - predictions
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.hist(errors, bins=20, alpha=0.7)
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Histogram of Prediction Errors')
        st.pyplot(fig2)

    # Additional Data Insights
    st.write("---")
    st.header('Additional Data Insights')
    st.write('Explore additional data insights with these interactive visualizations.')

    # Plot sales by product category
    sales_by_category = filtered_data.groupby('product_category')['total_sales'].sum().sort_values()
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sales_by_category.plot(kind='barh', ax=ax3)
    ax3.set_xlabel('Total Sales')
    ax3.set_ylabel('Product Category')
    ax3.set_title('Total Sales by Product Category')
    st.pyplot(fig3)

    # Plot sales over time
    if 'transaction_date' in filtered_data.columns:
        filtered_data['transaction_date'] = pd.to_datetime(filtered_data['transaction_date'])
        sales_over_time = filtered_data.set_index('transaction_date').resample('M')['total_sales'].sum()
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        sales_over_time.plot(ax=ax4)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Total Sales')
        ax4.set_title('Total Sales Over Time')
        st.pyplot(fig4)