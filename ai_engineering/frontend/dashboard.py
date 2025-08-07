#!/usr/bin/env python3
"""
Titans Finance Streamlit Dashboard

A comprehensive financial dashboard for transaction analysis, ML model predictions,
and financial insights.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/titans_finance")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Page configuration
st.set_page_config(
    page_title="Titans Finance Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .anomaly-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class DataService:
    """Service for data operations"""
    
    def __init__(self):
        self.engine = None
        self.redis_client = None
        self._init_connections()
    
    def _init_connections(self):
        """Initialize database and Redis connections"""
        try:
            self.engine = create_engine(DATABASE_URL)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            st.error("Database connection failed. Using sample data.")
        
        try:
            import redis as redis_lib
            self.redis_client = redis_lib.from_url(REDIS_URL)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            st.warning("Redis cache unavailable")
    
    def load_transactions(self, limit: int = 1000) -> pd.DataFrame:
        """Load transaction data from database or CSV file"""
        try:
            if self.engine:
                query = """
                SELECT * FROM transactions 
                ORDER BY date DESC 
                LIMIT %s
                """
                return pd.read_sql(query, self.engine, params=[limit])
            else:
                # Fallback to CSV file
                csv_path = "/app/data/all_transactions.csv"
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    return df.head(limit)
                else:
                    return self._generate_sample_data()
        except Exception as e:
            logger.error(f"Error loading transactions: {e}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample transaction data"""
        np.random.seed(42)
        n_transactions = 500
        
        categories = ['Food & Dining', 'Transportation', 'Shopping', 'Entertainment', 
                     'Bills & Utilities', 'Health & Fitness', 'Travel', 'Business Services']
        payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'Bank Transfer']
        
        data = {
            'Date': pd.date_range(start='2024-01-01', periods=n_transactions, freq='D')[:n_transactions],
            'Type': np.random.choice(['Expense', 'Income'], n_transactions, p=[0.8, 0.2]),
            'Description': [f"Transaction {i+1}" for i in range(n_transactions)],
            'Amount': np.random.uniform(-500, 1000, n_transactions),
            'Category': np.random.choice(categories, n_transactions),
            'Payment Method': np.random.choice(payment_methods, n_transactions)
        }
        
        # Make expenses negative
        df = pd.DataFrame(data)
        expense_mask = df['Type'] == 'Expense'
        df.loc[expense_mask, 'Amount'] = -abs(df.loc[expense_mask, 'Amount'])
        
        return df

class APIService:
    """Service for API interactions"""
    
    @staticmethod
    def predict_category(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict transaction category"""
        try:
            headers = {
                "Authorization": "Bearer dev-api-key-change-in-production",
                "Content-Type": "application/json"
            }
            response = requests.post(f"{API_URL}/predict/category", 
                                   json=transaction_data, 
                                   headers=headers,
                                   timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned status {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def predict_amount(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict transaction amount"""
        try:
            headers = {
                "Authorization": "Bearer dev-api-key-change-in-production",
                "Content-Type": "application/json"
            }
            response = requests.post(f"{API_URL}/predict/amount", 
                                   json=transaction_data, 
                                   headers=headers,
                                   timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned status {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def detect_anomaly(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect transaction anomalies"""
        try:
            headers = {
                "Authorization": "Bearer dev-api-key-change-in-production",
                "Content-Type": "application/json"
            }
            response = requests.post(f"{API_URL}/predict/anomaly", 
                                   json=transaction_data, 
                                   headers=headers,
                                   timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned status {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

# Initialize services
@st.cache_resource
def init_services():
    """Initialize services with caching"""
    return DataService(), APIService()

data_service, api_service = init_services()

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown("<h1 class='main-header'>💰 Titans Finance Dashboard</h1>", 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Overview",
        "📊 Transaction Analysis", 
        "🤖 ML Predictions",
        "⚠️ Anomaly Detection",
        "💹 Financial Insights"
    ])
    
    # Load data
    with st.spinner("Loading data..."):
        df = data_service.load_transactions()
    
    if df.empty:
        st.error("No transaction data available")
        return
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Page routing
    if page == "🏠 Overview":
        show_overview(df)
    elif page == "📊 Transaction Analysis":
        show_transaction_analysis(df)
    elif page == "🤖 ML Predictions":
        show_ml_predictions()
    elif page == "⚠️ Anomaly Detection":
        show_anomaly_detection(df)
    elif page == "💹 Financial Insights":
        show_financial_insights(df)

def show_overview(df: pd.DataFrame):
    """Show overview dashboard"""
    st.header("📊 Financial Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_transactions = len(df)
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Transactions</h3>
            <h2>{total_transactions:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_income = df[df['Amount'] > 0]['Amount'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Income</h3>
            <h2>${total_income:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Expenses</h3>
            <h2>${total_expenses:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        net_amount = total_income - total_expenses
        color = "green" if net_amount >= 0 else "red"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Net Amount</h3>
            <h2 style="color: {color};">${net_amount:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💳 Spending by Category")
        expenses_df = df[df['Amount'] < 0].copy()
        expenses_df['Amount'] = abs(expenses_df['Amount'])
        category_spending = expenses_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        
        fig = px.pie(values=category_spending.values, 
                     names=category_spending.index,
                     title="Expenses by Category")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Daily Balance")
        daily_balance = df.groupby(df['Date'].dt.date)['Amount'].sum().cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_balance.index, y=daily_balance.values,
                                mode='lines', name='Cumulative Balance',
                                line=dict(color='#1f77b4', width=2)))
        fig.update_layout(title="Daily Cumulative Balance")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent transactions
    st.subheader("🕒 Recent Transactions")
    recent_df = df.head(10)[['Date', 'Type', 'Description', 'Amount', 'Category']]
    recent_df['Amount'] = recent_df['Amount'].apply(lambda x: f"${x:.2f}")
    st.dataframe(recent_df, use_container_width=True)

def show_transaction_analysis(df: pd.DataFrame):
    """Show detailed transaction analysis"""
    st.header("📊 Transaction Analysis")
    
    # Filters
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_range = st.date_input(
            "Date Range",
            value=(df['Date'].min().date(), df['Date'].max().date()),
            min_value=df['Date'].min().date(),
            max_value=df['Date'].max().date()
        )
    
    with col2:
        categories = st.multiselect(
            "Categories",
            options=df['Category'].unique(),
            default=df['Category'].unique()
        )
    
    with col3:
        transaction_types = st.multiselect(
            "Transaction Types",
            options=df['Type'].unique(),
            default=df['Type'].unique()
        )
    
    # Filter data
    filtered_df = df[
        (df['Date'].dt.date >= date_range[0]) &
        (df['Date'].dt.date <= date_range[1]) &
        (df['Category'].isin(categories)) &
        (df['Type'].isin(transaction_types))
    ]
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters")
        return
    
    st.markdown("---")
    
    # Analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Monthly Trends")
        monthly_data = filtered_df.groupby([filtered_df['Date'].dt.to_period('M'), 'Type'])['Amount'].sum().unstack(fill_value=0)
        
        fig = go.Figure()
        if 'Expense' in monthly_data.columns:
            fig.add_trace(go.Bar(x=[str(x) for x in monthly_data.index], 
                               y=abs(monthly_data['Expense']),
                               name='Expenses', marker_color='red'))
        if 'Income' in monthly_data.columns:
            fig.add_trace(go.Bar(x=[str(x) for x in monthly_data.index], 
                               y=monthly_data['Income'],
                               name='Income', marker_color='green'))
        
        fig.update_layout(title="Monthly Income vs Expenses", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Amount Distribution")
        fig = px.histogram(filtered_df, x='Amount', nbins=50, 
                          title="Transaction Amount Distribution")
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    
    # Payment method analysis
    st.subheader("💳 Payment Method Analysis")
    payment_analysis = filtered_df.groupby('Payment Method')['Amount'].agg(['count', 'sum', 'mean'])
    payment_analysis.columns = ['Count', 'Total Amount', 'Average Amount']
    payment_analysis['Total Amount'] = payment_analysis['Total Amount'].apply(lambda x: f"${x:.2f}")
    payment_analysis['Average Amount'] = payment_analysis['Average Amount'].apply(lambda x: f"${x:.2f}")
    st.dataframe(payment_analysis, use_container_width=True)
    
    # Detailed transaction table
    st.subheader("📋 Transaction Details")
    display_df = filtered_df.copy()
    display_df['Amount'] = display_df['Amount'].apply(lambda x: f"${x:.2f}")
    st.dataframe(display_df, use_container_width=True)

def show_ml_predictions():
    """Show ML prediction interface"""
    st.header("🤖 ML Predictions")
    
    st.subheader("💡 Test Model Predictions")
    st.write("Enter transaction details to get ML model predictions:")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Amount", value=25.50, step=0.01)
            description = st.text_input("Description", value="Coffee shop purchase")
            date_input = st.date_input("Date", value=datetime.now().date())
        
        with col2:
            transaction_type = st.selectbox("Type", ["Expense", "Income"])
            payment_method = st.selectbox("Payment Method", 
                                        ["credit_card", "debit_card", "cash", "bank_transfer"])
            category = st.text_input("Category (optional)", value="")
        
        submit_button = st.form_submit_button("🔮 Get Predictions")
    
    if submit_button:
        # Prepare transaction data
        transaction_data = {
            "amount": -abs(amount) if transaction_type == "Expense" else abs(amount),
            "description": description,
            "date": date_input.isoformat(),
            "transaction_type": transaction_type,
            "payment_method": payment_method
        }
        
        if category:
            transaction_data["category"] = category
        
        # Make predictions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🏷️ Category Prediction")
            with st.spinner("Predicting category..."):
                category_result = api_service.predict_category(transaction_data)
            
            if "error" in category_result:
                st.error(f"Category prediction failed: {category_result['error']}")
            else:
                confidence = category_result.get('confidence', 0)
                prediction = category_result.get('prediction', 'Unknown')
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Predicted Category</h3>
                    <h2>{prediction}</h2>
                    <p>Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("💵 Amount Prediction")
            with st.spinner("Predicting amount..."):
                amount_result = api_service.predict_amount(transaction_data)
            
            if "error" in amount_result:
                st.error(f"Amount prediction failed: {amount_result['error']}")
            else:
                predicted_amount = amount_result.get('prediction', 0)
                confidence = amount_result.get('confidence', 0)
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Predicted Amount</h3>
                    <h2>${predicted_amount:.2f}</h2>
                    <p>Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.subheader("⚠️ Anomaly Detection")
            with st.spinner("Checking for anomalies..."):
                anomaly_result = api_service.detect_anomaly(transaction_data)
            
            if "error" in anomaly_result:
                st.error(f"Anomaly detection failed: {anomaly_result['error']}")
            else:
                is_anomaly = anomaly_result.get('is_anomaly', False)
                anomaly_score = anomaly_result.get('anomaly_score', 0)
                
                card_class = "danger-card" if is_anomaly else "prediction-card"
                status = "⚠️ ANOMALY" if is_anomaly else "✅ NORMAL"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <h3>Anomaly Status</h3>
                    <h2>{status}</h2>
                    <p>Score: {anomaly_score:.3f}</p>
                </div>
                """, unsafe_allow_html=True)

def show_anomaly_detection(df: pd.DataFrame):
    """Show anomaly detection analysis"""
    st.header("⚠️ Anomaly Detection")
    
    st.subheader("🔍 Potential Anomalies")
    st.write("Analyzing historical transactions for anomalous patterns...")
    
    # Simple rule-based anomaly detection for demonstration
    anomalies = []
    
    # Detect unusually large amounts
    amount_threshold = df['Amount'].std() * 3
    large_amounts = df[abs(df['Amount']) > amount_threshold]
    
    # Detect unusual timing (e.g., transactions at odd hours)
    # This would require time information which we don't have in sample data
    
    # Detect unusual frequency
    daily_counts = df.groupby(df['Date'].dt.date).size()
    high_frequency_days = daily_counts[daily_counts > daily_counts.quantile(0.95)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Large Amount Transactions")
        if not large_amounts.empty:
            display_df = large_amounts[['Date', 'Description', 'Amount', 'Category']].copy()
            display_df['Amount'] = display_df['Amount'].apply(lambda x: f"${x:.2f}")
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No unusually large transactions detected")
    
    with col2:
        st.subheader("📊 High Activity Days")
        if not high_frequency_days.empty:
            st.write("Days with unusually high transaction counts:")
            for date, count in high_frequency_days.items():
                st.write(f"• {date}: {count} transactions")
        else:
            st.info("No unusual activity patterns detected")
    
    # Anomaly score distribution
    st.subheader("📈 Anomaly Score Distribution")
    # Generate mock anomaly scores for visualization
    df_sample = df.sample(min(100, len(df)))
    mock_scores = np.random.beta(2, 8, len(df_sample))  # Most scores low, few high
    
    fig = px.histogram(x=mock_scores, nbins=20, 
                      title="Distribution of Anomaly Scores",
                      labels={'x': 'Anomaly Score', 'y': 'Count'})
    fig.add_vline(x=0.7, line_dash="dash", line_color="red", 
                  annotation_text="Anomaly Threshold")
    st.plotly_chart(fig, use_container_width=True)

def show_financial_insights(df: pd.DataFrame):
    """Show financial insights and analytics"""
    st.header("💹 Financial Insights")
    
    # Spending patterns
    st.subheader("💳 Spending Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Day of week spending
        df['DayOfWeek'] = df['Date'].dt.day_name()
        daily_spending = df[df['Amount'] < 0].groupby('DayOfWeek')['Amount'].sum().abs()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_spending = daily_spending.reindex(day_order)
        
        fig = px.bar(x=daily_spending.index, y=daily_spending.values,
                    title="Spending by Day of Week")
        fig.update_layout(xaxis_title="Day", yaxis_title="Amount ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly trend
        monthly_trend = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[str(x) for x in monthly_trend.index], 
                               y=monthly_trend.values,
                               mode='lines+markers', name='Net Amount'))
        fig.update_layout(title="Monthly Net Amount Trend")
        st.plotly_chart(fig, use_container_width=True)
    
    # Financial health metrics
    st.subheader("🏥 Financial Health Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Expense ratio
        total_income = df[df['Amount'] > 0]['Amount'].sum()
        total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())
        expense_ratio = (total_expenses / total_income) * 100 if total_income > 0 else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Expense Ratio</h3>
            <h2>{expense_ratio:.1f}%</h2>
            <p>Of total income</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Average daily spending
        expense_days = df[df['Amount'] < 0].groupby(df['Date'].dt.date)['Amount'].sum()
        avg_daily_spending = abs(expense_days.mean()) if not expense_days.empty else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Daily Spending</h3>
            <h2>${avg_daily_spending:.2f}</h2>
            <p>Per day</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Largest expense category
        expenses_by_category = df[df['Amount'] < 0].groupby('Category')['Amount'].sum().abs()
        if not expenses_by_category.empty:
            largest_category = expenses_by_category.idxmax()
            largest_amount = expenses_by_category.max()
        else:
            largest_category = "N/A"
            largest_amount = 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Top Expense Category</h3>
            <h2>{largest_category}</h2>
            <p>${largest_amount:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Budget analysis (mock data)
    st.subheader("📊 Budget Analysis")
    
    categories = df[df['Amount'] < 0]['Category'].unique()
    budget_data = []
    
    for category in categories:
        spent = abs(df[(df['Amount'] < 0) & (df['Category'] == category)]['Amount'].sum())
        # Mock budget (spending + 20%)
        budget = spent * 1.2
        budget_data.append({
            'Category': category,
            'Spent': spent,
            'Budget': budget,
            'Remaining': max(0, budget - spent),
            'Utilization': (spent / budget) * 100 if budget > 0 else 0
        })
    
    budget_df = pd.DataFrame(budget_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Spent', x=budget_df['Category'], y=budget_df['Spent']))
    fig.add_trace(go.Bar(name='Remaining', x=budget_df['Category'], y=budget_df['Remaining']))
    fig.update_layout(title="Budget vs Actual Spending", barmode='stack')
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()