"""
Utility functions for the Titans Finance Dashboard
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import streamlit as st

logger = logging.getLogger(__name__)

def format_currency(amount: float) -> str:
    """Format amount as currency"""
    return f"${amount:,.2f}"

def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:.1f}%"

def calculate_financial_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate key financial metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['total_transactions'] = len(df)
    metrics['total_income'] = df[df['Amount'] > 0]['Amount'].sum()
    metrics['total_expenses'] = abs(df[df['Amount'] < 0]['Amount'].sum())
    metrics['net_amount'] = metrics['total_income'] - metrics['total_expenses']
    
    # Ratios
    if metrics['total_income'] > 0:
        metrics['expense_ratio'] = (metrics['total_expenses'] / metrics['total_income']) * 100
        metrics['savings_rate'] = ((metrics['total_income'] - metrics['total_expenses']) / metrics['total_income']) * 100
    else:
        metrics['expense_ratio'] = 0
        metrics['savings_rate'] = 0
    
    # Daily averages
    if not df.empty:
        date_range = (df['Date'].max() - df['Date'].min()).days + 1
        if date_range > 0:
            metrics['avg_daily_income'] = metrics['total_income'] / date_range
            metrics['avg_daily_expenses'] = metrics['total_expenses'] / date_range
        else:
            metrics['avg_daily_income'] = 0
            metrics['avg_daily_expenses'] = 0
    else:
        metrics['avg_daily_income'] = 0
        metrics['avg_daily_expenses'] = 0
    
    return metrics

def detect_spending_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect spending patterns in transaction data"""
    patterns = {}
    
    if df.empty:
        return patterns
    
    # Day of week patterns
    df_copy = df.copy()
    df_copy['DayOfWeek'] = df_copy['Date'].dt.day_name()
    daily_spending = df_copy[df_copy['Amount'] < 0].groupby('DayOfWeek')['Amount'].sum().abs()
    
    if not daily_spending.empty:
        patterns['highest_spending_day'] = daily_spending.idxmax()
        patterns['lowest_spending_day'] = daily_spending.idxmin()
        patterns['weekend_vs_weekday'] = {
            'weekend': daily_spending[['Saturday', 'Sunday']].sum(),
            'weekday': daily_spending[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].sum()
        }
    
    # Category patterns
    category_spending = df_copy[df_copy['Amount'] < 0].groupby('Category')['Amount'].sum().abs()
    if not category_spending.empty:
        patterns['top_spending_category'] = category_spending.idxmax()
        patterns['category_distribution'] = category_spending.to_dict()
    
    # Monthly patterns
    monthly_spending = df_copy[df_copy['Amount'] < 0].groupby(df_copy['Date'].dt.to_period('M'))['Amount'].sum().abs()
    if len(monthly_spending) > 1:
        patterns['spending_trend'] = 'increasing' if monthly_spending.iloc[-1] > monthly_spending.iloc[0] else 'decreasing'
        patterns['monthly_variance'] = monthly_spending.var()
    
    return patterns

def generate_insights(df: pd.DataFrame, metrics: Dict[str, float], patterns: Dict[str, Any]) -> List[str]:
    """Generate financial insights based on data analysis"""
    insights = []
    
    # Savings rate insights
    if 'savings_rate' in metrics:
        if metrics['savings_rate'] > 20:
            insights.append("🎉 Excellent! You're saving more than 20% of your income.")
        elif metrics['savings_rate'] > 10:
            insights.append("👍 Good savings rate! You're saving more than 10% of your income.")
        elif metrics['savings_rate'] > 0:
            insights.append("💡 Consider increasing your savings rate to at least 10% of income.")
        else:
            insights.append("⚠️ You're spending more than you earn. Consider reducing expenses.")
    
    # Spending pattern insights
    if 'highest_spending_day' in patterns:
        insights.append(f"📅 Your highest spending day is {patterns['highest_spending_day']}.")
    
    if 'top_spending_category' in patterns:
        insights.append(f"💳 Your largest expense category is {patterns['top_spending_category']}.")
    
    # Weekend vs weekday spending
    if 'weekend_vs_weekday' in patterns:
        weekend_spending = patterns['weekend_vs_weekday']['weekend']
        weekday_spending = patterns['weekend_vs_weekday']['weekday']
        if weekend_spending > weekday_spending * 0.4:  # Weekends should be ~2/7 of weekdays
            insights.append("🎪 You tend to spend more on weekends. Consider budgeting for leisure activities.")
    
    # Expense ratio insights
    if 'expense_ratio' in metrics:
        if metrics['expense_ratio'] > 90:
            insights.append("🚨 High expense ratio! Consider ways to reduce spending or increase income.")
        elif metrics['expense_ratio'] > 80:
            insights.append("⚡ Your expense ratio is quite high. Look for optimization opportunities.")
    
    return insights

def create_budget_recommendations(df: pd.DataFrame, current_month_only: bool = True) -> Dict[str, Dict[str, float]]:
    """Create budget recommendations based on spending patterns"""
    recommendations = {}
    
    if df.empty:
        return recommendations
    
    # Filter to current month if requested
    if current_month_only:
        current_date = datetime.now()
        df_filtered = df[(df['Date'].dt.month == current_date.month) & 
                        (df['Date'].dt.year == current_date.year)]
    else:
        df_filtered = df
    
    # Calculate spending by category
    expenses_by_category = df_filtered[df_filtered['Amount'] < 0].groupby('Category')['Amount'].sum().abs()
    
    for category, spent in expenses_by_category.items():
        # Recommend budget as 110% of average spending (with some buffer)
        recommended_budget = spent * 1.1
        remaining = max(0, recommended_budget - spent)
        utilization = (spent / recommended_budget * 100) if recommended_budget > 0 else 0
        
        recommendations[category] = {
            'spent': spent,
            'budget': recommended_budget,
            'remaining': remaining,
            'utilization': utilization
        }
    
    return recommendations

@st.cache_data
def load_sample_transactions(num_transactions: int = 500) -> pd.DataFrame:
    """Load sample transaction data for demo purposes"""
    np.random.seed(42)
    
    categories = [
        'Food & Dining', 'Transportation', 'Shopping', 'Entertainment', 
        'Bills & Utilities', 'Health & Fitness', 'Travel', 'Business Services',
        'Education', 'Personal Care', 'Home & Garden', 'Technology'
    ]
    
    payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'Bank Transfer', 'Digital Wallet']
    
    # Generate realistic transaction descriptions
    food_descriptions = ['Restaurant', 'Coffee Shop', 'Grocery Store', 'Fast Food', 'Cafe']
    transport_descriptions = ['Gas Station', 'Uber', 'Public Transit', 'Parking', 'Car Service']
    shopping_descriptions = ['Online Purchase', 'Department Store', 'Clothing Store', 'Electronics Store']
    entertainment_descriptions = ['Movie Theater', 'Concert', 'Sports Event', 'Streaming Service']
    
    descriptions_by_category = {
        'Food & Dining': food_descriptions,
        'Transportation': transport_descriptions,
        'Shopping': shopping_descriptions,
        'Entertainment': entertainment_descriptions
    }
    
    data = []
    start_date = datetime.now() - timedelta(days=90)
    
    for i in range(num_transactions):
        # Generate random date within last 90 days
        random_date = start_date + timedelta(days=np.random.randint(0, 90))
        
        # Choose transaction type (80% expenses, 20% income)
        trans_type = np.random.choice(['Expense', 'Income'], p=[0.8, 0.2])
        
        # Choose category
        category = np.random.choice(categories)
        
        # Generate description based on category
        if category in descriptions_by_category:
            description = np.random.choice(descriptions_by_category[category])
        else:
            description = f"{category} Transaction"
        
        # Generate amount based on category and type
        if trans_type == 'Expense':
            if category in ['Food & Dining']:
                amount = -np.random.uniform(5, 100)
            elif category in ['Transportation']:
                amount = -np.random.uniform(10, 80)
            elif category in ['Shopping']:
                amount = -np.random.uniform(20, 300)
            elif category in ['Bills & Utilities']:
                amount = -np.random.uniform(50, 200)
            else:
                amount = -np.random.uniform(10, 150)
        else:
            # Income
            amount = np.random.uniform(500, 3000)
            category = 'Income'
            description = np.random.choice(['Salary', 'Freelance', 'Investment', 'Other Income'])
        
        data.append({
            'Date': random_date,
            'Type': trans_type,
            'Description': description,
            'Amount': round(amount, 2),
            'Category': category,
            'Payment Method': np.random.choice(payment_methods)
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Date', ascending=False)
    
    return df

def validate_api_connection(api_url: str) -> bool:
    """Validate connection to the API"""
    try:
        import requests
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"API connection failed: {e}")
        return False