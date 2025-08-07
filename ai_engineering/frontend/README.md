# Titans Finance Streamlit Dashboard

A comprehensive financial dashboard built with Streamlit for transaction analysis, ML model predictions, and financial insights.

## 🌟 Features

### 📊 Overview Dashboard
- **Key Financial Metrics**: Total transactions, income, expenses, and net amount
- **Spending by Category**: Interactive pie chart showing expense distribution
- **Daily Balance**: Cumulative balance trend over time
- **Recent Transactions**: List of latest financial activities

### 📈 Transaction Analysis
- **Advanced Filtering**: Filter by date range, categories, and transaction types
- **Monthly Trends**: Income vs expenses analysis by month
- **Amount Distribution**: Histogram of transaction amounts
- **Payment Method Analysis**: Breakdown by payment methods
- **Detailed Transaction Table**: Searchable and sortable transaction data

### 🤖 ML Predictions
- **Category Prediction**: AI-powered transaction categorization
- **Amount Prediction**: Predict transaction amounts
- **Anomaly Detection**: Identify unusual transactions
- **Real-time API Integration**: Live predictions using trained ML models

### ⚠️ Anomaly Detection
- **Large Amount Detection**: Identify unusually large transactions
- **High Activity Days**: Detect days with unusual transaction frequency
- **Anomaly Score Distribution**: Visualize risk patterns

### 💹 Financial Insights
- **Spending Patterns**: Analysis by day of week and monthly trends
- **Financial Health Metrics**: Expense ratios and daily spending averages
- **Budget Analysis**: Mock budget vs actual spending comparison
- **Top Categories**: Identification of highest spending categories

## 🚀 Getting Started

### Prerequisites
- Docker and Docker Compose
- Access to the Titans Finance API
- PostgreSQL database (optional, falls back to sample data)
- Redis cache (optional)

### Running the Dashboard

1. **Using Docker Compose** (Recommended):
```bash
# Start the dashboard with profile
COMPOSE_PROFILES=dashboard docker compose up dashboard -d

# Or start all services including dashboard
docker compose --profile dashboard up -d
```

2. **Access the Dashboard**:
   - Open your browser to: `http://localhost:8501`
   - The dashboard will be available immediately

3. **Stop the Dashboard**:
```bash
docker compose down dashboard
```

### Configuration

The dashboard uses the following environment variables:

- `API_URL`: URL of the Titans Finance API (default: `http://api:8000`)
- `DATABASE_URL`: PostgreSQL connection string (default: sample data if unavailable)
- `REDIS_URL`: Redis cache URL (default: `redis://redis:6379/0`)

## 🏗️ Architecture

### Components
- **Streamlit Frontend**: Interactive web interface
- **DataService**: Handles database connections and data loading
- **APIService**: Manages ML API interactions
- **Utility Functions**: Helper functions for calculations and formatting

### Data Flow
1. Dashboard loads transaction data from PostgreSQL or sample data
2. User interacts with filters and controls
3. ML predictions are fetched from the API service
4. Results are visualized using Plotly charts
5. Redis caching improves performance (when available)

## 📱 Page Structure

### 🏠 Overview
Main dashboard with key metrics and recent activity

### 📊 Transaction Analysis
Detailed transaction filtering and analysis tools

### 🤖 ML Predictions
Interactive form to test ML model predictions:
- Input transaction details
- Get category, amount, and anomaly predictions
- View confidence scores

### ⚠️ Anomaly Detection
Historical anomaly analysis and pattern detection

### 💹 Financial Insights
Advanced analytics and financial health indicators

## 🔧 API Integration

The dashboard integrates with the Titans Finance ML API:

- **Authentication**: Bearer token authentication
- **Endpoints**:
  - `/predict/category`: Transaction categorization
  - `/predict/amount`: Amount prediction
  - `/predict/anomaly`: Anomaly detection
  - `/health`: API health check

### Sample API Request
```python
headers = {
    "Authorization": "Bearer dev-api-key-change-in-production",
    "Content-Type": "application/json"
}

data = {
    "amount": -25.50,
    "description": "Coffee shop",
    "date": "2024-01-15",
    "transaction_type": "Expense",
    "payment_method": "credit_card"
}

response = requests.post(f"{API_URL}/predict/category", json=data, headers=headers)
```

## 🎨 Customization

### Styling
The dashboard uses custom CSS for:
- Color-coded metric cards
- Responsive layout
- Professional appearance
- Alert styling for anomalies

### Charts
All visualizations use Plotly for:
- Interactive charts
- Responsive design
- Export capabilities
- Professional styling

## 📊 Data Sources

### Primary Data
- **PostgreSQL Database**: Live transaction data
- **ML API**: Real-time predictions
- **Redis Cache**: Performance optimization

### Fallback Data
- **Sample Transactions**: 500 realistic sample transactions
- **Mock Predictions**: Fallback when API unavailable
- **Sample Categories**: Common expense categories

## 🔍 Troubleshooting

### Common Issues

1. **Dashboard Not Loading**:
   - Check if containers are running: `docker compose ps`
   - Verify port 8501 is available
   - Check logs: `docker compose logs dashboard`

2. **API Connection Failed**:
   - Ensure API container is healthy: `docker compose ps`
   - Check API health: `curl http://localhost:8000/health`
   - Verify network connectivity between containers

3. **No Transaction Data**:
   - Dashboard falls back to sample data when database unavailable
   - Check PostgreSQL connection string
   - Verify database contains transaction table

4. **ML Predictions Not Working**:
   - Check API authentication token
   - Verify ML models are loaded in API
   - Check API logs for errors

### Debug Commands
```bash
# Check container status
docker compose ps

# View dashboard logs
docker compose logs dashboard --tail 50

# Test API from dashboard container
docker exec titans_dashboard curl http://api:8000/health

# Check database connection
docker exec titans_postgres psql -U postgres -d titans_finance -c "SELECT COUNT(*) FROM transactions;"
```

## 🚀 Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export API_URL="http://localhost:8000"
export DATABASE_URL="postgresql://postgres:password@localhost:5432/titans_finance"

# Run Streamlit
streamlit run dashboard.py
```

### Adding Features
1. Create new page functions in `dashboard.py`
2. Add navigation item in sidebar
3. Implement data processing in `utils.py`
4. Add new API endpoints as needed

## 📈 Performance

### Optimization Features
- **Streamlit Caching**: Cached data loading and API calls
- **Redis Integration**: Cache frequently accessed data
- **Efficient Queries**: Optimized database queries with limits
- **Lazy Loading**: Load data only when needed

### Monitoring
- Health checks for all services
- Container resource monitoring
- API response time tracking
- Error logging and alerting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the Titans Finance suite and follows the same licensing terms.

---

**🎯 Quick Start**: `COMPOSE_PROFILES=dashboard docker compose up -d` then visit `http://localhost:8501`