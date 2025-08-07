# Data Engineering Tasks

## Overview
This document outlines the pending tasks required to complete the data engineering components of the Titans Finance project. The data engineering layer is responsible for data ingestion, transformation, quality validation, and warehouse management.

## Current Status
✅ **Completed:**
- CSV Extractor (`data_engineering/etl/extractors/csv_extractor.py`) - Full implementation
- ETL Pipeline Framework (`data_engineering/etl/run_pipeline.py`) - Framework exists
- Docker Infrastructure - PostgreSQL, Redis running
- Project Structure - Directories created

❌ **Missing/Incomplete:**
- Data Transformers (0% complete)
- Data Loaders (0% complete) 
- Database Schema (0% complete)
- Airflow DAGs (0% complete)
- Data Quality Framework (0% complete)
- Warehouse Initialization (0% complete)

## Pending Tasks

### 1. Database Schema and Warehouse Setup

#### 1.1 Create Database Initialization Scripts
**Priority:** High
**File:** `data_engineering/warehouse/init_scripts/01_create_schema.sql`
```sql
-- Create main schema
CREATE SCHEMA IF NOT EXISTS titans_finance;

-- Raw transactions table
CREATE TABLE titans_finance.raw_transactions (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    type VARCHAR(50) NOT NULL,
    description TEXT,
    amount DECIMAL(15,2) NOT NULL,
    category VARCHAR(100),
    payment_method VARCHAR(50),
    status VARCHAR(20),
    reference VARCHAR(100),
    receipt_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processed transactions table
CREATE TABLE titans_finance.processed_transactions (
    id SERIAL PRIMARY KEY,
    raw_transaction_id INTEGER REFERENCES titans_finance.raw_transactions(id),
    date DATE NOT NULL,
    type VARCHAR(50) NOT NULL,
    description_cleaned TEXT,
    amount DECIMAL(15,2) NOT NULL,
    category_predicted VARCHAR(100),
    category_confidence DECIMAL(5,4),
    payment_method VARCHAR(50),
    status VARCHAR(20),
    
    -- Feature engineering columns
    day_of_week INTEGER,
    month INTEGER,
    quarter INTEGER,
    is_weekend BOOLEAN,
    amount_category VARCHAR(20), -- small, medium, large
    
    -- Time-based features
    days_since_last_transaction INTEGER,
    rolling_avg_7d DECIMAL(15,2),
    rolling_avg_30d DECIMAL(15,2),
    
    -- Categorical features
    merchant_category VARCHAR(100),
    location_category VARCHAR(100),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Aggregated metrics table
CREATE TABLE titans_finance.transaction_metrics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    total_transactions INTEGER,
    total_amount DECIMAL(15,2),
    avg_amount DECIMAL(15,2),
    
    -- By category
    category VARCHAR(100),
    category_count INTEGER,
    category_amount DECIMAL(15,2),
    
    -- By type
    income_count INTEGER,
    expense_count INTEGER,
    income_amount DECIMAL(15,2),
    expense_amount DECIMAL(15,2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data quality tracking table
CREATE TABLE titans_finance.data_quality_reports (
    id SERIAL PRIMARY KEY,
    execution_date TIMESTAMP NOT NULL,
    pipeline_name VARCHAR(100),
    total_records INTEGER,
    passed_records INTEGER,
    failed_records INTEGER,
    quality_score DECIMAL(5,4),
    issues_detected TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_raw_transactions_date ON titans_finance.raw_transactions(date);
CREATE INDEX idx_processed_transactions_date ON titans_finance.processed_transactions(date);
CREATE INDEX idx_processed_transactions_category ON titans_finance.processed_transactions(category_predicted);
CREATE INDEX idx_transaction_metrics_date ON titans_finance.transaction_metrics(date);
```

#### 1.2 Create Data Warehouse Migration Scripts
**File:** `data_engineering/warehouse/migrations/001_initial_schema.py`
```python
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001'
down_revision = None

def upgrade():
    # Implementation of schema creation using Alembic
    pass

def downgrade():
    # Implementation of schema rollback
    pass
```

### 2. Data Transformation Layer

#### 2.1 Transaction Transformer Implementation
**Priority:** High
**File:** `data_engineering/etl/transformers/transaction_transformer.py`

**Required Features:**
- Data cleaning and normalization
- Date/time parsing and standardization
- Amount validation and formatting
- Category standardization
- Payment method normalization
- Feature engineering pipeline
- Data validation rules

**Implementation Tasks:**
```python
class TransactionTransformer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main transformation pipeline"""
        # 1. Clean and standardize data
        # 2. Parse and validate dates
        # 3. Normalize categories and payment methods
        # 4. Validate and clean amounts
        # 5. Create derived fields
        pass
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering pipeline"""
        # 1. Time-based features (day_of_week, month, quarter, is_weekend)
        # 2. Amount-based features (percentiles, categories)
        # 3. Rolling statistics (7d, 30d averages)
        # 4. Transaction frequency features
        # 5. Category-based features
        # 6. Seasonal features
        pass
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Data validation and quality checks"""
        # 1. Null value validation
        # 2. Data type validation
        # 3. Range validation for amounts and dates
        # 4. Referential integrity checks
        # 5. Business rule validation
        pass
```

#### 2.2 Additional Transformers
**Files to Create:**
- `data_engineering/etl/transformers/feature_transformer.py` - Feature engineering
- `data_engineering/etl/transformers/data_cleaner.py` - Data cleaning utilities
- `data_engineering/etl/transformers/validation_transformer.py` - Data validation

### 3. Data Loading Layer

#### 3.1 PostgreSQL Loader Implementation
**Priority:** High
**File:** `data_engineering/etl/loaders/postgres_loader.py`

**Required Features:**
- Bulk data insertion
- Upsert operations
- Transaction management
- Error handling and recovery
- Performance optimization
- Connection pooling

**Implementation Tasks:**
```python
class PostgresLoader:
    def load_raw_data(self, df: pd.DataFrame) -> bool:
        """Load raw transaction data"""
        # 1. Prepare data for insertion
        # 2. Handle duplicates and conflicts
        # 3. Bulk insert with error handling
        # 4. Log insertion statistics
        pass
    
    def load_processed_data(self, df: pd.DataFrame) -> bool:
        """Load processed transaction data"""
        # 1. Validate foreign key relationships
        # 2. Insert processed transactions
        # 3. Update aggregated metrics
        # 4. Handle data quality issues
        pass
    
    def load_aggregated_metrics(self, metrics: Dict) -> bool:
        """Load aggregated transaction metrics"""
        # 1. Calculate daily/monthly aggregations
        # 2. Insert or update metrics table
        # 3. Maintain historical data
        pass
```

#### 3.2 Additional Loaders
**Files to Create:**
- `data_engineering/etl/loaders/redis_loader.py` - Cache loading
- `data_engineering/etl/loaders/feature_store_loader.py` - Feature store integration

### 4. Apache Airflow DAGs

#### 4.1 Main Data Pipeline DAG
**Priority:** High
**File:** `data_engineering/airflow/dags/transaction_pipeline_dag.py`

**Pipeline Flow:**
```python
# Daily transaction processing pipeline
dag = DAG(
    'transaction_etl_pipeline',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
)

extract_task = PythonOperator(
    task_id='extract_transactions',
    python_callable=extract_csv_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_transactions', 
    python_callable=transform_transaction_data,
    dag=dag
)

load_raw_task = PythonOperator(
    task_id='load_raw_data',
    python_callable=load_raw_transactions,
    dag=dag
)

load_processed_task = PythonOperator(
    task_id='load_processed_data',
    python_callable=load_processed_transactions,
    dag=dag
)

quality_check_task = PythonOperator(
    task_id='data_quality_checks',
    python_callable=run_data_quality_checks,
    dag=dag
)

# Define task dependencies
extract_task >> transform_task >> [load_raw_task, load_processed_task] >> quality_check_task
```

#### 4.2 Additional DAGs
**Files to Create:**
- `data_engineering/airflow/dags/data_quality_dag.py` - Data quality monitoring
- `data_engineering/airflow/dags/feature_pipeline_dag.py` - Feature engineering pipeline
- `data_engineering/airflow/dags/aggregation_dag.py` - Metrics aggregation pipeline

### 5. Data Quality Framework

#### 5.1 Great Expectations Integration
**Priority:** Medium
**Directory:** `data_engineering/quality/`

**Files to Create:**
- `great_expectations.yml` - GE configuration
- `expectations/transaction_suite.json` - Data expectations
- `checkpoints/daily_validation.yml` - Validation checkpoints

#### 5.2 Custom Data Quality Checks
**File:** `data_engineering/quality/data_validator.py`
```python
class DataQualityValidator:
    def validate_completeness(self, df: pd.DataFrame) -> Dict:
        """Check data completeness"""
        pass
    
    def validate_accuracy(self, df: pd.DataFrame) -> Dict:
        """Check data accuracy"""
        pass
    
    def validate_consistency(self, df: pd.DataFrame) -> Dict:
        """Check data consistency"""
        pass
    
    def validate_timeliness(self, df: pd.DataFrame) -> Dict:
        """Check data freshness"""
        pass
```

### 6. Configuration and Utilities

#### 6.1 Configuration Management
**File:** `data_engineering/config/database_config.py`
```python
DATABASE_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', 5432),
    'database': os.getenv('POSTGRES_DB', 'titans_finance'),
    'username': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'password')
}
```

#### 6.2 Database Utilities
**File:** `data_engineering/utils/db_utils.py`
- Connection management
- Query utilities
- Performance monitoring
- Error handling

#### 6.3 Logging and Monitoring
**File:** `data_engineering/utils/logger.py`
- Structured logging
- Performance metrics
- Error tracking
- Pipeline monitoring

### 7. Testing Framework

#### 7.1 Unit Tests
**Directory:** `tests/data_engineering/`
**Files to Create:**
- `test_extractors.py` - CSV extractor tests
- `test_transformers.py` - Transformation logic tests
- `test_loaders.py` - Data loading tests
- `test_data_quality.py` - Quality validation tests

#### 7.2 Integration Tests
**Files to Create:**
- `test_pipeline_integration.py` - End-to-end pipeline tests
- `test_database_integration.py` - Database integration tests
- `test_airflow_dags.py` - DAG validation tests

### 8. Documentation

#### 8.1 Technical Documentation
**Files to Create:**
- `data_engineering/README.md` - Component overview
- `docs/data_pipeline_architecture.md` - Pipeline architecture
- `docs/data_quality_standards.md` - Quality standards
- `docs/troubleshooting_guide.md` - Common issues and solutions

## Implementation Priority

### Phase 1: Core Infrastructure (Week 1)
1. Database schema and initialization scripts
2. Transaction transformer implementation
3. PostgreSQL loader implementation
4. Basic pipeline testing

### Phase 2: Pipeline Automation (Week 2)
1. Airflow DAGs implementation
2. Data quality framework setup
3. Error handling and monitoring
4. Integration testing

### Phase 3: Optimization and Monitoring (Week 3)
1. Performance optimization
2. Advanced data quality checks
3. Monitoring and alerting
4. Documentation completion

## Success Criteria

✅ **Pipeline Functionality:**
- Raw CSV data successfully extracted
- Data transformed and cleaned properly
- All data loaded to PostgreSQL correctly
- Airflow DAGs execute without errors

✅ **Data Quality:**
- 95%+ data quality score
- Comprehensive validation rules implemented
- Quality reports generated automatically
- Data lineage tracking functional

✅ **Performance:**
- Pipeline processes 10K+ records efficiently
- Sub-minute execution time for daily batches
- Resource usage optimized
- Error recovery mechanisms working

✅ **Monitoring:**
- Pipeline status dashboard functional
- Alert system operational
- Performance metrics tracked
- Comprehensive logging implemented

## Dependencies

**External:**
- PostgreSQL database (✅ Running)
- Apache Airflow (✅ Running)
- Redis cache (✅ Running)

**Internal:**
- Transaction data (`data/all_transactions.csv`) (✅ Available)
- Python environment with required packages
- Database permissions and access

## Estimated Effort

- **Total Effort:** 15-20 days
- **Critical Path:** Database schema → Transformers → Loaders → DAGs
- **Team Size:** 1-2 data engineers
- **Risk Level:** Medium (standard ETL implementation)

---

**Next Step:** Begin with database schema creation and transaction transformer implementation as these are foundational for all other components.