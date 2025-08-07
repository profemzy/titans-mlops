-- Titans Finance Database Schema
-- This script creates the complete database schema for the titans finance project

-- Create main schema
CREATE SCHEMA IF NOT EXISTS titans_finance;

-- Create extension for generating UUIDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================
-- RAW TRANSACTIONS TABLE
-- =============================================
DROP TABLE IF EXISTS titans_finance.raw_transactions CASCADE;
CREATE TABLE titans_finance.raw_transactions (
    id SERIAL PRIMARY KEY,
    transaction_uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    
    -- Original CSV fields
    date DATE NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN ('income', 'expense')),
    description TEXT,
    amount DECIMAL(15,2) NOT NULL,
    category VARCHAR(100),
    payment_method VARCHAR(50),
    status VARCHAR(20) DEFAULT 'completed',
    reference VARCHAR(100),
    receipt_url TEXT,
    
    -- ETL metadata
    source_file VARCHAR(255),
    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_quality_score DECIMAL(5,4),
    validation_errors TEXT[],
    
    -- Auditing
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT check_amount_not_zero CHECK (amount != 0),
    CONSTRAINT check_date_reasonable CHECK (date >= '2000-01-01' AND date <= CURRENT_DATE + INTERVAL '1 year')
);

-- =============================================
-- PROCESSED TRANSACTIONS TABLE
-- =============================================
DROP TABLE IF EXISTS titans_finance.processed_transactions CASCADE;
CREATE TABLE titans_finance.processed_transactions (
    id SERIAL PRIMARY KEY,
    transaction_uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    raw_transaction_id INTEGER REFERENCES titans_finance.raw_transactions(id),
    
    -- Cleaned and standardized fields
    date DATE NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN ('income', 'expense')),
    description_original TEXT,
    description_cleaned TEXT,
    amount DECIMAL(15,2) NOT NULL,
    amount_abs DECIMAL(15,2) NOT NULL,
    
    -- Categorization (original and predicted)
    category_original VARCHAR(100),
    category_predicted VARCHAR(100),
    category_confidence DECIMAL(5,4),
    
    -- Payment method standardization
    payment_method_original VARCHAR(50),
    payment_method_standardized VARCHAR(50),
    
    -- Status and reference
    status VARCHAR(20) DEFAULT 'completed',
    reference VARCHAR(100),
    receipt_url TEXT,
    
    -- =============================================
    -- FEATURE ENGINEERING FIELDS
    -- =============================================
    
    -- Time-based features
    day_of_week INTEGER CHECK (day_of_week BETWEEN 0 AND 6), -- 0=Monday, 6=Sunday
    month INTEGER CHECK (month BETWEEN 1 AND 12),
    quarter INTEGER CHECK (quarter BETWEEN 1 AND 4),
    year INTEGER,
    is_weekend BOOLEAN,
    is_month_end BOOLEAN,
    is_month_start BOOLEAN,
    week_of_year INTEGER CHECK (week_of_year BETWEEN 1 AND 53),
    days_since_epoch INTEGER,
    
    -- Amount-based features
    amount_category VARCHAR(20) CHECK (amount_category IN ('small', 'medium', 'large', 'very_large')),
    amount_log DECIMAL(10,6),
    amount_percentile DECIMAL(5,4),
    
    -- Sequential features
    days_since_last_transaction INTEGER,
    transaction_sequence INTEGER,
    
    -- Rolling statistics (will be calculated by transformation pipeline)
    rolling_avg_7d DECIMAL(15,2),
    rolling_avg_30d DECIMAL(15,2),
    rolling_std_7d DECIMAL(15,2),
    rolling_sum_7d DECIMAL(15,2),
    
    -- Categorical encoding features
    merchant_category VARCHAR(100),
    location_category VARCHAR(100),
    
    -- Business logic features
    is_recurring BOOLEAN DEFAULT FALSE,
    recurring_pattern VARCHAR(50), -- 'weekly', 'monthly', 'quarterly', etc.
    
    -- Data quality and processing metadata
    processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feature_engineering_version VARCHAR(20),
    data_quality_score DECIMAL(5,4),
    anomaly_score DECIMAL(5,4),
    
    -- Auditing
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT check_processed_amount_not_zero CHECK (amount != 0),
    CONSTRAINT check_processed_date_reasonable CHECK (date >= '2000-01-01' AND date <= CURRENT_DATE + INTERVAL '1 year'),
    CONSTRAINT check_amount_abs_positive CHECK (amount_abs >= 0)
);

-- =============================================
-- TRANSACTION METRICS TABLE (Aggregated data)
-- =============================================
DROP TABLE IF EXISTS titans_finance.transaction_metrics CASCADE;
CREATE TABLE titans_finance.transaction_metrics (
    id SERIAL PRIMARY KEY,
    
    -- Time dimensions
    date DATE NOT NULL,
    week_start_date DATE,
    month_start_date DATE,
    quarter_start_date DATE,
    year INTEGER,
    
    -- Overall metrics
    total_transactions INTEGER NOT NULL DEFAULT 0,
    total_amount DECIMAL(15,2) NOT NULL DEFAULT 0,
    avg_amount DECIMAL(15,2),
    median_amount DECIMAL(15,2),
    min_amount DECIMAL(15,2),
    max_amount DECIMAL(15,2),
    std_amount DECIMAL(15,2),
    
    -- By transaction type
    income_count INTEGER DEFAULT 0,
    expense_count INTEGER DEFAULT 0,
    income_amount DECIMAL(15,2) DEFAULT 0,
    expense_amount DECIMAL(15,2) DEFAULT 0,
    net_amount DECIMAL(15,2) DEFAULT 0, -- income - expense
    
    -- By category (top categories only - stored as JSON)
    category_distribution JSONB,
    top_category VARCHAR(100),
    top_category_amount DECIMAL(15,2),
    
    -- By payment method
    payment_method_distribution JSONB,
    top_payment_method VARCHAR(50),
    
    -- Quality metrics
    data_quality_score DECIMAL(5,4),
    anomaly_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    
    -- Metadata
    calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(date) -- One record per date
);

-- =============================================
-- DATA QUALITY REPORTS TABLE
-- =============================================
DROP TABLE IF EXISTS titans_finance.data_quality_reports CASCADE;
CREATE TABLE titans_finance.data_quality_reports (
    id SERIAL PRIMARY KEY,
    
    -- Report metadata
    execution_date TIMESTAMP NOT NULL,
    report_type VARCHAR(50) NOT NULL, -- 'daily', 'weekly', 'monthly', 'pipeline_run'
    pipeline_name VARCHAR(100),
    data_source VARCHAR(100),
    
    -- Record counts
    total_records INTEGER NOT NULL DEFAULT 0,
    passed_records INTEGER NOT NULL DEFAULT 0,
    failed_records INTEGER NOT NULL DEFAULT 0,
    warning_records INTEGER DEFAULT 0,
    
    -- Quality scores
    overall_quality_score DECIMAL(5,4) NOT NULL,
    completeness_score DECIMAL(5,4),
    accuracy_score DECIMAL(5,4),
    consistency_score DECIMAL(5,4),
    validity_score DECIMAL(5,4),
    
    -- Specific checks
    null_value_percentage DECIMAL(5,4),
    duplicate_percentage DECIMAL(5,4),
    outlier_percentage DECIMAL(5,4),
    format_error_percentage DECIMAL(5,4),
    
    -- Issues detected (stored as JSON array)
    issues_detected JSONB,
    critical_issues TEXT[],
    warnings TEXT[],
    recommendations TEXT[],
    
    -- Performance metrics
    processing_time_seconds INTEGER,
    memory_usage_mb DECIMAL(10,2),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- PIPELINE EXECUTION LOG TABLE
-- =============================================
DROP TABLE IF EXISTS titans_finance.pipeline_execution_log CASCADE;
CREATE TABLE titans_finance.pipeline_execution_log (
    id SERIAL PRIMARY KEY,
    
    -- Execution metadata
    execution_id UUID DEFAULT uuid_generate_v4() UNIQUE,
    pipeline_name VARCHAR(100) NOT NULL,
    execution_date TIMESTAMP NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('running', 'success', 'failed', 'warning')),
    
    -- Execution details
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_seconds INTEGER,
    
    -- Data processing metrics
    records_extracted INTEGER DEFAULT 0,
    records_transformed INTEGER DEFAULT 0,
    records_loaded INTEGER DEFAULT 0,
    records_rejected INTEGER DEFAULT 0,
    
    -- Error tracking
    error_message TEXT,
    error_details JSONB,
    stack_trace TEXT,
    
    -- Performance metrics
    memory_peak_mb DECIMAL(10,2),
    cpu_usage_percent DECIMAL(5,2),
    
    -- Configuration
    pipeline_config JSONB,
    pipeline_version VARCHAR(20),
    
    -- Auditing
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- INDEXES FOR PERFORMANCE
-- =============================================

-- Raw transactions indexes
CREATE INDEX idx_raw_transactions_date ON titans_finance.raw_transactions(date);
CREATE INDEX idx_raw_transactions_type ON titans_finance.raw_transactions(type);
CREATE INDEX idx_raw_transactions_category ON titans_finance.raw_transactions(category);
CREATE INDEX idx_raw_transactions_ingestion_timestamp ON titans_finance.raw_transactions(ingestion_timestamp);
CREATE INDEX idx_raw_transactions_uuid ON titans_finance.raw_transactions(transaction_uuid);

-- Processed transactions indexes
CREATE INDEX idx_processed_transactions_date ON titans_finance.processed_transactions(date);
CREATE INDEX idx_processed_transactions_type ON titans_finance.processed_transactions(type);
CREATE INDEX idx_processed_transactions_category_predicted ON titans_finance.processed_transactions(category_predicted);
CREATE INDEX idx_processed_transactions_amount ON titans_finance.processed_transactions(amount);
CREATE INDEX idx_processed_transactions_raw_id ON titans_finance.processed_transactions(raw_transaction_id);
CREATE INDEX idx_processed_transactions_uuid ON titans_finance.processed_transactions(transaction_uuid);
CREATE INDEX idx_processed_transactions_day_of_week ON titans_finance.processed_transactions(day_of_week);
CREATE INDEX idx_processed_transactions_month ON titans_finance.processed_transactions(month);
CREATE INDEX idx_processed_transactions_is_weekend ON titans_finance.processed_transactions(is_weekend);

-- Transaction metrics indexes
CREATE INDEX idx_transaction_metrics_date ON titans_finance.transaction_metrics(date);
CREATE INDEX idx_transaction_metrics_month_start_date ON titans_finance.transaction_metrics(month_start_date);
CREATE INDEX idx_transaction_metrics_year ON titans_finance.transaction_metrics(year);

-- Data quality reports indexes
CREATE INDEX idx_data_quality_reports_execution_date ON titans_finance.data_quality_reports(execution_date);
CREATE INDEX idx_data_quality_reports_report_type ON titans_finance.data_quality_reports(report_type);
CREATE INDEX idx_data_quality_reports_pipeline_name ON titans_finance.data_quality_reports(pipeline_name);

-- Pipeline execution log indexes
CREATE INDEX idx_pipeline_execution_log_execution_date ON titans_finance.pipeline_execution_log(execution_date);
CREATE INDEX idx_pipeline_execution_log_pipeline_name ON titans_finance.pipeline_execution_log(pipeline_name);
CREATE INDEX idx_pipeline_execution_log_status ON titans_finance.pipeline_execution_log(status);
CREATE INDEX idx_pipeline_execution_log_execution_id ON titans_finance.pipeline_execution_log(execution_id);

-- =============================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
-- =============================================

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables with updated_at columns
CREATE TRIGGER update_raw_transactions_updated_at
    BEFORE UPDATE ON titans_finance.raw_transactions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_processed_transactions_updated_at
    BEFORE UPDATE ON titans_finance.processed_transactions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_transaction_metrics_updated_at
    BEFORE UPDATE ON titans_finance.transaction_metrics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_pipeline_execution_log_updated_at
    BEFORE UPDATE ON titans_finance.pipeline_execution_log
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================
-- VIEWS FOR COMMON QUERIES
-- =============================================

-- View for recent transactions with all relevant data
CREATE OR REPLACE VIEW titans_finance.v_recent_transactions AS
SELECT 
    p.id,
    p.transaction_uuid,
    p.date,
    p.type,
    p.description_cleaned,
    p.amount,
    p.category_predicted,
    p.category_confidence,
    p.payment_method_standardized,
    p.day_of_week,
    p.month,
    p.is_weekend,
    p.amount_category,
    p.anomaly_score,
    p.created_at
FROM titans_finance.processed_transactions p
WHERE p.date >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY p.date DESC, p.created_at DESC;

-- View for transaction summaries by month
CREATE OR REPLACE VIEW titans_finance.v_monthly_summary AS
SELECT 
    DATE_TRUNC('month', date) as month,
    COUNT(*) as transaction_count,
    SUM(CASE WHEN type = 'income' THEN amount ELSE 0 END) as total_income,
    SUM(CASE WHEN type = 'expense' THEN amount ELSE 0 END) as total_expense,
    SUM(CASE WHEN type = 'income' THEN amount ELSE -amount END) as net_amount,
    AVG(amount) as avg_amount,
    COUNT(DISTINCT category_predicted) as unique_categories
FROM titans_finance.processed_transactions
GROUP BY DATE_TRUNC('month', date)
ORDER BY month DESC;

-- View for data quality dashboard
CREATE OR REPLACE VIEW titans_finance.v_data_quality_dashboard AS
SELECT 
    DATE(execution_date) as report_date,
    report_type,
    pipeline_name,
    overall_quality_score,
    completeness_score,
    accuracy_score,
    total_records,
    failed_records,
    ROUND((failed_records::numeric / NULLIF(total_records, 0)) * 100, 2) as failure_percentage,
    array_length(critical_issues, 1) as critical_issue_count
FROM titans_finance.data_quality_reports
WHERE execution_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY execution_date DESC;

-- =============================================
-- SAMPLE DATA VALIDATION FUNCTION
-- =============================================

CREATE OR REPLACE FUNCTION titans_finance.validate_transaction_data()
RETURNS TABLE(
    validation_result TEXT,
    issue_count INTEGER,
    details TEXT
) AS $$
BEGIN
    -- Check for null required fields
    RETURN QUERY
    SELECT 
        'NULL_REQUIRED_FIELDS'::TEXT as validation_result,
        COUNT(*)::INTEGER as issue_count,
        'Transactions with null values in required fields (date, type, amount)'::TEXT as details
    FROM titans_finance.raw_transactions 
    WHERE date IS NULL OR type IS NULL OR amount IS NULL;
    
    -- Check for invalid transaction types
    RETURN QUERY
    SELECT 
        'INVALID_TRANSACTION_TYPES'::TEXT as validation_result,
        COUNT(*)::INTEGER as issue_count,
        'Transactions with invalid type values (not income or expense)'::TEXT as details
    FROM titans_finance.raw_transactions 
    WHERE type NOT IN ('income', 'expense');
    
    -- Check for zero or unreasonable amounts
    RETURN QUERY
    SELECT 
        'INVALID_AMOUNTS'::TEXT as validation_result,
        COUNT(*)::INTEGER as issue_count,
        'Transactions with zero or unreasonable amounts'::TEXT as details
    FROM titans_finance.raw_transactions 
    WHERE amount = 0 OR ABS(amount) > 1000000;
    
    -- Check for future dates
    RETURN QUERY
    SELECT 
        'FUTURE_DATES'::TEXT as validation_result,
        COUNT(*)::INTEGER as issue_count,
        'Transactions with future dates'::TEXT as details
    FROM titans_finance.raw_transactions 
    WHERE date > CURRENT_DATE;
    
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- COMPLETION MESSAGE
-- =============================================

-- Create a simple function to verify schema creation
CREATE OR REPLACE FUNCTION titans_finance.verify_schema_creation()
RETURNS TEXT AS $$
DECLARE
    table_count INTEGER;
    index_count INTEGER;
    view_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO table_count 
    FROM information_schema.tables 
    WHERE table_schema = 'titans_finance';
    
    SELECT COUNT(*) INTO index_count
    FROM pg_indexes
    WHERE schemaname = 'titans_finance';
    
    SELECT COUNT(*) INTO view_count
    FROM information_schema.views
    WHERE table_schema = 'titans_finance';
    
    RETURN format('✅ Schema creation completed successfully! Created %s tables, %s indexes, and %s views.', 
                  table_count, index_count, view_count);
END;
$$ LANGUAGE plpgsql;

-- Run verification
SELECT titans_finance.verify_schema_creation();