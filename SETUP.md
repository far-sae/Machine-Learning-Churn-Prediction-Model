# Setup Guide

This guide walks you through the complete setup process for the Machine Learning Churn Prediction Model system.

## Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 8GB, 16GB recommended
- **Storage**: 10GB free space minimum
- **CPU**: Multi-core processor recommended

### Software Dependencies
- **PostgreSQL**: Version 12 or higher
- **MongoDB**: Version 4.4 or higher
- **Git**: For version control

## Step-by-Step Setup

### 1. Database Setup

#### PostgreSQL Setup

1. Install PostgreSQL:
```bash
# macOS
brew install postgresql

# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# Windows: Download from https://www.postgresql.org/download/
```

2. Create database and tables:
```sql
CREATE DATABASE customer_db;

\c customer_db;

-- Customer table
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    registration_date DATE,
    account_status VARCHAR(20),
    customer_tier VARCHAR(20),
    country VARCHAR(50),
    city VARCHAR(100),
    age INTEGER,
    gender VARCHAR(10),
    referral_source VARCHAR(50),
    last_login_date DATE,
    is_premium BOOLEAN,
    subscription_type VARCHAR(50),
    account_balance DECIMAL(10,2)
);

-- Transactions table
CREATE TABLE transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    transaction_date TIMESTAMP,
    transaction_amount DECIMAL(10,2),
    transaction_type VARCHAR(50),
    payment_method VARCHAR(50),
    product_category VARCHAR(100),
    quantity INTEGER,
    discount_applied DECIMAL(5,2),
    is_refunded BOOLEAN
);

-- Support tickets table
CREATE TABLE support_tickets (
    ticket_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    created_date TIMESTAMP,
    status VARCHAR(20),
    issue_category VARCHAR(100),
    resolution_time_hours DECIMAL(10,2)
);

-- Subscription history table
CREATE TABLE subscription_history (
    subscription_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    subscription_start_date DATE,
    subscription_end_date DATE,
    subscription_type VARCHAR(50),
    monthly_fee DECIMAL(10,2),
    payment_frequency VARCHAR(20),
    auto_renewal_enabled BOOLEAN,
    upgrade_count INTEGER,
    downgrade_count INTEGER,
    cancellation_date DATE,
    reactivation_count INTEGER
);

-- Churn predictions table
CREATE TABLE churn_predictions (
    prediction_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    churn_probability DECIMAL(5,4),
    risk_category VARCHAR(20),
    prediction_date DATE,
    prediction_timestamp TIMESTAMP
);
```

#### MongoDB Setup

1. Install MongoDB:
```bash
# macOS
brew install mongodb-community

# Ubuntu/Debian
sudo apt-get install mongodb

# Start MongoDB
mongod --dbpath /path/to/data
```

2. Create collections:
```javascript
use engagement_db;

db.createCollection("user_events");

// Create indexes
db.user_events.createIndex({ "customer_id": 1 });
db.user_events.createIndex({ "event_timestamp": 1 });
db.user_events.createIndex({ "event_type": 1 });
```

### 2. Python Environment Setup

1. **Clone the repository** (or navigate to project directory):
```bash
cd "Machine Learning Churn Prediction Model"
```

2. **Create virtual environment**:
```bash
python3 -m venv venv
```

3. **Activate virtual environment**:
```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

4. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configuration

1. **Create environment file**:
```bash
cp .env.example .env
```

2. **Edit .env file** with your credentials:
```bash
# Database Credentials
DB_USER=your_postgres_user
DB_PASSWORD=your_postgres_password

# MongoDB
MONGO_CONNECTION_STRING=mongodb://localhost:27017

# CRM Integration
CRM_API_ENDPOINT=https://your-crm-api.com/v1
CRM_API_KEY=your_crm_api_key

# Monitoring & Alerts
ALERT_EMAIL=alerts@yourcompany.com
```

3. **Configure config.yaml** (optional):
- Review and adjust model parameters
- Set scheduling preferences
- Configure data source endpoints

### 4. Directory Structure

The system will automatically create necessary directories, but you can create them manually:

```bash
mkdir -p data/{raw,processed,predictions}
mkdir -p models
mkdir -p logs
mkdir -p mlruns
mkdir -p reports/figures
```

### 5. MLflow Setup

1. **Start MLflow server**:
```bash
python main.py mlflow
```

2. **Access MLflow UI**:
Open browser to: http://localhost:5000

3. **Verify connection**:
- Check that experiments appear in UI
- Verify artifact storage location

### 6. Initial Data Loading

If you have historical data, load it into your databases:

```bash
# Example for PostgreSQL
psql -d customer_db -U your_user -f initial_data.sql

# Example for MongoDB
mongoimport --db engagement_db --collection user_events --file events.json
```

### 7. Test the System

#### Test Data Collection
```bash
python main.py collect
```

Expected output:
- "Data collection completed successfully"
- Data files in `data/raw/`

#### Test Model Training
```bash
python main.py train
```

Expected output:
- Training progress logs
- Model files in `models/`
- MLflow run visible in UI
- Test AUC score logged

#### Test Prediction
```bash
python main.py predict
```

Expected output:
- Predictions saved to database and files
- Prediction summary in `data/predictions/`

### 8. Start Automated Scheduler

```bash
python main.py scheduler
```

The scheduler will run in the foreground. Press Ctrl+C to stop.

For production, run as a background service:

```bash
# Using nohup
nohup python main.py scheduler > scheduler.log 2>&1 &

# Or use systemd (Linux)
sudo systemctl start churn-predictor
```

## Verification Checklist

- [ ] PostgreSQL database created and tables exist
- [ ] MongoDB running and accessible
- [ ] Python virtual environment activated
- [ ] All dependencies installed successfully
- [ ] Environment variables configured in .env
- [ ] MLflow server running and accessible
- [ ] Data collection successful
- [ ] Model training completed with good metrics
- [ ] Predictions generated successfully
- [ ] Scheduler running and jobs scheduled

## Troubleshooting

### Database Connection Issues

**PostgreSQL connection error:**
```bash
# Test connection
psql -h localhost -U your_user -d customer_db

# Check if PostgreSQL is running
pg_isready
```

**MongoDB connection error:**
```bash
# Test connection
mongo

# Check if MongoDB is running
ps aux | grep mongod
```

### Python Package Issues

**If pip install fails:**
```bash
# Update pip
pip install --upgrade pip setuptools wheel

# Install packages one by one
pip install numpy pandas scikit-learn
pip install xgboost mlflow
```

### MLflow Issues

**If MLflow server won't start:**
```bash
# Check port availability
lsof -i :5000

# Use different port
mlflow server --port 5001
```

## Production Deployment

For production deployment:

1. **Use proper database hosting** (AWS RDS, MongoDB Atlas)
2. **Set up monitoring** (Prometheus, Grafana)
3. **Configure log aggregation** (ELK stack, CloudWatch)
4. **Use process manager** (systemd, supervisord)
5. **Set up SSL/TLS** for API endpoints
6. **Implement backup strategy** for models and data
7. **Configure alerts** for failures and anomalies

## Next Steps

Once setup is complete:

1. Review and adjust model parameters in `config.yaml`
2. Set up monitoring dashboards
3. Configure CRM integration endpoints
4. Test campaign triggering with sample data
5. Set up backup and recovery procedures
6. Document custom configurations
7. Train team on system usage

## Support

For setup issues, check:
- Logs in `logs/churn_prediction.log`
- MLflow UI for experiment details
- Database logs for connection issues

Contact the ML team for additional support.
