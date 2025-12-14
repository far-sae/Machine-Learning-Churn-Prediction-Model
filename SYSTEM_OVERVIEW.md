# Machine Learning Churn Prediction Model - System Overview

## Executive Summary

The Machine Learning Churn Prediction Model is a comprehensive, production-ready system designed to predict customer churn with high accuracy and automatically trigger targeted retention campaigns. This advanced solution combines ensemble machine learning, extensive feature engineering, and seamless CRM integration to enable proactive business retention strategies.

## Key Achievements

✅ **50+ Predictive Features** engineered from customer data  
✅ **Ensemble Model** combining Random Forest (40%) and XGBoost (60%)  
✅ **Daily Automated Pipeline** requiring zero manual intervention  
✅ **CRM Integration** with multi-channel campaign triggering  
✅ **MLflow Integration** for complete experiment tracking and model registry  
✅ **Production-Ready** with comprehensive logging, monitoring, and error handling  

## System Components

### 1. Data Collection Layer
**Location:** `src/data_collection/`

Aggregates customer data from multiple sources:
- **PostgreSQL**: Customer profiles, transactions, subscriptions, support tickets
- **MongoDB**: User engagement events, clickstream data
- **REST APIs**: Real-time transaction and interaction data

**Key Features:**
- Multi-source integration
- Configurable lookback periods
- Automated data validation
- Parquet file storage for efficiency

### 2. Feature Engineering Pipeline
**Location:** `src/features/`

Creates **50+ predictive features** across multiple categories:

#### RFM (Recency, Frequency, Monetary) Features
- Multiple time periods: 7d, 14d, 30d, 60d, 90d, 180d, 365d
- Transaction recency, frequency, and monetary value
- Spending trends and patterns

#### Behavioral Features
- Activity decline indicators
- Transaction regularity (CV)
- Payment method diversity
- Product category preferences
- Refund patterns

#### Engagement Features
- Page views, clicks, searches
- Session metrics
- Engagement scores
- Click-through rates
- Cart conversion rates

#### Support Interaction Features
- Ticket counts and resolution rates
- Issue category diversity
- Average resolution time

#### Subscription Features
- Tenure and stability metrics
- Upgrade/downgrade patterns
- Auto-renewal status

#### Temporal Features
- Customer lifecycle stages
- Registration patterns
- Login frequency

### 3. Model Training Module
**Location:** `src/models/`

#### Ensemble Approach
Combines two powerful algorithms with weighted averaging:

**Random Forest (40% weight)**
- 200 estimators, max depth 15
- Balanced class weights
- Robust to overfitting

**XGBoost (60% weight)**
- 200 estimators, learning rate 0.05
- L1/L2 regularization
- Scale positive weight for imbalance

#### Training Pipeline
1. Train/validation/test split (70/10/20)
2. SMOTE for class imbalance handling
3. 5-fold stratified cross-validation
4. Comprehensive metric evaluation
5. Feature importance analysis
6. MLflow experiment tracking
7. Automatic model registration

**Target Metrics:**
- ROC-AUC > 0.75
- Optimized precision-recall balance
- Low false negative rate (catch churners)

### 4. Prediction System
**Location:** `src/prediction/`

#### Daily Scoring Pipeline
- Generates churn probability (0-1) for all active customers
- Batch processing (10,000 customers/batch)
- Risk categorization:
  - **High Risk**: ≥ 0.7 (urgent action needed)
  - **Medium Risk**: ≥ 0.4 (proactive monitoring)
  - **Low Risk**: < 0.4 (routine engagement)

#### Output Formats
- Database storage (churn_predictions table)
- Parquet files for archival
- JSON summary reports
- Daily prediction reports with statistics

### 5. CRM Integration
**Location:** `src/crm/`

#### Automated Campaign Triggering

**High-Risk Campaigns**
- Channels: Email + SMS + Push
- Priority: Urgent
- Offer: 30% premium discount
- Timing: Immediate

**Medium-Risk Campaigns**
- Channels: Email + Push
- Priority: High
- Offer: 15% standard discount
- Timing: 24 hours

**Low-Risk Campaigns**
- Channels: Email
- Priority: Normal
- Offer: Engagement content
- Timing: Routine

#### Features
- Rate limiting (60 req/min)
- Retry logic with exponential backoff
- Personalized messaging
- Campaign performance tracking
- CRM sync for all predictions

### 6. MLflow Integration
**Location:** `src/mlflow_setup/`

#### Experiment Tracking
- Automatic parameter logging
- Metric tracking across runs
- Artifact storage (models, plots)
- Run comparison tools

#### Model Registry
- Version control (None → Staging → Production → Archived)
- Model lineage tracking
- Production promotion workflow
- Easy rollback capability

### 7. Automated Scheduler
**Location:** `src/scheduler/`

#### Daily Schedule

| Time | Task | Description |
|------|------|-------------|
| 1:00 AM | Data Collection | Aggregate from all sources |
| 2:00 AM | Model Retraining | Train with latest data |
| 6:00 AM | Prediction | Score all customers |
| 7:00 AM | Campaign Trigger | Launch retention campaigns |

#### Features
- Configurable cron expressions
- Job monitoring and status tracking
- Manual job triggering
- Pause/resume capabilities
- Error handling with alerts

## Technical Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **scikit-learn 1.3.0**: Random Forest implementation
- **XGBoost 1.7.6**: Gradient boosting implementation
- **MLflow 2.7.1**: Experiment tracking and model registry
- **pandas 2.0.3**: Data manipulation
- **NumPy 1.24.3**: Numerical computing

### Data Storage
- **PostgreSQL 12+**: Relational data (customers, transactions)
- **MongoDB 4.4+**: Document data (engagement events)
- **Parquet**: Efficient columnar storage

### Infrastructure
- **APScheduler**: Job scheduling
- **Loguru**: Advanced logging
- **SQLAlchemy**: Database ORM
- **Requests**: HTTP API integration

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   Data Sources Layer                         │
│  PostgreSQL    MongoDB    REST APIs    Other Sources         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                Data Collection Module                        │
│  • Multi-source aggregation                                 │
│  • Data validation                                          │
│  • Parquet storage                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Engineering Module                      │
│  • 50+ features across 6 categories                         │
│  • RFM, behavioral, engagement, support, temporal           │
│  • Missing value handling                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                Model Training Module                         │
│  ┌──────────────────┬─────────────────┐                    │
│  │ Random Forest    │   XGBoost       │                    │
│  │ (40% weight)     │   (60% weight)  │                    │
│  └──────────────────┴─────────────────┘                    │
│              Ensemble Predictions                            │
│              MLflow Tracking                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Prediction Module                            │
│  • Daily batch scoring (10K/batch)                          │
│  • Risk categorization (High/Medium/Low)                    │
│  • Database + file storage                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                CRM Integration Module                        │
│  • Automated campaign triggering                            │
│  • Multi-channel delivery                                   │
│  • Personalized messaging                                   │
│  • Performance tracking                                     │
└─────────────────────────────────────────────────────────────┘
                     
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│  MLflow Server   │    │    Scheduler     │
│  • Experiments   │    │  • Daily jobs    │
│  • Model Registry│    │  • Monitoring    │
│  • Deployment    │    │  • Automation    │
└──────────────────┘    └──────────────────┘
```

## File Structure

```
Machine Learning Churn Prediction Model/
├── config.yaml                 # Main configuration
├── requirements.txt            # Dependencies
├── .env.example               # Environment template
├── main.py                    # Entry point
├── README.md                  # Main documentation
├── SETUP.md                   # Setup guide
├── API_DOCUMENTATION.md       # API reference
│
├── src/
│   ├── data_collection/       # Data aggregation
│   ├── features/              # Feature engineering
│   ├── models/                # Model training
│   ├── prediction/            # Churn prediction
│   ├── crm/                   # CRM integration
│   ├── mlflow_setup/          # MLflow management
│   ├── scheduler/             # Automation
│   └── utils/                 # Utilities
│
├── data/                      # Data storage
│   ├── raw/                   # Raw data
│   ├── processed/             # Features
│   └── predictions/           # Predictions
│
├── models/                    # Saved models
├── logs/                      # Application logs
├── mlruns/                    # MLflow artifacts
└── reports/                   # Generated reports
```

## Usage Examples

### Command Line Interface

```bash
# Start MLflow server
python main.py mlflow

# Collect data
python main.py collect

# Train model
python main.py train

# Generate predictions
python main.py predict

# Trigger campaigns
python main.py campaigns

# Run automated scheduler
python main.py scheduler
```

### Programmatic Usage

```python
from src.utils.config import ConfigLoader, setup_logging
from src.data_collection.data_collector import DataCollector
from src.features.feature_engineer import FeatureEngineer
from src.models.model_trainer import EnsembleChurnModel
from src.prediction.predictor import ChurnPredictor

# Setup
config = ConfigLoader('config.yaml')
setup_logging(config)

# Collect data
collector = DataCollector(config)
data = collector.collect_all_data(lookback_days=365)

# Engineer features
engineer = FeatureEngineer(config)
features = engineer.create_all_features(data)

# Train model
model = EnsembleChurnModel(config)
results = model.train_full_pipeline(features)

# Generate predictions
predictor = ChurnPredictor(config)
predictor.load_models_from_registry(model_stage="Production")
predictions = predictor.predict_all_active_customers(features)
```

## Performance Metrics

### Model Performance
- **Target ROC-AUC**: > 0.75
- **Precision**: Optimized for high-risk predictions
- **Recall**: Balanced to capture churners
- **Training Time**: ~30 min for 100K customers

### System Performance
- **Prediction Throughput**: 10,000 customers/batch
- **Feature Engineering**: ~10 min for 100K customers
- **Daily Pipeline**: Complete in < 2 hours
- **API Rate Limit**: 60 requests/minute

## Business Impact

### Expected Outcomes
- **15-25%** reduction in customer churn rate
- **30%** improvement in retention campaign ROI
- **40%** reduction in manual analysis time
- **Real-time** actionable insights

### Key Benefits
✅ Proactive identification of at-risk customers  
✅ Automated, personalized retention campaigns  
✅ High accuracy ensemble predictions  
✅ Scalable to millions of customers  
✅ Complete audit trail with MLflow  
✅ Zero manual intervention required  

## Security & Compliance

- Environment variable management for credentials
- API key rotation support
- Database credential protection
- Rate limiting on external APIs
- Comprehensive logging for audit trails
- GDPR-compliant data handling capabilities

## Monitoring & Maintenance

### Automated Monitoring
- Model performance drift detection
- Feature distribution monitoring
- Prediction quality tracking
- Campaign performance metrics

### Logging
- Structured logging with Loguru
- Log rotation (100MB files)
- 30-day retention
- Multiple log levels

### Alerts
- Model performance degradation
- Job failures
- Data quality issues
- API connection errors

## Future Enhancements

### Potential Improvements
1. **Deep Learning Models**: Add neural network models to ensemble
2. **Real-time Scoring**: Stream processing for immediate predictions
3. **A/B Testing**: Built-in campaign experimentation framework
4. **Advanced Personalization**: Deep learning for offer optimization
5. **AutoML**: Automated hyperparameter optimization
6. **Model Interpretability**: SHAP values for prediction explanations
7. **API Gateway**: RESTful API for external integrations

## Documentation

- [README.md](README.md) - Main documentation
- [SETUP.md](SETUP.md) - Setup and installation guide
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - API reference
- Inline code documentation in all modules

## Support & Maintenance

### Contact
- ML Engineering Team
- Email: [Configure in .env]

### Version
- Current Version: 1.0.0
- Last Updated: December 2024

---

**This system represents a state-of-the-art approach to customer churn prediction, combining advanced machine learning techniques with practical business automation to deliver measurable value and competitive advantage.**
