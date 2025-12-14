# API Documentation

This document describes the APIs and interfaces available in the Machine Learning Churn Prediction Model system.

## Table of Contents
- [Data Collection API](#data-collection-api)
- [Feature Engineering API](#feature-engineering-api)
- [Model Training API](#model-training-api)
- [Prediction API](#prediction-api)
- [CRM Integration API](#crm-integration-api)
- [MLflow Management API](#mlflow-management-api)
- [Scheduler API](#scheduler-api)

---

## Data Collection API

### DataCollector Class

**Module:** `src.data_collection.data_collector`

#### Methods

##### `collect_all_data(lookback_days: int = 365) -> Dict[str, pd.DataFrame]`
Collects all customer data from configured sources.

**Parameters:**
- `lookback_days` (int): Number of days to look back for historical data

**Returns:**
- Dictionary containing DataFrames: `customers`, `transactions`, `engagement`, `support`, `subscriptions`

**Example:**
```python
from src.data_collection.data_collector import DataCollector

collector = DataCollector(config)
data = collector.collect_all_data(lookback_days=180)

print(f"Customers: {len(data['customers'])}")
print(f"Transactions: {len(data['transactions'])}")
```

##### `save_raw_data(data: Dict[str, pd.DataFrame], output_dir: str = 'data/raw')`
Saves collected data to disk in Parquet format.

**Parameters:**
- `data`: Dictionary of DataFrames to save
- `output_dir`: Directory path for saving

---

## Feature Engineering API

### FeatureEngineer Class

**Module:** `src.features.feature_engineer`

#### Methods

##### `create_all_features(data: Dict[str, pd.DataFrame]) -> pd.DataFrame`
Creates 50+ predictive features from raw data.

**Parameters:**
- `data`: Dictionary containing raw data DataFrames

**Returns:**
- DataFrame with engineered features (50+ columns)

**Feature Categories:**
- RFM metrics (Recency, Frequency, Monetary)
- Transaction behavior
- Engagement patterns
- Support interactions
- Subscription dynamics
- Temporal features
- Behavioral indicators

**Example:**
```python
from src.features.feature_engineer import FeatureEngineer

engineer = FeatureEngineer(config)
features_df = engineer.create_all_features(data)

print(f"Total features: {len(features_df.columns)}")
print(features_df.head())
```

##### `get_feature_importance_groups() -> Dict[str, List[str]]`
Returns feature groups for analysis.

**Returns:**
- Dictionary mapping group names to feature lists

---

## Model Training API

### EnsembleChurnModel Class

**Module:** `src.models.model_trainer`

#### Methods

##### `train_full_pipeline(features_df: pd.DataFrame, target_column: str = 'churned') -> Dict[str, Any]`
Executes complete training pipeline with MLflow tracking.

**Parameters:**
- `features_df`: DataFrame with features and target
- `target_column`: Name of target column (default: 'churned')

**Returns:**
- Dictionary containing:
  - `val_metrics`: Validation set metrics
  - `test_metrics`: Test set metrics
  - `cv_results`: Cross-validation results
  - `feature_importance`: Feature importance DataFrame

**Example:**
```python
from src.models.model_trainer import EnsembleChurnModel

model = EnsembleChurnModel(config)
results = model.train_full_pipeline(features_df)

print(f"Test AUC: {results['test_metrics']['roc_auc']:.4f}")
print(f"Test F1: {results['test_metrics']['f1_score']:.4f}")
```

##### `predict_ensemble(X: pd.DataFrame) -> np.ndarray`
Make predictions using weighted ensemble.

**Parameters:**
- `X`: Features DataFrame

**Returns:**
- Array of probability predictions

##### `get_feature_importance() -> pd.DataFrame`
Get combined feature importance from both models.

**Returns:**
- DataFrame with columns: `feature`, `rf_importance`, `xgb_importance`, `avg_importance`

---

## Prediction API

### ChurnPredictor Class

**Module:** `src.prediction.predictor`

#### Methods

##### `load_models_from_registry(model_stage: str = "Production") -> bool`
Load models from MLflow Model Registry.

**Parameters:**
- `model_stage`: Model stage to load (Production, Staging, None)

**Returns:**
- True if successful, False otherwise

##### `predict_all_active_customers(features_df: pd.DataFrame) -> pd.DataFrame`
Generate predictions for all active customers.

**Parameters:**
- `features_df`: DataFrame with customer features

**Returns:**
- DataFrame with columns:
  - `customer_id`
  - `churn_probability` (0-1)
  - `risk_category` (high/medium/low)
  - `prediction_date`
  - `prediction_timestamp`

**Example:**
```python
from src.prediction.predictor import ChurnPredictor

predictor = ChurnPredictor(config)
predictor.load_models_from_registry(model_stage="Production")

predictions = predictor.predict_all_active_customers(features_df)

# Get high-risk customers
high_risk = predictions[predictions['risk_category'] == 'high']
print(f"High-risk customers: {len(high_risk)}")
```

##### `save_predictions(predictions: pd.DataFrame, output_format: str = 'both')`
Save predictions to database and/or file.

**Parameters:**
- `predictions`: Predictions DataFrame
- `output_format`: 'database', 'file', or 'both'

##### `generate_daily_report(predictions: pd.DataFrame) -> Dict`
Generate daily prediction report with statistics.

**Returns:**
- Dictionary with report data including risk distribution and top customers

---

## CRM Integration API

### CRMIntegration Class

**Module:** `src.crm.crm_integration`

#### Methods

##### `create_retention_campaign(customer_id: str, risk_category: str, churn_probability: float, additional_data: Optional[Dict] = None) -> Dict`
Create a retention campaign for a customer.

**Parameters:**
- `customer_id`: Customer identifier
- `risk_category`: Risk level (high/medium/low)
- `churn_probability`: Predicted churn probability
- `additional_data`: Additional data for personalization

**Returns:**
- Campaign creation response

**Example:**
```python
from src.crm.crm_integration import CRMIntegration

crm = CRMIntegration(config)

response = crm.create_retention_campaign(
    customer_id="CUST123",
    risk_category="high",
    churn_probability=0.85
)

print(f"Campaign status: {response['status']}")
```

##### `trigger_high_risk_campaigns(predictions: pd.DataFrame) -> pd.DataFrame`
Trigger campaigns for all high-risk customers.

**Parameters:**
- `predictions`: Predictions DataFrame

**Returns:**
- DataFrame with campaign results

##### `sync_churn_predictions_to_crm(predictions: pd.DataFrame) -> Dict`
Synchronize all predictions to CRM system.

**Parameters:**
- `predictions`: Predictions DataFrame

**Returns:**
- Sync summary with success/failure counts

---

## MLflow Management API

### MLflowManager Class

**Module:** `src.mlflow_setup.mlflow_manager`

#### Methods

##### `start_mlflow_server(host: str = '0.0.0.0', port: int = 5000, backend_store_uri: Optional[str] = None, default_artifact_root: Optional[str] = None)`
Start MLflow tracking server.

**Parameters:**
- `host`: Server host address
- `port`: Server port
- `backend_store_uri`: Backend store URI
- `default_artifact_root`: Artifact storage location

##### `get_best_run(experiment_name: Optional[str] = None, metric: str = 'test_roc_auc', ascending: bool = False) -> Dict`
Get the best run from an experiment.

**Parameters:**
- `experiment_name`: Experiment name
- `metric`: Metric to optimize
- `ascending`: Whether lower is better

**Returns:**
- Dictionary with run information

**Example:**
```python
from src.mlflow_setup.mlflow_manager import MLflowManager

mlflow_mgr = MLflowManager(config)
best_run = mlflow_mgr.get_best_run(metric='test_roc_auc')

print(f"Best run ID: {best_run['run_id']}")
print(f"ROC-AUC: {best_run['metrics']['test_roc_auc']:.4f}")
```

##### `promote_model_to_production(model_name: Optional[str] = None, version: Optional[int] = None)`
Promote a model version to production stage.

**Parameters:**
- `model_name`: Model name in registry
- `version`: Specific version to promote (uses latest if None)

##### `get_production_model(model_name: Optional[str] = None) -> Dict`
Get current production model information.

**Returns:**
- Dictionary with model metadata

---

## Scheduler API

### ChurnPredictionScheduler Class

**Module:** `src.scheduler.scheduler`

#### Methods

##### `schedule_jobs()`
Schedule all automated jobs based on configuration.

**Example:**
```python
from src.scheduler.scheduler import ChurnPredictionScheduler

scheduler = ChurnPredictionScheduler(config)
scheduler.schedule_jobs()
scheduler.start()
```

##### `add_custom_job(job_func: Callable, cron_expression: str, job_id: str, job_name: str)`
Add a custom scheduled job.

**Parameters:**
- `job_func`: Function to execute
- `cron_expression`: Cron expression for scheduling
- `job_id`: Unique job identifier
- `job_name`: Human-readable name

**Example:**
```python
def custom_task():
    print("Running custom task")

scheduler.add_custom_job(
    job_func=custom_task,
    cron_expression="0 */2 * * *",  # Every 2 hours
    job_id="custom_task",
    job_name="Custom Task"
)
```

##### `run_job_now(job_id: str)`
Run a scheduled job immediately.

**Parameters:**
- `job_id`: Job identifier

##### `get_job_status() -> Dict`
Get status of all scheduled jobs.

**Returns:**
- Dictionary with scheduler status and job information

##### `pause_job(job_id: str)` / `resume_job(job_id: str)`
Pause or resume a scheduled job.

---

## Command Line Interface

### Main Script Commands

**Module:** `main.py`

#### Available Commands

```bash
# Start MLflow server
python main.py mlflow

# Collect data from sources
python main.py collect

# Train ensemble model
python main.py train

# Generate predictions
python main.py predict

# Trigger retention campaigns
python main.py campaigns

# Start automated scheduler
python main.py scheduler
```

#### Command Options

```bash
python main.py <command> [--config CONFIG_FILE]
```

**Options:**
- `--config`: Path to configuration file (default: config.yaml)

---

## Configuration API

### ConfigLoader Class

**Module:** `src.utils.config`

#### Methods

##### `get(*keys, default=None)`
Get configuration value using dot notation.

**Example:**
```python
from src.utils.config import ConfigLoader

config = ConfigLoader('config.yaml')

# Get nested configuration
rf_params = config.get('models', 'random_forest')
n_estimators = config.get('models', 'random_forest', 'n_estimators')

# With default value
threshold = config.get('models', 'ensemble', 'threshold', default=0.5)
```

---

## Data Formats

### Input Data Schemas

#### Customer Data
```python
{
    'customer_id': str,
    'registration_date': datetime,
    'account_status': str,
    'customer_tier': str,
    'age': int,
    'last_login_date': datetime,
    # ... additional fields
}
```

#### Transaction Data
```python
{
    'transaction_id': str,
    'customer_id': str,
    'transaction_date': datetime,
    'transaction_amount': float,
    'transaction_type': str,
    # ... additional fields
}
```

### Output Data Schemas

#### Predictions
```python
{
    'customer_id': str,
    'churn_probability': float,  # 0-1
    'risk_category': str,  # 'high', 'medium', 'low'
    'prediction_date': date,
    'prediction_timestamp': datetime
}
```

#### Campaign Results
```python
{
    'customer_id': str,
    'risk_category': str,
    'campaign_status': str,
    'campaign_id': str,
    'triggered_at': datetime
}
```

---

## Error Handling

All API methods raise exceptions on errors. Wrap calls in try-except blocks:

```python
try:
    predictions = predictor.predict_all_active_customers(features_df)
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    # Handle error appropriately
```

Common exceptions:
- `ValueError`: Invalid parameters or data
- `FileNotFoundError`: Missing files or models
- `ConnectionError`: Database or API connection issues
- `RuntimeError`: Execution errors

---

## Rate Limits

### CRM API
- Default: 60 requests/minute
- Automatic rate limiting implemented
- Exponential backoff on failures

### Database Queries
- Batch size: 10,000 records (configurable)
- Connection pooling enabled

---

## Best Practices

1. **Always close connections** when done:
```python
collector = DataCollector(config)
try:
    data = collector.collect_all_data()
finally:
    collector.close_connections()
```

2. **Use configuration** instead of hardcoded values:
```python
batch_size = config.get('prediction', 'batch_size', default=10000)
```

3. **Log important operations**:
```python
logger.info(f"Processing {len(predictions)} predictions")
```

4. **Handle errors gracefully**:
```python
try:
    results = model.train_full_pipeline(features_df)
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise
```

---

For additional API details, refer to the inline documentation in each module.
