"""
Automated Scheduler Module

Manages automated daily retraining schedule and orchestrates
the entire churn prediction pipeline.
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict
from loguru import logger
import pytz

from ..data_collection.data_collector import DataCollector
from ..features.feature_engineer import FeatureEngineer
from ..models.model_trainer import EnsembleChurnModel
from ..prediction.predictor import ChurnPredictor
from ..crm.crm_integration import CRMIntegration
from ..utils.config import ConfigLoader


class ChurnPredictionScheduler:
    """
    Automated scheduler for the churn prediction pipeline.
    
    Schedules:
    - Daily data collection (1 AM)
    - Daily model retraining (2 AM)
    - Daily predictions (6 AM)
    - Daily campaign triggering (7 AM)
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize the scheduler.
        
        Args:
            config: ConfigLoader instance
        """
        self.config = config
        self.scheduler = BackgroundScheduler()
        
        # Get timezone from config
        timezone_str = config.get('scheduler', 'timezone', default='UTC')
        self.timezone = pytz.timezone(timezone_str)
        
        # Initialize components
        self.data_collector = None
        self.feature_engineer = None
        self.model_trainer = None
        self.predictor = None
        self.crm_integration = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components")
        
        try:
            self.data_collector = DataCollector(self.config)
            self.feature_engineer = FeatureEngineer(self.config)
            self.model_trainer = EnsembleChurnModel(self.config)
            self.predictor = ChurnPredictor(self.config)
            self.crm_integration = CRMIntegration(self.config)
            
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _data_collection_job(self):
        """Daily data collection job."""
        logger.info("=" * 80)
        logger.info("STARTING DAILY DATA COLLECTION JOB")
        logger.info("=" * 80)
        
        try:
            # Collect data from all sources
            data = self.data_collector.collect_all_data(lookback_days=365)
            
            # Save raw data
            self.data_collector.save_raw_data(data)
            
            logger.info("Data collection job completed successfully")
            
        except Exception as e:
            logger.error(f"Data collection job failed: {e}")
            raise
    
    def _model_training_job(self):
        """Daily model training job."""
        logger.info("=" * 80)
        logger.info("STARTING DAILY MODEL TRAINING JOB")
        logger.info("=" * 80)
        
        try:
            # Load most recent raw data
            data = self._load_latest_data()
            
            # Create features
            features_df = self.feature_engineer.create_all_features(data)
            
            # Save processed features
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            features_path = Path('data/processed')
            features_path.mkdir(parents=True, exist_ok=True)
            features_df.to_parquet(features_path / f'features_{timestamp}.parquet')
            
            # Train model
            training_results = self.model_trainer.train_full_pipeline(features_df)
            
            logger.info(f"Model training completed. Test AUC: {training_results['test_metrics']['roc_auc']:.4f}")
            
        except Exception as e:
            logger.error(f"Model training job failed: {e}")
            raise
    
    def _prediction_job(self):
        """Daily prediction job."""
        logger.info("=" * 80)
        logger.info("STARTING DAILY PREDICTION JOB")
        logger.info("=" * 80)
        
        try:
            # Load latest features
            features_df = self._load_latest_features()
            
            # Load production models
            success = self.predictor.load_models_from_registry(model_stage="Production")
            
            if not success:
                logger.warning("Failed to load from registry, trying latest files")
                rf_path, xgb_path = self._get_latest_model_files()
                self.predictor.load_models_from_files(rf_path, xgb_path)
            
            # Generate predictions
            predictions = self.predictor.predict_all_active_customers(features_df)
            
            # Save predictions
            self.predictor.save_predictions(predictions, output_format='both')
            
            # Generate daily report
            report = self.predictor.generate_daily_report(predictions)
            self._save_daily_report(report)
            
            logger.info(f"Prediction job completed. Total customers: {len(predictions)}")
            
        except Exception as e:
            logger.error(f"Prediction job failed: {e}")
            raise
    
    def _campaign_trigger_job(self):
        """Daily campaign triggering job."""
        logger.info("=" * 80)
        logger.info("STARTING DAILY CAMPAIGN TRIGGER JOB")
        logger.info("=" * 80)
        
        try:
            # Load latest predictions
            predictions = self._load_latest_predictions()
            
            # Trigger campaigns for high-risk customers
            campaign_results = self.crm_integration.trigger_high_risk_campaigns(predictions)
            
            # Generate campaign report
            campaign_report = self.crm_integration.generate_campaign_report(campaign_results)
            self._save_campaign_report(campaign_report)
            
            # Sync all predictions to CRM
            sync_results = self.crm_integration.sync_churn_predictions_to_crm(predictions)
            
            logger.info(f"Campaign trigger job completed. Campaigns triggered: {len(campaign_results)}")
            
        except Exception as e:
            logger.error(f"Campaign trigger job failed: {e}")
            raise
    
    def _load_latest_data(self) -> Dict:
        """Load the most recent raw data files."""
        import pandas as pd
        
        data_dir = Path('data/raw')
        
        data = {}
        for data_type in ['customers', 'transactions', 'engagement', 'support', 'subscriptions']:
            # Find most recent file for this data type
            files = sorted(data_dir.glob(f'{data_type}_*.parquet'))
            if files:
                latest_file = files[-1]
                data[data_type] = pd.read_parquet(latest_file)
                logger.info(f"Loaded {data_type} from {latest_file}")
        
        return data
    
    def _load_latest_features(self):
        """Load the most recent feature file."""
        import pandas as pd
        
        features_dir = Path('data/processed')
        feature_files = sorted(features_dir.glob('features_*.parquet'))
        
        if not feature_files:
            raise FileNotFoundError("No feature files found")
        
        latest_file = feature_files[-1]
        logger.info(f"Loading features from {latest_file}")
        
        return pd.read_parquet(latest_file)
    
    def _load_latest_predictions(self):
        """Load the most recent predictions."""
        import pandas as pd
        
        predictions_dir = Path('data/predictions')
        prediction_files = sorted(predictions_dir.glob('predictions_*.parquet'))
        
        if not prediction_files:
            raise FileNotFoundError("No prediction files found")
        
        latest_file = prediction_files[-1]
        logger.info(f"Loading predictions from {latest_file}")
        
        return pd.read_parquet(latest_file)
    
    def _get_latest_model_files(self):
        """Get paths to the most recent model files."""
        models_dir = Path('models')
        
        rf_files = sorted(models_dir.glob('rf_model_*.pkl'))
        xgb_files = sorted(models_dir.glob('xgb_model_*.pkl'))
        
        if not rf_files or not xgb_files:
            raise FileNotFoundError("Model files not found")
        
        return str(rf_files[-1]), str(xgb_files[-1])
    
    def _save_daily_report(self, report: Dict):
        """Save daily prediction report."""
        import json
        
        reports_dir = Path('reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f'daily_report_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Daily report saved to {report_file}")
    
    def _save_campaign_report(self, report: Dict):
        """Save campaign report."""
        import json
        
        reports_dir = Path('reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f'campaign_report_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Campaign report saved to {report_file}")
    
    def schedule_jobs(self):
        """Schedule all automated jobs based on configuration."""
        logger.info("Scheduling automated jobs")
        
        scheduler_config = self.config.get('scheduler')
        
        # Schedule data collection
        self.scheduler.add_job(
            self._data_collection_job,
            CronTrigger.from_crontab(scheduler_config['data_collection_cron'], timezone=self.timezone),
            id='data_collection',
            name='Daily Data Collection',
            replace_existing=True
        )
        logger.info(f"Scheduled data collection: {scheduler_config['data_collection_cron']}")
        
        # Schedule model training
        self.scheduler.add_job(
            self._model_training_job,
            CronTrigger.from_crontab(scheduler_config['model_training_cron'], timezone=self.timezone),
            id='model_training',
            name='Daily Model Training',
            replace_existing=True
        )
        logger.info(f"Scheduled model training: {scheduler_config['model_training_cron']}")
        
        # Schedule predictions
        self.scheduler.add_job(
            self._prediction_job,
            CronTrigger.from_crontab(scheduler_config['prediction_cron'], timezone=self.timezone),
            id='prediction',
            name='Daily Predictions',
            replace_existing=True
        )
        logger.info(f"Scheduled predictions: {scheduler_config['prediction_cron']}")
        
        # Schedule campaign triggering
        self.scheduler.add_job(
            self._campaign_trigger_job,
            CronTrigger.from_crontab(scheduler_config['campaign_trigger_cron'], timezone=self.timezone),
            id='campaign_trigger',
            name='Daily Campaign Trigger',
            replace_existing=True
        )
        logger.info(f"Scheduled campaign trigger: {scheduler_config['campaign_trigger_cron']}")
    
    def add_custom_job(
        self, 
        job_func: Callable,
        cron_expression: str,
        job_id: str,
        job_name: str
    ):
        """
        Add a custom scheduled job.
        
        Args:
            job_func: Function to execute
            cron_expression: Cron expression for scheduling
            job_id: Unique job identifier
            job_name: Human-readable job name
        """
        self.scheduler.add_job(
            job_func,
            CronTrigger.from_crontab(cron_expression, timezone=self.timezone),
            id=job_id,
            name=job_name,
            replace_existing=True
        )
        logger.info(f"Added custom job '{job_name}': {cron_expression}")
    
    def start(self):
        """Start the scheduler."""
        logger.info("Starting automated scheduler")
        self.scheduler.start()
        logger.info("Scheduler started successfully")
        logger.info("Scheduled jobs:")
        for job in self.scheduler.get_jobs():
            logger.info(f"  - {job.name} (ID: {job.id})")
    
    def stop(self):
        """Stop the scheduler."""
        logger.info("Stopping scheduler")
        self.scheduler.shutdown()
        logger.info("Scheduler stopped")
    
    def get_job_status(self) -> Dict:
        """
        Get status of all scheduled jobs.
        
        Returns:
            Dictionary with job status information
        """
        jobs = self.scheduler.get_jobs()
        
        status = {
            'scheduler_running': self.scheduler.running,
            'total_jobs': len(jobs),
            'jobs': []
        }
        
        for job in jobs:
            job_info = {
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            }
            status['jobs'].append(job_info)
        
        return status
    
    def run_job_now(self, job_id: str):
        """
        Run a scheduled job immediately.
        
        Args:
            job_id: Job identifier
        """
        logger.info(f"Running job '{job_id}' immediately")
        job = self.scheduler.get_job(job_id)
        
        if job:
            job.func()
            logger.info(f"Job '{job_id}' executed successfully")
        else:
            logger.error(f"Job '{job_id}' not found")
    
    def pause_job(self, job_id: str):
        """Pause a scheduled job."""
        self.scheduler.pause_job(job_id)
        logger.info(f"Job '{job_id}' paused")
    
    def resume_job(self, job_id: str):
        """Resume a paused job."""
        self.scheduler.resume_job(job_id)
        logger.info(f"Job '{job_id}' resumed")
