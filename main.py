"""
Main Orchestration Script

Entry point for the Machine Learning Churn Prediction Model system.
"""

import argparse
from pathlib import Path
from loguru import logger

from src.utils.config import ConfigLoader, setup_logging, ensure_directories
from src.data_collection.data_collector import DataCollector
from src.features.feature_engineer import FeatureEngineer
from src.models.model_trainer import EnsembleChurnModel
from src.prediction.predictor import ChurnPredictor
from src.crm.crm_integration import CRMIntegration
from src.mlflow_setup.mlflow_manager import MLflowManager
from src.scheduler.scheduler import ChurnPredictionScheduler


def run_data_collection(config: ConfigLoader):
    """Run data collection pipeline."""
    logger.info("Starting data collection")
    
    collector = DataCollector(config)
    data = collector.collect_all_data(lookback_days=365)
    collector.save_raw_data(data)
    collector.close_connections()
    
    logger.info("Data collection completed")


def run_training(config: ConfigLoader):
    """Run model training pipeline."""
    logger.info("Starting model training")
    
    # Collect data
    collector = DataCollector(config)
    data = collector.collect_all_data(lookback_days=365)
    
    # Engineer features
    feature_engineer = FeatureEngineer(config)
    features_df = feature_engineer.create_all_features(data)
    
    # Train model
    model_trainer = EnsembleChurnModel(config)
    results = model_trainer.train_full_pipeline(features_df)
    
    logger.info(f"Training completed. Test AUC: {results['test_metrics']['roc_auc']:.4f}")
    
    collector.close_connections()


def run_prediction(config: ConfigLoader):
    """Run prediction pipeline."""
    logger.info("Starting prediction")
    
    # Collect data
    collector = DataCollector(config)
    data = collector.collect_all_data(lookback_days=365)
    
    # Engineer features
    feature_engineer = FeatureEngineer(config)
    features_df = feature_engineer.create_all_features(data)
    
    # Load predictor and generate predictions
    predictor = ChurnPredictor(config)
    predictor.load_models_from_registry(model_stage="Production")
    
    predictions = predictor.predict_all_active_customers(features_df)
    predictor.save_predictions(predictions, output_format='both')
    
    # Generate report
    report = predictor.generate_daily_report(predictions)
    logger.info(f"Predictions completed. High-risk customers: {report['risk_distribution']['high']}")
    
    collector.close_connections()


def run_campaigns(config: ConfigLoader):
    """Trigger retention campaigns."""
    logger.info("Starting campaign triggering")
    
    # Load latest predictions
    from datetime import datetime
    import pandas as pd
    
    predictions_dir = Path('data/predictions')
    prediction_files = sorted(predictions_dir.glob('predictions_*.parquet'))
    
    if not prediction_files:
        logger.error("No predictions found. Run prediction first.")
        return
    
    latest_predictions = pd.read_parquet(prediction_files[-1])
    
    # Trigger campaigns
    crm = CRMIntegration(config)
    campaign_results = crm.trigger_high_risk_campaigns(latest_predictions)
    
    # Generate report
    campaign_report = crm.generate_campaign_report(campaign_results)
    logger.info(f"Campaigns triggered. Success rate: {campaign_report['success_rate']:.2f}%")


def run_scheduler(config: ConfigLoader):
    """Run automated scheduler."""
    logger.info("Starting automated scheduler")
    
    scheduler = ChurnPredictionScheduler(config)
    scheduler.schedule_jobs()
    scheduler.start()
    
    logger.info("Scheduler is running. Press Ctrl+C to stop.")
    
    try:
        # Keep the script running
        import time
        while True:
            time.sleep(60)
            status = scheduler.get_job_status()
            logger.info(f"Scheduler status: {status['total_jobs']} jobs scheduled")
    except KeyboardInterrupt:
        logger.info("Stopping scheduler...")
        scheduler.stop()


def start_mlflow_server(config: ConfigLoader):
    """Start MLflow tracking server."""
    logger.info("Starting MLflow server")
    
    mlflow_manager = MLflowManager(config)
    mlflow_manager.create_experiment()
    mlflow_manager.start_mlflow_server(host='0.0.0.0', port=5000)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Machine Learning Churn Prediction Model System"
    )
    
    parser.add_argument(
        'command',
        choices=['collect', 'train', 'predict', 'campaigns', 'scheduler', 'mlflow'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigLoader(args.config)
    
    # Setup logging
    setup_logging(config)
    
    # Ensure directories exist
    ensure_directories()
    
    # Execute command
    commands = {
        'collect': run_data_collection,
        'train': run_training,
        'predict': run_prediction,
        'campaigns': run_campaigns,
        'scheduler': run_scheduler,
        'mlflow': start_mlflow_server
    }
    
    try:
        commands[args.command](config)
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        raise


if __name__ == '__main__':
    main()
