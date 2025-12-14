"""
MLflow Setup and Management

Handles MLflow server initialization, experiment tracking,
model registry management, and deployment orchestration.
"""

import subprocess
import os
from pathlib import Path
from typing import Optional, Dict, List
from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime


class MLflowManager:
    """
    Manages MLflow tracking server, experiments, and model registry.
    
    Features:
    - MLflow server initialization
    - Experiment management
    - Model registry operations
    - Model versioning and promotion
    - Deployment orchestration
    """
    
    def __init__(self, config):
        """
        Initialize MLflow manager.
        
        Args:
            config: ConfigLoader instance
        """
        self.config = config
        self.mlflow_config = config.get('mlflow')
        self.tracking_uri = self.mlflow_config['tracking_uri']
        self.experiment_name = self.mlflow_config['experiment_name']
        self.model_registry_name = self.mlflow_config['model_registry_name']
        
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()
    
    def start_mlflow_server(
        self, 
        host: str = '0.0.0.0',
        port: int = 5000,
        backend_store_uri: Optional[str] = None,
        default_artifact_root: Optional[str] = None
    ):
        """
        Start MLflow tracking server.
        
        Args:
            host: Server host address
            port: Server port
            backend_store_uri: Backend store URI (default: sqlite)
            default_artifact_root: Artifact storage location
        """
        logger.info(f"Starting MLflow server on {host}:{port}")
        
        # Set default paths
        if backend_store_uri is None:
            backend_store_uri = 'sqlite:///mlflow.db'
        
        if default_artifact_root is None:
            artifact_path = Path(self.mlflow_config.get('artifact_location', './mlruns'))
            artifact_path.mkdir(parents=True, exist_ok=True)
            default_artifact_root = str(artifact_path.absolute())
        
        # Start MLflow server
        command = [
            'mlflow', 'server',
            '--host', host,
            '--port', str(port),
            '--backend-store-uri', backend_store_uri,
            '--default-artifact-root', default_artifact_root
        ]
        
        logger.info(f"MLflow server command: {' '.join(command)}")
        logger.info(f"Access MLflow UI at: http://{host}:{port}")
        
        try:
            # Run server in background
            subprocess.Popen(command)
            logger.info("MLflow server started successfully")
        except Exception as e:
            logger.error(f"Failed to start MLflow server: {e}")
            raise
    
    def create_experiment(self, experiment_name: Optional[str] = None) -> str:
        """
        Create or get MLflow experiment.
        
        Args:
            experiment_name: Name of experiment (uses config default if None)
            
        Returns:
            Experiment ID
        """
        if experiment_name is None:
            experiment_name = self.experiment_name
        
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment:
                logger.info(f"Experiment '{experiment_name}' already exists (ID: {experiment.experiment_id})")
                return experiment.experiment_id
            else:
                experiment_id = self.client.create_experiment(experiment_name)
                logger.info(f"Created experiment '{experiment_name}' (ID: {experiment_id})")
                return experiment_id
        except Exception as e:
            logger.error(f"Error managing experiment: {e}")
            raise
    
    def list_experiments(self) -> List[Dict]:
        """
        List all MLflow experiments.
        
        Returns:
            List of experiment dictionaries
        """
        experiments = self.client.search_experiments()
        
        experiment_list = []
        for exp in experiments:
            experiment_list.append({
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'lifecycle_stage': exp.lifecycle_stage,
                'artifact_location': exp.artifact_location
            })
        
        logger.info(f"Found {len(experiment_list)} experiments")
        return experiment_list
    
    def get_best_run(
        self, 
        experiment_name: Optional[str] = None,
        metric: str = 'test_roc_auc',
        ascending: bool = False
    ) -> Dict:
        """
        Get the best run from an experiment based on a metric.
        
        Args:
            experiment_name: Experiment name
            metric: Metric to optimize
            ascending: Whether lower is better
            
        Returns:
            Best run information
        """
        if experiment_name is None:
            experiment_name = self.experiment_name
        
        experiment = self.client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        
        if not runs:
            logger.warning(f"No runs found in experiment '{experiment_name}'")
            return {}
        
        best_run = runs[0]
        
        run_info = {
            'run_id': best_run.info.run_id,
            'run_name': best_run.data.tags.get('mlflow.runName', 'N/A'),
            'start_time': datetime.fromtimestamp(best_run.info.start_time / 1000),
            'metrics': best_run.data.metrics,
            'parameters': best_run.data.params,
            'artifact_uri': best_run.info.artifact_uri
        }
        
        logger.info(f"Best run: {run_info['run_id']} with {metric}={run_info['metrics'].get(metric)}")
        
        return run_info
    
    def register_model(
        self, 
        run_id: str,
        model_path: str,
        model_name: Optional[str] = None
    ) -> Dict:
        """
        Register a model in MLflow Model Registry.
        
        Args:
            run_id: MLflow run ID
            model_path: Path to model within run artifacts
            model_name: Model name for registry
            
        Returns:
            Registered model information
        """
        if model_name is None:
            model_name = self.model_registry_name
        
        model_uri = f"runs:/{run_id}/{model_path}"
        
        try:
            model_version = mlflow.register_model(model_uri, model_name)
            
            logger.info(f"Registered model '{model_name}' version {model_version.version}")
            
            return {
                'name': model_version.name,
                'version': model_version.version,
                'run_id': model_version.run_id,
                'status': model_version.status
            }
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def promote_model_to_production(
        self, 
        model_name: Optional[str] = None,
        version: Optional[int] = None
    ):
        """
        Promote a model version to production stage.
        
        Args:
            model_name: Model name in registry
            version: Specific version to promote (uses latest if None)
        """
        if model_name is None:
            model_name = self.model_registry_name
        
        try:
            # Get latest version if not specified
            if version is None:
                latest_versions = self.client.get_latest_versions(model_name, stages=["None", "Staging"])
                if not latest_versions:
                    raise ValueError(f"No versions found for model '{model_name}'")
                version = latest_versions[0].version
            
            # Archive current production models
            production_models = self.client.get_latest_versions(model_name, stages=["Production"])
            for prod_model in production_models:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=prod_model.version,
                    stage="Archived"
                )
                logger.info(f"Archived production model version {prod_model.version}")
            
            # Promote new version to production
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            
            logger.info(f"Promoted model '{model_name}' version {version} to Production")
            
        except Exception as e:
            logger.error(f"Failed to promote model to production: {e}")
            raise
    
    def get_production_model(self, model_name: Optional[str] = None) -> Dict:
        """
        Get current production model.
        
        Args:
            model_name: Model name in registry
            
        Returns:
            Production model information
        """
        if model_name is None:
            model_name = self.model_registry_name
        
        try:
            production_models = self.client.get_latest_versions(model_name, stages=["Production"])
            
            if not production_models:
                logger.warning(f"No production model found for '{model_name}'")
                return {}
            
            prod_model = production_models[0]
            
            return {
                'name': prod_model.name,
                'version': prod_model.version,
                'run_id': prod_model.run_id,
                'stage': prod_model.current_stage,
                'creation_timestamp': datetime.fromtimestamp(prod_model.creation_timestamp / 1000),
                'last_updated_timestamp': datetime.fromtimestamp(prod_model.last_updated_timestamp / 1000)
            }
        except Exception as e:
            logger.error(f"Failed to get production model: {e}")
            raise
    
    def compare_models(self, run_ids: List[str], metrics: List[str]) -> Dict:
        """
        Compare multiple model runs based on metrics.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(run_ids)} models on metrics: {metrics}")
        
        comparison = {}
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            comparison[run_id] = {
                'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
                'metrics': {metric: run.data.metrics.get(metric) for metric in metrics}
            }
        
        return comparison
    
    def cleanup_old_runs(
        self, 
        experiment_name: Optional[str] = None,
        keep_last_n: int = 10
    ):
        """
        Clean up old runs, keeping only the most recent ones.
        
        Args:
            experiment_name: Experiment name
            keep_last_n: Number of recent runs to keep
        """
        if experiment_name is None:
            experiment_name = self.experiment_name
        
        experiment = self.client.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.warning(f"Experiment '{experiment_name}' not found")
            return
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        if len(runs) <= keep_last_n:
            logger.info(f"Only {len(runs)} runs found, no cleanup needed")
            return
        
        runs_to_delete = runs[keep_last_n:]
        
        for run in runs_to_delete:
            self.client.delete_run(run.info.run_id)
            logger.info(f"Deleted run {run.info.run_id}")
        
        logger.info(f"Cleaned up {len(runs_to_delete)} old runs")
    
    def export_model_metadata(self, model_name: Optional[str] = None) -> Dict:
        """
        Export model metadata and version information.
        
        Args:
            model_name: Model name in registry
            
        Returns:
            Model metadata dictionary
        """
        if model_name is None:
            model_name = self.model_registry_name
        
        try:
            # Get all versions
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            metadata = {
                'model_name': model_name,
                'total_versions': len(versions),
                'versions': []
            }
            
            for version in versions:
                metadata['versions'].append({
                    'version': version.version,
                    'stage': version.current_stage,
                    'run_id': version.run_id,
                    'creation_timestamp': datetime.fromtimestamp(version.creation_timestamp / 1000).isoformat(),
                    'last_updated': datetime.fromtimestamp(version.last_updated_timestamp / 1000).isoformat()
                })
            
            logger.info(f"Exported metadata for model '{model_name}' with {len(versions)} versions")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to export model metadata: {e}")
            raise
