"""
Model Training Module

Implements ensemble approach combining Random Forest and XGBoost algorithms
with MLflow integration for experiment tracking and model registry.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from loguru import logger
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class EnsembleChurnModel:
    """
    Ensemble model combining Random Forest and XGBoost for churn prediction.
    
    Features:
    - Weighted ensemble of RF and XGBoost
    - SMOTE for handling class imbalance
    - Cross-validation for robust evaluation
    - MLflow integration for experiment tracking
    - Model registry for deployment
    """
    
    def __init__(self, config):
        """
        Initialize ensemble model.
        
        Args:
            config: ConfigLoader instance
        """
        self.config = config
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.rf_weight = config.get('models', 'ensemble', 'rf_weight', default=0.4)
        self.xgb_weight = config.get('models', 'ensemble', 'xgb_weight', default=0.6)
        
        # Initialize MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking and experiment."""
        mlflow_config = self.config.get('mlflow')
        
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        mlflow.set_experiment(mlflow_config['experiment_name'])
        
        if mlflow_config.get('autolog', True):
            mlflow.sklearn.autolog()
            mlflow.xgboost.autolog()
        
        logger.info(f"MLflow tracking configured: {mlflow_config['tracking_uri']}")
    
    def prepare_data(
        self, 
        features_df: pd.DataFrame, 
        target_column: str = 'churned'
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare data for training with train/validation/test split.
        
        Args:
            features_df: DataFrame with features and target
            target_column: Name of target column
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info("Preparing data for training")
        
        # Separate features and target
        if target_column not in features_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in features")
        
        X = features_df.drop(columns=[target_column, 'customer_id'], errors='ignore')
        y = features_df[target_column]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        # Update feature names after encoding
        self.feature_names = X.columns.tolist()
        
        # Split configuration
        training_config = self.config.get('training')
        test_size = training_config.get('test_size', 0.2)
        val_size = training_config.get('validation_size', 0.1)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y if training_config.get('stratify', True) else None,
            random_state=42
        )
        
        # Second split: separate validation set from training
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=y_temp if training_config.get('stratify', True) else None,
            random_state=42
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def handle_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using SMOTE.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Balanced X_train and y_train
        """
        training_config = self.config.get('training')
        
        if not training_config.get('handle_imbalance', True):
            return X_train, y_train
        
        logger.info("Handling class imbalance with SMOTE")
        
        original_distribution = y_train.value_counts()
        logger.info(f"Original class distribution: {original_distribution.to_dict()}")
        
        smote_strategy = training_config.get('smote_sampling_strategy', 0.5)
        smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
        
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        new_distribution = pd.Series(y_train_balanced).value_counts()
        logger.info(f"Balanced class distribution: {new_distribution.to_dict()}")
        
        return X_train_balanced, y_train_balanced
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained Random Forest model
        """
        logger.info("Training Random Forest model")
        
        rf_config = self.config.get('models', 'random_forest')
        
        self.rf_model = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 200),
            max_depth=rf_config.get('max_depth', 15),
            min_samples_split=rf_config.get('min_samples_split', 5),
            min_samples_leaf=rf_config.get('min_samples_leaf', 2),
            max_features=rf_config.get('max_features', 'sqrt'),
            class_weight=rf_config.get('class_weight', 'balanced'),
            random_state=rf_config.get('random_state', 42),
            n_jobs=rf_config.get('n_jobs', -1)
        )
        
        self.rf_model.fit(X_train, y_train)
        
        logger.info("Random Forest training completed")
        
        return self.rf_model
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost model")
        
        xgb_config = self.config.get('models', 'xgboost')
        
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=xgb_config.get('n_estimators', 200),
            max_depth=xgb_config.get('max_depth', 10),
            learning_rate=xgb_config.get('learning_rate', 0.05),
            subsample=xgb_config.get('subsample', 0.8),
            colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
            gamma=xgb_config.get('gamma', 1),
            reg_alpha=xgb_config.get('reg_alpha', 0.1),
            reg_lambda=xgb_config.get('reg_lambda', 1),
            scale_pos_weight=xgb_config.get('scale_pos_weight', 3),
            random_state=xgb_config.get('random_state', 42),
            n_jobs=xgb_config.get('n_jobs', -1),
            eval_metric='logloss'
        )
        
        self.xgb_model.fit(X_train, y_train)
        
        logger.info("XGBoost training completed")
        
        return self.xgb_model
    
    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using weighted ensemble.
        
        Args:
            X: Features for prediction
            
        Returns:
            Ensemble probability predictions
        """
        if self.rf_model is None or self.xgb_model is None:
            raise ValueError("Models must be trained before making predictions")
        
        # Get probability predictions from both models
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        
        # Weighted ensemble
        ensemble_proba = (self.rf_weight * rf_proba + self.xgb_weight * xgb_proba)
        
        return ensemble_proba
    
    def evaluate_model(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        dataset_name: str = "validation"
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True labels
            dataset_name: Name of dataset being evaluated
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {dataset_name} set")
        
        # Get predictions
        y_proba = self.predict_ensemble(X)
        y_pred = (y_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba)
        }
        
        # Log metrics
        for metric_name, value in metrics.items():
            logger.info(f"{dataset_name} {metric_name}: {value:.4f}")
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dictionary of cross-validation scores
        """
        logger.info("Performing cross-validation")
        
        cv_folds = self.config.get('training', 'cross_validation_folds', default=5)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validate Random Forest
        rf_scores = cross_val_score(
            self.rf_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1
        )
        
        # Cross-validate XGBoost
        xgb_scores = cross_val_score(
            self.xgb_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1
        )
        
        cv_results = {
            'rf_cv_scores': rf_scores,
            'xgb_cv_scores': xgb_scores,
            'rf_cv_mean': rf_scores.mean(),
            'rf_cv_std': rf_scores.std(),
            'xgb_cv_mean': xgb_scores.mean(),
            'xgb_cv_std': xgb_scores.std()
        }
        
        logger.info(f"RF CV AUC: {cv_results['rf_cv_mean']:.4f} (+/- {cv_results['rf_cv_std']:.4f})")
        logger.info(f"XGB CV AUC: {cv_results['xgb_cv_mean']:.4f} (+/- {cv_results['xgb_cv_std']:.4f})")
        
        return cv_results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from both models.
        
        Returns:
            DataFrame with feature importances
        """
        rf_importance = pd.DataFrame({
            'feature': self.feature_names,
            'rf_importance': self.rf_model.feature_importances_
        })
        
        xgb_importance = pd.DataFrame({
            'feature': self.feature_names,
            'xgb_importance': self.xgb_model.feature_importances_
        })
        
        # Merge and calculate average
        importance_df = rf_importance.merge(xgb_importance, on='feature')
        importance_df['avg_importance'] = (
            self.rf_weight * importance_df['rf_importance'] + 
            self.xgb_weight * importance_df['xgb_importance']
        )
        
        importance_df = importance_df.sort_values('avg_importance', ascending=False)
        
        return importance_df
    
    def train_full_pipeline(
        self, 
        features_df: pd.DataFrame, 
        target_column: str = 'churned'
    ) -> Dict[str, Any]:
        """
        Execute complete training pipeline with MLflow tracking.
        
        Args:
            features_df: DataFrame with features and target
            target_column: Name of target column
            
        Returns:
            Dictionary containing training results and metrics
        """
        logger.info("Starting full training pipeline")
        
        with mlflow.start_run(run_name=f"churn_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            mlflow.log_params(self.config.get('models', 'random_forest'))
            mlflow.log_params(self.config.get('models', 'xgboost'))
            mlflow.log_param("rf_weight", self.rf_weight)
            mlflow.log_param("xgb_weight", self.xgb_weight)
            
            # Prepare data
            X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(
                features_df, target_column
            )
            
            # Handle imbalance
            X_train, y_train = self.handle_imbalance(X_train, y_train)
            
            # Train models
            self.train_random_forest(X_train, y_train)
            self.train_xgboost(X_train, y_train)
            
            # Cross-validation
            cv_results = self.cross_validate(X_train, y_train)
            mlflow.log_metrics({
                'rf_cv_auc_mean': cv_results['rf_cv_mean'],
                'xgb_cv_auc_mean': cv_results['xgb_cv_mean']
            })
            
            # Evaluate on validation set
            val_metrics = self.evaluate_model(X_val, y_val, "validation")
            mlflow.log_metrics({f'val_{k}': v for k, v in val_metrics.items()})
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(X_test, y_test, "test")
            mlflow.log_metrics({f'test_{k}': v for k, v in test_metrics.items()})
            
            # Feature importance
            importance_df = self.get_feature_importance()
            
            # Save models
            model_dir = Path('models')
            model_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            joblib.dump(self.rf_model, model_dir / f'rf_model_{timestamp}.pkl')
            joblib.dump(self.xgb_model, model_dir / f'xgb_model_{timestamp}.pkl')
            joblib.dump(self.scaler, model_dir / f'scaler_{timestamp}.pkl')
            
            # Log models to MLflow
            mlflow.sklearn.log_model(self.rf_model, "random_forest_model")
            mlflow.xgboost.log_model(self.xgb_model, "xgboost_model")
            
            # Register model
            self._register_model(test_metrics['roc_auc'])
            
            logger.info("Training pipeline completed successfully")
            
            return {
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'cv_results': cv_results,
                'feature_importance': importance_df
            }
    
    def _register_model(self, test_auc: float):
        """
        Register model in MLflow Model Registry if performance is sufficient.
        
        Args:
            test_auc: Test set ROC-AUC score
        """
        threshold = self.config.get('monitoring', 'model_performance_threshold', default=0.75)
        
        if test_auc >= threshold:
            model_name = self.config.get('mlflow', 'model_registry_name')
            
            # Register both models
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/random_forest_model",
                f"{model_name}_rf"
            )
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/xgboost_model",
                f"{model_name}_xgb"
            )
            
            logger.info(f"Models registered in MLflow Model Registry (AUC: {test_auc:.4f})")
        else:
            logger.warning(f"Model not registered. AUC {test_auc:.4f} below threshold {threshold}")
