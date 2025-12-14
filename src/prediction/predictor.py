"""
Prediction Module

Generates daily churn risk scores for all active customers with
risk categorization and actionable insights.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from loguru import logger
import joblib
import mlflow
from sqlalchemy import create_engine


class ChurnPredictor:
    """
    Churn prediction system for scoring customers daily.
    
    Features:
    - Batch prediction for all active customers
    - Risk categorization (High/Medium/Low)
    - Confidence scores
    - Database integration for results storage
    - MLflow model loading from registry
    """
    
    def __init__(self, config):
        """
        Initialize churn predictor.
        
        Args:
            config: ConfigLoader instance
        """
        self.config = config
        self.rf_model = None
        self.xgb_model = None
        self.rf_weight = config.get('models', 'ensemble', 'rf_weight', default=0.4)
        self.xgb_weight = config.get('models', 'ensemble', 'xgb_weight', default=0.6)
        self.threshold_high = config.get('models', 'ensemble', 'threshold_high_risk', default=0.7)
        self.threshold_medium = config.get('models', 'ensemble', 'threshold_medium_risk', default=0.4)
        self.batch_size = config.get('prediction', 'batch_size', default=10000)
        
        # Database connection for saving results
        self.db_engine = None
        self._initialize_db_connection()
    
    def _initialize_db_connection(self):
        """Initialize database connection for saving predictions."""
        try:
            pg_config = self.config.get('data', 'postgres')
            connection_string = (
                f"postgresql://{pg_config['user']}:{pg_config['password']}"
                f"@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
            )
            self.db_engine = create_engine(connection_string)
            logger.info("Database connection for predictions established")
        except Exception as e:
            logger.warning(f"Could not establish database connection: {e}")
    
    def load_models_from_registry(self, model_stage: str = "Production") -> bool:
        """
        Load models from MLflow Model Registry.
        
        Args:
            model_stage: Model stage to load (Production, Staging, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Loading models from MLflow Registry (stage: {model_stage})")
        
        try:
            mlflow.set_tracking_uri(self.config.get('mlflow', 'tracking_uri'))
            model_name = self.config.get('mlflow', 'model_registry_name')
            
            # Load Random Forest model
            rf_model_uri = f"models:/{model_name}_rf/{model_stage}"
            self.rf_model = mlflow.sklearn.load_model(rf_model_uri)
            logger.info(f"Loaded Random Forest model from {rf_model_uri}")
            
            # Load XGBoost model
            xgb_model_uri = f"models:/{model_name}_xgb/{model_stage}"
            self.xgb_model = mlflow.xgboost.load_model(xgb_model_uri)
            logger.info(f"Loaded XGBoost model from {xgb_model_uri}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models from registry: {e}")
            return False
    
    def load_models_from_files(self, rf_model_path: str, xgb_model_path: str) -> bool:
        """
        Load models from saved files.
        
        Args:
            rf_model_path: Path to Random Forest model file
            xgb_model_path: Path to XGBoost model file
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Loading models from files")
        
        try:
            self.rf_model = joblib.load(rf_model_path)
            self.xgb_model = joblib.load(xgb_model_path)
            logger.info("Models loaded successfully from files")
            return True
        except Exception as e:
            logger.error(f"Error loading models from files: {e}")
            return False
    
    def predict_churn_probability(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict churn probability using ensemble model.
        
        Args:
            features: Customer features DataFrame
            
        Returns:
            Array of churn probabilities
        """
        if self.rf_model is None or self.xgb_model is None:
            raise ValueError("Models must be loaded before making predictions")
        
        # Get predictions from both models
        rf_proba = self.rf_model.predict_proba(features)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(features)[:, 1]
        
        # Weighted ensemble
        ensemble_proba = (self.rf_weight * rf_proba + self.xgb_weight * xgb_proba)
        
        return ensemble_proba
    
    def categorize_risk(self, churn_probability: float) -> str:
        """
        Categorize churn risk level.
        
        Args:
            churn_probability: Predicted churn probability
            
        Returns:
            Risk category: 'high', 'medium', or 'low'
        """
        if churn_probability >= self.threshold_high:
            return 'high'
        elif churn_probability >= self.threshold_medium:
            return 'medium'
        else:
            return 'low'
    
    def generate_prediction_insights(
        self, 
        customer_id: str,
        churn_probability: float,
        features: pd.Series,
        feature_importance: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Generate actionable insights for a customer's churn prediction.
        
        Args:
            customer_id: Customer identifier
            churn_probability: Predicted churn probability
            features: Customer feature values
            feature_importance: Feature importance DataFrame (optional)
            
        Returns:
            Dictionary with prediction insights
        """
        risk_category = self.categorize_risk(churn_probability)
        
        insights = {
            'customer_id': customer_id,
            'churn_probability': round(float(churn_probability), 4),
            'risk_category': risk_category,
            'prediction_date': datetime.now().isoformat(),
            'key_risk_factors': []
        }
        
        # Identify key risk factors
        risk_factors = []
        
        # Check recency
        if 'recency_days_30d' in features.index and features['recency_days_30d'] > 30:
            risk_factors.append({
                'factor': 'high_recency',
                'description': 'No recent transactions',
                'value': features['recency_days_30d']
            })
        
        # Check declining activity
        if 'activity_decline_30_90' in features.index and features['activity_decline_30_90'] > 0.5:
            risk_factors.append({
                'factor': 'declining_activity',
                'description': 'Significant activity decline',
                'value': features['activity_decline_30_90']
            })
        
        # Check support tickets
        if 'open_tickets' in features.index and features['open_tickets'] > 0:
            risk_factors.append({
                'factor': 'open_support_tickets',
                'description': 'Unresolved support issues',
                'value': int(features['open_tickets'])
            })
        
        # Check engagement
        if 'engagement_score' in features.index:
            engagement_percentile = 50  # Would calculate from historical data
            if features['engagement_score'] < engagement_percentile:
                risk_factors.append({
                    'factor': 'low_engagement',
                    'description': 'Below average engagement',
                    'value': features['engagement_score']
                })
        
        insights['key_risk_factors'] = risk_factors
        
        return insights
    
    def predict_batch(
        self, 
        features_df: pd.DataFrame,
        include_insights: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions for a batch of customers.
        
        Args:
            features_df: DataFrame with customer features
            include_insights: Whether to generate detailed insights
            
        Returns:
            DataFrame with predictions and risk scores
        """
        logger.info(f"Generating predictions for {len(features_df)} customers")
        
        # Ensure customer_id is preserved
        customer_ids = features_df['customer_id'].values
        
        # Prepare features (same preprocessing as training)
        features_for_prediction = features_df.drop(columns=['customer_id'], errors='ignore')
        features_for_prediction = pd.get_dummies(features_for_prediction, drop_first=True)
        
        # Generate predictions
        churn_probabilities = self.predict_churn_probability(features_for_prediction)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'customer_id': customer_ids,
            'churn_probability': churn_probabilities,
            'risk_category': [self.categorize_risk(p) for p in churn_probabilities],
            'prediction_date': datetime.now().date(),
            'prediction_timestamp': datetime.now()
        })
        
        # Sort by churn probability (highest risk first)
        results = results.sort_values('churn_probability', ascending=False)
        
        logger.info(f"Predictions generated. Risk distribution: "
                   f"High: {(results['risk_category']=='high').sum()}, "
                   f"Medium: {(results['risk_category']=='medium').sum()}, "
                   f"Low: {(results['risk_category']=='low').sum()}")
        
        return results
    
    def predict_all_active_customers(
        self, 
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate predictions for all active customers in batches.
        
        Args:
            features_df: DataFrame with all customer features
            
        Returns:
            DataFrame with all predictions
        """
        logger.info(f"Predicting churn for all {len(features_df)} active customers")
        
        all_predictions = []
        
        # Process in batches
        for i in range(0, len(features_df), self.batch_size):
            batch = features_df.iloc[i:i + self.batch_size]
            batch_predictions = self.predict_batch(batch, include_insights=False)
            all_predictions.append(batch_predictions)
            
            logger.info(f"Processed batch {i//self.batch_size + 1}: "
                       f"{len(batch)} customers")
        
        # Combine all batches
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        
        logger.info(f"All predictions completed: {len(final_predictions)} customers scored")
        
        return final_predictions
    
    def save_predictions(
        self, 
        predictions: pd.DataFrame,
        output_format: str = 'both'  # 'database', 'file', or 'both'
    ):
        """
        Save predictions to database and/or file.
        
        Args:
            predictions: DataFrame with predictions
            output_format: Where to save ('database', 'file', or 'both')
        """
        logger.info(f"Saving predictions ({output_format})")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save to database
        if output_format in ['database', 'both']:
            if self.db_engine:
                try:
                    predictions.to_sql(
                        'churn_predictions',
                        self.db_engine,
                        if_exists='append',
                        index=False
                    )
                    logger.info(f"Predictions saved to database: {len(predictions)} records")
                except Exception as e:
                    logger.error(f"Error saving to database: {e}")
            else:
                logger.warning("Database engine not available, skipping database save")
        
        # Save to file
        if output_format in ['file', 'both']:
            output_dir = Path('data/predictions')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as parquet
            parquet_path = output_dir / f'predictions_{timestamp}.parquet'
            predictions.to_parquet(parquet_path, index=False)
            logger.info(f"Predictions saved to file: {parquet_path}")
            
            # Save summary as JSON
            summary = {
                'prediction_date': datetime.now().isoformat(),
                'total_customers': len(predictions),
                'high_risk_count': int((predictions['risk_category'] == 'high').sum()),
                'medium_risk_count': int((predictions['risk_category'] == 'medium').sum()),
                'low_risk_count': int((predictions['risk_category'] == 'low').sum()),
                'avg_churn_probability': float(predictions['churn_probability'].mean()),
                'high_risk_customers': predictions[predictions['risk_category'] == 'high']['customer_id'].tolist()[:100]
            }
            
            json_path = output_dir / f'prediction_summary_{timestamp}.json'
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Prediction summary saved: {json_path}")
    
    def get_high_risk_customers(
        self, 
        predictions: pd.DataFrame,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get high-risk customers for immediate action.
        
        Args:
            predictions: Predictions DataFrame
            top_n: Number of top customers to return (None for all high risk)
            
        Returns:
            DataFrame with high-risk customers
        """
        high_risk = predictions[predictions['risk_category'] == 'high'].copy()
        
        if top_n:
            high_risk = high_risk.head(top_n)
        
        logger.info(f"Identified {len(high_risk)} high-risk customers")
        
        return high_risk
    
    def generate_daily_report(self, predictions: pd.DataFrame) -> Dict:
        """
        Generate daily prediction report with statistics.
        
        Args:
            predictions: Predictions DataFrame
            
        Returns:
            Dictionary with report statistics
        """
        report = {
            'report_date': datetime.now().isoformat(),
            'total_customers_scored': len(predictions),
            'risk_distribution': {
                'high': int((predictions['risk_category'] == 'high').sum()),
                'medium': int((predictions['risk_category'] == 'medium').sum()),
                'low': int((predictions['risk_category'] == 'low').sum())
            },
            'statistics': {
                'avg_churn_probability': float(predictions['churn_probability'].mean()),
                'median_churn_probability': float(predictions['churn_probability'].median()),
                'max_churn_probability': float(predictions['churn_probability'].max()),
                'std_churn_probability': float(predictions['churn_probability'].std())
            },
            'top_10_high_risk': predictions.nlargest(10, 'churn_probability')[
                ['customer_id', 'churn_probability', 'risk_category']
            ].to_dict('records')
        }
        
        logger.info("Daily prediction report generated")
        
        return report
