"""
CRM Integration Module

Automatically triggers targeted retention campaigns based on churn predictions.
Integrates with external CRM systems to execute personalized interventions.
"""

import pandas as pd
import requests
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
import time


class CRMIntegration:
    """
    CRM integration for automated retention campaign triggering.
    
    Features:
    - Automatic campaign triggering based on risk level
    - Multi-channel campaign support (email, SMS, push)
    - Campaign tracking and logging
    - Batch campaign creation
    - Rate limiting and retry logic
    """
    
    def __init__(self, config):
        """
        Initialize CRM integration.
        
        Args:
            config: ConfigLoader instance
        """
        self.config = config
        self.api_endpoint = config.get('crm', 'api_endpoint')
        self.api_key = config.get('crm', 'api_key')
        self.campaign_templates = config.get('crm', 'campaign_templates')
        
        # Rate limiting
        self.max_requests_per_minute = 60
        self.request_interval = 60 / self.max_requests_per_minute
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Implement rate limiting for API calls."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.request_interval:
            time.sleep(self.request_interval - time_since_last_request)
        
        self.last_request_time = time.time()
    
    def _make_api_request(
        self, 
        endpoint: str, 
        method: str = 'POST',
        data: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Dict:
        """
        Make API request with retry logic.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            data: Request payload
            max_retries: Maximum number of retry attempts
            
        Returns:
            API response as dictionary
        """
        self._rate_limit()
        
        url = f"{self.api_endpoint}/{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        for attempt in range(max_retries):
            try:
                if method == 'POST':
                    response = requests.post(url, json=data, headers=headers, timeout=30)
                elif method == 'GET':
                    response = requests.get(url, headers=headers, timeout=30)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"API request failed after {max_retries} attempts")
                    raise
    
    def create_retention_campaign(
        self, 
        customer_id: str,
        risk_category: str,
        churn_probability: float,
        additional_data: Optional[Dict] = None
    ) -> Dict:
        """
        Create a retention campaign for a customer.
        
        Args:
            customer_id: Customer identifier
            risk_category: Risk level (high/medium/low)
            churn_probability: Predicted churn probability
            additional_data: Additional customer data for personalization
            
        Returns:
            Campaign creation response
        """
        logger.info(f"Creating retention campaign for customer {customer_id} ({risk_category} risk)")
        
        # Get campaign template based on risk category
        if risk_category not in self.campaign_templates:
            logger.warning(f"No campaign template for risk category: {risk_category}")
            return {'status': 'skipped', 'reason': 'no_template'}
        
        template = self.campaign_templates[risk_category]
        
        # Prepare campaign data
        campaign_data = {
            'customer_id': customer_id,
            'campaign_id': template['campaign_id'],
            'channels': template['channel'],
            'priority': template['priority'],
            'churn_probability': churn_probability,
            'risk_category': risk_category,
            'created_at': datetime.now().isoformat(),
            'personalization': self._generate_personalization(
                customer_id, risk_category, churn_probability, additional_data
            )
        }
        
        try:
            response = self._make_api_request('campaigns/create', method='POST', data=campaign_data)
            logger.info(f"Campaign created successfully for customer {customer_id}")
            return response
        except Exception as e:
            logger.error(f"Failed to create campaign for customer {customer_id}: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _generate_personalization(
        self,
        customer_id: str,
        risk_category: str,
        churn_probability: float,
        additional_data: Optional[Dict] = None
    ) -> Dict:
        """
        Generate personalized content for retention campaigns.
        
        Args:
            customer_id: Customer identifier
            risk_category: Risk level
            churn_probability: Churn probability
            additional_data: Additional customer data
            
        Returns:
            Personalization data dictionary
        """
        personalization = {
            'customer_id': customer_id,
            'urgency_level': self._map_risk_to_urgency(risk_category)
        }
        
        # Add personalized offers based on risk level
        if risk_category == 'high':
            personalization.update({
                'offer_type': 'premium_discount',
                'discount_percentage': 30,
                'message_tone': 'urgent',
                'incentive': 'exclusive_benefits'
            })
        elif risk_category == 'medium':
            personalization.update({
                'offer_type': 'standard_discount',
                'discount_percentage': 15,
                'message_tone': 'friendly',
                'incentive': 'loyalty_rewards'
            })
        else:
            personalization.update({
                'offer_type': 'engagement',
                'discount_percentage': 0,
                'message_tone': 'informative',
                'incentive': 'feature_awareness'
            })
        
        # Include additional data if provided
        if additional_data:
            personalization.update(additional_data)
        
        return personalization
    
    def _map_risk_to_urgency(self, risk_category: str) -> str:
        """Map risk category to urgency level."""
        mapping = {
            'high': 'critical',
            'medium': 'moderate',
            'low': 'routine'
        }
        return mapping.get(risk_category, 'routine')
    
    def trigger_campaigns_batch(
        self, 
        predictions: pd.DataFrame,
        risk_filter: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Trigger retention campaigns for multiple customers.
        
        Args:
            predictions: DataFrame with customer predictions
            risk_filter: List of risk categories to include (None for all)
            
        Returns:
            DataFrame with campaign triggering results
        """
        logger.info(f"Triggering batch campaigns for {len(predictions)} customers")
        
        # Filter by risk category if specified
        if risk_filter:
            predictions = predictions[predictions['risk_category'].isin(risk_filter)]
            logger.info(f"Filtered to {len(predictions)} customers with risk: {risk_filter}")
        
        results = []
        
        for idx, row in predictions.iterrows():
            try:
                campaign_result = self.create_retention_campaign(
                    customer_id=row['customer_id'],
                    risk_category=row['risk_category'],
                    churn_probability=row['churn_probability']
                )
                
                results.append({
                    'customer_id': row['customer_id'],
                    'risk_category': row['risk_category'],
                    'churn_probability': row['churn_probability'],
                    'campaign_status': campaign_result.get('status', 'unknown'),
                    'campaign_id': campaign_result.get('campaign_id'),
                    'triggered_at': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error triggering campaign for customer {row['customer_id']}: {e}")
                results.append({
                    'customer_id': row['customer_id'],
                    'risk_category': row['risk_category'],
                    'campaign_status': 'error',
                    'error_message': str(e),
                    'triggered_at': datetime.now().isoformat()
                })
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Batch campaign triggering completed. "
                   f"Success: {(results_df['campaign_status']=='success').sum()}, "
                   f"Failed: {(results_df['campaign_status']=='error').sum()}")
        
        return results_df
    
    def trigger_high_risk_campaigns(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Trigger campaigns specifically for high-risk customers.
        
        Args:
            predictions: DataFrame with customer predictions
            
        Returns:
            DataFrame with campaign results
        """
        logger.info("Triggering campaigns for high-risk customers")
        return self.trigger_campaigns_batch(predictions, risk_filter=['high'])
    
    def get_campaign_status(self, campaign_id: str) -> Dict:
        """
        Check the status of a triggered campaign.
        
        Args:
            campaign_id: Campaign identifier
            
        Returns:
            Campaign status information
        """
        try:
            response = self._make_api_request(f'campaigns/{campaign_id}/status', method='GET')
            return response
        except Exception as e:
            logger.error(f"Failed to get campaign status for {campaign_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def update_customer_segment(
        self, 
        customer_id: str,
        segment: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Update customer segment in CRM based on churn risk.
        
        Args:
            customer_id: Customer identifier
            segment: Segment name (e.g., 'at_risk', 'loyal')
            metadata: Additional metadata
            
        Returns:
            Update response
        """
        logger.info(f"Updating CRM segment for customer {customer_id} to '{segment}'")
        
        data = {
            'customer_id': customer_id,
            'segment': segment,
            'updated_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        try:
            response = self._make_api_request('customers/segment', method='POST', data=data)
            logger.info(f"Segment updated for customer {customer_id}")
            return response
        except Exception as e:
            logger.error(f"Failed to update segment for customer {customer_id}: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def sync_churn_predictions_to_crm(self, predictions: pd.DataFrame) -> Dict:
        """
        Synchronize all churn predictions to CRM system.
        
        Args:
            predictions: DataFrame with customer predictions
            
        Returns:
            Sync summary
        """
        logger.info(f"Syncing {len(predictions)} churn predictions to CRM")
        
        sync_results = {
            'total_customers': len(predictions),
            'successful_syncs': 0,
            'failed_syncs': 0,
            'sync_timestamp': datetime.now().isoformat()
        }
        
        for idx, row in predictions.iterrows():
            try:
                # Prepare prediction data for CRM
                prediction_data = {
                    'customer_id': row['customer_id'],
                    'churn_probability': float(row['churn_probability']),
                    'risk_category': row['risk_category'],
                    'prediction_date': row['prediction_date'].isoformat() if hasattr(row['prediction_date'], 'isoformat') else str(row['prediction_date']),
                    'model_version': 'ensemble_v1'
                }
                
                # Send to CRM
                response = self._make_api_request(
                    'predictions/sync',
                    method='POST',
                    data=prediction_data
                )
                
                if response.get('status') == 'success':
                    sync_results['successful_syncs'] += 1
                else:
                    sync_results['failed_syncs'] += 1
                    
            except Exception as e:
                logger.error(f"Failed to sync prediction for customer {row['customer_id']}: {e}")
                sync_results['failed_syncs'] += 1
        
        logger.info(f"CRM sync completed: {sync_results['successful_syncs']} successful, "
                   f"{sync_results['failed_syncs']} failed")
        
        return sync_results
    
    def generate_campaign_report(self, campaign_results: pd.DataFrame) -> Dict:
        """
        Generate report on campaign triggering results.
        
        Args:
            campaign_results: DataFrame with campaign results
            
        Returns:
            Campaign report dictionary
        """
        report = {
            'report_date': datetime.now().isoformat(),
            'total_campaigns': len(campaign_results),
            'campaigns_by_risk': campaign_results['risk_category'].value_counts().to_dict(),
            'campaigns_by_status': campaign_results['campaign_status'].value_counts().to_dict(),
            'success_rate': (
                (campaign_results['campaign_status'] == 'success').sum() / 
                len(campaign_results) * 100
            ) if len(campaign_results) > 0 else 0,
            'avg_churn_probability': float(campaign_results['churn_probability'].mean())
        }
        
        logger.info(f"Campaign report generated: {report['total_campaigns']} campaigns, "
                   f"{report['success_rate']:.2f}% success rate")
        
        return report
    
    def create_custom_campaign(
        self,
        customer_ids: List[str],
        campaign_name: str,
        channels: List[str],
        message: str,
        offer: Optional[Dict] = None
    ) -> Dict:
        """
        Create a custom retention campaign.
        
        Args:
            customer_ids: List of customer identifiers
            campaign_name: Name of the campaign
            channels: List of channels to use
            message: Campaign message
            offer: Optional offer details
            
        Returns:
            Campaign creation response
        """
        logger.info(f"Creating custom campaign '{campaign_name}' for {len(customer_ids)} customers")
        
        campaign_data = {
            'campaign_name': campaign_name,
            'customer_ids': customer_ids,
            'channels': channels,
            'message': message,
            'offer': offer or {},
            'created_at': datetime.now().isoformat(),
            'campaign_type': 'custom_retention'
        }
        
        try:
            response = self._make_api_request('campaigns/custom', method='POST', data=campaign_data)
            logger.info(f"Custom campaign '{campaign_name}' created successfully")
            return response
        except Exception as e:
            logger.error(f"Failed to create custom campaign: {e}")
            return {'status': 'failed', 'error': str(e)}
