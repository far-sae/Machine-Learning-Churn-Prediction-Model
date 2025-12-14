"""
Feature Engineering Module

Creates 50+ predictive features including:
- RFM (Recency, Frequency, Monetary) metrics
- Behavioral features
- Engagement features
- Transaction patterns
- Temporal features
- Demographic features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from loguru import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Comprehensive feature engineering for churn prediction.
    
    Generates 50+ features across multiple categories:
    - RFM metrics (Recency, Frequency, Monetary)
    - Transaction behavior
    - Engagement patterns
    - Support interactions
    - Subscription dynamics
    - Temporal patterns
    """
    
    def __init__(self, config):
        """
        Initialize feature engineer.
        
        Args:
            config: ConfigLoader instance
        """
        self.config = config
        self.lookback_periods = config.get('features', 'lookback_periods', 
                                          default=[7, 14, 30, 60, 90, 180, 365])
        self.scalers = {}
        self.encoders = {}
    
    def create_all_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create all features from collected data.
        
        Args:
            data: Dictionary containing all collected DataFrames
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting comprehensive feature engineering")
        
        # Start with customer base
        features_df = data['customers'].copy()
        
        # Create feature groups
        features_df = self._create_rfm_features(features_df, data['transactions'])
        features_df = self._create_transaction_features(features_df, data['transactions'])
        features_df = self._create_engagement_features(features_df, data['engagement'])
        features_df = self._create_support_features(features_df, data['support'])
        features_df = self._create_subscription_features(features_df, data['subscriptions'])
        features_df = self._create_temporal_features(features_df)
        features_df = self._create_behavioral_features(features_df, data)
        features_df = self._create_demographic_features(features_df)
        
        # Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        logger.info(f"Feature engineering completed: {len(features_df.columns)} total features")
        
        return features_df
    
    def _create_rfm_features(self, df: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) features.
        
        Features created (per lookback period):
        - Recency: Days since last transaction
        - Frequency: Number of transactions
        - Monetary: Total transaction value, average transaction value
        """
        logger.info("Creating RFM features")
        
        if transactions.empty:
            logger.warning("No transaction data available for RFM features")
            return df
        
        # Ensure date columns are datetime
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        current_date = datetime.now()
        
        for period in self.lookback_periods:
            period_start = current_date - timedelta(days=period)
            period_transactions = transactions[transactions['transaction_date'] >= period_start]
            
            # Aggregate by customer
            rfm_agg = period_transactions.groupby('customer_id').agg({
                'transaction_date': ['max', 'count'],
                'transaction_amount': ['sum', 'mean', 'std', 'min', 'max']
            }).reset_index()
            
            rfm_agg.columns = ['customer_id',
                              f'last_transaction_date_{period}d',
                              f'transaction_count_{period}d',
                              f'total_spent_{period}d',
                              f'avg_transaction_{period}d',
                              f'std_transaction_{period}d',
                              f'min_transaction_{period}d',
                              f'max_transaction_{period}d']
            
            # Calculate recency
            rfm_agg[f'recency_days_{period}d'] = (
                current_date - pd.to_datetime(rfm_agg[f'last_transaction_date_{period}d'])
            ).dt.days
            rfm_agg.drop(f'last_transaction_date_{period}d', axis=1, inplace=True)
            
            # Merge with main dataframe
            df = df.merge(rfm_agg, on='customer_id', how='left')
        
        # Create RFM ratios and trends
        df['spending_trend_30_90'] = df['total_spent_30d'] / (df['total_spent_90d'] + 1)
        df['frequency_trend_30_90'] = df['transaction_count_30d'] / (df['transaction_count_90d'] + 1)
        df['recency_change_30_90'] = df['recency_days_30d'] - df['recency_days_90d']
        
        logger.info(f"Created RFM features for {len(self.lookback_periods)} periods")
        
        return df
    
    def _create_transaction_features(self, df: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Create transaction behavior features.
        
        Features:
        - Payment method diversity
        - Product category diversity
        - Refund rate
        - Discount usage
        - Transaction regularity
        """
        logger.info("Creating transaction behavior features")
        
        if transactions.empty:
            return df
        
        # Payment method features
        payment_diversity = transactions.groupby('customer_id')['payment_method'].nunique().reset_index()
        payment_diversity.columns = ['customer_id', 'payment_method_count']
        df = df.merge(payment_diversity, on='customer_id', how='left')
        
        # Product category features
        category_diversity = transactions.groupby('customer_id')['product_category'].nunique().reset_index()
        category_diversity.columns = ['customer_id', 'product_category_count']
        df = df.merge(category_diversity, on='customer_id', how='left')
        
        # Refund rate
        refund_stats = transactions.groupby('customer_id').agg({
            'is_refunded': ['sum', 'mean'],
            'discount_applied': 'mean',
            'quantity': ['sum', 'mean']
        }).reset_index()
        refund_stats.columns = ['customer_id', 'total_refunds', 'refund_rate', 
                               'avg_discount_rate', 'total_quantity', 'avg_quantity']
        df = df.merge(refund_stats, on='customer_id', how='left')
        
        # Transaction regularity (coefficient of variation of transaction intervals)
        transactions_sorted = transactions.sort_values(['customer_id', 'transaction_date'])
        transactions_sorted['days_between_transactions'] = (
            transactions_sorted.groupby('customer_id')['transaction_date']
            .diff().dt.days
        )
        
        regularity = transactions_sorted.groupby('customer_id')['days_between_transactions'].agg([
            ('avg_days_between_transactions', 'mean'),
            ('std_days_between_transactions', 'std')
        ]).reset_index()
        
        regularity['transaction_regularity_cv'] = (
            regularity['std_days_between_transactions'] / 
            (regularity['avg_days_between_transactions'] + 1)
        )
        
        df = df.merge(regularity, on='customer_id', how='left')
        
        logger.info("Transaction behavior features created")
        
        return df
    
    def _create_engagement_features(self, df: pd.DataFrame, engagement: pd.DataFrame) -> pd.DataFrame:
        """
        Create user engagement features.
        
        Features:
        - Total events, page views, clicks
        - Session metrics
        - Time spent patterns
        - Event diversity
        """
        logger.info("Creating engagement features")
        
        if engagement.empty:
            logger.warning("No engagement data available")
            return df
        
        # Merge engagement data
        df = df.merge(engagement, on='customer_id', how='left')
        
        # Calculate engagement rates
        df['page_view_rate'] = df['page_views'] / (df['total_events'] + 1)
        df['click_rate'] = df['clicks'] / (df['page_views'] + 1)
        df['search_rate'] = df['searches'] / (df['total_events'] + 1)
        df['cart_conversion_rate'] = df['add_to_cart'] / (df['page_views'] + 1)
        
        # Average time per session
        df['avg_time_per_session'] = df['total_time_spent'] / (df['session_count'] + 1)
        
        # Engagement score (composite metric)
        df['engagement_score'] = (
            df['total_events'] * 0.3 +
            df['page_views'] * 0.2 +
            df['clicks'] * 0.2 +
            df['searches'] * 0.15 +
            df['add_to_cart'] * 0.15
        )
        
        logger.info("Engagement features created")
        
        return df
    
    def _create_support_features(self, df: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer support interaction features.
        
        Features:
        - Support ticket count
        - Resolution rate
        - Issue diversity
        """
        logger.info("Creating support interaction features")
        
        if support.empty:
            logger.warning("No support data available")
            return df
        
        df = df.merge(support, on='customer_id', how='left')
        
        # Support interaction rate
        df['support_resolution_rate'] = df['resolved_tickets'] / (df['support_ticket_count'] + 1)
        df['support_open_rate'] = df['open_tickets'] / (df['support_ticket_count'] + 1)
        
        # Days since last ticket
        df['last_ticket_date'] = pd.to_datetime(df['last_ticket_date'])
        df['days_since_last_ticket'] = (datetime.now() - df['last_ticket_date']).dt.days
        
        logger.info("Support features created")
        
        return df
    
    def _create_subscription_features(self, df: pd.DataFrame, subscriptions: pd.DataFrame) -> pd.DataFrame:
        """
        Create subscription-related features.
        
        Features:
        - Subscription tenure
        - Upgrade/downgrade patterns
        - Auto-renewal status
        - Payment frequency
        """
        logger.info("Creating subscription features")
        
        if subscriptions.empty:
            return df
        
        # Get most recent subscription per customer
        latest_subscription = subscriptions.sort_values('subscription_start_date', ascending=False)\
                                          .groupby('customer_id').first().reset_index()
        
        # Calculate subscription tenure
        latest_subscription['subscription_start_date'] = pd.to_datetime(
            latest_subscription['subscription_start_date']
        )
        latest_subscription['subscription_tenure_days'] = (
            datetime.now() - latest_subscription['subscription_start_date']
        ).dt.days
        
        # Select relevant columns
        subscription_features = latest_subscription[[
            'customer_id',
            'subscription_tenure_days',
            'monthly_fee',
            'auto_renewal_enabled',
            'upgrade_count',
            'downgrade_count',
            'reactivation_count'
        ]]
        
        df = df.merge(subscription_features, on='customer_id', how='left')
        
        # Subscription stability score
        df['upgrade_downgrade_ratio'] = df['upgrade_count'] / (df['downgrade_count'] + 1)
        df['subscription_stability_score'] = (
            df['subscription_tenure_days'] * 0.4 +
            df['auto_renewal_enabled'].astype(int) * 100 * 0.3 +
            df['upgrade_downgrade_ratio'] * 50 * 0.3
        )
        
        logger.info("Subscription features created")
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Features:
        - Customer age
        - Days since last login
        - Registration day of week, month
        """
        logger.info("Creating temporal features")
        
        # Ensure date columns are datetime
        df['registration_date'] = pd.to_datetime(df['registration_date'])
        df['last_login_date'] = pd.to_datetime(df['last_login_date'])
        
        # Registration temporal features
        df['registration_day_of_week'] = df['registration_date'].dt.dayofweek
        df['registration_month'] = df['registration_date'].dt.month
        df['registration_quarter'] = df['registration_date'].dt.quarter
        
        # Login activity
        df['login_frequency_score'] = 365 / (df['days_since_last_login'] + 1)
        df['customer_lifecycle_stage'] = pd.cut(
            df['customer_age_days'],
            bins=[0, 30, 90, 180, 365, float('inf')],
            labels=['new', 'growing', 'established', 'mature', 'veteran']
        )
        
        logger.info("Temporal features created")
        
        return df
    
    def _create_behavioral_features(self, df: pd.DataFrame, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create advanced behavioral features combining multiple data sources.
        
        Features:
        - Customer value score
        - Churn risk indicators
        - Activity trends
        """
        logger.info("Creating behavioral features")
        
        # Customer Lifetime Value (CLV) estimate
        df['estimated_clv'] = (
            df['total_spent_365d'] / (df['customer_age_days'] + 1) * 365 * 2
        )
        
        # Activity decline indicator
        df['activity_decline_30_90'] = (
            (df['transaction_count_90d'] - df['transaction_count_30d'] * 3) /
            (df['transaction_count_90d'] + 1)
        )
        
        # Engagement decline
        if 'total_events' in df.columns:
            df['engagement_per_day'] = df['total_events'] / (df['customer_age_days'] + 1)
        
        # Multi-channel engagement
        df['is_multi_channel'] = (df['payment_method_count'] > 1).astype(int)
        
        # Premium customer indicator
        df['is_high_value'] = (
            df['total_spent_365d'] > df['total_spent_365d'].quantile(0.75)
        ).astype(int)
        
        # At-risk indicators
        df['high_recency_flag'] = (df['recency_days_30d'] > 30).astype(int)
        df['declining_frequency_flag'] = (df['frequency_trend_30_90'] < 0.5).astype(int)
        df['declining_spending_flag'] = (df['spending_trend_30_90'] < 0.5).astype(int)
        
        # Composite risk score
        df['preliminary_risk_score'] = (
            df['high_recency_flag'] * 0.3 +
            df['declining_frequency_flag'] * 0.3 +
            df['declining_spending_flag'] * 0.2 +
            (1 - df['login_frequency_score'] / df['login_frequency_score'].max()) * 0.2
        )
        
        logger.info("Behavioral features created")
        
        return df
    
    def _create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process demographic features.
        
        Features:
        - Age groups
        - Geographic features
        - Customer tier encoding
        """
        logger.info("Creating demographic features")
        
        # Age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0, 18, 25, 35, 45, 55, 65, 100],
                labels=['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
            )
        
        # Encode categorical variables
        categorical_cols = ['gender', 'country', 'customer_tier', 'subscription_type', 
                          'referral_source', 'customer_lifecycle_stage']
        
        for col in categorical_cols:
            if col in df.columns:
                # Create frequency encoding
                freq_encoding = df[col].value_counts(normalize=True).to_dict()
                df[f'{col}_frequency'] = df[col].map(freq_encoding)
        
        logger.info("Demographic features created")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values intelligently.
        
        Strategy:
        - Numeric: Fill with 0 or median based on feature type
        - Categorical: Fill with 'unknown'
        """
        logger.info("Handling missing values")
        
        # Fill numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Features that should be 0 when missing (counts, amounts)
        zero_fill_patterns = ['count', 'total', 'sum', 'rate', 'score', 'days_since']
        
        for col in numeric_cols:
            if any(pattern in col.lower() for pattern in zero_fill_patterns):
                df[col].fillna(0, inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            df[col].fillna('unknown', inplace=True)
        
        # Handle boolean columns
        boolean_cols = df.select_dtypes(include=['bool']).columns
        for col in boolean_cols:
            df[col].fillna(False, inplace=True)
        
        logger.info(f"Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")
        
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Return feature groups for analysis and interpretation.
        
        Returns:
            Dictionary mapping feature group names to feature lists
        """
        return {
            'rfm': [col for col in ['recency_days', 'transaction_count', 'total_spent', 
                                   'avg_transaction'] if any(p in col for p in ['30d', '90d', '365d'])],
            'engagement': ['page_views', 'clicks', 'searches', 'engagement_score', 
                          'page_view_rate', 'click_rate'],
            'support': ['support_ticket_count', 'support_resolution_rate', 
                       'days_since_last_ticket'],
            'subscription': ['subscription_tenure_days', 'auto_renewal_enabled', 
                           'upgrade_downgrade_ratio'],
            'behavioral': ['activity_decline_30_90', 'preliminary_risk_score', 
                          'estimated_clv'],
            'demographic': ['age', 'gender_frequency', 'country_frequency', 
                          'customer_tier_frequency']
        }
