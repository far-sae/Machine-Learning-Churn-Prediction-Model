"""
Data Collection Module

This module aggregates customer behavior, transaction, and engagement data 
from multiple sources including PostgreSQL, MongoDB, and external APIs.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
from sqlalchemy import create_engine, text
from pymongo import MongoClient
import requests


class DataCollector:
    """
    Comprehensive data collector that aggregates customer data from various sources.
    
    Sources include:
    - PostgreSQL: Customer profiles, transactions, subscriptions
    - MongoDB: User engagement events, clickstream data
    - REST APIs: Real-time transaction data, customer interactions
    """
    
    def __init__(self, config):
        """
        Initialize data collector with configuration.
        
        Args:
            config: ConfigLoader instance containing data source configurations
        """
        self.config = config
        self.postgres_engine = None
        self.mongo_client = None
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections."""
        try:
            # PostgreSQL connection
            pg_config = self.config.get('data', 'postgres')
            connection_string = (
                f"postgresql://{pg_config['user']}:{pg_config['password']}"
                f"@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
            )
            self.postgres_engine = create_engine(connection_string)
            logger.info("PostgreSQL connection established")
            
            # MongoDB connection
            mongo_config = self.config.get('data', 'mongodb')
            self.mongo_client = MongoClient(mongo_config['connection_string'])
            self.mongo_db = self.mongo_client[mongo_config['database']]
            logger.info("MongoDB connection established")
            
        except Exception as e:
            logger.error(f"Error initializing database connections: {e}")
            raise
    
    def collect_customer_profiles(self, lookback_days: int = 365) -> pd.DataFrame:
        """
        Collect customer profile data from PostgreSQL.
        
        Args:
            lookback_days: Number of days to look back for customer data
            
        Returns:
            DataFrame containing customer profile information
        """
        logger.info(f"Collecting customer profiles (lookback: {lookback_days} days)")
        
        query = text("""
            SELECT 
                customer_id,
                registration_date,
                account_status,
                customer_tier,
                country,
                city,
                age,
                gender,
                referral_source,
                last_login_date,
                is_premium,
                subscription_type,
                account_balance,
                CURRENT_DATE - registration_date::date as customer_age_days,
                CURRENT_DATE - last_login_date::date as days_since_last_login
            FROM customers
            WHERE registration_date >= CURRENT_DATE - :lookback_days
                AND account_status IN ('active', 'at_risk')
            ORDER BY customer_id
        """)
        
        try:
            df = pd.read_sql(query, self.postgres_engine, params={'lookback_days': lookback_days})
            logger.info(f"Retrieved {len(df)} customer profiles")
            return df
        except Exception as e:
            logger.error(f"Error collecting customer profiles: {e}")
            raise
    
    def collect_transaction_data(self, lookback_days: int = 365) -> pd.DataFrame:
        """
        Collect transaction data from PostgreSQL and API.
        
        Args:
            lookback_days: Number of days to look back for transactions
            
        Returns:
            DataFrame containing transaction information
        """
        logger.info(f"Collecting transaction data (lookback: {lookback_days} days)")
        
        query = text("""
            SELECT 
                customer_id,
                transaction_id,
                transaction_date,
                transaction_amount,
                transaction_type,
                payment_method,
                product_category,
                quantity,
                discount_applied,
                is_refunded
            FROM transactions
            WHERE transaction_date >= CURRENT_DATE - :lookback_days
            ORDER BY customer_id, transaction_date DESC
        """)
        
        try:
            df = pd.read_sql(query, self.postgres_engine, params={'lookback_days': lookback_days})
            logger.info(f"Retrieved {len(df)} transactions")
            return df
        except Exception as e:
            logger.error(f"Error collecting transaction data: {e}")
            raise
    
    def collect_engagement_data(self, lookback_days: int = 365) -> pd.DataFrame:
        """
        Collect user engagement data from MongoDB.
        
        Args:
            lookback_days: Number of days to look back for engagement events
            
        Returns:
            DataFrame containing engagement metrics
        """
        logger.info(f"Collecting engagement data (lookback: {lookback_days} days)")
        
        try:
            collection_name = self.config.get('data', 'mongodb', 'collection')
            collection = self.mongo_db[collection_name]
            
            # Calculate date threshold
            date_threshold = datetime.now() - timedelta(days=lookback_days)
            
            # Query MongoDB for engagement events
            pipeline = [
                {
                    '$match': {
                        'event_timestamp': {'$gte': date_threshold}
                    }
                },
                {
                    '$group': {
                        '_id': '$customer_id',
                        'total_events': {'$sum': 1},
                        'unique_event_types': {'$addToSet': '$event_type'},
                        'page_views': {
                            '$sum': {'$cond': [{'$eq': ['$event_type', 'page_view']}, 1, 0]}
                        },
                        'clicks': {
                            '$sum': {'$cond': [{'$eq': ['$event_type', 'click']}, 1, 0]}
                        },
                        'searches': {
                            '$sum': {'$cond': [{'$eq': ['$event_type', 'search']}, 1, 0]}
                        },
                        'add_to_cart': {
                            '$sum': {'$cond': [{'$eq': ['$event_type', 'add_to_cart']}, 1, 0]}
                        },
                        'session_count': {'$sum': '$session_id'},
                        'total_time_spent': {'$sum': '$time_spent_seconds'},
                        'last_event_date': {'$max': '$event_timestamp'}
                    }
                }
            ]
            
            results = list(collection.aggregate(pipeline))
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            if not df.empty:
                df.rename(columns={'_id': 'customer_id'}, inplace=True)
                df['unique_event_type_count'] = df['unique_event_types'].apply(len)
                df.drop('unique_event_types', axis=1, inplace=True)
            
            logger.info(f"Retrieved engagement data for {len(df)} customers")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting engagement data: {e}")
            raise
    
    def collect_support_interactions(self, lookback_days: int = 365) -> pd.DataFrame:
        """
        Collect customer support interaction data.
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame containing support interaction metrics
        """
        logger.info(f"Collecting support interaction data (lookback: {lookback_days} days)")
        
        query = text("""
            SELECT 
                customer_id,
                COUNT(*) as support_ticket_count,
                SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved_tickets,
                SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open_tickets,
                AVG(resolution_time_hours) as avg_resolution_time,
                MAX(created_date) as last_ticket_date,
                COUNT(DISTINCT issue_category) as unique_issue_categories
            FROM support_tickets
            WHERE created_date >= CURRENT_DATE - :lookback_days
            GROUP BY customer_id
        """)
        
        try:
            df = pd.read_sql(query, self.postgres_engine, params={'lookback_days': lookback_days})
            logger.info(f"Retrieved support data for {len(df)} customers")
            return df
        except Exception as e:
            logger.error(f"Error collecting support data: {e}")
            raise
    
    def collect_subscription_history(self) -> pd.DataFrame:
        """
        Collect subscription history and changes.
        
        Returns:
            DataFrame containing subscription history
        """
        logger.info("Collecting subscription history")
        
        query = text("""
            SELECT 
                customer_id,
                subscription_start_date,
                subscription_end_date,
                subscription_type,
                monthly_fee,
                payment_frequency,
                auto_renewal_enabled,
                upgrade_count,
                downgrade_count,
                cancellation_date,
                reactivation_count
            FROM subscription_history
            WHERE subscription_start_date IS NOT NULL
            ORDER BY customer_id, subscription_start_date DESC
        """)
        
        try:
            df = pd.read_sql(query, self.postgres_engine)
            logger.info(f"Retrieved subscription history for {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error collecting subscription history: {e}")
            raise
    
    def collect_all_data(self, lookback_days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Collect all data from various sources.
        
        Args:
            lookback_days: Number of days to look back for historical data
            
        Returns:
            Dictionary containing all collected DataFrames
        """
        logger.info(f"Starting comprehensive data collection (lookback: {lookback_days} days)")
        
        data = {}
        
        try:
            # Collect from all sources
            data['customers'] = self.collect_customer_profiles(lookback_days)
            data['transactions'] = self.collect_transaction_data(lookback_days)
            data['engagement'] = self.collect_engagement_data(lookback_days)
            data['support'] = self.collect_support_interactions(lookback_days)
            data['subscriptions'] = self.collect_subscription_history()
            
            logger.info("Data collection completed successfully")
            logger.info(f"Summary: {len(data['customers'])} customers, "
                       f"{len(data['transactions'])} transactions, "
                       f"{len(data['engagement'])} engagement records")
            
            return data
            
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
            raise
    
    def save_raw_data(self, data: Dict[str, pd.DataFrame], output_dir: str = 'data/raw'):
        """
        Save collected data to disk.
        
        Args:
            data: Dictionary of DataFrames to save
            output_dir: Directory to save data files
        """
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, df in data.items():
            filename = output_path / f"{name}_{timestamp}.parquet"
            df.to_parquet(filename, index=False)
            logger.info(f"Saved {name} data to {filename}")
    
    def close_connections(self):
        """Close all database connections."""
        if self.postgres_engine:
            self.postgres_engine.dispose()
            logger.info("PostgreSQL connection closed")
        
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("MongoDB connection closed")
