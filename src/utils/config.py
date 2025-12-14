"""
Utility functions for loading configuration and setting up logging.
"""

import os
import yaml
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from string import Template


class ConfigLoader:
    """Load and manage configuration from YAML files with environment variable substitution."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        load_dotenv()  # Load environment variables from .env file
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file with environment variable substitution."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_content = f.read()
        
        # Substitute environment variables
        config_content = Template(config_content).safe_substitute(os.environ)
        
        return yaml.safe_load(config_content)
    
    def get(self, *keys, default=None):
        """
        Get configuration value using dot notation.
        
        Args:
            *keys: Configuration keys (e.g., 'data', 'postgres', 'host')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def get_all(self) -> dict:
        """Get the entire configuration dictionary."""
        return self.config


def setup_logging(config: ConfigLoader):
    """
    Configure logging based on configuration settings.
    
    Args:
        config: ConfigLoader instance
    """
    log_config = config.get('logging')
    
    # Create log directory if it doesn't exist
    log_dir = Path(log_config.get('log_dir', './logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sink=lambda msg: print(msg, end=''),
        format=log_config.get('format'),
        level=log_config.get('level', 'INFO'),
        colorize=True
    )
    
    # Add file logger with rotation
    logger.add(
        sink=log_dir / "churn_prediction.log",
        format=log_config.get('format'),
        level=log_config.get('level', 'INFO'),
        rotation=log_config.get('rotation', '100 MB'),
        retention=log_config.get('retention', '30 days'),
        compression="zip"
    )
    
    logger.info("Logging initialized successfully")
    return logger


def ensure_directories():
    """Create necessary directories for the project."""
    directories = [
        'data/raw',
        'data/processed',
        'data/predictions',
        'models',
        'logs',
        'mlruns',
        'reports/figures'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
