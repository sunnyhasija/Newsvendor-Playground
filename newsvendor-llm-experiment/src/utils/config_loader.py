"""
Configuration Management for Newsvendor Experiment

Loads and validates configuration from YAML files with intelligent defaults
and environment variable overrides.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML files with intelligent defaults.
    
    Args:
        config_path: Optional path to main config file
        
    Returns:
        Combined configuration dictionary
    """
    
    # Default configuration
    default_config = {
        'game': {
            'selling_price': 100,
            'production_cost': 30,
            'demand_mean': 40,      # CORRECTED: Normal(40, 10)
            'demand_std': 10,       # CORRECTED: Normal(40, 10)  
            'optimal_price': 65,    # Fair split-the-difference
            'max_rounds': 10,
            'timeout_seconds': 60
        },
        'technical': {
            'max_concurrent_models': 1,
            'memory_limit_gb': 40,
            'retry_attempts': 2,
            'validation_checks': True
        },
        'metrics': {
            'target_completion_rate': 0.95,
            'target_convergence_rate': 0.85,
            'target_price_optimality': 8,
            'target_efficiency': 2000,
            'target_invalid_rate': 0.05
        },
        'storage': {
            'output_dir': './data',
            'backup_enabled': True,
            'compression_enabled': True,
            'export_formats': ['csv', 'json']
        },
        'models': {
            'tinyllama:latest': {
                'tier': 'ultra',
                'token_limit': 256,
                'temperature': 0.3,
                'top_p': 0.8
            },
            'qwen2:1.5b': {
                'tier': 'ultra', 
                'token_limit': 256,
                'temperature': 0.3,
                'top_p': 0.8
            },
            'gemma2:2b': {
                'tier': 'compact',
                'token_limit': 384,
                'temperature': 0.4,
                'top_p': 0.85
            },
            'phi3:mini': {
                'tier': 'compact',
                'token_limit': 384,
                'temperature': 0.4,
                'top_p': 0.85
            },
            'llama3.2:latest': {
                'tier': 'compact',
                'token_limit': 384,
                'temperature': 0.4,
                'top_p': 0.85
            },
            'mistral:instruct': {
                'tier': 'mid',
                'token_limit': 512,
                'temperature': 0.5,
                'top_p': 0.9
            },
            'qwen:7b': {
                'tier': 'mid',
                'token_limit': 512,
                'temperature': 0.5,
                'top_p': 0.9
            },
            'qwen3:latest': {
                'tier': 'large',
                'token_limit': 512,
                'temperature': 0.6,
                'top_p': 0.95
            }
        }
    }
    
    # Try to find config files
    config_files = []
    
    # 1. Explicit config path
    if config_path and Path(config_path).exists():
        config_files.append(Path(config_path))
    
    # 2. Standard locations
    standard_locations = [
        Path('./config/experiment.yaml'),
        Path('./config/experiment.yml'),
        Path('./experiment.yaml'),
        Path('./experiment.yml')
    ]
    
    for location in standard_locations:
        if location.exists():
            config_files.append(location)
            break
    
    # Load and merge configurations
    config = default_config.copy()
    
    for config_file in config_files:
        try:
            logger.info(f"Loading configuration from: {config_file}")
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                config = deep_merge(config, file_config)
                
        except Exception as e:
            logger.error(f"Error loading config from {config_file}: {e}")
    
    # Apply environment variable overrides
    config = apply_env_overrides(config)
    
    # Validate configuration
    validate_config(config)
    
    logger.info("Configuration loaded successfully")
    return config


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary to merge in
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration.
    
    Environment variables should be prefixed with NEWSVENDOR_ and use
    double underscores to indicate nesting (e.g., NEWSVENDOR_GAME__MAX_ROUNDS=15)
    """
    
    env_overrides = {}
    prefix = "NEWSVENDOR_"
    
    for env_var, value in os.environ.items():
        if not env_var.startswith(prefix):
            continue
        
        # Remove prefix and convert to nested dict structure
        key_path = env_var[len(prefix):].lower().split('__')
        
        # Convert string values to appropriate types
        converted_value = convert_env_value(value)
        
        # Build nested dictionary
        current = env_overrides
        for key in key_path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[key_path[-1]] = converted_value
    
    if env_overrides:
        logger.info(f"Applying environment overrides: {list(env_overrides.keys())}")
        config = deep_merge(config, env_overrides)
    
    return config


def convert_env_value(value: str) -> Any:
    """Convert environment variable string to appropriate Python type."""
    
    # Boolean conversion
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    
    # Integer conversion
    try:
        if '.' not in value:
            return int(value)
    except ValueError:
        pass
    
    # Float conversion
    try:
        return float(value)
    except ValueError:
        pass
    
    # List conversion (comma-separated)
    if ',' in value:
        return [item.strip() for item in value.split(',')]
    
    # Return as string
    return value


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration for correctness and consistency.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    
    # Validate game parameters
    game = config.get('game', {})
    
    if game.get('selling_price', 0) <= 0:
        raise ValueError("selling_price must be positive")
    
    if game.get('production_cost', 0) <= 0:
        raise ValueError("production_cost must be positive")
    
    if game.get('production_cost', 0) >= game.get('selling_price', 100):
        raise ValueError("production_cost must be less than selling_price")
    
    if game.get('demand_mean', 0) <= 0:
        raise ValueError("demand_mean must be positive")
    
    if game.get('demand_std', 0) <= 0:
        raise ValueError("demand_std must be positive")
    
    if game.get('max_rounds', 0) <= 0:
        raise ValueError("max_rounds must be positive")
    
    # Validate technical parameters
    technical = config.get('technical', {})
    
    if technical.get('max_concurrent_models', 0) <= 0:
        raise ValueError("max_concurrent_models must be positive")
    
    if technical.get('memory_limit_gb', 0) <= 0:
        raise ValueError("memory_limit_gb must be positive")
    
    # Validate model configurations
    models = config.get('models', {})
    
    for model_name, model_config in models.items():
        if not isinstance(model_config, dict):
            raise ValueError(f"Model config for {model_name} must be a dictionary")
        
        if model_config.get('token_limit', 0) <= 0:
            raise ValueError(f"token_limit for {model_name} must be positive")
        
        temp = model_config.get('temperature', 0.5)
        if not (0.0 <= temp <= 2.0):
            raise ValueError(f"temperature for {model_name} must be between 0.0 and 2.0")
        
        top_p = model_config.get('top_p', 0.9)
        if not (0.0 <= top_p <= 1.0):
            raise ValueError(f"top_p for {model_name} must be between 0.0 and 1.0")
    
    logger.info("Configuration validation passed")


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration saved to: {output_file}")


def get_model_list(config: Dict[str, Any]) -> list[str]:
    """Get list of available models from configuration."""
    return list(config.get('models', {}).keys())


def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    models = config.get('models', {})
    return models.get(model_name, {})


def create_default_config_files() -> None:
    """Create default configuration files in the config directory."""
    
    config_dir = Path('./config')
    config_dir.mkdir(exist_ok=True)
    
    # Load default config
    default_config = load_config()
    
    # Split into separate files
    files_to_create = {
        'experiment.yaml': {
            'game': default_config['game'],
            'technical': default_config['technical'],
            'metrics': default_config['metrics'],
            'storage': default_config['storage']
        },
        'models.yaml': {
            'models': default_config['models']
        }
    }
    
    for filename, file_config in files_to_create.items():
        file_path = config_dir / filename
        if not file_path.exists():
            with open(file_path, 'w') as f:
                yaml.dump(file_config, f, default_flow_style=False, indent=2)
            logger.info(f"Created default config: {file_path}")


if __name__ == '__main__':
    # Create default configuration files
    create_default_config_files()
    
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully:")
    print(yaml.dump(config, default_flow_style=False, indent=2))