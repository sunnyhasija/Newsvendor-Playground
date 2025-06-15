#!/usr/bin/env python3
"""
Validation Suite for Newsvendor Experiment

Validates experimental setup, model availability, and basic functionality
before running the full experiment.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import click
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from ..core.model_manager import OptimizedModelManager
from ..core.negotiation_engine import NegotiationEngine, NegotiationConfig
from ..utils.config_loader import load_config
from ..parsing.price_extractor import RobustPriceExtractor
from ..parsing.acceptance_detector import AcceptanceDetector


class ValidationSuite:
    """Comprehensive validation suite for the newsvendor experiment."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize validation suite."""
        self.config = load_config(config_path)
        self.model_manager = None
        self.negotiation_engine = None
        
        # Standard models to validate
        self.models = [
            "tinyllama:latest",
            "qwen2:1.5b", 
            "gemma2:2b",
            "phi3:mini",
            "llama3.2:latest",
            "mistral:instruct",
            "qwen:7b",
            "qwen3:latest"
        ]
        
        logger.info("Initialized ValidationSuite")
    
    async def run_dry_run(self) -> Dict[str, Any]:
        """Run dry-run validation without actually loading models."""
        logger.info("=== DRY RUN VALIDATION ===")
        
        results = {
            "status": "success",
            "checks": {},
            "warnings": [],
            "errors": []
        }
        
        # Check 1: Configuration validation
        logger.info("Checking configuration...")
        try:
            config_check = self._validate_configuration()
            results["checks"]["configuration"] = config_check
            if not config_check["valid"]:
                results["errors"].extend(config_check["errors"])
        except Exception as e:
            results["errors"].append(f"Configuration check failed: {e}")
        
        # Check 2: Module imports
        logger.info("Checking module imports...")
        try:
            import_check = self._validate_imports()
            results["checks"]["imports"] = import_check
            if not import_check["valid"]:
                results["errors"].extend(import_check["errors"])
        except Exception as e:
            results["errors"].append(f"Import check failed: {e}")
        
        # Check 3: Ollama availability
        logger.info("Checking Ollama availability...")
        try:
            ollama_check = await self._check_ollama_availability()
            results["checks"]["ollama"] = ollama_check
            if not ollama_check["available"]:
                results["errors"].append("Ollama is not available")
        except Exception as e:
            results["errors"].append(f"Ollama check failed: {e}")
        
        # Check 4: System resources
        logger.info("Checking system resources...")
        try:
            resource_check = self._check_system_resources()
            results["checks"]["resources"] = resource_check
            if resource_check["warnings"]:
                results["warnings"].extend(resource_check["warnings"])
        except Exception as e:
            results["errors"].append(f"Resource check failed: {e}")
        
        # Check 5: Parsing components
        logger.info("Checking parsing components...")
        try:
            parsing_check = self._validate_parsing_components()
            results["checks"]["parsing"] = parsing_check
            if not parsing_check["valid"]:
                results["errors"].extend(parsing_check["errors"])
        except Exception as e:
            results["errors"].append(f"Parsing check failed: {e}")
        
        # Determine overall status
        if results["errors"]:
            results["status"] = "failed"
        elif results["warnings"]:
            results["status"] = "warnings"
        
        return results
    
    async def run_full_validation(self, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run complete validation including model loading and test negotiations."""
        logger.info("=== FULL VALIDATION ===")
        
        if models is None:
            models = self.models
        
        results = {
            "status": "success",
            "checks": {},
            "model_validation": {},
            "test_negotiations": {},
            "warnings": [],
            "errors": []
        }
        
        try:
            # First run dry-run checks
            dry_run_results = await self.run_dry_run()
            results["checks"] = dry_run_results["checks"]
            results["warnings"].extend(dry_run_results["warnings"])
            results["errors"].extend(dry_run_results["errors"])
            
            if dry_run_results["status"] == "failed":
                results["status"] = "failed"
                return results
            
            # Initialize components
            logger.info("Initializing components...")
            await self._initialize_components()
            
            # Validate model availability
            logger.info("Validating model availability...")
            model_validation = await self._validate_models(models)
            results["model_validation"] = model_validation
            
            # Count working models
            working_models = [
                model for model, result in model_validation.items() 
                if result.get("success", False)
            ]
            
            if len(working_models) < 2:
                results["errors"].append("Need at least 2 working models for validation")
                results["status"] = "failed"
                return results
            
            # Run test negotiations
            logger.info("Running test negotiations...")
            test_results = await self._run_test_negotiations(working_models[:4])  # Test with first 4 working models
            results["test_negotiations"] = test_results
            
            # Analyze results
            if test_results["success_rate"] < 0.5:
                results["warnings"].append("Low success rate in test negotiations")
            
            # Final status determination
            if results["errors"]:
                results["status"] = "failed"
            elif results["warnings"]:
                results["status"] = "warnings"
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            results["errors"].append(f"Validation error: {e}")
            results["status"] = "failed"
        
        finally:
            # Cleanup
            if self.model_manager:
                await self.model_manager.shutdown()
        
        return results
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration settings."""
        errors = []
        warnings = []
        
        # Check game parameters
        game_config = self.config.get('game', {})
        
        required_game_params = ['selling_price', 'production_cost', 'demand_mean', 'demand_std', 'optimal_price']
        for param in required_game_params:
            if param not in game_config:
                errors.append(f"Missing game parameter: {param}")
        
        # Validate parameter values
        if game_config.get('selling_price', 0) <= 0:
            errors.append("selling_price must be positive")
        
        if game_config.get('production_cost', 0) <= 0:
            errors.append("production_cost must be positive")
        
        if game_config.get('production_cost', 0) >= game_config.get('selling_price', 100):
            errors.append("production_cost must be less than selling_price")
        
        # Check corrected parameters
        if game_config.get('demand_mean') != 40:
            warnings.append(f"demand_mean is {game_config.get('demand_mean')}, expected 40")
        
        if game_config.get('demand_std') != 10:
            warnings.append(f"demand_std is {game_config.get('demand_std')}, expected 10")
        
        if game_config.get('optimal_price') != 65:
            warnings.append(f"optimal_price is {game_config.get('optimal_price')}, expected 65")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "config_summary": {
                "selling_price": game_config.get('selling_price'),
                "production_cost": game_config.get('production_cost'),
                "demand_mean": game_config.get('demand_mean'),
                "demand_std": game_config.get('demand_std'),
                "optimal_price": game_config.get('optimal_price')
            }
        }
    
    def _validate_imports(self) -> Dict[str, Any]:
        """Validate that all required modules can be imported."""
        errors = []
        
        required_modules = [
            'pandas',
            'numpy',
            'scipy',
            'matplotlib',
            'seaborn',
            'pydantic',
            'yaml',
            'asyncio',
            'logging',
            'pathlib',
            'dataclasses'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError as e:
                errors.append(f"Failed to import {module}: {e}")
        
        # Check optional modules
        optional_modules = {
            'pyarrow': 'Parquet export will not be available',
            'plotly': 'Interactive visualizations will not be available'
        }
        
        warnings = []
        for module, warning in optional_modules.items():
            try:
                __import__(module)
            except ImportError:
                warnings.append(f"{module} not available: {warning}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def _check_ollama_availability(self) -> Dict[str, Any]:
        """Check if Ollama is available and running."""
        try:
            import ollama
            client = ollama.Client()
            
            # Try to list models
            models = client.list()
            
            return {
                "available": True,
                "installed_models": [model['name'] for model in models.get('models', [])],
                "model_count": len(models.get('models', []))
            }
            
        except ImportError:
            return {
                "available": False,
                "error": "Ollama Python package not installed"
            }
        except Exception as e:
            return {
                "available": False,
                "error": f"Ollama not running or accessible: {e}"
            }
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resources."""
        try:
            import psutil
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            # CPU check
            cpu_count = psutil.cpu_count()
            
            # Disk space check
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024**3)
            
            warnings = []
            
            if memory_gb < 16:
                warnings.append(f"Low RAM: {memory_gb:.1f}GB (16GB+ recommended)")
            
            if cpu_count < 4:
                warnings.append(f"Limited CPU cores: {cpu_count} (4+ recommended)")
            
            if disk_free_gb < 30:
                warnings.append(f"Low disk space: {disk_free_gb:.1f}GB (30GB+ recommended)")
            
            return {
                "memory_gb": round(memory_gb, 1),
                "cpu_count": cpu_count,
                "disk_free_gb": round(disk_free_gb, 1),
                "warnings": warnings,
                "adequate": len(warnings) == 0
            }
            
        except ImportError:
            return {
                "error": "psutil not available for system checks",
                "warnings": ["Cannot check system resources"]
            }
    
    def _validate_parsing_components(self) -> Dict[str, Any]:
        """Validate price extraction and acceptance detection."""
        errors = []
        
        try:
            # Test price extractor
            price_extractor = RobustPriceExtractor()
            test_cases = price_extractor.validate_test_cases()
            
            failed_price_tests = [case for case, passed in test_cases.items() if not passed]
            if failed_price_tests:
                errors.append(f"Price extraction tests failed: {len(failed_price_tests)} cases")
        
        except Exception as e:
            errors.append(f"Price extractor validation failed: {e}")
        
        try:
            # Test acceptance detector
            acceptance_detector = AcceptanceDetector()
            test_cases = acceptance_detector.validate_test_cases()
            
            failed_acceptance_tests = [case for case, passed in test_cases.items() if not passed]
            if failed_acceptance_tests:
                errors.append(f"Acceptance detection tests failed: {len(failed_acceptance_tests)} cases")
        
        except Exception as e:
            errors.append(f"Acceptance detector validation failed: {e}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def _initialize_components(self) -> None:
        """Initialize model manager and negotiation engine."""
        self.model_manager = OptimizedModelManager(
            max_concurrent_models=1,  # Conservative for validation
            config=self.config
        )
        
        self.negotiation_engine = NegotiationEngine(
            model_manager=self.model_manager,
            config=self.config
        )
    
    async def _validate_models(self, models: List[str]) -> Dict[str, Any]:
        """Validate model availability and basic functionality."""
        return await self.model_manager.validate_models(models)
    
    async def _run_test_negotiations(self, models: List[str]) -> Dict[str, Any]:
        """Run basic test negotiations."""
        if len(models) < 2:
            return {"error": "Need at least 2 models for test negotiations"}
        
        # Create simple test configurations
        test_configs = [
            NegotiationConfig(models[0], models[1], "00"),  # No reflection
            NegotiationConfig(models[1], models[0], "11") if len(models) >= 2 else None,  # Full reflection
        ]
        
        test_configs = [config for config in test_configs if config is not None]
        
        if len(models) >= 4:
            test_configs.extend([
                NegotiationConfig(models[2], models[3], "01"),  # Buyer reflection only
                NegotiationConfig(models[3], models[2], "10"),  # Supplier reflection only
            ])
        
        # Run test negotiations
        results = await self.negotiation_engine.run_batch_negotiations(
            test_configs,
            max_concurrent=1
        )
        
        # Analyze results
        successful = [r for r in results if r.completed]
        
        return {
            "total_tests": len(results),
            "successful": len(successful),
            "success_rate": len(successful) / len(results) if results else 0,
            "avg_rounds": sum(r.total_rounds for r in successful) / len(successful) if successful else 0,
            "avg_tokens": sum(r.total_tokens for r in successful) / len(successful) if successful else 0,
            "price_range": [r.agreed_price for r in successful if r.agreed_price]
        }


@click.command()
@click.option('--dry-run', is_flag=True, help='Run dry-run validation only')
@click.option('--models', type=str, help='Comma-separated list of models to validate')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.option('--output', type=click.Path(), help='Output file for validation results')
def main(dry_run: bool, models: Optional[str], config: Optional[str], output: Optional[str]):
    """Run validation suite for newsvendor experiment."""
    
    try:
        # Create validation suite
        validator = ValidationSuite(config)
        
        # Override models if specified
        if models:
            validator.models = [model.strip() for model in models.split(',')]
        
        # Run validation
        if dry_run:
            results = asyncio.run(validator.run_dry_run())
        else:
            results = asyncio.run(validator.run_full_validation(validator.models))
        
        # Print results
        click.echo("\n=== VALIDATION RESULTS ===")
        click.echo(json.dumps(results, indent=2, default=str))
        
        # Save results if output specified
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            click.echo(f"\nResults saved to: {output}")
        
        # Print summary
        status = results.get("status", "unknown")
        if status == "success":
            click.echo("\n✅ Validation PASSED - Ready to run experiments!")
        elif status == "warnings":
            click.echo("\n⚠️  Validation passed with WARNINGS - Check issues above")
        else:
            click.echo("\n❌ Validation FAILED - Fix errors before proceeding")
            
        # Print specific recommendations
        if results.get("errors"):
            click.echo("\n🔴 ERRORS TO FIX:")
            for error in results["errors"]:
                click.echo(f"  - {error}")
        
        if results.get("warnings"):
            click.echo("\n🟡 WARNINGS:")
            for warning in results["warnings"]:
                click.echo(f"  - {warning}")
        
        # Exit with appropriate code
        if status == "failed":
            sys.exit(1)
        elif status == "warnings":
            sys.exit(2)
        else:
            sys.exit(0)
            
    except Exception as e:
        click.echo(f"❌ Validation suite failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()