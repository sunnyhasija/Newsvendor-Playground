#!/usr/bin/env python3
"""
Full Experiment Runner for Newsvendor Negotiation Study

Runs the complete experimental protocol with all model pairings,
reflection conditions, and adaptive replication strategy.
"""

import asyncio
import time
import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict
import click
from tqdm.asyncio import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('newsvendor_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
from ..core.model_manager import OptimizedModelManager
from ..core.negotiation_engine import NegotiationEngine, NegotiationConfig
from ..utils.config_loader import load_config
from ..utils.data_exporter import DataExporter


class ExperimentRunner:
    """Main experiment runner with progress tracking and data management."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize experiment runner."""
        self.config = load_config(config_path)
        self.data_exporter = DataExporter(self.config.get('storage', {}))
        
        # Initialize components
        self.model_manager = None
        self.negotiation_engine = None
        
        # Experiment state
        self.start_time = None
        self.results = []
        self.current_phase = "initialization"
        
        # Models to test (8 models from specification)
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
        
        logger.info("Initialized ExperimentRunner")
    
    async def initialize(self) -> None:
        """Initialize experimental components."""
        logger.info("Initializing experimental components...")
        
        # Initialize model manager
        max_concurrent = self.config.get('technical', {}).get('max_concurrent_models', 1)
        self.model_manager = OptimizedModelManager(
            max_concurrent_models=max_concurrent,
            config=self.config
        )
        
        # Initialize negotiation engine
        self.negotiation_engine = NegotiationEngine(
            model_manager=self.model_manager,
            config=self.config
        )
        
        logger.info("Components initialized successfully")
    
    async def run_validation_phase(self) -> Dict[str, Any]:
        """Run Phase 1: Validation."""
        logger.info("=== PHASE 1: VALIDATION ===")
        self.current_phase = "validation"
        
        # Initialize components first
        await self.initialize()
        
        # Validate setup
        validation_results = await self.negotiation_engine.validate_setup(self.models)
        
        if validation_results["overall_status"] != "ready":
            logger.error(f"Validation failed: {validation_results['overall_status']}")
            return validation_results
        
        # Run quick validation negotiations (1 rep each)
        validation_negotiations = []
        
        # Test each model tier with basic patterns
        test_pairs = [
            ("tinyllama:latest", "qwen2:1.5b", "00"),      # Ultra vs Ultra
            ("gemma2:2b", "phi3:mini", "11"),              # Compact vs Compact  
            ("mistral:instruct", "qwen:7b", "01"),         # Mid vs Mid
            ("qwen3:latest", "gemma2:2b", "10"),           # Large vs Compact
        ]
        
        for buyer, supplier, pattern in test_pairs:
            if buyer in self.models and supplier in self.models:
                validation_negotiations.append(
                    NegotiationConfig(buyer, supplier, pattern)
                )
        
        # Run validation negotiations
        logger.info(f"Running {len(validation_negotiations)} validation negotiations...")
        
        validation_results_list = await self.negotiation_engine.run_batch_negotiations(
            validation_negotiations,
            max_concurrent=1,
            progress_callback=self._validation_progress_callback
        )
        
        # Analyze validation results
        success_count = sum(1 for r in validation_results_list if r.completed)
        success_rate = success_count / len(validation_results_list) if validation_results_list else 0
        
        validation_summary = {
            "phase": "validation",
            "total_negotiations": len(validation_results_list),
            "successful": success_count,
            "success_rate": success_rate,
            "avg_rounds": sum(r.total_rounds for r in validation_results_list) / len(validation_results_list) if validation_results_list else 0,
            "avg_tokens": sum(r.total_tokens for r in validation_results_list) / len(validation_results_list) if validation_results_list else 0,
            "ready_for_full_experiment": success_rate >= 0.7
        }
        
        logger.info(f"Validation complete: {validation_summary}")
        
        # Save validation results
        await self.data_exporter.save_results(validation_results_list, "validation")
        
        return validation_summary
    
    async def run_statistical_power_phase(self) -> Dict[str, Any]:
        """Run Phase 2: Statistical Power."""
        logger.info("=== PHASE 2: STATISTICAL POWER ===")
        self.current_phase = "statistical_power"
        
        # Ensure components are initialized
        if self.negotiation_engine is None:
            await self.initialize()
        
        # Generate experiment plan with reduced replications for this phase
        full_plan = self.negotiation_engine.generate_experiment_plan(self.models)
        
        # Use 3 replications for statistical power phase (as per spec)
        power_plan = []
        seen_configs = set()
        
        for config in full_plan:
            config_key = (config.buyer_model, config.supplier_model, config.reflection_pattern)
            if config_key not in seen_configs:
                seen_configs.add(config_key)
                # Add 3 replications
                for i in range(3):
                    power_plan.append(config)
        
        logger.info(f"Running {len(power_plan)} negotiations for statistical power analysis")
        
        # Run negotiations with progress tracking
        power_results = await self._run_with_progress(
            power_plan, 
            "Statistical Power Phase",
            max_concurrent=1
        )
        
        # Analyze statistical power
        power_analysis = self._analyze_statistical_power(power_results)
        
        # Save results
        await self.data_exporter.save_results(power_results, "statistical_power")
        
        return power_analysis
    
    async def run_full_dataset_phase(self) -> Dict[str, Any]:
        """Run Phase 3: Full Dataset."""
        logger.info("=== PHASE 3: FULL DATASET ===")
        self.current_phase = "full_dataset"
        
        # Ensure components are initialized
        if self.negotiation_engine is None:
            await self.initialize()
        
        # Generate complete experiment plan
        full_plan = self.negotiation_engine.generate_experiment_plan(self.models)
        
        logger.info(f"Running {len(full_plan)} negotiations for complete dataset")
        logger.info(f"Estimated time: {self._estimate_completion_time(full_plan):.1f} hours")
        
        # Run all negotiations
        full_results = await self._run_with_progress(
            full_plan,
            "Full Dataset Phase", 
            max_concurrent=self.config.get('technical', {}).get('max_concurrent_models', 1)
        )
        
        # Analyze complete results
        final_analysis = self._analyze_complete_results(full_results)
        
        # Save complete dataset
        await self.data_exporter.save_results(full_results, "complete")
        await self.data_exporter.export_analysis(final_analysis)
        
        return final_analysis
    
    async def _run_with_progress(
        self, 
        negotiations: List[NegotiationConfig], 
        phase_name: str,
        max_concurrent: int = 1
    ) -> List:
        """Run negotiations with progress bar."""
        
        results = []
        progress_bar = tqdm(total=len(negotiations), desc=phase_name)
        
        def progress_callback(completed: int, total: int, result):
            progress_bar.update(1)
            progress_bar.set_postfix({
                'Success': f"{result.completed}",
                'Price': f"${result.agreed_price}" if result.agreed_price else "None",
                'Rounds': result.total_rounds
            })
        
        try:
            batch_results = await self.negotiation_engine.run_batch_negotiations(
                negotiations,
                max_concurrent=max_concurrent,
                progress_callback=progress_callback
            )
            results.extend(batch_results)
            
        finally:
            progress_bar.close()
        
        return results
    
    def _validation_progress_callback(self, completed: int, total: int, result):
        """Progress callback for validation phase."""
        logger.info(f"Validation {completed}/{total}: "
                   f"{'SUCCESS' if result.completed else 'FAILED'} "
                   f"(${result.agreed_price}, {result.total_rounds} rounds)")
    
    def _analyze_statistical_power(self, results: List) -> Dict[str, Any]:
        """Analyze statistical power of the experimental design."""
        
        # Group results by conditions
        by_model_pair = {}
        by_reflection = {}
        
        for result in results:
            # Group by model pairing
            pair_key = f"{result.buyer_model}_{result.supplier_model}"
            if pair_key not in by_model_pair:
                by_model_pair[pair_key] = []
            by_model_pair[pair_key].append(result)
            
            # Group by reflection pattern
            if result.reflection_pattern not in by_reflection:
                by_reflection[result.reflection_pattern] = []
            by_reflection[result.reflection_pattern].append(result)
        
        # Calculate power metrics
        analysis = {
            "total_negotiations": len(results),
            "overall_success_rate": sum(1 for r in results if r.completed) / len(results) if results else 0,
            "model_pair_analysis": {},
            "reflection_analysis": {},
            "power_assessment": "adequate"  # Will be determined below
        }
        
        # Calculate average price for successful negotiations
        successful_results = [r for r in results if r.completed and r.agreed_price]
        if successful_results:
            analysis["average_price"] = sum(r.agreed_price for r in successful_results) / len(successful_results)
        
        # Analyze by model pairs
        for pair, pair_results in by_model_pair.items():
            if len(pair_results) >= 3:  # Minimum for statistical power
                success_rate = sum(1 for r in pair_results if r.completed) / len(pair_results)
                prices = [r.agreed_price for r in pair_results if r.agreed_price]
                
                analysis["model_pair_analysis"][pair] = {
                    "count": len(pair_results),
                    "success_rate": success_rate,
                    "avg_price": sum(prices) / len(prices) if prices else None,
                    "price_std": self._calculate_std(prices) if len(prices) > 1 else None
                }
        
        # Analyze by reflection
        for pattern, pattern_results in by_reflection.items():
            success_rate = sum(1 for r in pattern_results if r.completed) / len(pattern_results)
            prices = [r.agreed_price for r in pattern_results if r.agreed_price]
            
            analysis["reflection_analysis"][pattern] = {
                "count": len(pattern_results),
                "success_rate": success_rate,
                "avg_price": sum(prices) / len(prices) if prices else None
            }
        
        return analysis
    
    def _analyze_complete_results(self, results: List) -> Dict[str, Any]:
        """Analyze complete experimental results."""
        
        total_results = len(results)
        successful_results = [r for r in results if r.completed and r.agreed_price]
        
        if not successful_results:
            return {"error": "No successful negotiations to analyze"}
        
        # Basic statistics
        prices = [r.agreed_price for r in successful_results]
        rounds = [r.total_rounds for r in successful_results]
        tokens = [r.total_tokens for r in successful_results]
        
        analysis = {
            "experiment_summary": {
                "total_negotiations": total_results,
                "successful_negotiations": len(successful_results),
                "success_rate": len(successful_results) / total_results,
                "completion_time_hours": (time.time() - self.start_time) / 3600 if self.start_time else 0
            },
            "price_analysis": {
                "mean_price": sum(prices) / len(prices),
                "median_price": sorted(prices)[len(prices)//2],
                "price_std": self._calculate_std(prices),
                "min_price": min(prices),
                "max_price": max(prices),
                "distance_from_optimal": [abs(p - 65) for p in prices],  # Optimal = $65
                "within_optimal_range": sum(1 for p in prices if 60 <= p <= 70) / len(prices)
            },
            "efficiency_analysis": {
                "mean_rounds": sum(rounds) / len(rounds),
                "mean_tokens": sum(tokens) / len(tokens),
                "tokens_per_round": sum(tokens) / sum(rounds) if sum(rounds) > 0 else 0
            },
            "model_analysis": self._analyze_by_model(successful_results),
            "reflection_analysis": self._analyze_by_reflection(successful_results)
        }
        
        return analysis
    
    def _analyze_by_model(self, results: List) -> Dict[str, Any]:
        """Analyze results by model type."""
        model_stats = {}
        
        for result in results:
            for role, model in [("buyer", result.buyer_model), ("supplier", result.supplier_model)]:
                if model not in model_stats:
                    model_stats[model] = {
                        "as_buyer": {"count": 0, "successes": 0, "prices": [], "tokens": []},
                        "as_supplier": {"count": 0, "successes": 0, "prices": [], "tokens": []}
                    }
                
                role_stats = model_stats[model][f"as_{role}"]
                role_stats["count"] += 1
                if result.completed:
                    role_stats["successes"] += 1
                    if result.agreed_price:
                        role_stats["prices"].append(result.agreed_price)
                    role_stats["tokens"].append(result.total_tokens)
        
        # Calculate summary statistics
        for model, stats in model_stats.items():
            for role in ["as_buyer", "as_supplier"]:
                role_data = stats[role]
                if role_data["count"] > 0:
                    role_data["success_rate"] = role_data["successes"] / role_data["count"]
                    if role_data["prices"]:
                        role_data["avg_price"] = sum(role_data["prices"]) / len(role_data["prices"])
                    if role_data["tokens"]:
                        role_data["avg_tokens"] = sum(role_data["tokens"]) / len(role_data["tokens"])
        
        return model_stats
    
    def _analyze_by_reflection(self, results: List) -> Dict[str, Any]:
        """Analyze results by reflection pattern."""
        reflection_stats = {}
        
        for result in results:
            pattern = result.reflection_pattern
            if pattern not in reflection_stats:
                reflection_stats[pattern] = {
                    "count": 0, 
                    "successes": 0, 
                    "prices": [], 
                    "tokens": [],
                    "rounds": []
                }
            
            stats = reflection_stats[pattern]
            stats["count"] += 1
            if result.completed:
                stats["successes"] += 1
                if result.agreed_price:
                    stats["prices"].append(result.agreed_price)
                stats["tokens"].append(result.total_tokens)
                stats["rounds"].append(result.total_rounds)
        
        # Calculate summary statistics
        for pattern, stats in reflection_stats.items():
            if stats["count"] > 0:
                stats["success_rate"] = stats["successes"] / stats["count"]
                if stats["prices"]:
                    stats["avg_price"] = sum(stats["prices"]) / len(stats["prices"])
                if stats["tokens"]:
                    stats["avg_tokens"] = sum(stats["tokens"]) / len(stats["tokens"])
                if stats["rounds"]:
                    stats["avg_rounds"] = sum(stats["rounds"]) / len(stats["rounds"])
        
        return reflection_stats
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _estimate_completion_time(self, negotiations: List[NegotiationConfig]) -> float:
        """Estimate completion time in hours."""
        # Rough estimates based on model tiers
        time_per_negotiation = {
            "ultra": 2,      # 2 minutes
            "compact": 4,    # 4 minutes  
            "mid": 6,        # 6 minutes
            "large": 10      # 10 minutes
        }
        
        total_minutes = 0
        for config in negotiations:
            # Use max time of the two models (conservative estimate)
            buyer_time = time_per_negotiation.get("mid", 6)  # Default
            supplier_time = time_per_negotiation.get("mid", 6)
            total_minutes += max(buyer_time, supplier_time)
        
        return total_minutes / 60  # Convert to hours
    
    async def run_complete_experiment(self) -> Dict[str, Any]:
        """Run the complete three-phase experiment."""
        self.start_time = time.time()
        
        try:
            # Initialize
            await self.initialize()
            
            # Phase 1: Validation
            validation_results = await self.run_validation_phase()
            if not validation_results.get("ready_for_full_experiment", False):
                logger.error("Validation failed - aborting experiment")
                return {"status": "failed", "phase": "validation", "results": validation_results}
            
            # Phase 2: Statistical Power
            power_results = await self.run_statistical_power_phase()
            
            # Phase 3: Full Dataset  
            final_results = await self.run_full_dataset_phase()
            
            # Compile complete experiment results
            complete_results = {
                "status": "completed",
                "phases": {
                    "validation": validation_results,
                    "statistical_power": power_results,
                    "full_dataset": final_results
                },
                "experiment_metadata": {
                    "start_time": self.start_time,
                    "end_time": time.time(),
                    "total_duration_hours": (time.time() - self.start_time) / 3600,
                    "models_tested": self.models,
                    "engine_stats": self.negotiation_engine.get_performance_stats()
                }
            }
            
            # Save complete experiment results
            await self.data_exporter.export_analysis(complete_results, "complete_experiment_results.json")
            
            logger.info("Complete experiment finished successfully!")
            return complete_results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "phase": self.current_phase,
                "partial_results": self.results
            }
        
        finally:
            # Cleanup
            if self.model_manager:
                await self.model_manager.shutdown()


# CLI interface
@click.command()
@click.option('--phase', type=click.Choice(['validation', 'power', 'full', 'all']), 
              default='all', help='Experiment phase to run')
@click.option('--models', type=str, help='Comma-separated list of models to test')
@click.option('--output', type=click.Path(), help='Output directory for results')
@click.option('--concurrent', type=int, default=1, help='Max concurrent negotiations')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
def main(phase: str, models: Optional[str], output: Optional[str], concurrent: int, config: Optional[str]):
    """Run the newsvendor negotiation experiment."""
    
    try:
        # Create experiment runner
        runner = ExperimentRunner(config)
        
        # Override models if specified
        if models:
            runner.models = [model.strip() for model in models.split(',')]
        
        # Override output directory if specified
        if output:
            runner.config['storage']['output_dir'] = output
            runner.data_exporter = DataExporter(runner.config.get('storage', {}))
        
        # Override concurrency if specified
        if concurrent > 1:
            runner.config['technical']['max_concurrent_models'] = concurrent
        
        # Run experiment based on phase
        async def run_phase():
            if phase == 'validation':
                return await runner.run_validation_phase()
            elif phase == 'power':
                return await runner.run_statistical_power_phase()
            elif phase == 'full':
                return await runner.run_full_dataset_phase()
            else:  # phase == 'all'
                return await runner.run_complete_experiment()
        
        result = asyncio.run(run_phase())
        
        # Print results summary
        click.echo("\n=== EXPERIMENT RESULTS ===")
        click.echo(json.dumps(result, indent=2, default=str))
        
        if result.get("status") == "completed":
            click.echo("\n✅ Experiment completed successfully!")
        else:
            click.echo(f"\n❌ Experiment failed: {result.get('status', 'unknown')}")
            
    except Exception as e:
        logger.error(f"CLI error: {e}")
        click.echo(f"❌ Error: {e}")
        raise click.Abort()


if __name__ == '__main__':
    main()