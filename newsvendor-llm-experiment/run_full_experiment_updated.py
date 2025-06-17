#!/usr/bin/env python3
"""
run_full_experiment_updated.py
Full Experiment Runner for Updated Newsvendor Study
Runs complete experiment: 10 models × 10 models × 4 reflection patterns × 20 runs = 8,000 negotiations

Place this file in the ROOT directory of your project
"""

import asyncio
import json
import logging
import time
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import click
try:
    from tqdm.asyncio import tqdm
except ImportError:
    # Fallback if tqdm not available
    print("Installing tqdm for progress bars...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm.asyncio import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('newsvendor_full_experiment_updated.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our modules (updated paths for your project structure)
from core.unified_model_manager import create_unified_model_manager
from agents.standardized_agents import StandardizedBuyerAgent, StandardizedSupplierAgent
from core.conversation_tracker import ConversationTracker, NegotiationResult
from parsing.acceptance_detector import TerminationType


@dataclass
class ExperimentConfig:
    """Configuration for a single experimental condition."""
    buyer_model: str
    supplier_model: str
    reflection_pattern: str  # "00", "01", "10", "11"
    replications: int = 20
    max_rounds: int = 10
    timeout_seconds: int = 120


class FullExperimentRunner:
    """Runs the complete experimental protocol across all models and conditions."""
    
    def __init__(self, max_concurrent: int = 2):
        """Initialize full experiment runner."""
        self.model_manager = None
        self.max_concurrent = max_concurrent
        self.results = []
        self.start_time = None
        
        # Game configuration (corrected parameters)
        self.game_config = {
            'selling_price': 100,
            'production_cost': 30,
            'demand_mean': 40,      # CORRECTED: Normal(40, 10)
            'demand_std': 10,       # CORRECTED: Normal(40, 10)
            'optimal_price': 65,    # Fair split-the-difference
            'max_rounds': 10,
            'timeout_seconds': 120
        }
        
        # All 10 models in complexity order
        self.all_models = [
            # Tiny models
            "tinyllama:latest",
            "qwen2:1.5b",
            # Small models  
            "gemma2:2b",
            "phi3:mini", 
            "llama3.2:latest",
            # Medium models
            "mistral:instruct",
            "qwen:7b",
            # Large models
            "qwen3:latest",
            # Remote models
            "claude-sonnet-4-remote",
            "o3-remote"
        ]
        
        # All reflection patterns
        self.reflection_patterns = ["00", "01", "10", "11"]
        
        logger.info(f"Initialized FullExperimentRunner for {len(self.all_models)} models with {max_concurrent} concurrency")
    
    async def initialize(self) -> None:
        """Initialize the model manager and validate models."""
        logger.info("Initializing unified model manager...")
        self.model_manager = create_unified_model_manager()
        
        # Validate all models are available
        validation_results = await self.model_manager.validate_all_models()
        
        working_models = [model for model, result in validation_results.items() if result["success"]]
        failed_models = [model for model, result in validation_results.items() if not result["success"]]
        
        logger.info(f"✅ Working models: {len(working_models)}")
        for model in working_models:
            cost = validation_results[model].get("cost", 0)
            if cost > 0:
                logger.info(f"   🌐 {model} (${cost:.6f} per validation)")
            else:
                logger.info(f"   💻 {model}")
        
        if failed_models:
            logger.error(f"❌ Failed models: {failed_models}")
            raise RuntimeError(f"Some models failed validation: {failed_models}")
        
        logger.info("All models validated successfully")
    
    def generate_experiment_plan(self) -> List[ExperimentConfig]:
        """Generate the complete experiment plan: 10×10×4×20 = 8,000 negotiations."""
        
        experiment_plan = []
        
        for buyer_model in self.all_models:
            for supplier_model in self.all_models:
                for pattern in self.reflection_patterns:
                    config = ExperimentConfig(
                        buyer_model=buyer_model,
                        supplier_model=supplier_model,
                        reflection_pattern=pattern,
                        replications=20  # Exactly 20 for all combinations
                    )
                    experiment_plan.append(config)
        
        total_negotiations = len(experiment_plan) * 20
        logger.info(f"Generated experiment plan:")
        logger.info(f"  Model combinations: {len(self.all_models)}² = {len(self.all_models)**2}")
        logger.info(f"  Reflection patterns: {len(self.reflection_patterns)}")
        logger.info(f"  Replications per condition: 20")
        logger.info(f"  Total configurations: {len(experiment_plan)}")
        logger.info(f"  Total negotiations: {total_negotiations}")
        
        return experiment_plan
    
    def estimate_experiment_cost(self, experiment_plan: List[ExperimentConfig]) -> Dict[str, float]:
        """Estimate the total cost of the experiment."""
        
        # Cost estimates per token (from model configs)
        costs = {
            "claude-sonnet-4-remote": 0.000075,
            "o3-remote": 0.000240
        }
        
        # Estimate tokens per negotiation (based on validation data)
        estimated_tokens_per_negotiation = {
            "claude-sonnet-4-remote": 400,  # Conservative estimate
            "o3-remote": 600  # Higher due to reasoning tokens
        }
        
        total_cost = 0.0
        cost_breakdown = {}
        
        for config in experiment_plan:
            config_cost = 0.0
            
            for model in [config.buyer_model, config.supplier_model]:
                if model in costs:
                    model_cost = (
                        costs[model] * 
                        estimated_tokens_per_negotiation[model] * 
                        config.replications
                    )
                    config_cost += model_cost
                    
                    if model not in cost_breakdown:
                        cost_breakdown[model] = 0.0
                    cost_breakdown[model] += model_cost
            
            total_cost += config_cost
        
        return {
            "total_estimated_cost": total_cost,
            "breakdown": cost_breakdown,
            "negotiations_with_remote": sum(
                config.replications for config in experiment_plan
                if "remote" in config.buyer_model or "remote" in config.supplier_model
            )
        }
    
    async def run_full_experiment(self, experiment_plan: Optional[List[ExperimentConfig]] = None) -> Dict[str, Any]:
        """Run the complete experiment with progress tracking."""
        
        if experiment_plan is None:
            experiment_plan = self.generate_experiment_plan()
        
        # Estimate cost
        cost_estimate = self.estimate_experiment_cost(experiment_plan)
        logger.info(f"💰 Estimated total cost: ${cost_estimate['total_estimated_cost']:.2f}")
        for model, cost in cost_estimate['breakdown'].items():
            logger.info(f"   {model}: ${cost:.2f}")
        
        # Confirm before proceeding
        confirm = input(f"\nProceed with experiment costing ~${cost_estimate['total_estimated_cost']:.2f}? [y/N]: ")
        if confirm.lower() != 'y':
            logger.info("Experiment cancelled by user")
            return {"status": "cancelled"}
        
        self.start_time = time.time()
        logger.info(f"🚀 Starting full experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create all individual negotiations from the experiment plan
        all_negotiations = []
        for config in experiment_plan:
            for rep in range(config.replications):
                negotiation_id = f"{config.buyer_model}_{config.supplier_model}_{config.reflection_pattern}_rep{rep:02d}"
                all_negotiations.append((config, negotiation_id))
        
        logger.info(f"Total negotiations to run: {len(all_negotiations)}")
        
        # Run negotiations with controlled concurrency and progress tracking
        results = await self._run_with_progress(all_negotiations)
        
        # Analyze complete results
        analysis = self._analyze_complete_results(results)
        
        # Save complete dataset
        await self._save_complete_results(results, analysis)
        
        return analysis
    
    async def _run_with_progress(self, negotiations: List[Tuple[ExperimentConfig, str]]) -> List[NegotiationResult]:
        """Run all negotiations with progress tracking and controlled concurrency."""
        
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Progress tracking
        progress_bar = tqdm(total=len(negotiations), desc="Negotiations")
        completed_count = 0
        total_cost = 0.0
        
        async def run_single_with_semaphore(config_and_id: Tuple[ExperimentConfig, str]) -> NegotiationResult:
            nonlocal completed_count, total_cost
            
            config, negotiation_id = config_and_id
            
            async with semaphore:
                try:
                    result = await self._run_single_negotiation(config, negotiation_id)
                    
                    # Update progress
                    completed_count += 1
                    negotiation_cost = result.metadata.get('total_cost', 0.0)
                    total_cost += negotiation_cost
                    
                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'Success': f"{result.completed}",
                        'Price': f"${result.agreed_price}" if result.agreed_price else "None",
                        'Cost': f"${total_cost:.2f}",
                        'ETA': f"{self._estimate_remaining_time(completed_count, len(negotiations))}"
                    })
                    
                    # Log milestone progress
                    if completed_count % 100 == 0:
                        logger.info(f"✅ Completed {completed_count}/{len(negotiations)} negotiations (${total_cost:.2f} spent)")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Failed negotiation {negotiation_id}: {e}")
                    # Return failed result
                    return NegotiationResult(
                        negotiation_id=negotiation_id,
                        buyer_model=config.buyer_model,
                        supplier_model=config.supplier_model,
                        reflection_pattern=config.reflection_pattern,
                        completed=False,
                        agreed_price=None,
                        termination_type=TerminationType.FAILURE,
                        total_rounds=0,
                        total_tokens=0,
                        total_time=0.0,
                        buyer_profit=None,
                        supplier_profit=None,
                        distance_from_optimal=None,
                        turns=[],
                        metadata={"error": str(e)}
                    )
        
        try:
            # Create all tasks
            tasks = [run_single_with_semaphore(neg) for neg in negotiations]
            
            # Execute with controlled concurrency
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Negotiation {i} failed with exception: {result}")
                    config, negotiation_id = negotiations[i]
                    failed_result = NegotiationResult(
                        negotiation_id=negotiation_id,
                        buyer_model=config.buyer_model,
                        supplier_model=config.supplier_model,
                        reflection_pattern=config.reflection_pattern,
                        completed=False,
                        agreed_price=None,
                        termination_type=TerminationType.FAILURE,
                        total_rounds=0,
                        total_tokens=0,
                        total_time=0.0,
                        buyer_profit=None,
                        supplier_profit=None,
                        distance_from_optimal=None,
                        turns=[],
                        metadata={"error": str(result)}
                    )
                    final_results.append(failed_result)
                else:
                    final_results.append(result)
            
            return final_results
            
        finally:
            progress_bar.close()
    
    def _estimate_remaining_time(self, completed: int, total: int) -> str:
        """Estimate remaining time based on current progress."""
        if completed == 0 or not self.start_time:
            return "Unknown"
        
        elapsed = time.time() - self.start_time
        rate = completed / elapsed  # negotiations per second
        remaining = total - completed
        
        if rate > 0:
            remaining_seconds = remaining / rate
            hours = int(remaining_seconds // 3600)
            minutes = int((remaining_seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
        else:
            return "Unknown"
    
    async def _run_single_negotiation(self, config: ExperimentConfig, negotiation_id: str) -> NegotiationResult:
        """Run a single negotiation between two models."""
        
        try:
            # Parse reflection pattern
            buyer_reflection = config.reflection_pattern[0] == "1"
            supplier_reflection = config.reflection_pattern[1] == "1"
            
            # Initialize conversation tracker
            tracker = ConversationTracker(
                negotiation_id=negotiation_id,
                buyer_model=config.buyer_model,
                supplier_model=config.supplier_model,
                reflection_pattern=config.reflection_pattern,
                config=self.game_config
            )
            
            # Initialize standardized agents
            buyer_agent = StandardizedBuyerAgent(
                model_name=config.buyer_model,
                model_manager=self.model_manager,
                reflection_enabled=buyer_reflection,
                config=self.game_config
            )
            
            supplier_agent = StandardizedSupplierAgent(
                model_name=config.supplier_model,
                model_manager=self.model_manager,
                reflection_enabled=supplier_reflection,
                config=self.game_config
            )
            
            # Conduct negotiation
            result = await self._conduct_negotiation(tracker, buyer_agent, supplier_agent)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in negotiation {negotiation_id}: {e}")
            
            # Return failed result
            return NegotiationResult(
                negotiation_id=negotiation_id,
                buyer_model=config.buyer_model,
                supplier_model=config.supplier_model,
                reflection_pattern=config.reflection_pattern,
                completed=False,
                agreed_price=None,
                termination_type=TerminationType.FAILURE,
                total_rounds=0,
                total_tokens=0,
                total_time=0.0,
                buyer_profit=None,
                supplier_profit=None,
                distance_from_optimal=None,
                turns=[],
                metadata={"error": str(e)}
            )
    
    async def _conduct_negotiation(self, tracker, buyer_agent, supplier_agent) -> NegotiationResult:
        """Conduct the actual negotiation between two agents."""
        
        timeout_seconds = self.game_config.get('timeout_seconds', 120)
        max_rounds = self.game_config.get('max_rounds', 10)
        
        start_time = time.time()
        total_cost = 0.0
        
        try:
            # Main negotiation loop
            while not tracker.completed and tracker.round_number < max_rounds:
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    tracker.force_termination("timeout")
                    break
                
                # Determine current agent
                if tracker.current_speaker == "buyer":
                    current_agent = buyer_agent
                    agent_role = "buyer"
                else:
                    current_agent = supplier_agent
                    agent_role = "supplier"
                
                # Generate conversation context
                context = tracker.get_conversation_history()
                
                # Get agent response
                response = await current_agent.generate_response(
                    context=context,
                    negotiation_history=[asdict(turn) for turn in tracker.turns],
                    round_number=tracker.round_number + 1,
                    max_rounds=max_rounds
                )
                
                if not response.success:
                    tracker.force_termination("generation_failure")
                    break
                
                # Track cost
                total_cost += getattr(response, 'cost_estimate', 0.0)
                
                # Add turn to conversation
                turn_added = tracker.add_turn(
                    speaker=agent_role,
                    message=response.text,
                    reflection=None,
                    tokens_used=response.tokens_used,
                    generation_time=response.generation_time
                )
                
                if not turn_added:
                    tracker.force_termination("turn_error")
                    break
                
                # Check for early termination
                if tracker.completed:
                    break
                
                # Brief delay for remote API rate limiting
                if "remote" in current_agent.model_name:
                    await asyncio.sleep(0.5)
                else:
                    await asyncio.sleep(0.1)
            
            # Finalize negotiation if not already completed
            if not tracker.completed and tracker.round_number >= max_rounds:
                tracker.force_termination("max_rounds_reached")
            
            # Get final result and add cost information
            result = tracker.get_final_result()
            result.metadata['total_cost'] = total_cost
            
            return result
            
        except Exception as e:
            tracker.force_termination(f"error: {str(e)}")
            result = tracker.get_final_result()
            result.metadata['total_cost'] = total_cost
            return result
    
    def _analyze_complete_results(self, results: List[NegotiationResult]) -> Dict[str, Any]:
        """Analyze complete experimental results."""
        
        total_results = len(results)
        successful_results = [r for r in results if r.completed and r.agreed_price]
        
        if not results:
            return {"error": "No results to analyze"}
        
        analysis = {
            "experiment_summary": {
                "total_negotiations": total_results,
                "successful_negotiations": len(successful_results),
                "success_rate": len(successful_results) / total_results,
                "completion_time_hours": (time.time() - self.start_time) / 3600 if self.start_time else 0,
                "total_cost": sum(r.metadata.get('total_cost', 0.0) for r in results)
            },
            "price_analysis": {},
            "efficiency_analysis": {},
            "model_analysis": {},
            "reflection_analysis": {},
            "cost_analysis": {}
        }
        
        if successful_results:
            # Price analysis
            prices = [r.agreed_price for r in successful_results]
            rounds = [r.total_rounds for r in successful_results]
            tokens = [r.total_tokens for r in successful_results]
            
            analysis["price_analysis"] = {
                "mean_price": sum(prices) / len(prices),
                "median_price": sorted(prices)[len(prices)//2],
                "price_std": self._calculate_std(prices),
                "min_price": min(prices),
                "max_price": max(prices),
                "optimal_convergence_rate": sum(1 for p in prices if 60 <= p <= 70) / len(prices),
                "mean_distance_from_optimal": sum(abs(p - 65) for p in prices) / len(prices)
            }
            
            analysis["efficiency_analysis"] = {
                "mean_rounds": sum(rounds) / len(rounds),
                "mean_tokens": sum(tokens) / len(tokens),
                "tokens_per_round": sum(tokens) / sum(rounds) if sum(rounds) > 0 else 0,
                "mean_time_per_negotiation": sum(r.total_time for r in successful_results) / len(successful_results)
            }
        
        # Model-by-model analysis
        analysis["model_analysis"] = self._analyze_by_model(results)
        
        # Reflection pattern analysis
        analysis["reflection_analysis"] = self._analyze_by_reflection(results)
        
        # Cost analysis
        analysis["cost_analysis"] = self._analyze_costs(results)
        
        return analysis
    
    def _analyze_by_model(self, results: List[NegotiationResult]) -> Dict[str, Any]:
        """Analyze results by individual models."""
        model_stats = {}
        
        for result in results:
            for role, model in [("buyer", result.buyer_model), ("supplier", result.supplier_model)]:
                if model not in model_stats:
                    model_stats[model] = {
                        "as_buyer": {"count": 0, "successes": 0, "prices": [], "tokens": [], "costs": []},
                        "as_supplier": {"count": 0, "successes": 0, "prices": [], "tokens": [], "costs": []}
                    }
                
                role_stats = model_stats[model][f"as_{role}"]
                role_stats["count"] += 1
                
                if result.completed:
                    role_stats["successes"] += 1
                    if result.agreed_price:
                        role_stats["prices"].append(result.agreed_price)
                    role_stats["tokens"].append(result.total_tokens)
                    role_stats["costs"].append(result.metadata.get('total_cost', 0.0))
        
        # Calculate summary statistics
        for model, stats in model_stats.items():
            for role in ["as_buyer", "as_supplier"]:
                role_data = stats[role]
                if role_data["count"] > 0:
                    role_data["success_rate"] = role_data["successes"] / role_data["count"]
                    if role_data["prices"]:
                        role_data["avg_price"] = sum(role_data["prices"]) / len(role_data["prices"])
                        role_data["price_std"] = self._calculate_std(role_data["prices"])
                    if role_data["tokens"]:
                        role_data["avg_tokens"] = sum(role_data["tokens"]) / len(role_data["tokens"])
                    if role_data["costs"]:
                        role_data["avg_cost"] = sum(role_data["costs"]) / len(role_data["costs"])
        
        return model_stats
    
    def _analyze_by_reflection(self, results: List[NegotiationResult]) -> Dict[str, Any]:
        """Analyze results by reflection pattern."""
        reflection_stats = {}
        
        pattern_names = {
            "00": "No Reflection",
            "01": "Buyer Reflects Only", 
            "10": "Supplier Reflects Only",
            "11": "Both Reflect"
        }
        
        for result in results:
            pattern = result.reflection_pattern
            if pattern not in reflection_stats:
                reflection_stats[pattern] = {
                    "name": pattern_names.get(pattern, pattern),
                    "count": 0,
                    "successes": 0,
                    "prices": [],
                    "tokens": [],
                    "rounds": [],
                    "costs": []
                }
            
            stats = reflection_stats[pattern]
            stats["count"] += 1
            
            if result.completed:
                stats["successes"] += 1
                if result.agreed_price:
                    stats["prices"].append(result.agreed_price)
                stats["tokens"].append(result.total_tokens)
                stats["rounds"].append(result.total_rounds)
                stats["costs"].append(result.metadata.get('total_cost', 0.0))
        
        # Calculate summary statistics
        for pattern, stats in reflection_stats.items():
            if stats["count"] > 0:
                stats["success_rate"] = stats["successes"] / stats["count"]
                if stats["prices"]:
                    stats["avg_price"] = sum(stats["prices"]) / len(stats["prices"])
                    stats["price_std"] = self._calculate_std(stats["prices"])
                    stats["optimal_convergence_rate"] = sum(1 for p in stats["prices"] if 60 <= p <= 70) / len(stats["prices"])
                if stats["tokens"]:
                    stats["avg_tokens"] = sum(stats["tokens"]) / len(stats["tokens"])
                if stats["rounds"]:
                    stats["avg_rounds"] = sum(stats["rounds"]) / len(stats["rounds"])
                if stats["costs"]:
                    stats["total_cost"] = sum(stats["costs"])
        
        return reflection_stats
    
    def _analyze_costs(self, results: List[NegotiationResult]) -> Dict[str, Any]:
        """Analyze cost breakdown."""
        
        total_cost = sum(r.metadata.get('total_cost', 0.0) for r in results)
        
        # Cost by model
        model_costs = {}
        for result in results:
            cost = result.metadata.get('total_cost', 0.0)
            
            for model in [result.buyer_model, result.supplier_model]:
                if "remote" in model:  # Only remote models have costs
                    if model not in model_costs:
                        model_costs[model] = 0.0
                    model_costs[model] += cost / 2  # Split cost between buyer and supplier
        
        # Cost by negotiation type
        local_vs_local = sum(
            r.metadata.get('total_cost', 0.0) for r in results 
            if "remote" not in r.buyer_model and "remote" not in r.supplier_model
        )
        local_vs_remote = sum(
            r.metadata.get('total_cost', 0.0) for r in results 
            if ("remote" in r.buyer_model) != ("remote" in r.supplier_model)
        )
        remote_vs_remote = sum(
            r.metadata.get('total_cost', 0.0) for r in results 
            if "remote" in r.buyer_model and "remote" in r.supplier_model
        )
        
        return {
            "total_experiment_cost": total_cost,
            "cost_by_model": model_costs,
            "cost_by_negotiation_type": {
                "local_vs_local": local_vs_local,
                "local_vs_remote": local_vs_remote,
                "remote_vs_remote": remote_vs_remote
            },
            "avg_cost_per_negotiation": total_cost / len(results) if results else 0
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    async def _save_complete_results(self, results: List[NegotiationResult], analysis: Dict[str, Any]):
        """Save complete experimental results."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('./full_experiment_results_updated')
        output_dir.mkdir(exist_ok=True)
        
        # Save complete results as JSON
        results_data = {
            "metadata": {
                "experiment_type": "full_newsvendor_experiment_updated",
                "timestamp": timestamp,
                "total_models": len(self.all_models),
                "total_negotiations": len(results),
                "game_config": self.game_config,
                "reflection_patterns": self.reflection_patterns,
                "replications_per_condition": 20,
                "unified_token_budget": 2000,
                "standardized_reflection": True
            },
            "analysis": analysis,
            "results": [asdict(result) for result in results]
        }
        
        results_file = output_dir / f"complete_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save as CSV for analysis
        csv_file = output_dir / f"complete_results_{timestamp}.csv"
        self._save_results_as_csv(results, csv_file)
        
        # Save conversation transcripts (sample)
        transcripts_file = output_dir / f"sample_conversations_{timestamp}.txt"
        await self._save_sample_conversations(results, transcripts_file)
        
        # Save analysis summary
        summary_file = output_dir / f"experiment_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Complete results saved to:")
        logger.info(f"  JSON: {results_file}")
        logger.info(f"  CSV: {csv_file}")
        logger.info(f"  Summary: {summary_file}")
        logger.info(f"  Sample conversations: {transcripts_file}")
    
    def _save_results_as_csv(self, results: List[NegotiationResult], csv_file: Path):
        """Save results as CSV for statistical analysis."""
        
        with open(csv_file, 'w', newline='') as f:
            fieldnames = [
                'negotiation_id', 'buyer_model', 'supplier_model', 'reflection_pattern',
                'completed', 'agreed_price', 'total_rounds', 'total_tokens', 'total_time',
                'buyer_profit', 'supplier_profit', 'distance_from_optimal',
                'termination_type', 'total_cost'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    'negotiation_id': result.negotiation_id,
                    'buyer_model': result.buyer_model,
                    'supplier_model': result.supplier_model,
                    'reflection_pattern': result.reflection_pattern,
                    'completed': result.completed,
                    'agreed_price': result.agreed_price,
                    'total_rounds': result.total_rounds,
                    'total_tokens': result.total_tokens,
                    'total_time': result.total_time,
                    'buyer_profit': result.buyer_profit,
                    'supplier_profit': result.supplier_profit,
                    'distance_from_optimal': result.distance_from_optimal,
                    'termination_type': result.termination_type.value if result.termination_type else '',
                    'total_cost': result.metadata.get('total_cost', 0.0)
                })
    
    async def _save_sample_conversations(self, results: List[NegotiationResult], transcripts_file: Path):
        """Save sample conversations for qualitative analysis."""
        
        # Select interesting conversations to save
        samples = []
        
        # Get successful negotiations from each reflection pattern
        for pattern in ["00", "01", "10", "11"]:
            pattern_results = [r for r in results if r.reflection_pattern == pattern and r.completed]
            if pattern_results:
                # Take first successful one as sample
                samples.append(pattern_results[0])
        
        # Add some failed negotiations for comparison
        failed_results = [r for r in results if not r.completed]
        if failed_results:
            samples.extend(failed_results[:3])  # First 3 failures
        
        with open(transcripts_file, 'w') as f:
            f.write("NEWSVENDOR EXPERIMENT - SAMPLE CONVERSATIONS (UPDATED)\n")
            f.write("=" * 80 + "\n\n")
            
            for result in samples:
                f.write(f"NEGOTIATION: {result.negotiation_id}\n")
                f.write(f"BUYER: {result.buyer_model}\n")
                f.write(f"SUPPLIER: {result.supplier_model}\n")
                f.write(f"REFLECTION: {result.reflection_pattern}\n")
                f.write(f"RESULT: {'SUCCESS' if result.completed else 'FAILED'}\n")
                f.write(f"PRICE: ${result.agreed_price}\n")
                f.write(f"ROUNDS: {result.total_rounds}\n")
                f.write(f"COST: ${result.metadata.get('total_cost', 0):.6f}\n")
                f.write("-" * 80 + "\n")
                
                for turn in result.turns:
                    speaker = turn.speaker.upper()
                    message = turn.message
                    price = f" (${turn.price})" if turn.price else ""
                    f.write(f"Round {turn.round_number} - {speaker}{price}:\n")
                    f.write(f"  {message}\n\n")
                
                f.write("=" * 80 + "\n\n")
    
    async def shutdown(self):
        """Clean shutdown."""
        if self.model_manager:
            await self.model_manager.shutdown()


@click.command()
@click.option('--models', type=str, help='Comma-separated list of models to test (default: all 10)')
@click.option('--patterns', type=str, help='Comma-separated reflection patterns (default: 00,01,10,11)')
@click.option('--reps', type=int, default=20, help='Replications per condition (default: 20)')
@click.option('--concurrent', type=int, default=2, help='Max concurrent negotiations (default: 2)')
@click.option('--output', type=click.Path(), help='Output directory for results')
@click.option('--dry-run', is_flag=True, help='Show experiment plan without running')
def main(models: Optional[str], patterns: Optional[str], reps: int, concurrent: int, 
         output: Optional[str], dry_run: bool):
    """Run the complete newsvendor experiment across all models and conditions."""
    
    async def run_experiment():
        runner = FullExperimentRunner(max_concurrent=concurrent)
        
        # Override models if specified
        if models:
            runner.all_models = [model.strip() for model in models.split(',')]
        
        # Override reflection patterns if specified
        if patterns:
            runner.reflection_patterns = [pattern.strip() for pattern in patterns.split(',')]
        
        # Override output directory if specified
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
        
        try:
            await runner.initialize()
            
            # Generate experiment plan
            experiment_plan = runner.generate_experiment_plan()
            
            # Update replications if different from default
            if reps != 20:
                for config in experiment_plan:
                    config.replications = reps
            
            # Show cost estimate
            cost_estimate = runner.estimate_experiment_cost(experiment_plan)
            
            print("\n" + "=" * 80)
            print("NEWSVENDOR EXPERIMENT PLAN (UPDATED)")
            print("=" * 80)
            print(f"Models: {len(runner.all_models)}")
            print(f"Reflection patterns: {len(runner.reflection_patterns)}")
            print(f"Replications per condition: {reps}")
            print(f"Total negotiations: {len(experiment_plan) * reps}")
            print(f"Estimated cost: ${cost_estimate['total_estimated_cost']:.2f}")
            print(f"Max concurrent: {concurrent}")
            print(f"Token budget per model: 2000 (unlimited thinking)")
            print(f"Standardized reflection: Yes")
            
            if dry_run:
                print("\n🔍 DRY RUN - Experiment plan generated but not executed")
                return
            
            # Run full experiment
            analysis = await runner.run_full_experiment(experiment_plan)
            
            # Print final summary
            print("\n" + "=" * 80)
            print("EXPERIMENT COMPLETE")
            print("=" * 80)
            
            summary = analysis["experiment_summary"]
            print(f"Total negotiations: {summary['total_negotiations']}")
            print(f"Successful: {summary['successful_negotiations']} ({summary['success_rate']*100:.1f}%)")
            print(f"Duration: {summary['completion_time_hours']:.1f} hours")
            print(f"Total cost: ${summary['total_cost']:.2f}")
            
            if "price_analysis" in analysis:
                price_stats = analysis["price_analysis"]
                print(f"Average price: ${price_stats['mean_price']:.2f}")
                print(f"Optimal convergence: {price_stats['optimal_convergence_rate']*100:.1f}%")
            
            print("\n✅ Full experiment complete! Check output files for detailed results.")
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            raise
        
        finally:
            await runner.shutdown()
    
    asyncio.run(run_experiment())


if __name__ == '__main__':
    main()