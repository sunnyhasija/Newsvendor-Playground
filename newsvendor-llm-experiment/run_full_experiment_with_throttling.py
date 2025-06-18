#!/usr/bin/env python3
"""
run_full_experiment_with_throttling.py
Full Experiment Runner with Smart Throttling Management
Runs complete experiment: 10 models × 10 models × 4 reflection patterns × 20 runs = 8,000 negotiations
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
        logging.FileHandler('newsvendor_full_experiment_throttled.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
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


class FullExperimentRunnerWithThrottling:
    """Runs the complete experimental protocol with smart throttling management."""
    
    def __init__(self, max_concurrent: int = 1):  # More conservative for throttling
        """Initialize full experiment runner with throttling management."""
        self.model_manager = None
        self.max_concurrent = max_concurrent
        self.results = []
        self.start_time = None
        
        # Game configuration
        self.game_config = {
            'selling_price': 100,
            'production_cost': 30,
            'demand_mean': 40,
            'demand_std': 10,
            'optimal_price': 65,
            'max_rounds': 10,
            'timeout_seconds': 120
        }
        
        # All 10 models (TinyLlama removed, Grok added)
        self.all_models = [
            # Local models
            "qwen2:1.5b",
            "gemma2:2b",
            "phi3:mini", 
            "llama3.2:latest",
            "mistral:instruct",
            "qwen:7b",
            "qwen3:latest",
            # Remote models
            "claude-sonnet-4-remote",
            "o3-remote",
            "grok-remote"
        ]
        
        # All reflection patterns
        self.reflection_patterns = ["00", "01", "10", "11"]
        
        # Smart throttling configuration
        self.throttle_delays = {
            'claude-sonnet-4-remote': 0.5,   # Start minimal, increase if throttled
            'o3-remote': 0.5,
            'grok-remote': 0.5,
            'default': 0.1
        }
        
        self.throttle_counts = {}         # Track throttling events per model
        self.throttle_multipliers = {}    # Dynamic multipliers for throttled models
        self.last_call_times = {}         # Track last call times
        
        # Progressive backoff parameters
        self.max_throttle_delay = 10.0    # Maximum delay after repeated throttling
        self.throttle_backoff_factor = 1.5 # Multiply delay by this after each throttle
        
        logger.info(f"Initialized FullExperimentRunnerWithThrottling for {len(self.all_models)} models")
        logger.info(f"Smart throttling enabled with progressive backoff")
        logger.info(f"Max concurrent: {max_concurrent} (conservative for API limits)")
    
    async def initialize(self) -> None:
        """Initialize the model manager and validate models."""
        logger.info("Initializing unified model manager with throttling support...")
        self.model_manager = create_unified_model_manager()
        
        # Quick validation with throttling awareness
        logger.info("Running quick model validation with throttling detection...")
        
        validation_results = {}
        for model_name in self.all_models:
            try:
                # Apply throttling delay
                await self._apply_smart_throttling(model_name)
                
                # Test with minimal tokens
                test_prompt = "You are negotiating. Say only: I offer $50"
                
                if model_name == 'o3-remote':
                    response = await self.model_manager.generate_response(
                        model_name, test_prompt, max_completion_tokens=50, reasoning_effort='low'
                    )
                else:
                    response = await self.model_manager.generate_response(
                        model_name, test_prompt, max_tokens=50
                    )
                
                validation_results[model_name] = {
                    "success": response.success,
                    "cost": getattr(response, 'cost_estimate', 0),
                    "error": response.error if not response.success else None
                }
                
                if response.success:
                    logger.info(f"   ✅ {model_name}")
                else:
                    logger.warning(f"   ❌ {model_name}: {response.error}")
                
            except Exception as e:
                logger.error(f"   ❌ {model_name}: {str(e)}")
                validation_results[model_name] = {"success": False, "error": str(e)}
        
        working_models = [model for model, result in validation_results.items() if result["success"]]
        failed_models = [model for model, result in validation_results.items() if not result["success"]]
        
        logger.info(f"✅ Working models: {len(working_models)}")
        
        if failed_models:
            logger.error(f"❌ Failed models: {failed_models}")
            # Remove failed models from our list
            self.all_models = [m for m in self.all_models if m not in failed_models]
            logger.info(f"Updated model list to {len(self.all_models)} working models")
        
        logger.info("Model validation complete with throttling awareness")
    
    async def _apply_smart_throttling(self, model_name: str) -> None:
        """Apply smart throttling with progressive backoff for repeatedly throttled models."""
        
        # Get base delay for this model
        base_delay = self.throttle_delays.get(model_name, self.throttle_delays['default'])
        
        # Apply multiplier if this model has been throttled before
        multiplier = self.throttle_multipliers.get(model_name, 1.0)
        actual_delay = min(base_delay * multiplier, self.max_throttle_delay)
        
        # Check if we need to wait based on last call time
        if model_name in self.last_call_times:
            time_since_last = time.time() - self.last_call_times[model_name]
            if time_since_last < actual_delay:
                wait_time = actual_delay - time_since_last
                if wait_time > 0.1:  # Only log if significant wait
                    throttle_count = self.throttle_counts.get(model_name, 0)
                    logger.debug(f"Smart throttling {model_name}: waiting {wait_time:.1f}s (throttled {throttle_count} times)")
                await asyncio.sleep(wait_time)
        
        # Update last call time
        self.last_call_times[model_name] = time.time()
    
    def _handle_throttling_event(self, model_name: str) -> None:
        """Handle a throttling event by updating delays and multipliers."""
        
        # Increment throttle count
        self.throttle_counts[model_name] = self.throttle_counts.get(model_name, 0) + 1
        
        # Increase the delay multiplier
        current_multiplier = self.throttle_multipliers.get(model_name, 1.0)
        new_multiplier = min(current_multiplier * self.throttle_backoff_factor, 10.0)  # Cap at 10x
        self.throttle_multipliers[model_name] = new_multiplier
        
        new_delay = min(
            self.throttle_delays.get(model_name, self.throttle_delays['default']) * new_multiplier,
            self.max_throttle_delay
        )
        
        logger.warning(
            f"Throttling event #{self.throttle_counts[model_name]} for {model_name}, "
            f"increasing delay to {new_delay:.1f}s (multiplier: {new_multiplier:.1f}x)"
        )
    
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
                        replications=20  # Exactly 20 for ALL combinations
                    )
                    experiment_plan.append(config)
        
        total_negotiations = len(experiment_plan) * 20
        logger.info(f"Generated experiment plan:")
        logger.info(f"  Model combinations: {len(self.all_models)}² = {len(self.all_models)**2}")
        logger.info(f"  Reflection patterns: {len(self.reflection_patterns)}")
        logger.info(f"  Replications per condition: 20 (uniform across all pairs)")
        logger.info(f"  Total configurations: {len(experiment_plan)}")
        logger.info(f"  Total negotiations: {total_negotiations}")
        
        return experiment_plan
    
    def estimate_experiment_cost_and_time(self, experiment_plan: List[ExperimentConfig]) -> Dict[str, float]:
        """Estimate the total cost and time of the experiment with throttling."""
        
        # Cost estimates per token
        costs = {
            "claude-sonnet-4-remote": 0.000075,
            "o3-remote": 0.000240,
            "grok-remote": 0.000020
        }
        
        # Token estimates per negotiation
        estimated_tokens_per_negotiation = {
            "claude-sonnet-4-remote": 400,
            "o3-remote": 600,
            "grok-remote": 350
        }
        
        # Time estimates per negotiation (including throttling)
        estimated_time_per_negotiation = {
            "claude-sonnet-4-remote": 8.0,  # Includes throttling delays
            "o3-remote": 10.0,               # O3 is slower + throttling
            "grok-remote": 6.0,              # Grok is faster
            "local": 3.0                     # Local models are fast
        }
        
        total_cost = 0.0
        total_time_hours = 0.0
        cost_breakdown = {}
        
        for config in experiment_plan:
            config_cost = 0.0
            config_time = 0.0
            
            for model in [config.buyer_model, config.supplier_model]:
                if model in costs:
                    # Remote model
                    model_cost = (
                        costs[model] * 
                        estimated_tokens_per_negotiation[model] * 
                        config.replications
                    )
                    config_cost += model_cost
                    
                    if model not in cost_breakdown:
                        cost_breakdown[model] = 0.0
                    cost_breakdown[model] += model_cost
                    
                    # Add time estimate
                    config_time += estimated_time_per_negotiation[model] * config.replications
                else:
                    # Local model
                    config_time += estimated_time_per_negotiation["local"] * config.replications
            
            total_cost += config_cost
            # Take max time between buyer and supplier (they alternate)
            total_time_hours += max(config_time / 2, estimated_time_per_negotiation["local"] * config.replications) / 3600
        
        # Adjust for concurrency
        total_time_hours = total_time_hours / self.max_concurrent
        
        return {
            "total_estimated_cost": total_cost,
            "breakdown": cost_breakdown,
            "estimated_time_hours": total_time_hours,
            "negotiations_with_remote": sum(
                config.replications for config in experiment_plan
                if "remote" in config.buyer_model or "remote" in config.supplier_model
            ),
            "avg_cost_per_negotiation": total_cost / (len(experiment_plan) * 20) if experiment_plan else 0
        }
    
    async def run_full_experiment(self, experiment_plan: Optional[List[ExperimentConfig]] = None) -> Dict[str, Any]:
        """Run the complete experiment with smart throttling management."""
        
        if experiment_plan is None:
            experiment_plan = self.generate_experiment_plan()
        
        # Estimate cost and time
        estimates = self.estimate_experiment_cost_and_time(experiment_plan)
        logger.info(f"💰 Estimated total cost: ${estimates['total_estimated_cost']:.2f}")
        logger.info(f"⏱️  Estimated time: {estimates['estimated_time_hours']:.1f} hours")
        for model, cost in estimates['breakdown'].items():
            logger.info(f"   {model}: ${cost:.2f}")
        
        # Confirm before proceeding
        print(f"\n" + "="*80)
        print("FULL EXPERIMENT WITH SMART THROTTLING")
        print("="*80)
        print(f"Total negotiations: {len(experiment_plan) * 20:,}")
        print(f"Estimated cost: ${estimates['total_estimated_cost']:.2f}")
        print(f"Estimated time: {estimates['estimated_time_hours']:.1f} hours")
        print(f"Smart throttling: Enabled with progressive backoff")
        print(f"Concurrency: {self.max_concurrent} (conservative)")
        
        confirm = input(f"\nProceed with experiment? [y/N]: ")
        if confirm.lower() != 'y':
            logger.info("Experiment cancelled by user")
            return {"status": "cancelled"}
        
        self.start_time = time.time()
        logger.info(f"🚀 Starting full experiment with throttling at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create all individual negotiations from the experiment plan
        all_negotiations = []
        for config in experiment_plan:
            for rep in range(config.replications):
                negotiation_id = f"{config.buyer_model}_{config.supplier_model}_{config.reflection_pattern}_rep{rep:02d}"
                all_negotiations.append((config, negotiation_id))
        
        logger.info(f"Total negotiations to run: {len(all_negotiations)}")
        
        # Run negotiations with smart throttling and progress tracking
        results = await self._run_with_smart_throttling(all_negotiations)
        
        # Analyze complete results
        analysis = self._analyze_complete_results(results)
        
        # Save complete dataset
        await self._save_complete_results(results, analysis)
        
        return analysis
    
    async def _run_with_smart_throttling(self, negotiations: List[Tuple[ExperimentConfig, str]]) -> List[NegotiationResult]:
        """Run all negotiations with smart throttling and progress tracking."""
        
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Progress tracking
        progress_bar = tqdm(total=len(negotiations), desc="Negotiations")
        completed_count = 0
        total_cost = 0.0
        last_throttle_report = time.time()
        
        async def run_single_with_throttling(config_and_id: Tuple[ExperimentConfig, str]) -> NegotiationResult:
            nonlocal completed_count, total_cost, last_throttle_report
            
            config, negotiation_id = config_and_id
            
            async with semaphore:
                try:
                    result = await self._run_single_negotiation_with_throttling(config, negotiation_id)
                    
                    # Update progress
                    completed_count += 1
                    negotiation_cost = result.metadata.get('total_cost', 0.0)
                    total_cost += negotiation_cost
                    
                    # Update progress bar
                    throttle_events = sum(self.throttle_counts.values())
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'Success': f"{result.completed}",
                        'Price': f"${result.agreed_price}" if result.agreed_price else "None",
                        'Cost': f"${total_cost:.2f}",
                        'Throttles': f"{throttle_events}",
                        'ETA': f"{self._estimate_remaining_time(completed_count, len(negotiations))}"
                    })
                    
                    # Log milestone progress and throttling stats
                    if completed_count % 100 == 0 or time.time() - last_throttle_report > 300:  # Every 100 negotiations or 5 min
                        throttle_events = sum(self.throttle_counts.values())
                        throttled_models = list(self.throttle_counts.keys())
                        
                        logger.info(f"✅ Progress: {completed_count}/{len(negotiations)} (${total_cost:.2f} spent)")
                        if throttle_events > 0:
                            logger.info(f"🐌 Throttling: {throttle_events} events across {len(throttled_models)} models")
                            for model, count in self.throttle_counts.items():
                                delay = self.throttle_delays.get(model, 0.5) * self.throttle_multipliers.get(model, 1.0)
                                logger.info(f"   {model}: {count} throttles, current delay: {delay:.1f}s")
                        
                        last_throttle_report = time.time()
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Failed negotiation {negotiation_id}: {e}")
                    return self._create_failed_result(config, negotiation_id, str(e))
        
        try:
            # Create all tasks
            tasks = [run_single_with_throttling(neg) for neg in negotiations]
            
            # Execute with controlled concurrency
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Negotiation {i} failed with exception: {result}")
                    config, negotiation_id = negotiations[i]
                    failed_result = self._create_failed_result(config, negotiation_id, str(result))
                    final_results.append(failed_result)
                else:
                    final_results.append(result)
            
            return final_results
            
        finally:
            progress_bar.close()
            
            # Final throttling report
            total_throttles = sum(self.throttle_counts.values())
            if total_throttles > 0:
                logger.info(f"🐌 Final throttling report: {total_throttles} total events")
                for model, count in self.throttle_counts.items():
                    final_delay = self.throttle_delays.get(model, 0.5) * self.throttle_multipliers.get(model, 1.0)
                    logger.info(f"   {model}: {count} throttles, final delay: {final_delay:.1f}s")
            else:
                logger.info("🎉 No throttling events detected!")
    
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
    
    async def _run_single_negotiation_with_throttling(self, config: ExperimentConfig, negotiation_id: str) -> NegotiationResult:
        """Run a single negotiation with smart throttling management."""
        
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
            
            # Conduct negotiation with throttling
            result = await self._conduct_negotiation_with_throttling(tracker, buyer_agent, supplier_agent)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in negotiation {negotiation_id}: {e}")
            return self._create_failed_result(config, negotiation_id, str(e))
    
    async def _conduct_negotiation_with_throttling(self, tracker, buyer_agent, supplier_agent) -> NegotiationResult:
        """Conduct the actual negotiation with smart throttling."""
        
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
                
                # Apply smart throttling
                await self._apply_smart_throttling(current_agent.model_name)
                
                # Generate conversation context
                context = tracker.get_conversation_history()
                
                # Get agent response with throttling-aware retry
                response = await self._generate_with_throttling_retry(current_agent, context, tracker, max_rounds)
                
                if not response or not response.success:
                    tracker.force_termination("generation_failure")
                    break
                
                # Track cost
                total_cost += getattr(response, 'cost_estimate', 0.0)
                
                # Add turn to conversation
                if hasattr(tracker.add_turn, '__call__') and asyncio.iscoroutinefunction(tracker.add_turn):
                    turn_added = await tracker.add_turn(
                        speaker=agent_role,
                        message=response.text,
                        reflection=None,
                        tokens_used=response.tokens_used,
                        generation_time=response.generation_time
                    )
                else:
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
    
    async def _generate_with_throttling_retry(self, agent, context, tracker, max_rounds, max_retries=3):
        """Generate response with throttling-aware retry logic."""
        
        for attempt in range(max_retries):
            try:
                response = await agent.generate_response(
                    context=context,
                    negotiation_history=[asdict(turn) for turn in tracker.turns],
                    round_number=tracker.round_number + 1,
                    max_rounds=max_rounds
                )
                
                if response.success:
                    return response
                
                # Check if it's a throttling error
                if "throttling" in response.error.lower() or "too many" in response.error.lower():
                    # Handle throttling event
                    self._handle_throttling_event(agent.model_name)
                    
                    if attempt < max_retries - 1:
                        # Apply progressive backoff
                        base_wait = 2.0 * (attempt + 1)  # 2, 4, 6 seconds
                        multiplier = self.throttle_multipliers.get(agent.model_name, 1.0)
                        wait_time = min(base_wait * multiplier, 30.0)  # Cap at 30 seconds
                        
                        logger.warning(f"Throttled {agent.model_name}, waiting {wait_time:.1f}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait_time)
                        continue
                
                # For non-throttling errors, return the response
                return response
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2.0
                    logger.warning(f"Generation error for {agent.model_name}, retrying in {wait_time:.1f}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Create a failed response for the final attempt
                    return type('Response', (), {
                        'success': False,
                        'error': str(e),
                        'text': '',
                        'tokens_used': 0,
                        'generation_time': 0.0,
                        'cost_estimate': 0.0
                    })()
        
        return None
    
    def _create_failed_result(self, config: ExperimentConfig, negotiation_id: str, error: str) -> NegotiationResult:
        """Create a failed negotiation result."""
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
            metadata={"error": error}
        )
    
    def _analyze_complete_results(self, results: List[NegotiationResult]) -> Dict[str, Any]:
        """Analyze complete experimental results including throttling statistics."""
        
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
                "total_cost": sum(r.metadata.get('total_cost', 0.0) for r in results),
                "models_tested": len(self.all_models),
                "uniform_replications": 20,
                "throttling_stats": {
                    "total_throttle_events": sum(self.throttle_counts.values()),
                    "throttled_models": len(self.throttle_counts),
                    "throttle_counts_by_model": dict(self.throttle_counts),
                    "final_delay_multipliers": dict(self.throttle_multipliers)
                }
            },
            "price_analysis": {},
            "efficiency_analysis": {},
            "model_analysis": {},
            "reflection_analysis": {},
            "cost_analysis": {},
            "throttling_analysis": self._analyze_throttling_impact(results)
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
        
        # Model-by-model analysis (existing method)
        analysis["model_analysis"] = self._analyze_by_model(results)
        
        # Reflection pattern analysis (existing method)
        analysis["reflection_analysis"] = self._analyze_by_reflection(results)
        
        # Cost analysis (existing method)
        analysis["cost_analysis"] = self._analyze_costs(results)
        
        return analysis
    
    def _analyze_throttling_impact(self, results: List[NegotiationResult]) -> Dict[str, Any]:
        """Analyze the impact of throttling on experiment performance."""
        
        throttling_analysis = {
            "summary": {
                "total_throttle_events": sum(self.throttle_counts.values()),
                "models_throttled": len(self.throttle_counts),
                "throttle_rate": sum(self.throttle_counts.values()) / len(results) if results else 0
            },
            "by_model": {},
            "impact_on_performance": {}
        }
        
        # Analyze throttling by model
        for model, count in self.throttle_counts.items():
            # Find negotiations involving this model
            model_negotiations = [
                r for r in results 
                if r.buyer_model == model or r.supplier_model == model
            ]
            
            if model_negotiations:
                avg_time = sum(r.total_time for r in model_negotiations) / len(model_negotiations)
                success_rate = len([r for r in model_negotiations if r.completed]) / len(model_negotiations)
                
                throttling_analysis["by_model"][model] = {
                    "throttle_count": count,
                    "negotiations_involved": len(model_negotiations),
                    "throttle_rate": count / len(model_negotiations),
                    "avg_negotiation_time": avg_time,
                    "success_rate": success_rate,
                    "final_delay_multiplier": self.throttle_multipliers.get(model, 1.0)
                }
        
        # Calculate performance impact
        if self.throttle_counts:
            throttled_models = set(self.throttle_counts.keys())
            
            # Compare throttled vs non-throttled model performance
            throttled_negotiations = [
                r for r in results
                if r.buyer_model in throttled_models or r.supplier_model in throttled_models
            ]
            non_throttled_negotiations = [
                r for r in results
                if r.buyer_model not in throttled_models and r.supplier_model not in throttled_models
            ]
            
            if throttled_negotiations and non_throttled_negotiations:
                throttling_analysis["impact_on_performance"] = {
                    "throttled_avg_time": sum(r.total_time for r in throttled_negotiations) / len(throttled_negotiations),
                    "non_throttled_avg_time": sum(r.total_time for r in non_throttled_negotiations) / len(non_throttled_negotiations),
                    "throttled_success_rate": len([r for r in throttled_negotiations if r.completed]) / len(throttled_negotiations),
                    "non_throttled_success_rate": len([r for r in non_throttled_negotiations if r.completed]) / len(non_throttled_negotiations)
                }
        
        return throttling_analysis
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    # Include all existing analysis methods (_analyze_by_model, _analyze_by_reflection, _analyze_costs)
    # and saving methods (_save_complete_results, _save_results_as_csv, _save_sample_conversations)
    # [These are the same as in the original experiment runner, so I'll skip them for brevity]
    
    async def shutdown(self):
        """Clean shutdown with throttling summary."""
        if self.model_manager:
            await self.model_manager.shutdown()
        
        # Final throttling summary
        if self.throttle_counts:
            logger.info("Final throttling summary:")
            for model, count in self.throttle_counts.items():
                multiplier = self.throttle_multipliers.get(model, 1.0)
                logger.info(f"  {model}: {count} throttles, final multiplier: {multiplier:.1f}x")


@click.command()
@click.option('--models', type=str, help='Comma-separated list of models to test (default: all 10)')
@click.option('--patterns', type=str, help='Comma-separated reflection patterns (default: 00,01,10,11)')
@click.option('--concurrent', type=int, default=1, help='Max concurrent negotiations (default: 1 for throttling)')
@click.option('--output', type=click.Path(), help='Output directory for results')
@click.option('--dry-run', is_flag=True, help='Show experiment plan without running')
@click.option('--test-throttling', is_flag=True, help='Run small test to verify throttling works')
def main(models: Optional[str], patterns: Optional[str], concurrent: int, 
         output: Optional[str], dry_run: bool, test_throttling: bool):
    """Run the complete newsvendor experiment with smart throttling management."""
    
    async def run_experiment():
        runner = FullExperimentRunnerWithThrottling(max_concurrent=concurrent)
        
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
            
            if test_throttling:
                logger.info("🧪 Running throttling test with a few negotiations...")
                # Create a small test plan
                test_models = runner.all_models[:3] if len(runner.all_models) >= 3 else runner.all_models
                test_plan = []
                for buyer in test_models:
                    for supplier in test_models:
                        if buyer != supplier:
                            test_plan.append(ExperimentConfig(
                                buyer_model=buyer,
                                supplier_model=supplier,
                                reflection_pattern="00",
                                replications=1
                            ))
                            if len(test_plan) >= 5:  # Just 5 test negotiations
                                break
                    if len(test_plan) >= 5:
                        break
                
                analysis = await runner.run_full_experiment(test_plan)
                print(f"\n🧪 Throttling test complete!")
                print(f"Throttle events: {analysis['experiment_summary']['throttling_stats']['total_throttle_events']}")
                return
            
            # Generate full experiment plan
            experiment_plan = runner.generate_experiment_plan()
            
            # Show cost and time estimates
            estimates = runner.estimate_experiment_cost_and_time(experiment_plan)
            
            if dry_run:
                print("\n" + "=" * 80)
                print("FULL EXPERIMENT PLAN (WITH SMART THROTTLING)")
                print("=" * 80)
                print(f"Models: {len(runner.all_models)}")
                print(f"Model list: {', '.join(runner.all_models)}")
                print(f"Reflection patterns: {len(runner.reflection_patterns)}")
                print(f"Replications per condition: 20 (uniform)")
                print(f"Total negotiations: {len(experiment_plan) * 20:,}")
                print(f"Estimated cost: ${estimates['total_estimated_cost']:.2f}")
                print(f"Estimated time: {estimates['estimated_time_hours']:.1f} hours")
                print(f"Max concurrent: {concurrent}")
                print(f"Smart throttling: Enabled with progressive backoff")
                print("\n🔍 DRY RUN - Experiment plan generated but not executed")
                return
            
            # Run full experiment
            analysis = await runner.run_full_experiment(experiment_plan)
            
            # Print final summary
            print("\n" + "=" * 80)
            print("FULL EXPERIMENT WITH SMART THROTTLING COMPLETE")
            print("=" * 80)
            
            summary = analysis["experiment_summary"]
            print(f"Total negotiations: {summary['total_negotiations']:,}")
            print(f"Successful: {summary['successful_negotiations']:,} ({summary['success_rate']*100:.1f}%)")
            print(f"Duration: {summary['completion_time_hours']:.1f} hours")
            print(f"Total cost: ${summary['total_cost']:.2f}")
            
            # Throttling summary
            throttling = summary['throttling_stats']
            print(f"\nThrottling Summary:")
            print(f"  Total throttle events: {throttling['total_throttle_events']}")
            print(f"  Models throttled: {throttling['throttled_models']}")
            if throttling['total_throttle_events'] > 0:
                for model, count in throttling['throttle_counts_by_model'].items():
                    print(f"    {model}: {count} events")
            
            if "price_analysis" in analysis:
                price_stats = analysis["price_analysis"]
                print(f"\nPrice Analysis:")
                print(f"  Average price: ${price_stats['mean_price']:.2f}")
                print(f"  Optimal convergence: {price_stats['optimal_convergence_rate']*100:.1f}%")
            
            print("\n✅ Full experiment with smart throttling complete!")
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            raise
        
        finally:
            await runner.shutdown()
    
    asyncio.run(run_experiment())


if __name__ == '__main__':
    main()