#!/usr/bin/env python3
"""
run_turn_order_experiment.py
Turn Order Control Experiment Runner v0.6 - Literature Bias vs Anchoring Research

This experiment runner implements configurable turn order to separate literature bias 
from anchoring effects in LLM negotiations. Key innovation: identical prompts with 
only context assignment changes.
"""

import asyncio
import json
import logging
import time
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
        logging.FileHandler('newsvendor_turn_order_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
from core.unified_model_manager import create_unified_model_manager
from agents.standardized_agents import StandardizedBuyerAgent, StandardizedSupplierAgent
from core.conversation_tracker import ConversationTracker, NegotiationResult
from parsing.acceptance_detector import TerminationType
from utils.config_loader import load_config, get_turn_order_strategy, set_turn_order_strategy
from utils.data_exporter import DataExporter


@dataclass
class TurnOrderExperimentConfig:
    """Configuration for a turn order controlled experiment."""
    buyer_model: str
    supplier_model: str
    reflection_pattern: str  # "00", "01", "10", "11"
    turn_order_strategy: str  # "buyer_first", "supplier_first", "random"
    replications: int = 20
    max_rounds: int = 10
    timeout_seconds: int = 120


class TurnOrderExperimentRunner:
    """Runs turn order controlled experiments to separate literature bias from anchoring effects."""
    
    def __init__(self, max_concurrent: int = 1):
        """Initialize turn order experiment runner."""
        self.model_manager = None
        self.max_concurrent = max_concurrent
        self.results = []
        self.start_time = None
        
        # Load configuration with turn order support
        self.config = load_config()
        
        # Game configuration
        self.game_config = {
            'selling_price': self.config['game']['selling_price'],
            'production_cost': self.config['game']['production_cost'],
            'demand_mean': self.config['game']['demand_mean'],
            'demand_std': self.config['game']['demand_std'],
            'optimal_price': self.config['game']['optimal_wholesale_price'],
            'max_rounds': self.config['negotiation']['max_rounds'],
            'timeout_seconds': self.config['negotiation']['timeout_seconds']
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
        
        # v0.6 Turn order strategies
        self.turn_order_strategies = ["buyer_first", "supplier_first"]
        
        # Initialize data exporter with turn order support
        self.data_exporter = DataExporter(self.config.get('storage', {}))
        
        # Smart throttling configuration
        self.throttle_delays = {
            'claude-sonnet-4-remote': 0.5,
            'o3-remote': 0.5,
            'grok-remote': 0.5,
            'default': 0.1
        }
        self.throttle_counts = {}
        self.throttle_multipliers = {}
        self.last_call_times = {}
        self.max_throttle_delay = 10.0
        self.throttle_backoff_factor = 1.5
        logger.info(f"Smart throttling enabled with progressive backoff")
        
        logger.info(f"Initialized TurnOrderExperimentRunner v0.6")
        logger.info(f"Models: {len(self.all_models)}")
        logger.info(f"Turn order strategies: {self.turn_order_strategies}")
        logger.info(f"Research focus: Literature bias vs anchoring effects")
    
    async def initialize(self) -> None:
        """Initialize the model manager and validate models."""
        logger.info("Initializing unified model manager...")
        self.model_manager = create_unified_model_manager()
        
        # Quick validation with a subset of models
        logger.info("Running quick model validation...")
        
        validation_results = {}
        test_models = self.all_models[:3]  # Test first 3 models
        
        for model_name in test_models:
            try:
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
        
        if len(working_models) < 3:
            logger.error("Insufficient working models for experiment")
            raise RuntimeError("Model validation failed")
        
        logger.info("Model validation complete")
    
    async def _apply_smart_throttling(self, model_name: str) -> None:
        """Apply smart throttling with progressive backoff."""
        base_delay = self.throttle_delays.get(model_name, self.throttle_delays['default'])
        multiplier = self.throttle_multipliers.get(model_name, 1.0)
        actual_delay = min(base_delay * multiplier, self.max_throttle_delay)
        
        if model_name in self.last_call_times:
            time_since_last = time.time() - self.last_call_times[model_name]
            if time_since_last < actual_delay:
                wait_time = actual_delay - time_since_last
                if wait_time > 0.1:
                    throttle_count = self.throttle_counts.get(model_name, 0)
                    logger.debug(f"Smart throttling {model_name}: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        self.last_call_times[model_name] = time.time()
    
    def _handle_throttling_event(self, model_name: str) -> None:
        """Handle a throttling event by updating delays and multipliers."""
        self.throttle_counts[model_name] = self.throttle_counts.get(model_name, 0) + 1
        current_multiplier = self.throttle_multipliers.get(model_name, 1.0)
        new_multiplier = min(current_multiplier * self.throttle_backoff_factor, 10.0)
        self.throttle_multipliers[model_name] = new_multiplier
        
        new_delay = min(
            self.throttle_delays.get(model_name, self.throttle_delays['default']) * new_multiplier,
            self.max_throttle_delay
        )
        
        logger.warning(
            f"Throttling event #{self.throttle_counts[model_name]} for {model_name}, "
            f"increasing delay to {new_delay:.1f}s (multiplier: {new_multiplier:.1f}x)"
        )
    
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
        for model_name, count in self.throttle_counts.items():
            throttling_analysis["by_model"][model_name] = {
                "throttle_events": count,
                "current_multiplier": self.throttle_multipliers.get(model_name, 1.0),
                "current_delay": min(
                    self.throttle_delays.get(model_name, self.throttle_delays['default']) * 
                    self.throttle_multipliers.get(model_name, 1.0),
                    self.max_throttle_delay
                )
            }
        
        # Calculate performance impact
        total_throttle_events = sum(self.throttle_counts.values())
        if total_throttle_events > 0:
            throttling_analysis["impact_on_performance"] = {
                "throttling_overhead_estimate_seconds": total_throttle_events * 2.0,  # Conservative estimate
                "efficiency_impact": "low" if total_throttle_events < len(results) * 0.1 else "moderate" if total_throttle_events < len(results) * 0.3 else "high",
                "recommendations": self._generate_throttling_recommendations(total_throttle_events, len(results))
            }
        
        return throttling_analysis
    
    def _generate_throttling_recommendations(self, throttle_events: int, total_negotiations: int) -> List[str]:
        """Generate recommendations based on throttling performance."""
        recommendations = []
        throttle_rate = throttle_events / total_negotiations if total_negotiations > 0 else 0
        
        if throttle_rate > 0.3:
            recommendations.append("Consider increasing base throttle delays for remote models")
            recommendations.append("Reduce concurrent negotiations for remote models")
        elif throttle_rate > 0.1:
            recommendations.append("Monitor throttling patterns for optimization opportunities")
        else:
            recommendations.append("Throttling performance is optimal")
        
        if len(self.throttle_counts) > 3:
            recommendations.append("Multiple models experiencing throttling - consider model rotation")
        
        return recommendations
    
    def generate_experiment_plan(
        self, 
        models_subset: Optional[List[str]] = None,
        strategies_subset: Optional[List[str]] = None,
        patterns_subset: Optional[List[str]] = None,
        replications: int = 20
    ) -> List[TurnOrderExperimentConfig]:
        """
        Generate experiment plan with turn order control.
        
        Args:
            models_subset: Subset of models to test (default: all)
            strategies_subset: Subset of turn order strategies (default: all)
            patterns_subset: Subset of reflection patterns (default: all)
            replications: Number of replications per condition
            
        Returns:
            List of experiment configurations
        """
        
        models = models_subset or self.all_models
        strategies = strategies_subset or self.turn_order_strategies
        patterns = patterns_subset or self.reflection_patterns
        
        experiment_plan = []
        
        for buyer_model in models:
            for supplier_model in models:
                for strategy in strategies:
                    for pattern in patterns:
                        config = TurnOrderExperimentConfig(
                            buyer_model=buyer_model,
                            supplier_model=supplier_model,
                            reflection_pattern=pattern,
                            turn_order_strategy=strategy,
                            replications=replications
                        )
                        experiment_plan.append(config)
        
        total_negotiations = len(experiment_plan) * replications
        logger.info(f"Generated turn order experiment plan:")
        logger.info(f"  Model combinations: {len(models)}² = {len(models)**2}")
        logger.info(f"  Turn order strategies: {len(strategies)}")
        logger.info(f"  Reflection patterns: {len(patterns)}")
        logger.info(f"  Replications per condition: {replications}")
        logger.info(f"  Total configurations: {len(experiment_plan)}")
        logger.info(f"  Total negotiations: {total_negotiations}")
        
        return experiment_plan
    
    def estimate_experiment_cost_and_time(self, experiment_plan: List[TurnOrderExperimentConfig]) -> Dict[str, float]:
        """Estimate the total cost and time of the turn order experiment."""
        
        # Cost estimates per token (same as original)
        costs = {
            "claude-sonnet-4-remote": 0.000075,
            "o3-remote": 0.000240,
            "grok-remote": 0.000020
        }
        
        # Token estimates per negotiation
        estimated_tokens_per_negotiation = {
            "claude-sonnet-4-remote": 400,
            "o3-remote": 600,
            "grok-remote": 350,
            "local": 200
        }
        
        # Time estimates per negotiation
        estimated_time_per_negotiation = {
            "claude-sonnet-4-remote": 8.0,
            "o3-remote": 10.0,
            "grok-remote": 6.0,
            "local": 3.0
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
                    
                    config_time += estimated_time_per_negotiation[model] * config.replications
                else:
                    # Local model
                    config_time += estimated_time_per_negotiation["local"] * config.replications
            
            total_cost += config_cost
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
    
    async def run_turn_order_experiment(
        self, 
        experiment_plan: Optional[List[TurnOrderExperimentConfig]] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run the turn order controlled experiment."""
        
        if experiment_plan is None:
            # Default: quick validation experiment with subset
            experiment_plan = self.generate_experiment_plan(
                models_subset=self.all_models[:3],  # First 3 models
                replications=5  # Fewer replications for testing
            )
        
        # Estimate cost and time
        estimates = self.estimate_experiment_cost_and_time(experiment_plan)
        logger.info(f"💰 Estimated total cost: ${estimates['total_estimated_cost']:.2f}")
        logger.info(f"⏱️  Estimated time: {estimates['estimated_time_hours']:.1f} hours")
        
        self.start_time = time.time()
        logger.info(f"🚀 Starting turn order experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create all individual negotiations from the experiment plan
        all_negotiations = []
        for config in experiment_plan:
            for rep in range(config.replications):
                # v0.6 CRITICAL: Include turn order strategy in negotiation ID
                negotiation_id = f"{config.buyer_model}_{config.supplier_model}_{config.reflection_pattern}_{config.turn_order_strategy}_rep{rep:02d}"
                all_negotiations.append((config, negotiation_id))
        
        logger.info(f"Total negotiations to run: {len(all_negotiations)}")
        
        # Run negotiations with progress tracking
        results = await self._run_all_negotiations(all_negotiations)
        
        # Analyze results with turn order focus
        analysis = self._analyze_turn_order_results(results)
        
        # Save results if requested
        if save_results:
            await self._save_turn_order_results(results, analysis)
        
        return analysis
    
    async def _run_all_negotiations(self, negotiations: List[Tuple[TurnOrderExperimentConfig, str]]) -> List[NegotiationResult]:
        """Run all negotiations with progress tracking."""
        
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Progress tracking
        progress_bar = tqdm(total=len(negotiations), desc="Turn Order Negotiations")
        completed_count = 0
        total_cost = 0.0
        
        async def run_single_negotiation(config_and_id: Tuple[TurnOrderExperimentConfig, str]) -> NegotiationResult:
            nonlocal completed_count, total_cost
            
            config, negotiation_id = config_and_id
            
            async with semaphore:
                try:
                    result = await self._run_single_turn_order_negotiation(config, negotiation_id)
                    
                    # Update progress
                    completed_count += 1
                    negotiation_cost = result.metadata.get('total_cost', 0.0)
                    total_cost += negotiation_cost
                    
                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'Success': f"{result.completed}",
                        'Price': f"${result.agreed_price}" if result.agreed_price else "None",
                        'Turn Order': f"{result.turn_order_strategy}",
                        'First': f"{result.first_speaker}",
                        'Cost': f"${total_cost:.2f}"
                    })
                    
                    # Log milestone progress
                    if completed_count % 50 == 0:
                        logger.info(f"✅ Progress: {completed_count}/{len(negotiations)} (${total_cost:.2f} spent)")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Failed negotiation {negotiation_id}: {e}")
                    return self._create_failed_result(config, negotiation_id, str(e))
        
        try:
            # Create all tasks
            tasks = [run_single_negotiation(neg) for neg in negotiations]
            
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
    
    async def _run_single_turn_order_negotiation(
        self, 
        config: TurnOrderExperimentConfig, 
        negotiation_id: str
    ) -> NegotiationResult:
        """Run a single negotiation with turn order control."""
        
        try:
            # Parse reflection pattern
            buyer_reflection = config.reflection_pattern[0] == "1"
            supplier_reflection = config.reflection_pattern[1] == "1"
            
            # v0.6 CRITICAL: Initialize conversation tracker with turn order strategy
            tracker = ConversationTracker(
                negotiation_id=negotiation_id,
                buyer_model=config.buyer_model,
                supplier_model=config.supplier_model,
                reflection_pattern=config.reflection_pattern,
                turn_order_strategy=config.turn_order_strategy,  # v0.6 NEW
                config=self.game_config
            )
            
            # Initialize standardized agents (identical to original)
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
            
            # Conduct negotiation with turn order awareness
            result = await self._conduct_turn_order_negotiation(tracker, buyer_agent, supplier_agent)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in turn order negotiation {negotiation_id}: {e}")
            return self._create_failed_result(config, negotiation_id, str(e))
    
    async def _conduct_turn_order_negotiation(
        self, 
        tracker: ConversationTracker, 
        buyer_agent: StandardizedBuyerAgent, 
        supplier_agent: StandardizedSupplierAgent
    ) -> NegotiationResult:
        """Conduct the actual negotiation with turn order control."""
        
        timeout_seconds = self.game_config.get('timeout_seconds', 120)
        max_rounds = self.game_config.get('max_rounds', 10)
        
        start_time = time.time()
        total_cost = 0.0
        
        try:
            # Main negotiation loop (identical logic, but tracker handles turn order)
            while not tracker.completed and tracker.round_number < max_rounds:
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    tracker.force_termination("timeout")
                    break
                
                # Determine current agent based on tracker's current_speaker
                if tracker.current_speaker == "buyer":
                    current_agent = buyer_agent
                    agent_role = "buyer"
                else:
                    current_agent = supplier_agent
                    agent_role = "supplier"
                
                # Generate conversation context (tracker handles turn order specific context)
                context = tracker.get_conversation_history()
                
                # Apply smart throttling before API call
                await self._apply_smart_throttling(current_agent.model_name)
                
                # Get agent response with throttling retry logic
                response = await self._generate_with_throttling_retry(
                    current_agent,
                    context,
                    tracker,
                    max_rounds,
                    max_retries=3
                )
                
                if not response or not response.success:
                    tracker.force_termination("generation_failure")
                    break
                
                # Track cost
                total_cost += getattr(response, 'cost_estimate', 0.0)
                
                # Add turn to conversation
                turn_added = await tracker.add_turn(
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
            
            # Get final result with turn order information
            result = tracker.get_final_result()
            result.metadata['total_cost'] = total_cost
            
            return result
            
        except Exception as e:
            tracker.force_termination(f"error: {str(e)}")
            result = tracker.get_final_result()
            result.metadata['total_cost'] = total_cost
            return result
    
    def _create_failed_result(
        self, 
        config: TurnOrderExperimentConfig, 
        negotiation_id: str, 
        error: str
    ) -> NegotiationResult:
        """Create a failed negotiation result with turn order information."""
        return NegotiationResult(
            negotiation_id=negotiation_id,
            buyer_model=config.buyer_model,
            supplier_model=config.supplier_model,
            reflection_pattern=config.reflection_pattern,
            turn_order_strategy=config.turn_order_strategy,  # v0.6 NEW
            first_speaker="unknown",  # v0.6 NEW
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
    
    def _analyze_turn_order_results(self, results: List[NegotiationResult]) -> Dict[str, Any]:
        """Analyze results with focus on turn order effects."""
        
        total_results = len(results)
        successful_results = [r for r in results if r.completed and r.agreed_price]
        
        if not results:
            return {"error": "No results to analyze"}
        
        analysis = {
            "experiment_summary": {
                "version": "0.6",
                "experiment_type": "turn_order_control",
                "total_negotiations": total_results,
                "successful_negotiations": len(successful_results),
                "success_rate": len(successful_results) / total_results,
                "completion_time_hours": (time.time() - self.start_time) / 3600 if self.start_time else 0
            },
            "turn_order_analysis": self._analyze_turn_order_effects(results),
            "throttling_analysis": self._analyze_throttling_impact(results),
            "research_findings": {},
            "comparative_analysis": {},
            "statistical_tests": {}
        }
        
        # Detailed turn order analysis
        if successful_results:
            analysis["research_findings"] = self._generate_research_findings(successful_results)
            analysis["comparative_analysis"] = self._compare_turn_order_strategies(successful_results)
        
        return analysis
    
    def _analyze_turn_order_effects(self, results: List[NegotiationResult]) -> Dict[str, Any]:
        """Analyze the effects of different turn order strategies."""
        
        turn_order_analysis = {
            "distribution": {},
            "performance_by_strategy": {},
            "first_speaker_analysis": {}
        }
        
        # Analyze distribution
        for result in results:
            strategy = result.turn_order_strategy
            first_speaker = result.first_speaker
            
            # Count strategies
            if strategy not in turn_order_analysis["distribution"]:
                turn_order_analysis["distribution"][strategy] = 0
            turn_order_analysis["distribution"][strategy] += 1
            
            # Count first speakers
            if first_speaker not in turn_order_analysis["first_speaker_analysis"]:
                turn_order_analysis["first_speaker_analysis"][first_speaker] = {
                    "count": 0, "successes": 0, "avg_price": 0, "price_count": 0
                }
            
            turn_order_analysis["first_speaker_analysis"][first_speaker]["count"] += 1
            
            if result.completed and result.agreed_price:
                turn_order_analysis["first_speaker_analysis"][first_speaker]["successes"] += 1
                turn_order_analysis["first_speaker_analysis"][first_speaker]["avg_price"] += result.agreed_price
                turn_order_analysis["first_speaker_analysis"][first_speaker]["price_count"] += 1
        
        # Calculate averages
        for speaker, stats in turn_order_analysis["first_speaker_analysis"].items():
            if stats["count"] > 0:
                stats["success_rate"] = stats["successes"] / stats["count"]
            if stats["price_count"] > 0:
                stats["avg_price"] = stats["avg_price"] / stats["price_count"]
        
        # Performance by strategy
        strategies = set(r.turn_order_strategy for r in results)
        for strategy in strategies:
            strategy_results = [r for r in results if r.turn_order_strategy == strategy]
            successful_strategy = [r for r in strategy_results if r.completed and r.agreed_price]
            
            if strategy_results:
                performance = {
                    "count": len(strategy_results),
                    "success_count": len(successful_strategy),
                    "success_rate": len(successful_strategy) / len(strategy_results),
                    "avg_rounds": sum(r.total_rounds for r in successful_strategy) / len(successful_strategy) if successful_strategy else 0,
                    "avg_tokens": sum(r.total_tokens for r in successful_strategy) / len(successful_strategy) if successful_strategy else 0
                }
                
                prices = [r.agreed_price for r in successful_strategy if r.agreed_price]
                if prices:
                    performance["price_stats"] = {
                        "mean": sum(prices) / len(prices),
                        "median": sorted(prices)[len(prices)//2],
                        "min": min(prices),
                        "max": max(prices)
                    }
                
                turn_order_analysis["performance_by_strategy"][strategy] = performance
        
        return turn_order_analysis
    
    def _generate_research_findings(self, successful_results: List[NegotiationResult]) -> Dict[str, Any]:
        """Generate research findings about literature bias vs anchoring effects."""
        
        # Separate by turn order strategy
        buyer_first_results = [r for r in successful_results if r.turn_order_strategy == "buyer_first"]
        supplier_first_results = [r for r in successful_results if r.turn_order_strategy == "supplier_first"]
        
        research_findings = {
            "buyer_first_analysis": self._analyze_strategy_results(buyer_first_results),
            "supplier_first_analysis": self._analyze_strategy_results(supplier_first_results),
            "bias_decomposition": {},
            "hypothesis_testing": {}
        }
        
        # Compare the two strategies
        if buyer_first_results and supplier_first_results:
            bf_prices = [r.agreed_price for r in buyer_first_results if r.agreed_price]
            sf_prices = [r.agreed_price for r in supplier_first_results if r.agreed_price]
            
            if bf_prices and sf_prices:
                bf_mean = sum(bf_prices) / len(bf_prices)
                sf_mean = sum(sf_prices) / len(sf_prices)
                
                buyer_advantage_amount = bf_mean - sf_mean  # Positive = buyers still advantaged when suppliers go first
                
                research_findings["bias_decomposition"] = {
                    "buyer_first_mean_price": bf_mean,
                    "supplier_first_mean_price": sf_mean,
                    "buyer_advantage_when_supplier_goes_first": buyer_advantage_amount,
                    "percentage_advantage": (buyer_advantage_amount / sf_mean * 100) if sf_mean > 0 else 0,
                    "anchoring_effect_size": abs(buyer_advantage_amount),
                    "direction": "buyer_favored" if buyer_advantage_amount > 0 else "supplier_favored"
                }
                
                # Hypothesis testing
                research_findings["hypothesis_testing"] = {
                    "H1_literature_bias_only": {
                        "hypothesis": "Buyer advantage persists fully when suppliers go first",
                        "evidence": buyer_advantage_amount > 8.0,  # Similar to original $14.25 advantage
                        "strength": "strong" if buyer_advantage_amount > 8.0 else "weak"
                    },
                    "H2_anchoring_only": {
                        "hypothesis": "Buyer advantage disappears when suppliers go first", 
                        "evidence": abs(buyer_advantage_amount) < 2.0,
                        "strength": "strong" if abs(buyer_advantage_amount) < 1.0 else "moderate" if abs(buyer_advantage_amount) < 2.0 else "weak"
                    },
                    "H3_mixed_effects": {
                        "hypothesis": "Buyer advantage reduces but remains when suppliers go first",
                        "evidence": 2.0 <= abs(buyer_advantage_amount) <= 8.0,
                        "strength": "strong" if 4.0 <= abs(buyer_advantage_amount) <= 6.0 else "moderate"
                    }
                }
                
                # Interpretation
                if buyer_advantage_amount > 8.0:
                    interpretation = "Strong literature bias detected. LLMs favor buyers regardless of turn order."
                elif buyer_advantage_amount > 2.0:
                    interpretation = "Mixed effects detected. Both literature bias and anchoring contribute to buyer advantage."
                elif abs(buyer_advantage_amount) < 2.0:
                    interpretation = "Primary anchoring effect detected. Turn order was the main factor in original findings."
                else:
                    interpretation = "Supplier advantage detected when suppliers go first. Strong anchoring effect."
                
                research_findings["interpretation"] = interpretation
        
        return research_findings
    
    def _analyze_strategy_results(self, results: List[NegotiationResult]) -> Dict[str, Any]:
        """Analyze results for a specific turn order strategy."""
        
        if not results:
            return {"count": 0}
        
        prices = [r.agreed_price for r in results if r.agreed_price]
        
        analysis = {
            "count": len(results),
            "avg_rounds": sum(r.total_rounds for r in results) / len(results),
            "avg_tokens": sum(r.total_tokens for r in results) / len(results)
        }
        
        if prices:
            analysis["price_statistics"] = {
                "count": len(prices),
                "mean": sum(prices) / len(prices),
                "median": sorted(prices)[len(prices)//2],
                "std": self._calculate_std(prices),
                "optimal_convergence_rate": sum(1 for p in prices if 60 <= p <= 70) / len(prices)
            }
        
        return analysis
    
    def _compare_turn_order_strategies(self, results: List[NegotiationResult]) -> Dict[str, Any]:
        """Create detailed comparison between turn order strategies."""
        
        strategies = set(r.turn_order_strategy for r in results)
        comparison = {}
        
        for strategy in strategies:
            strategy_results = [r for r in results if r.turn_order_strategy == strategy]
            comparison[strategy] = self._analyze_strategy_results(strategy_results)
        
        return comparison
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    async def _save_turn_order_results(self, results: List[NegotiationResult], analysis: Dict[str, Any]) -> None:
        """Save turn order experiment results."""
        
        try:
            # Convert results to dictionaries
            results_dicts = [asdict(result) for result in results]
            
            # Save main results
            saved_files = await self.data_exporter.save_results(
                results_dicts, 
                "turn_order_experiment",
                metadata={
                    "experiment_type": "turn_order_control",
                    "version": "0.6",
                    "research_focus": "literature_bias_vs_anchoring"
                }
            )
            
            # Save analysis
            analysis_path = await self.data_exporter.export_analysis(
                analysis, 
                f"turn_order_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            # Export comprehensive turn order analysis
            comprehensive_path = await self.data_exporter.export_turn_order_analysis(results_dicts)
            
            logger.info("Turn order experiment results saved successfully:")
            for format_name, path in saved_files.items():
                logger.info(f"  {format_name}: {path}")
            logger.info(f"  Analysis: {analysis_path}")
            logger.info(f"  Comprehensive: {comprehensive_path}")
            
        except Exception as e:
            logger.error(f"Error saving turn order results: {e}")
    
    async def shutdown(self):
        """Clean shutdown."""
        if self.model_manager:
            await self.model_manager.shutdown()


@click.command()
@click.option('--models', type=str, help='Comma-separated list of models to test (default: first 3)')
@click.option('--strategies', type=str, help='Comma-separated turn order strategies (default: buyer_first,supplier_first)')
@click.option('--patterns', type=str, help='Comma-separated reflection patterns (default: 00,11)')
@click.option('--replications', type=int, default=5, help='Replications per condition (default: 5)')
@click.option('--concurrent', type=int, default=1, help='Max concurrent negotiations (default: 1)')
@click.option('--dry-run', is_flag=True, help='Show experiment plan without running')
@click.option('--full-experiment', is_flag=True, help='Run full experiment with all models')
def main(models: Optional[str], strategies: Optional[str], patterns: Optional[str], 
         replications: int, concurrent: int, dry_run: bool, full_experiment: bool):
    """Run turn order controlled experiment to separate literature bias from anchoring effects."""
    
    async def run_experiment():
        runner = TurnOrderExperimentRunner(max_concurrent=concurrent)
        
        try:
            await runner.initialize()
            
            # Determine experiment scope
            if full_experiment:
                # Full experiment with all models
                models_list = runner.all_models
                strategies_list = runner.turn_order_strategies
                patterns_list = runner.reflection_patterns
                reps = 20
            else:
                # Default or custom subset
                models_list = models.split(',') if models else runner.all_models[:3]
                strategies_list = strategies.split(',') if strategies else runner.turn_order_strategies
                patterns_list = patterns.split(',') if patterns else ["00", "11"]  # No reflection vs full reflection
                reps = replications
            
            # Generate experiment plan
            experiment_plan = runner.generate_experiment_plan(
                models_subset=models_list,
                strategies_subset=strategies_list,
                patterns_subset=patterns_list,
                replications=reps
            )
            
            # Show cost and time estimates
            estimates = runner.estimate_experiment_cost_and_time(experiment_plan)
            
            if dry_run:
                print("\n" + "=" * 80)
                print("TURN ORDER EXPERIMENT PLAN (v0.6)")
                print("=" * 80)
                print(f"Research Question: Literature bias vs anchoring effects")
                print(f"Models: {len(models_list)} ({', '.join(models_list[:3])}{'...' if len(models_list) > 3 else ''})")
                print(f"Turn order strategies: {len(strategies_list)} ({', '.join(strategies_list)})")
                print(f"Reflection patterns: {len(patterns_list)} ({', '.join(patterns_list)})")
                print(f"Replications per condition: {reps}")
                print(f"Total negotiations: {len(experiment_plan) * reps:,}")
                print(f"Estimated cost: ${estimates['total_estimated_cost']:.2f}")
                print(f"Estimated time: {estimates['estimated_time_hours']:.1f} hours")
                print(f"Max concurrent: {concurrent}")
                print(f"\nKey Innovation: Identical prompts, only context assignment changes")
                print(f"🔍 DRY RUN - Experiment plan generated but not executed")
                return
            
            # Run experiment
            analysis = await runner.run_turn_order_experiment(experiment_plan)
            
            # Print results summary
            print("\n" + "=" * 80)
            print("TURN ORDER EXPERIMENT COMPLETE (v0.6)")
            print("=" * 80)
            
            summary = analysis["experiment_summary"]
            print(f"Total negotiations: {summary['total_negotiations']:,}")
            print(f"Successful: {summary['successful_negotiations']:,} ({summary['success_rate']*100:.1f}%)")
            print(f"Duration: {summary['completion_time_hours']:.1f} hours")
            
            # Research findings
            if "research_findings" in analysis:
                findings = analysis["research_findings"]
                if "bias_decomposition" in findings:
                    bias = findings["bias_decomposition"]
                    print(f"\n🔬 Research Findings:")
                    print(f"  Buyer advantage when suppliers go first: ${bias.get('buyer_advantage_when_supplier_goes_first', 0):.2f}")
                    print(f"  Direction: {bias.get('direction', 'unknown')}")
                    print(f"  Interpretation: {findings.get('interpretation', 'Analysis pending')}")
            
            print("\n✅ Turn order experiment complete!")
            print("📊 Detailed analysis saved to data/turn_order_analysis/")
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            raise
        
        finally:
            await runner.shutdown()
    
    asyncio.run(run_experiment())


if __name__ == '__main__':
    main()