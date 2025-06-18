#!/usr/bin/env python3
"""
run_validation_updated.py
Enhanced Validation Experiment Runner for Updated Newsvendor Study
Tests all 10 models with smart pairing strategy, generous token limits, and enhanced price extraction

Place this file in the ROOT directory of your project
"""

import asyncio
import json
import logging
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import click

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('newsvendor_validation_enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our modules with proper error handling
try:
    from core.unified_model_manager import create_unified_model_manager
    from agents.standardized_agents import StandardizedBuyerAgent, StandardizedSupplierAgent
    from core.conversation_tracker import ConversationTracker, NegotiationResult
    from parsing.acceptance_detector import TerminationType
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📋 Make sure you've created the required files:")
    print("   - src/core/unified_model_manager.py")
    print("   - src/agents/standardized_agents.py")
    print("\nAlso check that your src/ modules don't have circular imports.")
    sys.exit(1)


@dataclass
class ValidationConfig:
    """Configuration for validation run."""
    buyer_model: str
    supplier_model: str
    reflection_pattern: str  # "00", "01", "10", "11"
    max_rounds: int = 10
    timeout_seconds: int = 120  # Longer timeout for generous token limits


class EnhancedValidationRunner:
    """Runs comprehensive validation experiments with generous token limits."""
    
    def __init__(self):
        """Initialize enhanced validation runner."""
        self.model_manager = None
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
        
        # All 10 models organized by tiers
        self.model_tiers = {
            'ultra': ["tinyllama:latest", "qwen2:1.5b"],
            'compact': ["gemma2:2b", "phi3:mini", "llama3.2:latest"],
            'mid': ["mistral:instruct", "qwen:7b"],
            'large': ["qwen3:latest"],
            'remote': ["claude-sonnet-4-remote", "o3-remote"]
        }
        
        self.all_models = []
        for tier_models in self.model_tiers.values():
            self.all_models.extend(tier_models)
        
        logger.info("Initialized EnhancedValidationRunner with generous token limits")
    
    async def initialize(self) -> None:
        """Initialize the model manager."""
        logger.info("Initializing unified model manager with generous token limits...")
        self.model_manager = create_unified_model_manager()
        
        # Validate all models are available
        validation_results = await self.model_manager.validate_all_models()
        
        working_models = [model for model, result in validation_results.items() if result["success"]]
        failed_models = [model for model, result in validation_results.items() if not result["success"]]
        
        logger.info(f"✅ Working models: {len(working_models)}")
        for model in working_models:
            cost = validation_results[model].get("cost", 0)
            if cost > 0:
                logger.info(f"   🌐 {model} (${cost:.6f})")
            else:
                logger.info(f"   💻 {model}")
        
        if failed_models:
            logger.warning(f"❌ Failed models: {failed_models}")
            
            # Update model lists to exclude failed models
            self._remove_failed_models(failed_models)
        
        logger.info("Model manager initialized successfully")
    
    def _remove_failed_models(self, failed_models: List[str]) -> None:
        """Remove failed models from tier lists and all_models."""
        for tier, models in self.model_tiers.items():
            self.model_tiers[tier] = [m for m in models if m not in failed_models]
        
        self.all_models = [m for m in self.all_models if m not in failed_models]
        
        logger.info(f"Updated model lists to exclude {len(failed_models)} failed models")
    
    def _generate_validation_pairs(self) -> List[ValidationConfig]:
        """Generate smart validation pairs without relying on a single base model."""
        validation_configs = []
        
        # Strategy 1: Each model as buyer paired with a model from different tier
        logger.info("Generating validation pairs using tier-based strategy...")
        
        for buyer_model in self.all_models:
            buyer_tier = self._get_model_tier(buyer_model)
            
            # Find a suitable supplier from a different tier (if possible)
            supplier_model = self._find_suitable_supplier(buyer_model, buyer_tier)
            
            if supplier_model:
                config = ValidationConfig(
                    buyer_model=buyer_model,
                    supplier_model=supplier_model,
                    reflection_pattern="11"  # Both agents reflect for validation
                )
                validation_configs.append(config)
                logger.debug(f"Paired {buyer_model} ({buyer_tier}) with {supplier_model}")
            else:
                logger.warning(f"Could not find suitable supplier for {buyer_model}")
        
        # Strategy 2: Add some reverse pairs (each model as supplier)
        # This ensures we test both buyer and supplier capabilities
        reverse_pairs = []
        for original_config in validation_configs[:5]:  # Test first 5 as suppliers too
            reverse_config = ValidationConfig(
                buyer_model=original_config.supplier_model,
                supplier_model=original_config.buyer_model,
                reflection_pattern="11"
            )
            reverse_pairs.append(reverse_config)
        
        validation_configs.extend(reverse_pairs)
        
        # Strategy 3: Add some intra-tier pairs for diversity
        intra_tier_pairs = self._generate_intra_tier_pairs()
        validation_configs.extend(intra_tier_pairs)
        
        logger.info(f"Generated {len(validation_configs)} validation pairs:")
        logger.info(f"  - {len(self.all_models)} primary buyer tests")
        logger.info(f"  - {len(reverse_pairs)} reverse supplier tests")
        logger.info(f"  - {len(intra_tier_pairs)} intra-tier diversity tests")
        
        return validation_configs
    
    def _get_model_tier(self, model_name: str) -> str:
        """Get the tier of a model."""
        for tier, models in self.model_tiers.items():
            if model_name in models:
                return tier
        return 'unknown'
    
    def _find_suitable_supplier(self, buyer_model: str, buyer_tier: str) -> Optional[str]:
        """Find a suitable supplier model for the given buyer."""
        
        # If buyer is remote, prefer local suppliers
        if buyer_tier == 'remote':
            for tier in ['ultra', 'compact', 'mid', 'large']:
                if self.model_tiers[tier]:
                    return self.model_tiers[tier][0]  # First available
        
        # If buyer is local, try different local tiers first
        tier_preference = {
            'ultra': ['compact', 'mid', 'large', 'ultra'],
            'compact': ['ultra', 'mid', 'large', 'compact'],
            'mid': ['compact', 'ultra', 'large', 'mid'],
            'large': ['mid', 'compact', 'ultra', 'large']
        }
        
        preferences = tier_preference.get(buyer_tier, ['ultra', 'compact', 'mid', 'large'])
        
        for preferred_tier in preferences:
            available_models = [m for m in self.model_tiers[preferred_tier] if m != buyer_model]
            if available_models:
                return available_models[0]
        
        # Last resort: use any available model except the buyer itself
        for model in self.all_models:
            if model != buyer_model:
                return model
        
        return None
    
    def _generate_intra_tier_pairs(self) -> List[ValidationConfig]:
        """Generate some intra-tier pairs for diversity testing."""
        intra_pairs = []
        
        # Test within tiers that have multiple models
        for tier, models in self.model_tiers.items():
            if len(models) >= 2:
                # Pair first with second in the tier
                config = ValidationConfig(
                    buyer_model=models[0],
                    supplier_model=models[1],
                    reflection_pattern="01"  # Mix reflection patterns
                )
                intra_pairs.append(config)
                
                # If tier has 3+ models, add another pair
                if len(models) >= 3:
                    config2 = ValidationConfig(
                        buyer_model=models[1],
                        supplier_model=models[2],
                        reflection_pattern="10"  # Different reflection pattern
                    )
                    intra_pairs.append(config2)
        
        return intra_pairs
    
    async def run_validation_phase(self) -> Dict[str, Any]:
        """Run comprehensive validation with generous token limits."""
        logger.info("=== ENHANCED VALIDATION PHASE ===")
        logger.info("Testing all models with generous token limits and smart pairing")
        self.start_time = time.time()
        
        # Generate validation configurations
        validation_configs = self._generate_validation_pairs()
        
        if not validation_configs:
            logger.error("No validation configurations generated!")
            return {"error": "No validation pairs could be generated"}
        
        logger.info(f"Running {len(validation_configs)} validation negotiations...")
        
        # Run validation negotiations with progress tracking
        validation_results = []
        total_cost = 0.0
        
        for i, config in enumerate(validation_configs, 1):
            logger.info(f"\n[{i}/{len(validation_configs)}] Testing {config.buyer_model} vs {config.supplier_model} ({config.reflection_pattern})")
            
            start_nego_time = time.time()
            result = await self._run_single_negotiation(config, f"validation_{i:03d}")
            nego_time = time.time() - start_nego_time
            
            validation_results.append(result)
            
            # Track cost
            nego_cost = result.metadata.get('total_cost', 0.0)
            total_cost += nego_cost
            
            # Show immediate results
            if result.completed:
                logger.info(f"✅ SUCCESS: Agreed on ${result.agreed_price} in {result.total_rounds} rounds ({nego_time:.1f}s)")
                if nego_cost > 0:
                    logger.info(f"   Cost: ${nego_cost:.6f}")
            else:
                logger.info(f"❌ FAILED: {result.termination_type.value} ({nego_time:.1f}s)")
            
            # Brief pause between negotiations
            await asyncio.sleep(0.5)
        
        # Analyze validation results
        analysis = self._analyze_validation_results(validation_results, total_cost)
        
        # Save validation results
        await self._save_validation_results(validation_results, analysis)
        
        return analysis
    
    async def _run_single_negotiation(self, config: ValidationConfig, negotiation_id: str) -> NegotiationResult:
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
                    logger.warning(f"Negotiation {tracker.negotiation_id} timed out after {timeout_seconds}s")
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
                    logger.error(f"Agent {agent_role} failed to generate response: {response.error}")
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
                    logger.error(f"Failed to add turn for {agent_role}")
                    tracker.force_termination("turn_error")
                    break
                
                # Show progress (debug level)
                logger.debug(f"  Round {tracker.round_number}: {agent_role.capitalize()} -> '{response.text[:50]}...'")
                
                # Check for early termination
                if tracker.completed:
                    break
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
            
            # Finalize negotiation if not already completed
            if not tracker.completed and tracker.round_number >= max_rounds:
                tracker.force_termination("max_rounds_reached")
            
            # Get final result and add cost information
            result = tracker.get_final_result()
            result.metadata['total_cost'] = total_cost
            
            return result
            
        except Exception as e:
            logger.error(f"Error during negotiation {tracker.negotiation_id}: {e}")
            tracker.force_termination(f"error: {str(e)}")
            result = tracker.get_final_result()
            result.metadata['total_cost'] = total_cost
            return result
    
    def _analyze_validation_results(self, results: List[NegotiationResult], total_cost: float) -> Dict[str, Any]:
        """Analyze validation results with comprehensive metrics."""
        
        total = len(results)
        successful = [r for r in results if r.completed and r.agreed_price]
        
        analysis = {
            "validation_summary": {
                "total_negotiations": total,
                "successful_negotiations": len(successful),
                "success_rate": len(successful) / total if total > 0 else 0,
                "completion_time": time.time() - self.start_time if self.start_time else 0,
                "total_cost": total_cost
            },
            "model_performance": {},
            "tier_analysis": {},
            "pairing_analysis": {},
            "reflection_analysis": {},
            "cost_analysis": {}
        }
        
        # Basic price and efficiency stats
        if successful:
            prices = [r.agreed_price for r in successful]
            analysis["price_statistics"] = {
                "mean_price": sum(prices) / len(prices),
                "median_price": sorted(prices)[len(prices)//2],
                "min_price": min(prices),
                "max_price": max(prices),
                "optimal_convergence": sum(1 for p in prices if 60 <= p <= 70) / len(prices),
                "price_std": self._calculate_std(prices)
            }
            
            tokens = [r.total_tokens for r in successful]
            rounds = [r.total_rounds for r in successful]
            
            analysis["efficiency_statistics"] = {
                "avg_tokens": sum(tokens) / len(tokens),
                "avg_rounds": sum(rounds) / len(rounds),
                "tokens_per_round": sum(tokens) / sum(rounds) if sum(rounds) > 0 else 0
            }
        
        # Per-model analysis (both as buyer and supplier)
        model_stats = {}
        for result in results:
            for role, model in [("buyer", result.buyer_model), ("supplier", result.supplier_model)]:
                if model not in model_stats:
                    model_stats[model] = {
                        "total_negotiations": 0,
                        "as_buyer": {"count": 0, "successes": 0, "prices": [], "tokens": []},
                        "as_supplier": {"count": 0, "successes": 0, "prices": [], "tokens": []},
                        "tier": self._get_model_tier(model)
                    }
                
                model_stats[model]["total_negotiations"] += 1
                role_stats = model_stats[model][f"as_{role}"]
                role_stats["count"] += 1
                
                if result.completed:
                    role_stats["successes"] += 1
                    if result.agreed_price:
                        role_stats["prices"].append(result.agreed_price)
                    role_stats["tokens"].append(result.total_tokens)
        
        # Calculate model performance metrics
        for model, stats in model_stats.items():
            for role in ["as_buyer", "as_supplier"]:
                role_data = stats[role]
                if role_data["count"] > 0:
                    role_data["success_rate"] = role_data["successes"] / role_data["count"]
                    if role_data["prices"]:
                        role_data["avg_price"] = sum(role_data["prices"]) / len(role_data["prices"])
                    if role_data["tokens"]:
                        role_data["avg_tokens"] = sum(role_data["tokens"]) / len(role_data["tokens"])
                else:
                    # Add missing success_rate for models not tested in this role
                    role_data["success_rate"] = 0.0
        
        analysis["model_performance"] = model_stats
        
        # Tier-based analysis
        tier_stats = {}
        for tier, models in self.model_tiers.items():
            if not models:  # Skip empty tiers
                continue
                
            tier_results = [r for r in results if r.buyer_model in models or r.supplier_model in models]
            tier_successful = [r for r in tier_results if r.completed]
            
            tier_stats[tier] = {
                "models": models,
                "negotiations": len(tier_results),
                "successes": len(tier_successful),
                "success_rate": len(tier_successful) / len(tier_results) if tier_results else 0,
                "avg_cost": sum(r.metadata.get('total_cost', 0) for r in tier_results) / len(tier_results) if tier_results else 0
            }
        
        analysis["tier_analysis"] = tier_stats
        
        # Cost analysis
        remote_cost = sum(r.metadata.get('total_cost', 0) for r in results 
                         if 'remote' in r.buyer_model or 'remote' in r.supplier_model)
        
        analysis["cost_analysis"] = {
            "total_validation_cost": total_cost,
            "remote_model_cost": remote_cost,
            "local_model_cost": total_cost - remote_cost,
            "avg_cost_per_negotiation": total_cost / total if total > 0 else 0,
            "cost_per_success": total_cost / len(successful) if successful else 0
        }
        
        return analysis
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    async def _save_validation_results(self, results: List[NegotiationResult], analysis: Dict[str, Any]):
        """Save comprehensive validation results to files."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('./validation_results_enhanced')
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_data = {
            "metadata": {
                "phase": "enhanced_validation",
                "timestamp": timestamp,
                "strategy": "smart_pairing_generous_tokens",
                "total_models": len(self.all_models),
                "game_config": self.game_config,
                "token_limits": {
                    "local_models": "4000 tokens",
                    "claude": "5000 tokens", 
                    "o3": "8000 tokens"
                },
                "enhanced_features": [
                    "generous_token_limits",
                    "smart_tier_pairing",
                    "comprehensive_analysis"
                ]
            },
            "model_tiers": self.model_tiers,
            "analysis": analysis,
            "results": [asdict(result) for result in results]
        }
        
        results_file = output_dir / f"validation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save conversation transcripts
        transcripts_file = output_dir / f"validation_conversations_{timestamp}.txt"
        with open(transcripts_file, 'w') as f:
            f.write("NEWSVENDOR EXPERIMENT - ENHANCED VALIDATION CONVERSATIONS\n")
            f.write("=" * 80 + "\n\n")
            
            for result in results:
                buyer_tier = self._get_model_tier(result.buyer_model)
                supplier_tier = self._get_model_tier(result.supplier_model)
                
                f.write(f"NEGOTIATION: {result.negotiation_id}\n")
                f.write(f"BUYER: {result.buyer_model} ({buyer_tier})\n")
                f.write(f"SUPPLIER: {result.supplier_model} ({supplier_tier})\n")
                f.write(f"REFLECTION: {result.reflection_pattern}\n")
                f.write(f"RESULT: {'SUCCESS' if result.completed else 'FAILED'}\n")
                f.write(f"PRICE: ${result.agreed_price}\n")
                f.write(f"ROUNDS: {result.total_rounds}\n")
                f.write(f"TOKENS: {result.total_tokens}\n")
                f.write(f"COST: ${result.metadata.get('total_cost', 0):.6f}\n")
                f.write("-" * 80 + "\n")
                
                for turn in result.turns:
                    speaker = turn.speaker.upper()
                    message = turn.message
                    price = f" (${turn.price})" if turn.price else ""
                    f.write(f"Round {turn.round_number} - {speaker}{price}:\n")
                    f.write(f"  {message}\n\n")
                
                f.write("=" * 80 + "\n\n")
        
        # Save summary report
        summary_file = output_dir / f"validation_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("ENHANCED VALIDATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            summary = analysis["validation_summary"]
            f.write(f"Total negotiations: {summary['total_negotiations']}\n")
            f.write(f"Successful: {summary['successful_negotiations']} ({summary['success_rate']*100:.1f}%)\n")
            f.write(f"Duration: {summary['completion_time']:.1f} seconds\n")
            f.write(f"Total cost: ${summary['total_cost']:.6f}\n\n")
            
            # Token limits info
            f.write("TOKEN LIMITS:\n")
            f.write("-" * 30 + "\n")
            f.write("Local models: 4,000 tokens (2x increase)\n")
            f.write("Claude: 5,000 tokens\n")
            f.write("O3: 8,000 tokens (for reasoning)\n\n")
            
            # Model performance summary
            f.write("MODEL PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            model_perf = analysis["model_performance"]
            for model, stats in model_perf.items():
                buyer_rate = stats["as_buyer"]["success_rate"] if stats["as_buyer"]["count"] > 0 else 0
                supplier_rate = stats["as_supplier"]["success_rate"] if stats["as_supplier"]["count"] > 0 else 0
                f.write(f"{model} ({stats['tier']}):\n")
                f.write(f"  As buyer: {buyer_rate*100:.1f}% ({stats['as_buyer']['count']} tests)\n")
                f.write(f"  As supplier: {supplier_rate*100:.1f}% ({stats['as_supplier']['count']} tests)\n\n")
            
            # Tier analysis
            f.write("TIER ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            tier_analysis = analysis["tier_analysis"]
            for tier, stats in tier_analysis.items():
                f.write(f"{tier.upper()} tier:\n")
                f.write(f"  Models: {len(stats['models'])}\n")
                f.write(f"  Success rate: {stats['success_rate']*100:.1f}%\n")
                f.write(f"  Avg cost: ${stats['avg_cost']:.6f}\n\n")
        
        logger.info(f"Enhanced validation results saved to:")
        logger.info(f"  Results: {results_file}")
        logger.info(f"  Conversations: {transcripts_file}")
        logger.info(f"  Summary: {summary_file}")
    
    async def shutdown(self):
        """Clean shutdown."""
        if self.model_manager:
            await self.model_manager.shutdown()


@click.command()
@click.option('--models', type=str, help='Comma-separated list of models to test (default: all 10)')
@click.option('--pattern', type=str, default='11', help='Primary reflection pattern to test (default: 11)')
@click.option('--output', type=click.Path(), help='Output directory for results')
@click.option('--quick', is_flag=True, help='Quick validation with fewer pairs')
def main(models: Optional[str], pattern: str, output: Optional[str], quick: bool):
    """Run enhanced validation phase for newsvendor experiment with generous token limits."""
    
    async def run_validation():
        runner = EnhancedValidationRunner()
        
        # Override models if specified
        if models:
            specified_models = [model.strip() for model in models.split(',')]
            # Update all tier lists to only include specified models
            for tier in runner.model_tiers:
                runner.model_tiers[tier] = [m for m in runner.model_tiers[tier] if m in specified_models]
            runner.all_models = [m for m in runner.all_models if m in specified_models]
        
        # Override output directory if specified
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
        
        try:
            await runner.initialize()
            
            # Modify strategy for quick mode
            if quick:
                logger.info("🚀 Quick validation mode: reducing test pairs")
                # Keep only primary strategy (each model as buyer once)
                original_method = runner._generate_validation_pairs
                def quick_pairs():
                    configs = []
                    for buyer_model in runner.all_models:
                        buyer_tier = runner._get_model_tier(buyer_model)
                        supplier_model = runner._find_suitable_supplier(buyer_model, buyer_tier)
                        if supplier_model:
                            configs.append(ValidationConfig(
                                buyer_model=buyer_model,
                                supplier_model=supplier_model,
                                reflection_pattern=pattern
                            ))
                    return configs
                runner._generate_validation_pairs = quick_pairs
            
            analysis = await runner.run_validation_phase()
            
            # Print comprehensive summary
            print("\n" + "=" * 80)
            print("ENHANCED VALIDATION PHASE COMPLETE")
            print("=" * 80)
            
            summary = analysis["validation_summary"]
            print(f"Total negotiations: {summary['total_negotiations']}")
            print(f"Successful: {summary['successful_negotiations']} ({summary['success_rate']*100:.1f}%)")
            print(f"Duration: {summary['completion_time']:.1f} seconds")
            
            if "cost_analysis" in analysis:
                cost = analysis["cost_analysis"]
                print(f"Total cost: ${cost['total_validation_cost']:.6f}")
                print(f"Remote model cost: ${cost['remote_model_cost']:.6f}")
            
            if "price_statistics" in analysis:
                prices = analysis["price_statistics"]
                print(f"Average price: ${prices['mean_price']:.2f}")
                print(f"Optimal convergence: {prices['optimal_convergence']*100:.1f}%")
            
            # Model performance highlights
            if "model_performance" in analysis:
                print(f"\nMODEL PERFORMANCE HIGHLIGHTS:")
                model_perf = analysis["model_performance"]
                
                # Find best overall model (handle missing success_rate)
                def get_combined_success_rate(model_stats):
                    buyer_rate = model_stats[1]['as_buyer'].get('success_rate', 0)
                    supplier_rate = model_stats[1]['as_supplier'].get('success_rate', 0)
                    return (buyer_rate + supplier_rate) / 2
                
                if model_perf:  # Check if there are any models
                    best_overall = max(model_perf.items(), key=get_combined_success_rate)
                    print(f"Best overall: {best_overall[0]} ({runner._get_model_tier(best_overall[0])} tier)")
                
                # Show tier summary
                if "tier_analysis" in analysis:
                    print(f"\nTIER PERFORMANCE:")
                    for tier, stats in analysis["tier_analysis"].items():
                        print(f"  {tier.upper()}: {stats['success_rate']*100:.1f}% success rate ({len(stats['models'])} models)")
            
            print(f"\nTOKEN LIMITS USED:")
            print(f"  Local models: 4,000 tokens (2x increase)")
            print(f"  Claude: 5,000 tokens")
            print(f"  O3: 8,000 tokens (for reasoning)")
            
            print("\n✅ Enhanced validation complete! Ready for full experiment.")
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            raise
        
        finally:
            await runner.shutdown()
    
    asyncio.run(run_validation())


if __name__ == '__main__':
    main()