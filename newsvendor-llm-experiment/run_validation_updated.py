#!/usr/bin/env python3
"""
run_validation_with_grok.py
Enhanced Validation Experiment Runner - Now with Grok support!
Tests all 11 models including Grok with smart pairing strategy and generous token limits
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
        logging.FileHandler('newsvendor_validation_with_grok.log'),
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
    print("📋 Make sure you've created the required files and updated unified_model_manager.py")
    sys.exit(1)


@dataclass
class ValidationConfig:
    """Configuration for validation run."""
    buyer_model: str
    supplier_model: str
    reflection_pattern: str  # "00", "01", "10", "11"
    max_rounds: int = 10
    timeout_seconds: int = 120  # Longer timeout for generous token limits


class EnhancedValidationRunnerWithGrok:
    """Runs comprehensive validation experiments with Grok and generous token limits."""
    
    def __init__(self):
        """Initialize enhanced validation runner with Grok support."""
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
        
        # All 11 models organized by tiers (now including Grok!)
        self.model_tiers = {
            'ultra': ["tinyllama:latest", "qwen2:1.5b"],
            'compact': ["gemma2:2b", "phi3:mini", "llama3.2:latest"],
            'mid': ["mistral:instruct", "qwen:7b"],
            'large': ["qwen3:latest"],
            'premium': ["claude-sonnet-4-remote", "o3-remote", "grok-remote"]  # Added Grok!
        }
        
        self.all_models = []
        for tier_models in self.model_tiers.values():
            self.all_models.extend(tier_models)
        
        logger.info("Initialized EnhancedValidationRunnerWithGrok - now with Grok support!")
        logger.info(f"Total models: {len(self.all_models)} (including Grok)")
    
    async def initialize(self) -> None:
        """Initialize the unified model manager."""
        logger.info("Initializing unified model manager with Grok support...")
        self.model_manager = create_unified_model_manager()
        
        # Validate all models are available
        validation_results = await self.model_manager.validate_all_models()
        
        working_models = [model for model, result in validation_results.items() if result["success"]]
        failed_models = [model for model, result in validation_results.items() if not result["success"]]
        
        logger.info(f"✅ Working models: {len(working_models)}")
        for model in working_models:
            cost = validation_results[model].get("cost", 0)
            if 'grok' in model.lower():
                logger.info(f"   🤖 {model} (GROK) - ${cost:.6f}")
            elif cost > 0:
                logger.info(f"   🌐 {model} (${cost:.6f})")
            else:
                logger.info(f"   💻 {model}")
        
        if failed_models:
            logger.warning(f"❌ Failed models: {failed_models}")
            
            # Update model lists to exclude failed models
            self._remove_failed_models(failed_models)
        
        # Special check for Grok
        if 'grok-remote' in working_models:
            logger.info("🎉 Grok is ready for validation!")
        else:
            logger.warning("⚠️  Grok not available - check AZURE_GROK3_MINI_KEY in .env")
        
        logger.info("Model manager initialized successfully")
    
    def _remove_failed_models(self, failed_models: List[str]) -> None:
        """Remove failed models from tier lists and all_models."""
        for tier, models in self.model_tiers.items():
            self.model_tiers[tier] = [m for m in models if m not in failed_models]
        
        self.all_models = [m for m in self.all_models if m not in failed_models]
        
        logger.info(f"Updated model lists to exclude {len(failed_models)} failed models")
    
    def _generate_validation_pairs(self) -> List[ValidationConfig]:
        """Generate smart validation pairs including Grok combinations."""
        validation_configs = []
        
        logger.info("Generating validation pairs with Grok support...")
        
        # Strategy 1: Each model as buyer paired with a model from different tier
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
                
                # Log Grok pairings specially
                if 'grok' in buyer_model.lower() or 'grok' in supplier_model.lower():
                    logger.info(f"🤖 Grok pairing: {buyer_model} vs {supplier_model}")
                else:
                    logger.debug(f"Paired {buyer_model} ({buyer_tier}) with {supplier_model}")
            else:
                logger.warning(f"Could not find suitable supplier for {buyer_model}")
        
        # Strategy 2: Special Grok vs other premium models tests
        if 'grok-remote' in self.all_models:
            premium_models = [m for m in self.model_tiers['premium'] if m != 'grok-remote']
            
            for other_premium in premium_models:
                # Grok as buyer vs other premium as supplier
                grok_vs_premium = ValidationConfig(
                    buyer_model='grok-remote',
                    supplier_model=other_premium,
                    reflection_pattern="11"
                )
                validation_configs.append(grok_vs_premium)
                
                # Other premium as buyer vs Grok as supplier  
                premium_vs_grok = ValidationConfig(
                    buyer_model=other_premium,
                    supplier_model='grok-remote',
                    reflection_pattern="11"
                )
                validation_configs.append(premium_vs_grok)
                
                logger.info(f"🏆 Premium battle: {other_premium} ⚔️  grok-remote")
        
        # Strategy 3: Add some intra-tier pairs for diversity
        intra_tier_pairs = self._generate_intra_tier_pairs()
        validation_configs.extend(intra_tier_pairs)
        
        logger.info(f"Generated {len(validation_configs)} validation pairs:")
        logger.info(f"  - {len(self.all_models)} primary buyer tests")
        logger.info(f"  - Special Grok premium battles")
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
        
        # If buyer is premium (including Grok), prefer local suppliers for cost efficiency
        if buyer_tier == 'premium':
            for tier in ['ultra', 'compact', 'mid', 'large']:
                if self.model_tiers[tier]:
                    return self.model_tiers[tier][0]  # First available
        
        # If buyer is local, try different local tiers first, then premium
        tier_preference = {
            'ultra': ['compact', 'mid', 'large', 'premium', 'ultra'],
            'compact': ['ultra', 'mid', 'large', 'premium', 'compact'],
            'mid': ['compact', 'ultra', 'large', 'premium', 'mid'],
            'large': ['mid', 'compact', 'ultra', 'premium', 'large']
        }
        
        preferences = tier_preference.get(buyer_tier, ['ultra', 'compact', 'mid', 'large', 'premium'])
        
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
        """Run comprehensive validation with Grok and generous token limits."""
        logger.info("=== ENHANCED VALIDATION PHASE WITH GROK ===")
        logger.info("Testing all models including Grok with generous token limits")
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
        grok_negotiations = 0
        
        for i, config in enumerate(validation_configs, 1):
            is_grok_negotiation = 'grok' in config.buyer_model.lower() or 'grok' in config.supplier_model.lower()
            if is_grok_negotiation:
                grok_negotiations += 1
                logger.info(f"\n[{i}/{len(validation_configs)}] 🤖 GROK TEST: {config.buyer_model} vs {config.supplier_model} ({config.reflection_pattern})")
            else:
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
                success_emoji = "🤖✅" if is_grok_negotiation else "✅"
                logger.info(f"{success_emoji} SUCCESS: Agreed on ${result.agreed_price} in {result.total_rounds} rounds ({nego_time:.1f}s)")
                if nego_cost > 0:
                    logger.info(f"   💰 Cost: ${nego_cost:.6f}")
            else:
                fail_emoji = "🤖❌" if is_grok_negotiation else "❌"
                logger.info(f"{fail_emoji} FAILED: {result.termination_type.value} ({nego_time:.1f}s)")
            
            # Brief pause between negotiations
            await asyncio.sleep(0.5)
        
        # Analyze validation results
        analysis = self._analyze_validation_results(validation_results, total_cost)
        
        # Add Grok-specific analysis
        analysis['grok_analysis'] = self._analyze_grok_performance(validation_results)
        
        # Save validation results
        await self._save_validation_results(validation_results, analysis)
        
        logger.info(f"\n🤖 Grok participated in {grok_negotiations} negotiations")
        return analysis
    
    def _analyze_grok_performance(self, results: List[NegotiationResult]) -> Dict[str, Any]:
        """Analyze Grok's specific performance."""
        
        grok_results = [r for r in results if 'grok' in r.buyer_model.lower() or 'grok' in r.supplier_model.lower()]
        
        if not grok_results:
            return {"error": "No Grok negotiations found"}
        
        grok_as_buyer = [r for r in grok_results if 'grok' in r.buyer_model.lower()]
        grok_as_supplier = [r for r in grok_results if 'grok' in r.supplier_model.lower()]
        
        successful_grok = [r for r in grok_results if r.completed and r.agreed_price]
        
        analysis = {
            "total_grok_negotiations": len(grok_results),
            "grok_as_buyer": len(grok_as_buyer),
            "grok_as_supplier": len(grok_as_supplier),
            "grok_success_rate": len(successful_grok) / len(grok_results) if grok_results else 0,
            "grok_vs_other_premium": {},
            "grok_cost_analysis": {},
            "grok_efficiency": {}
        }
        
        if successful_grok:
            # Price analysis
            grok_prices = [r.agreed_price for r in successful_grok]
            analysis["grok_price_performance"] = {
                "avg_price": sum(grok_prices) / len(grok_prices),
                "price_range": f"${min(grok_prices)} - ${max(grok_prices)}",
                "optimal_convergence": sum(1 for p in grok_prices if 60 <= p <= 70) / len(grok_prices)
            }
            
            # Efficiency analysis
            grok_tokens = [r.total_tokens for r in successful_grok]
            grok_rounds = [r.total_rounds for r in successful_grok]
            grok_costs = [r.metadata.get('total_cost', 0) for r in successful_grok]
            
            analysis["grok_efficiency"] = {
                "avg_tokens": sum(grok_tokens) / len(grok_tokens),
                "avg_rounds": sum(grok_rounds) / len(grok_rounds),
                "avg_cost_per_negotiation": sum(grok_costs) / len(grok_costs),
                "total_grok_cost": sum(grok_costs)
            }
            
            # Vs other premium models
            premium_opponents = [r for r in grok_results if 
                               ('claude' in r.buyer_model or 'claude' in r.supplier_model) or
                               ('o3' in r.buyer_model or 'o3' in r.supplier_model)]
            
            if premium_opponents:
                premium_successful = [r for r in premium_opponents if r.completed]
                analysis["grok_vs_other_premium"] = {
                    "total_premium_battles": len(premium_opponents),
                    "premium_battle_success_rate": len(premium_successful) / len(premium_opponents),
                    "avg_premium_battle_cost": sum(r.metadata.get('total_cost', 0) for r in premium_opponents) / len(premium_opponents)
                }
        
        return analysis
    
    # Rest of the methods remain the same as the original validation runner
    # Just copied from the original file...
    
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
            }
        }
        
        # Add basic analysis (copied from original)
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
        
        return analysis
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    async def _save_validation_results(self, results: List[NegotiationResult], analysis: Dict[str, Any]):
        """Save comprehensive validation results including Grok analysis."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('./validation_results_with_grok')
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results with Grok metadata
        results_data = {
            "metadata": {
                "phase": "enhanced_validation_with_grok",
                "timestamp": timestamp,
                "strategy": "smart_pairing_generous_tokens_plus_grok",
                "total_models": len(self.all_models),
                "includes_grok": True,
                "game_config": self.game_config,
                "token_limits": {
                    "local_models": "4000 tokens",
                    "claude": "5000 tokens", 
                    "o3": "8000 tokens",
                    "grok": "6000 tokens"
                }
            },
            "model_tiers": self.model_tiers,
            "analysis": analysis,
            "results": [asdict(result) for result in results]
        }
        
        results_file = output_dir / f"validation_results_with_grok_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save Grok-specific summary
        if 'grok_analysis' in analysis:
            grok_summary_file = output_dir / f"grok_performance_summary_{timestamp}.json"
            with open(grok_summary_file, 'w') as f:
                json.dump(analysis['grok_analysis'], f, indent=2, default=str)
        
        logger.info(f"Enhanced validation results with Grok saved to:")
        logger.info(f"  Main results: {results_file}")
        if 'grok_analysis' in analysis:
            logger.info(f"  Grok analysis: {grok_summary_file}")
    
    async def shutdown(self):
        """Clean shutdown."""
        if self.model_manager:
            await self.model_manager.shutdown()


@click.command()
@click.option('--models', type=str, help='Comma-separated list of models to test (default: all 11)')
@click.option('--pattern', type=str, default='11', help='Primary reflection pattern to test (default: 11)')
@click.option('--output', type=click.Path(), help='Output directory for results')
@click.option('--quick', is_flag=True, help='Quick validation with fewer pairs')
@click.option('--grok-only', is_flag=True, help='Test only Grok-related negotiations')
def main(models: Optional[str], pattern: str, output: Optional[str], quick: bool, grok_only: bool):
    """Run enhanced validation phase for newsvendor experiment with Grok support."""
    
    async def run_validation():
        runner = EnhancedValidationRunnerWithGrok()
        
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
            
            # Modify strategy for special modes
            if grok_only:
                logger.info("🤖 Grok-only mode: testing only Grok-related negotiations")
                original_method = runner._generate_validation_pairs
                def grok_only_pairs():
                    configs = []
                    if 'grok-remote' in runner.all_models:
                        # Grok vs each other model
                        for other_model in runner.all_models:
                            if other_model != 'grok-remote':
                                # Grok as buyer
                                configs.append(ValidationConfig(
                                    buyer_model='grok-remote',
                                    supplier_model=other_model,
                                    reflection_pattern=pattern
                                ))
                                # Grok as supplier
                                configs.append(ValidationConfig(
                                    buyer_model=other_model,
                                    supplier_model='grok-remote',
                                    reflection_pattern=pattern
                                ))
                    return configs
                runner._generate_validation_pairs = grok_only_pairs
                
            elif quick:
                logger.info("🚀 Quick validation mode: reducing test pairs")
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
            print("ENHANCED VALIDATION PHASE WITH GROK COMPLETE")
            print("=" * 80)
            
            summary = analysis["validation_summary"]
            print(f"Total negotiations: {summary['total_negotiations']}")
            print(f"Successful: {summary['successful_negotiations']} ({summary['success_rate']*100:.1f}%)")
            print(f"Duration: {summary['completion_time']:.1f} seconds")
            
            if "cost_analysis" in analysis:
                cost = analysis.get("cost_analysis", {})
                total_cost = cost.get('total_validation_cost', summary.get('total_cost', 0))
                print(f"Total cost: ${total_cost:.6f}")
            
            if "price_statistics" in analysis:
                prices = analysis["price_statistics"]
                print(f"Average price: ${prices['mean_price']:.2f}")
                print(f"Optimal convergence: {prices['optimal_convergence']*100:.1f}%")
            
            # Grok-specific results
            if "grok_analysis" in analysis:
                grok_stats = analysis["grok_analysis"]
                print(f"\n🤖 GROK PERFORMANCE:")
                if 'error' not in grok_stats:
                    print(f"  Total Grok negotiations: {grok_stats['total_grok_negotiations']}")
                    print(f"  Grok success rate: {grok_stats['grok_success_rate']*100:.1f}%")
                    
                    if 'grok_efficiency' in grok_stats:
                        eff = grok_stats['grok_efficiency']
                        print(f"  Grok avg cost per negotiation: ${eff.get('avg_cost_per_negotiation', 0):.6f}")
                        print(f"  Total Grok cost: ${eff.get('total_grok_cost', 0):.6f}")
                    
                    if 'grok_price_performance' in grok_stats:
                        price_perf = grok_stats['grok_price_performance']
                        print(f"  Grok avg price: ${price_perf['avg_price']:.2f}")
                        print(f"  Grok optimal convergence: {price_perf['optimal_convergence']*100:.1f}%")
                else:
                    print(f"  ❌ {grok_stats['error']}")
            
            print(f"\nTOKEN LIMITS USED:")
            print(f"  Local models: 4,000 tokens (2x increase)")
            print(f"  Claude: 5,000 tokens")
            print(f"  O3: 8,000 tokens (for reasoning)")
            print(f"  🤖 Grok: 6,000 tokens (NEW!)")
            
            if grok_only:
                print("\n🤖 Grok-only validation complete! Ready for full Grok experiments.")
            else:
                print("\n✅ Enhanced validation with Grok complete! Ready for full experiment.")
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            raise
        
        finally:
            await runner.shutdown()
    
    asyncio.run(run_validation())


if __name__ == '__main__':
    main()