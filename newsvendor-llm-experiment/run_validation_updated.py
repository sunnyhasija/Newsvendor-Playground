#!/usr/bin/env python3
"""
run_validation_updated.py
Validation Experiment Runner for Updated Newsvendor Study
Tests all 10 models (8 local + 2 remote) with standardized reflection prompts

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
        logging.FileHandler('newsvendor_validation_updated.log'),
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
    timeout_seconds: int = 120  # Longer timeout for remote models


class ValidationRunner:
    """Runs validation experiments across all 10 models."""
    
    def __init__(self):
        """Initialize validation runner."""
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
        
        logger.info("Initialized ValidationRunner for 10 models")
    
    async def initialize(self) -> None:
        """Initialize the model manager."""
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
                logger.info(f"   🌐 {model} (${cost:.6f})")
            else:
                logger.info(f"   💻 {model}")
        
        if failed_models:
            logger.warning(f"❌ Failed models: {failed_models}")
        
        logger.info("Model manager initialized successfully")
    
    async def run_validation_phase(self) -> Dict[str, Any]:
        """Run Phase 1: Validation with 10 models, 1 negotiation each."""
        logger.info("=== VALIDATION PHASE ===")
        logger.info("Testing all 10 models with 1 negotiation each")
        self.start_time = time.time()
        
        # Create validation configurations
        # Test each model as buyer vs first supplier (for simplicity)
        validation_configs = []
        
        for buyer_model in self.all_models:
            # Use tinyllama as standard supplier for validation
            supplier_model = "tinyllama:latest"
            
            # Test with full reflection (pattern "11") to see thinking
            config = ValidationConfig(
                buyer_model=buyer_model,
                supplier_model=supplier_model,
                reflection_pattern="11"  # Both agents reflect
            )
            validation_configs.append(config)
        
        logger.info(f"Running {len(validation_configs)} validation negotiations...")
        
        # Run validation negotiations sequentially (safer for debugging)
        validation_results = []
        for i, config in enumerate(validation_configs, 1):
            logger.info(f"\n[{i}/{len(validation_configs)}] Testing {config.buyer_model} vs {config.supplier_model}")
            
            result = await self._run_single_negotiation(config, f"validation_{i:02d}")
            validation_results.append(result)
            
            # Show immediate results
            if result.completed:
                logger.info(f"✅ SUCCESS: Agreed on ${result.agreed_price} in {result.total_rounds} rounds")
                logger.info(f"   Tokens: {result.total_tokens}, Cost: ${result.metadata.get('total_cost', 0):.6f}")
            else:
                logger.info(f"❌ FAILED: {result.termination_type.value}")
            
            # Brief pause between negotiations
            await asyncio.sleep(1)
        
        # Analyze validation results
        analysis = self._analyze_validation_results(validation_results)
        
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
                turn_added = tracker.add_turn(
                    speaker=agent_role,
                    message=response.text,
                    reflection=None,  # Could extract reflection from response if needed
                    tokens_used=response.tokens_used,
                    generation_time=response.generation_time
                )
                
                if not turn_added:
                    logger.error(f"Failed to add turn for {agent_role}")
                    tracker.force_termination("turn_error")
                    break
                
                # Show progress
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
    
    def _analyze_validation_results(self, results: List[NegotiationResult]) -> Dict[str, Any]:
        """Analyze validation results."""
        
        total = len(results)
        successful = [r for r in results if r.completed and r.agreed_price]
        
        analysis = {
            "validation_summary": {
                "total_negotiations": total,
                "successful_negotiations": len(successful),
                "success_rate": len(successful) / total if total > 0 else 0,
                "completion_time": time.time() - self.start_time if self.start_time else 0
            },
            "model_performance": {},
            "cost_analysis": {},
            "conversation_analysis": {}
        }
        
        if successful:
            # Price analysis
            prices = [r.agreed_price for r in successful]
            analysis["price_statistics"] = {
                "mean_price": sum(prices) / len(prices),
                "median_price": sorted(prices)[len(prices)//2],
                "min_price": min(prices),
                "max_price": max(prices),
                "optimal_convergence": sum(1 for p in prices if 60 <= p <= 70) / len(prices)
            }
            
            # Efficiency analysis
            tokens = [r.total_tokens for r in successful]
            rounds = [r.total_rounds for r in successful]
            
            analysis["efficiency_statistics"] = {
                "avg_tokens": sum(tokens) / len(tokens),
                "avg_rounds": sum(rounds) / len(rounds),
                "tokens_per_round": sum(tokens) / sum(rounds) if sum(rounds) > 0 else 0
            }
        
        # Per-model analysis
        for result in results:
            model = result.buyer_model  # All tested as buyers
            
            analysis["model_performance"][model] = {
                "success": result.completed,
                "agreed_price": result.agreed_price,
                "rounds": result.total_rounds,
                "tokens": result.total_tokens,
                "cost": result.metadata.get('total_cost', 0.0),
                "generation_time": result.total_time,
                "model_type": "remote" if "remote" in model else "local"
            }
        
        # Cost analysis
        total_cost = sum(r.metadata.get('total_cost', 0.0) for r in results)
        remote_cost = sum(
            r.metadata.get('total_cost', 0.0) for r in results 
            if "remote" in r.buyer_model
        )
        
        analysis["cost_analysis"] = {
            "total_validation_cost": total_cost,
            "remote_model_cost": remote_cost,
            "local_model_cost": total_cost - remote_cost,
            "avg_cost_per_negotiation": total_cost / total if total > 0 else 0
        }
        
        return analysis
    
    async def _save_validation_results(self, results: List[NegotiationResult], analysis: Dict[str, Any]):
        """Save validation results to files."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('./validation_results_updated')
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_data = {
            "metadata": {
                "phase": "validation_updated",
                "timestamp": timestamp,
                "total_models": len(self.all_models),
                "game_config": self.game_config,
                "unified_token_budget": 2000,
                "standardized_reflection": True
            },
            "analysis": analysis,
            "results": [asdict(result) for result in results]
        }
        
        results_file = output_dir / f"validation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save conversation transcripts
        transcripts_file = output_dir / f"validation_conversations_{timestamp}.txt"
        with open(transcripts_file, 'w') as f:
            f.write("NEWSVENDOR EXPERIMENT - VALIDATION CONVERSATIONS (UPDATED)\n")
            f.write("=" * 80 + "\n\n")
            
            for result in results:
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
        
        logger.info(f"Validation results saved to:")
        logger.info(f"  Results: {results_file}")
        logger.info(f"  Conversations: {transcripts_file}")
    
    async def shutdown(self):
        """Clean shutdown."""
        if self.model_manager:
            await self.model_manager.shutdown()


@click.command()
@click.option('--models', type=str, help='Comma-separated list of models to test (default: all 10)')
@click.option('--pattern', type=str, default='11', help='Reflection pattern to test (default: 11)')
@click.option('--output', type=click.Path(), help='Output directory for results')
def main(models: Optional[str], pattern: str, output: Optional[str]):
    """Run validation phase for updated newsvendor experiment."""
    
    async def run_validation():
        runner = ValidationRunner()
        
        # Override models if specified
        if models:
            runner.all_models = [model.strip() for model in models.split(',')]
        
        # Override output directory if specified
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
        
        try:
            await runner.initialize()
            analysis = await runner.run_validation_phase()
            
            # Print summary
            print("\n" + "=" * 80)
            print("VALIDATION PHASE COMPLETE")
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
            
            print("\n✅ Validation complete! Ready for full experiment.")
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            raise
        
        finally:
            await runner.shutdown()
    
    asyncio.run(run_validation())


if __name__ == '__main__':
    main()