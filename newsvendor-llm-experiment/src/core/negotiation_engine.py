"""
Negotiation Engine for Newsvendor Experiment

Orchestrates negotiations between buyer and supplier agents,
manages conversation flow, and ensures proper experimental protocol.
"""

import asyncio
import uuid
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import time

from .model_manager import OptimizedModelManager
from .conversation_tracker import ConversationTracker, NegotiationResult
from ..agents.buyer_agent import BuyerAgent
from ..agents.supplier_agent import SupplierAgent
from ..parsing.acceptance_detector import TerminationType

logger = logging.getLogger(__name__)


@dataclass
class NegotiationConfig:
    """Configuration for a single negotiation."""
    buyer_model: str
    supplier_model: str
    reflection_pattern: str  # "00", "01", "10", "11"
    max_rounds: int = 10
    timeout_seconds: int = 60
    game_config: Optional[Dict[str, Any]] = None


def get_replication_count(buyer_model: str, supplier_model: str) -> int:
    """Determine replications based on computational cost."""
    
    MODEL_TIERS = {
        'tinyllama:latest': 'ultra',
        'qwen2:1.5b': 'ultra',
        'gemma2:2b': 'compact',
        'phi3:mini': 'compact',
        'llama3.2:latest': 'compact',
        'mistral:instruct': 'mid',
        'qwen:7b': 'mid',
        'qwen3:latest': 'large'
    }
    
    # UPDATED: New replication matrix (50-40-30-20)
    REPLICATION_MATRIX = {
        ('ultra', 'ultra'): 50,
        ('compact', 'ultra'): 40,
        ('large', 'ultra'): 20,
        ('mid', 'ultra'): 30,
        ('compact', 'compact'): 40,
        ('compact', 'large'): 20,
        ('compact', 'mid'): 30,
        ('mid', 'mid'): 30,
        ('large', 'mid'): 20,
        ('large', 'large'): 20
    }
    
    buyer_tier = MODEL_TIERS.get(buyer_model, 'mid')
    supplier_tier = MODEL_TIERS.get(supplier_model, 'mid')
    
    # Always sort for consistent lookup
    key = tuple(sorted([buyer_tier, supplier_tier]))
    
    return REPLICATION_MATRIX.get(key, 30)  # Default to 30 reps


class NegotiationEngine:
    """Main engine for conducting newsvendor negotiations."""
    
    def __init__(self, model_manager: OptimizedModelManager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize negotiation engine.
        
        Args:
            model_manager: Model manager for LLM operations
            config: Configuration dictionary
        """
        self.model_manager = model_manager
        self.config = config or {}
        
        # Game configuration (CORRECTED)
        self.game_config = {
            'selling_price': 100,
            'production_cost': 30,
            'demand_mean': 40,      # CORRECTED: Normal(40, 10)
            'demand_std': 10,       # CORRECTED: Normal(40, 10)
            'optimal_price': 65,    # Fair split-the-difference
            'max_rounds': 10,
            'timeout_seconds': 60
        }
        self.game_config.update(self.config.get('game', {}))
        
        # Performance tracking
        self.total_negotiations = 0
        self.successful_negotiations = 0
        self.failed_negotiations = 0
        
        logger.info("Initialized NegotiationEngine with corrected game parameters")
        logger.info(f"Game config: {self.game_config}")
    
    async def run_single_negotiation(
        self, 
        buyer_model: str, 
        supplier_model: str,
        reflection_pattern: str,
        negotiation_id: Optional[str] = None
    ) -> NegotiationResult:
        """
        Run a single negotiation between two models.
        
        Args:
            buyer_model: Name of buyer's LLM model
            supplier_model: Name of supplier's LLM model  
            reflection_pattern: "00", "01", "10", "11" (buyer reflection, supplier reflection)
            negotiation_id: Optional custom ID for negotiation
            
        Returns:
            NegotiationResult with complete negotiation outcome
        """
        if negotiation_id is None:
            negotiation_id = f"neg_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Starting negotiation {negotiation_id}: {buyer_model} vs {supplier_model} ({reflection_pattern})")
        
        try:
            # Parse reflection pattern
            buyer_reflection = reflection_pattern[0] == "1"
            supplier_reflection = reflection_pattern[1] == "1"
            
            # Initialize conversation tracker
            tracker = ConversationTracker(
                negotiation_id=negotiation_id,
                buyer_model=buyer_model,
                supplier_model=supplier_model,
                reflection_pattern=reflection_pattern,
                config=self.game_config
            )
            
            # Initialize agents
            buyer_agent = BuyerAgent(
                model_name=buyer_model,
                model_manager=self.model_manager,
                reflection_enabled=buyer_reflection,
                config=self.game_config
            )
            
            supplier_agent = SupplierAgent(
                model_name=supplier_model,
                model_manager=self.model_manager,
                reflection_enabled=supplier_reflection,
                config=self.game_config
            )
            
            # Conduct negotiation
            result = await self._conduct_negotiation(tracker, buyer_agent, supplier_agent)
            
            # Update statistics
            self.total_negotiations += 1
            if result.completed and result.agreed_price:
                self.successful_negotiations += 1
            else:
                self.failed_negotiations += 1
            
            logger.info(f"Completed negotiation {negotiation_id}: "
                       f"{'SUCCESS' if result.completed else 'FAILED'}, "
                       f"price=${result.agreed_price}, rounds={result.total_rounds}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in negotiation {negotiation_id}: {e}")
            self.total_negotiations += 1
            self.failed_negotiations += 1
            
            # Return failed result
            return NegotiationResult(
                negotiation_id=negotiation_id,
                buyer_model=buyer_model,
                supplier_model=supplier_model,
                reflection_pattern=reflection_pattern,
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
    
    async def _conduct_negotiation(
        self, 
        tracker: ConversationTracker, 
        buyer_agent: BuyerAgent, 
        supplier_agent: SupplierAgent
    ) -> NegotiationResult:
        """
        Conduct the actual negotiation between two agents.
        
        Args:
            tracker: Conversation tracker
            buyer_agent: Buyer agent instance
            supplier_agent: Supplier agent instance
            
        Returns:
            Complete negotiation result
        """
        timeout_seconds = self.game_config.get('timeout_seconds', 60)
        max_rounds = self.game_config.get('max_rounds', 10)
        
        start_time = time.time()
        
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
                
                # Check for early termination
                if tracker.completed:
                    break
                
                # Small delay to prevent overwhelming the model manager
                await asyncio.sleep(0.1)
            
            # Finalize negotiation if not already completed
            if not tracker.completed and tracker.round_number >= max_rounds:
                tracker.force_termination("max_rounds_reached")
            
            return tracker.get_final_result()
            
        except Exception as e:
            logger.error(f"Error during negotiation {tracker.negotiation_id}: {e}")
            tracker.force_termination(f"error: {str(e)}")
            return tracker.get_final_result()
    
    async def run_batch_negotiations(
        self,
        negotiations: List[NegotiationConfig],
        max_concurrent: int = 1,
        progress_callback: Optional[callable] = None
    ) -> List[NegotiationResult]:
        """
        Run multiple negotiations with controlled concurrency.
        
        Args:
            negotiations: List of negotiation configurations
            max_concurrent: Maximum concurrent negotiations
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of negotiation results
        """
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_single_with_semaphore(config: NegotiationConfig) -> NegotiationResult:
            async with semaphore:
                result = await self.run_single_negotiation(
                    buyer_model=config.buyer_model,
                    supplier_model=config.supplier_model,
                    reflection_pattern=config.reflection_pattern
                )
                
                if progress_callback:
                    progress_callback(len(results) + 1, len(negotiations), result)
                
                return result
        
        # Create all tasks
        tasks = [run_single_with_semaphore(config) for config in negotiations]
        
        # Execute with controlled concurrency
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Negotiation {i} failed with exception: {result}")
                # Create a failed result
                config = negotiations[i]
                failed_result = NegotiationResult(
                    negotiation_id=f"failed_{i}",
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
    
    def generate_experiment_plan(self, models: List[str]) -> List[NegotiationConfig]:
        """
        Generate complete experiment plan with adaptive replication.
        
        Args:
            models: List of model names to test
            
        Returns:
            List of negotiation configurations
        """
        experiment_plan = []
        
        # All reflection patterns
        reflection_patterns = ["00", "01", "10", "11"]
        
        # Generate all model pairings
        for buyer_model in models:
            for supplier_model in models:
                # Determine replication count for this pairing
                reps = get_replication_count(buyer_model, supplier_model)
                
                # Create negotiations for each reflection pattern
                for pattern in reflection_patterns:
                    for rep in range(reps):
                        config = NegotiationConfig(
                            buyer_model=buyer_model,
                            supplier_model=supplier_model,
                            reflection_pattern=pattern,
                            max_rounds=self.game_config.get('max_rounds', 10),
                            timeout_seconds=self.game_config.get('timeout_seconds', 60),
                            game_config=self.game_config.copy()
                        )
                        experiment_plan.append(config)
        
        logger.info(f"Generated experiment plan with {len(experiment_plan)} negotiations")
        return experiment_plan
    
    async def validate_setup(self, models: List[str]) -> Dict[str, Any]:
        """
        Validate experimental setup and model availability.
        
        Args:
            models: List of models to validate
            
        Returns:
            Validation result dictionary
        """
        validation_results = {
            "overall_status": "unknown",
            "model_validation": {},
            "system_resources": {},
            "test_negotiations": {},
            "recommendations": []
        }
        
        try:
            # Validate model availability
            logger.info("Validating model availability...")
            model_validation = await self.model_manager.validate_models(models)
            validation_results["model_validation"] = model_validation
            
            # Check system resources
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()
            
            validation_results["system_resources"] = {
                "total_memory_gb": memory_gb,
                "cpu_count": cpu_count,
                "memory_adequate": memory_gb >= 16,
                "cpu_adequate": cpu_count >= 4
            }
            
            # Test basic negotiation functionality
            if any(result["success"] for result in model_validation.values()):
                # Pick the first working model for testing
                working_models = [
                    model for model, result in model_validation.items() 
                    if result["success"]
                ]
                
                if len(working_models) >= 2:
                    test_config = NegotiationConfig(
                        buyer_model=working_models[0],
                        supplier_model=working_models[1],
                        reflection_pattern="00"
                    )
                    
                    test_result = await self.run_single_negotiation(
                        test_config.buyer_model,
                        test_config.supplier_model,
                        test_config.reflection_pattern,
                        "validation_test"
                    )
                    
                    validation_results["test_negotiations"]["basic_test"] = {
                        "completed": test_result.completed,
                        "rounds": test_result.total_rounds,
                        "tokens": test_result.total_tokens,
                        "price": test_result.agreed_price
                    }
            
            # Determine overall status
            working_model_count = sum(1 for result in model_validation.values() if result["success"])
            
            if working_model_count >= 2:  # Fixed: Need at least 2 models, not 6
                validation_results["overall_status"] = "ready"
            elif working_model_count >= 1:  # Some models working
                validation_results["overall_status"] = "partial"
                validation_results["recommendations"].append("Some models failed validation - experiment will run with reduced model set")
            else:  # No models working
                validation_results["overall_status"] = "failed"
                validation_results["recommendations"].append("No working models available - check Ollama installation and model availability")
            
            # Resource recommendations
            if not validation_results["system_resources"]["memory_adequate"]:
                validation_results["recommendations"].append("Low system memory - consider reducing concurrent negotiations")
            
            if not validation_results["system_resources"]["cpu_adequate"]:
                validation_results["recommendations"].append("Limited CPU cores - experiment may run slowly")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results["overall_status"] = "error"
            validation_results["error"] = str(e)
        
        return validation_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the negotiation engine."""
        return {
            "total_negotiations": self.total_negotiations,
            "successful_negotiations": self.successful_negotiations,
            "failed_negotiations": self.failed_negotiations,
            "success_rate": (
                self.successful_negotiations / max(self.total_negotiations, 1)
            ),
            "model_manager_stats": self.model_manager.get_model_stats(),
            "game_config": self.game_config
        }


# Add missing import for asdict
from dataclasses import asdict