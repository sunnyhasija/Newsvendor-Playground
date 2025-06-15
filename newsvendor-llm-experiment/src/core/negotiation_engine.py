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
    
    # FIXED: Use sorted keys for symmetric lookup
    REPLICATION_MATRIX = {
        ('ultra', 'ultra'): 20,
        ('compact', 'ultra'): 15,  # Fixed: sorted order
        ('large', 'ultra'): 5,     # Fixed: sorted order
        ('mid', 'ultra'): 10,      # Fixed: sorted order
        ('compact', 'compact'): 15,
        ('compact', 'large'): 5,   # Fixed: sorted order
        ('compact', 'mid'): 10,    # Fixed: sorted order
        ('mid', 'mid'): 10,
        ('large', 'mid'): 5,       # Fixed: sorted order
        ('large', 'large'): 5
    }
    
    buyer_tier = MODEL_TIERS.get(buyer_model, 'mid')
    supplier_tier = MODEL_TIERS.get(supplier_model, 'mid')
    
    # Always sort for consistent lookup
    key = tuple(sorted([buyer_tier, supplier_tier]))
    
    return REPLICATION_MATRIX.get(key, 10)  # Default to 10 reps


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
                metadata=