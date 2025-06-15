"""
Optimized Model Manager for Newsvendor Experiment

Handles Ollama model loading, unloading, and generation with memory
optimization and concurrent model management.
"""

import asyncio
import time
import logging
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    import ollama
except ImportError:
    print("Please install ollama: pip install ollama")
    raise

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    loaded_at: float
    last_used: float
    memory_usage_mb: Optional[float] = None
    total_generations: int = 0
    total_tokens: int = 0
    total_time: float = 0.0


@dataclass
class GenerationResponse:
    """Response from model generation."""
    text: str
    tokens_used: int
    generation_time: float
    success: bool
    error: Optional[str] = None
    model_name: str = ""
    timestamp: float = 0.0


class OptimizedModelManager:
    """Efficient model loading with memory management."""
    
    def __init__(self, max_concurrent_models: int = 1, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model manager.
        
        Args:
            max_concurrent_models: Maximum number of models to keep loaded
            config: Configuration dictionary with model settings
        """
        self.ollama_client = ollama.Client()
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.max_concurrent = max_concurrent_models
        self.config = config or {}
        
        # Load model configurations
        self.model_configs = self._load_model_configs()
        
        # Performance tracking
        self.total_generations = 0
        self.total_errors = 0
        self.memory_limit_gb = self.config.get('memory_limit_gb', 40)
        
        # Async lock for model operations
        self._model_lock = asyncio.Lock()
        
        logger.info(f"Initialized ModelManager with max_concurrent={max_concurrent_models}")
    
    def _load_model_configs(self) -> Dict[str, Any]:
        """Load model configurations from config."""
        # Default configurations if not provided
        default_configs = {
            "tinyllama:latest": {
                "token_limit": 256,
                "temperature": 0.3,
                "top_p": 0.8,
                "tier": "ultra"
            },
            "qwen2:1.5b": {
                "token_limit": 256,
                "temperature": 0.3,
                "top_p": 0.8,
                "tier": "ultra"
            },
            "gemma2:2b": {
                "token_limit": 384,
                "temperature": 0.4,
                "top_p": 0.85,
                "tier": "compact"
            },
            "phi3:mini": {
                "token_limit": 384,
                "temperature": 0.4,
                "top_p": 0.85,
                "tier": "compact"
            },
            "llama3.2:latest": {
                "token_limit": 384,
                "temperature": 0.4,
                "top_p": 0.85,
                "tier": "compact"
            },
            "mistral:instruct": {
                "token_limit": 512,
                "temperature": 0.5,
                "top_p": 0.9,
                "tier": "mid"
            },
            "qwen:7b": {
                "token_limit": 512,
                "temperature": 0.5,
                "top_p": 0.9,
                "tier": "mid"
            },
            "qwen3:latest": {
                "token_limit": 512,
                "temperature": 0.6,
                "top_p": 0.95,
                "tier": "large"
            }
        }
        
        # Use provided config or defaults
        model_configs = self.config.get('models', default_configs)
        return model_configs
    
    async def generate_response(
        self, 
        model_name: str, 
        prompt: str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerationResponse:
        """
        Generate response with automatic model management.
        
        Args:
            model_name: Name of the model to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResponse with result and metadata
        """
        start_time = time.time()
        
        try:
            # Ensure model is loaded
            await self._ensure_model_loaded(model_name)
            
            # Get model-specific configuration
            config = self.model_configs.get(model_name, {})
            max_tokens = max_tokens or config.get('token_limit', 512)
            
            # Prepare generation options
            options = {
                'num_predict': max_tokens,
                'temperature': config.get('temperature', 0.5),
                'top_p': config.get('top_p', 0.9),
                'stop': ['<END>', '\n\nHuman:', '\n\nAssistant:'],
                **kwargs  # Allow override of options
            }
            
            # Generate response using asyncio.to_thread for thread safety
            response = await asyncio.to_thread(
                self.ollama_client.generate,
                model=model_name,
                prompt=prompt,
                options=options
            )
            
            # Extract response data
            response_text = response.get('response', '').strip()
            tokens_used = response.get('eval_count', 0)
            generation_time = time.time() - start_time
            
            # Update model statistics
            await self._update_model_stats(model_name, tokens_used, generation_time)
            
            # Update global statistics
            self.total_generations += 1
            
            logger.debug(f"Generated {tokens_used} tokens from {model_name} in {generation_time:.2f}s")
            
            return GenerationResponse(
                text=response_text,
                tokens_used=tokens_used,
                generation_time=generation_time,
                success=True,
                model_name=model_name,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.total_errors += 1
            error_msg = f"Generation failed for {model_name}: {str(e)}"
            logger.error(error_msg)
            
            return GenerationResponse(
                text='',
                tokens_used=0,
                generation_time=time.time() - start_time,
                success=False,
                error=error_msg,
                model_name=model_name,
                timestamp=time.time()
            )
    
    async def _ensure_model_loaded(self, model_name: str) -> None:
        """Load model if not already loaded, managing memory."""
        async with self._model_lock:
            # Check if already loaded
            if model_name in self.loaded_models:
                self.loaded_models[model_name].last_used = time.time()
                return
            
            # Check memory before loading
            current_memory_gb = psutil.virtual_memory().used / (1024**3)
            if current_memory_gb > self.memory_limit_gb * 0.8:  # 80% threshold
                logger.warning(f"Memory usage high ({current_memory_gb:.1f}GB), forcing cleanup")
                await self._cleanup_models(force=True)
            
            # Unload models if at capacity
            if len(self.loaded_models) >= self.max_concurrent:
                await self._unload_oldest_model()
            
            # Load the requested model
            try:
                logger.info(f"Loading model: {model_name}")
                
                # Use asyncio.to_thread for the blocking pull operation
                await asyncio.to_thread(self.ollama_client.pull, model_name)
                
                # Record model info
                model_info = ModelInfo(
                    name=model_name,
                    loaded_at=time.time(),
                    last_used=time.time()
                )
                
                self.loaded_models[model_name] = model_info
                logger.info(f"Successfully loaded model: {model_name}")
                
            except Exception as e:
                error_msg = f"Failed to load {model_name}: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
    
    async def _unload_oldest_model(self) -> None:
        """Unload the least recently used model."""
        if not self.loaded_models:
            return
        
        # Find oldest model by last_used timestamp
        oldest_model_name = min(
            self.loaded_models.keys(),
            key=lambda x: self.loaded_models[x].last_used
        )
        
        await self._unload_model(oldest_model_name)
    
    async def _unload_model(self, model_name: str) -> None:
        """Unload a specific model."""
        try:
            if model_name in self.loaded_models:
                logger.info(f"Unloading model: {model_name}")
                
                # Note: Ollama doesn't have explicit unload, but we remove from tracking
                model_info = self.loaded_models.pop(model_name)
                
                logger.info(
                    f"Unloaded {model_name} - "
                    f"generated {model_info.total_generations} responses, "
                    f"{model_info.total_tokens} tokens"
                )
                
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
    
    async def _update_model_stats(self, model_name: str, tokens: int, time_taken: float) -> None:
        """Update model performance statistics."""
        if model_name in self.loaded_models:
            model_info = self.loaded_models[model_name]
            model_info.last_used = time.time()
            model_info.total_generations += 1
            model_info.total_tokens += tokens
            model_info.total_time += time_taken
    
    async def _cleanup_models(self, force: bool = False) -> None:
        """Clean up models to free memory."""
        if not self.loaded_models:
            return
        
        if force:
            # Unload all models
            model_names = list(self.loaded_models.keys())
            for model_name in model_names:
                await self._unload_model(model_name)
        else:
            # Unload half of the models (oldest first)
            models_to_unload = len(self.loaded_models) // 2
            if models_to_unload > 0:
                sorted_models = sorted(
                    self.loaded_models.keys(),
                    key=lambda x: self.loaded_models[x].last_used
                )
                
                for model_name in sorted_models[:models_to_unload]:
                    await self._unload_model(model_name)
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all models."""
        stats = {
            "total_generations": self.total_generations,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / max(self.total_generations, 1),
            "loaded_models": len(self.loaded_models),
            "max_concurrent": self.max_concurrent,
            "memory_usage_gb": psutil.virtual_memory().used / (1024**3),
            "memory_limit_gb": self.memory_limit_gb,
            "models": {}
        }
        
        # Add per-model statistics
        for model_name, model_info in self.loaded_models.items():
            stats["models"][model_name] = {
                "loaded_at": model_info.loaded_at,
                "last_used": model_info.last_used,
                "total_generations": model_info.total_generations,
                "total_tokens": model_info.total_tokens,
                "total_time": model_info.total_time,
                "avg_tokens_per_generation": (
                    model_info.total_tokens / max(model_info.total_generations, 1)
                ),
                "avg_time_per_generation": (
                    model_info.total_time / max(model_info.total_generations, 1)
                ),
                "tokens_per_second": (
                    model_info.total_tokens / max(model_info.total_time, 0.001)
                )
            }
        
        return stats
    
    async def preload_models(self, model_names: List[str]) -> Dict[str, bool]:
        """Preload multiple models for faster access."""
        results = {}
        
        for model_name in model_names:
            try:
                await self._ensure_model_loaded(model_name)
                results[model_name] = True
                logger.info(f"Preloaded model: {model_name}")
            except Exception as e:
                results[model_name] = False
                logger.error(f"Failed to preload {model_name}: {e}")
        
        return results
    
    async def validate_models(self, model_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Validate that models work correctly with test prompts."""
        test_prompt = "You are negotiating. Say only: I offer $50"
        results = {}
        
        for model_name in model_names:
            try:
                response = await self.generate_response(
                    model_name, 
                    test_prompt, 
                    max_tokens=50
                )
                
                results[model_name] = {
                    "success": response.success,
                    "response_length": len(response.text),
                    "tokens_used": response.tokens_used,
                    "generation_time": response.generation_time,
                    "response_preview": response.text[:100] if response.text else "",
                    "error": response.error
                }
                
            except Exception as e:
                results[model_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    async def test_concurrency(self, model_names: List[str], num_concurrent: int = 2) -> Dict[str, Any]:
        """Test concurrent model operations."""
        start_time = time.time()
        memory_before = psutil.virtual_memory().used / (1024**3)
        
        # Create concurrent tasks
        tasks = []
        for i in range(num_concurrent):
            model_name = model_names[i % len(model_names)]
            task = self.generate_response(
                model_name,
                f"Test prompt {i}: respond with number {i}",
                max_tokens=20
            )
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        memory_after = psutil.virtual_memory().used / (1024**3)
        total_time = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if isinstance(r, GenerationResponse) and r.success)
        failed = len(results) - successful
        
        return {
            "num_concurrent": num_concurrent,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(results),
            "total_time": total_time,
            "memory_used_gb": memory_after - memory_before,
            "memory_before_gb": memory_before,
            "memory_after_gb": memory_after,
            "avg_time_per_request": total_time / len(results),
            "results": [r for r in results if isinstance(r, GenerationResponse)]
        }
    
    async def shutdown(self) -> None:
        """Clean shutdown of model manager."""
        logger.info("Shutting down ModelManager...")
        await self._cleanup_models(force=True)
        
        final_stats = self.get_model_stats()
        logger.info(f"Final stats: {final_stats['total_generations']} generations, "
                   f"{final_stats['total_errors']} errors")
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'loaded_models'):
            logger.info("ModelManager being destroyed")