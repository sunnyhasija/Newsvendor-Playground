#!/usr/bin/env python3
"""
src/core/unified_model_manager.py
Unified Model Manager for Newsvendor Experiment - integrates with existing architecture
Handles both local Ollama models and remote models (Claude, O3) with generous token limits
"""

import asyncio
import time
import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import ollama
except ImportError:
    print("Please install ollama: pip install ollama")
    raise

logger = logging.getLogger(__name__)


@dataclass
class GenerationResponse:
    """Unified response from any model generation."""
    text: str
    tokens_used: int
    generation_time: float
    success: bool
    error: Optional[str] = None
    model_name: str = ""
    timestamp: float = 0.0
    cost_estimate: float = 0.0
    
    # O3-specific fields
    reasoning_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class UnifiedModelManager:
    """Manages both local and remote models with generous token limits for natural behavior."""
    
    def __init__(self, max_concurrent_models: int = 2, config: Optional[Dict[str, Any]] = None):
        """Initialize unified model manager."""
        self.config = config or {}
        self.max_concurrent = max_concurrent_models
        
        # Initialize clients
        self.ollama_client = ollama.Client()
        self.loaded_models: Dict[str, float] = {}  # model_name -> last_used_time
        
        # Remote model configurations
        self.remote_configs = self._load_remote_configs()
        
        # Initialize remote clients
        self._init_remote_clients()
        
        # Model configurations with generous token budgets
        self.model_configs = self._load_unified_configs()
        
        # Performance tracking
        self.total_cost = 0.0
        self.total_generations = 0
        
        logger.info(f"Initialized UnifiedModelManager with {len(self.get_available_models())} models")
    
    def _load_remote_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load remote model configurations from environment."""
        return {
            'claude-sonnet-4-remote': {
                'provider': 'claude',
                'type': 'remote',
                'cost_per_token': 0.000075,
                'endpoint': os.getenv('claude_endpoint'),
                'aws_access_key': os.getenv('AWS_ACCESS_KEY'),
                'aws_secret_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
                'aws_region': os.getenv('AWS_REGION', 'us-east-1')
            },
            'o3-remote': {
                'provider': 'azure',
                'type': 'remote', 
                'cost_per_token': 0.000240,
                'api_key': os.getenv('AZURE_o3_KEY'),
                'base_url': os.getenv('AZURE_o3_BASE')
            }
        }
    
    def _init_remote_clients(self):
        """Initialize remote model clients."""
        # Claude (AWS Bedrock)
        try:
            import boto3
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=self.remote_configs['claude-sonnet-4-remote']['aws_access_key'],
                aws_secret_access_key=self.remote_configs['claude-sonnet-4-remote']['aws_secret_key'],
                region_name=self.remote_configs['claude-sonnet-4-remote']['aws_region']
            )
            logger.info("✅ Initialized AWS Bedrock client for Claude")
        except Exception as e:
            logger.warning(f"Failed to initialize Bedrock client: {e}")
            self.bedrock_client = None
        
        # O3 (Azure OpenAI)
        try:
            from openai import AzureOpenAI
            self.azure_client = AzureOpenAI(
                api_key=self.remote_configs['o3-remote']['api_key'],
                api_version="2024-12-01-preview",
                azure_endpoint=self.remote_configs['o3-remote']['base_url']
            )
            logger.info("✅ Initialized Azure OpenAI client for O3")
        except Exception as e:
            logger.warning(f"Failed to initialize Azure client: {e}")
            self.azure_client = None
    
    def _load_unified_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load unified model configurations with generous token budgets for natural behavior."""
        
        # All models get generous token budgets to express themselves naturally
        base_config = {
            'max_tokens': 4000,      # 2x increase - let models think and express fully
            'temperature': 0.5,
            'top_p': 0.9
        }
        
        configs = {}
        
        # Local models - all get generous limits
        local_models = [
            "tinyllama:latest", "qwen2:1.5b", "gemma2:2b", "phi3:mini",
            "llama3.2:latest", "mistral:instruct", "qwen:7b", "qwen3:latest"
        ]
        
        for model in local_models:
            configs[model] = {
                'provider': 'ollama',
                'type': 'local',
                'cost_per_token': 0.0,
                **base_config
            }
        
        # Remote models - even more generous for complex reasoning
        for model_name, remote_config in self.remote_configs.items():
            configs[model_name] = {
                **remote_config,
                **base_config,
                'max_tokens': 5000 if model_name != 'o3-remote' else 8000  # O3 gets extra for reasoning
            }
        
        return configs
    
    def get_available_models(self) -> List[str]:
        """Get list of all available models."""
        return list(self.model_configs.keys())
    
    def get_model_details(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all models."""
        return self.model_configs.copy()
    
    async def generate_response(
        self, 
        model_name: str, 
        prompt: str,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs
    ) -> GenerationResponse:
        """
        Generate response from any model with unified interface and generous limits.
        
        Args:
            model_name: Name of model to use
            prompt: Input prompt
            max_tokens: Maximum tokens (for local/Claude)
            max_completion_tokens: Maximum completion tokens (for O3)
            reasoning_effort: Reasoning effort for O3 ('high', 'medium', 'low')
            **kwargs: Additional parameters
        """
        start_time = time.time()
        
        if model_name not in self.model_configs:
            return GenerationResponse(
                text="", tokens_used=0, generation_time=0.0,
                success=False, error=f"Unknown model: {model_name}",
                model_name=model_name, timestamp=time.time()
            )
        
        config = self.model_configs[model_name]
        
        try:
            if config['provider'] == 'ollama':
                return await self._generate_ollama(model_name, prompt, max_tokens, **kwargs)
            elif config['provider'] == 'claude':
                return await self._generate_claude(model_name, prompt, max_tokens, **kwargs)
            elif config['provider'] == 'azure':
                return await self._generate_o3(model_name, prompt, max_completion_tokens, reasoning_effort, **kwargs)
            else:
                raise ValueError(f"Unknown provider: {config['provider']}")
                
        except Exception as e:
            error_msg = f"Generation failed for {model_name}: {str(e)}"
            logger.error(error_msg)
            
            return GenerationResponse(
                text="", tokens_used=0, generation_time=time.time() - start_time,
                success=False, error=error_msg, model_name=model_name,
                timestamp=time.time()
            )
    
    async def _generate_ollama(self, model_name: str, prompt: str, max_tokens: Optional[int], **kwargs) -> GenerationResponse:
        """Generate response from Ollama model with generous token limits."""
        config = self.model_configs[model_name]
        max_tokens = max_tokens or config.get('max_tokens', 4000)  # Default to generous limit
        
        # Prepare generation options
        options = {
            'num_predict': max_tokens,
            'temperature': config.get('temperature', 0.5),
            'top_p': config.get('top_p', 0.9),
            'stop': ['<END>', '\n\nHuman:', '\n\nAssistant:'],
            **kwargs
        }
        
        # Ensure model is loaded (simplified - assume models are available)
        self.loaded_models[model_name] = time.time()
        
        start_time = time.time()
        
        # Generate response using asyncio.to_thread for thread safety
        response = await asyncio.to_thread(
            self.ollama_client.generate,
            model=model_name,
            prompt=prompt,
            options=options
        )
        
        response_text = response.get('response', '').strip()
        tokens_used = response.get('eval_count', 0)
        generation_time = time.time() - start_time
        
        self.total_generations += 1
        
        return GenerationResponse(
            text=response_text,
            tokens_used=tokens_used,
            generation_time=generation_time,
            success=True,
            model_name=model_name,
            timestamp=time.time(),
            cost_estimate=0.0  # Local models are free
        )
    
    async def _generate_claude(self, model_name: str, prompt: str, max_tokens: Optional[int], **kwargs) -> GenerationResponse:
        """Generate response from Claude via AWS Bedrock with generous token limits."""
        if not self.bedrock_client:
            raise RuntimeError("Bedrock client not initialized")
        
        config = self.model_configs[model_name]
        max_tokens = max_tokens or config.get('max_tokens', 5000)  # Generous default
        
        import json
        
        # Prepare request body for Claude
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": config.get('temperature', 0.5),
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        start_time = time.time()
        
        # Make request to Bedrock
        response = await asyncio.to_thread(
            self.bedrock_client.invoke_model,
            modelId="us.anthropic.claude-sonnet-4-20250514-v1:0",  # Your correct endpoint
            body=json.dumps(request_body),
            contentType="application/json"
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        if response_body.get('type') == 'error':
            raise RuntimeError(f"Claude API error: {response_body.get('error', {}).get('message', 'Unknown error')}")
        
        response_text = response_body['content'][0]['text']
        tokens_used = response_body['usage']['input_tokens'] + response_body['usage']['output_tokens']
        generation_time = time.time() - start_time
        
        # Calculate cost
        cost = tokens_used * config['cost_per_token']
        self.total_cost += cost
        self.total_generations += 1
        
        return GenerationResponse(
            text=response_text,
            tokens_used=tokens_used,
            generation_time=generation_time,
            success=True,
            model_name=model_name,
            timestamp=time.time(),
            cost_estimate=cost
        )
    
    async def _generate_o3(self, model_name: str, prompt: str, max_completion_tokens: Optional[int], reasoning_effort: Optional[str], **kwargs) -> GenerationResponse:
        """Generate response from O3 via Azure OpenAI with very generous token limits."""
        if not self.azure_client:
            raise RuntimeError("Azure client not initialized")
        
        config = self.model_configs[model_name]
        max_completion_tokens = max_completion_tokens or config.get('max_tokens', 8000)  # Very generous for O3
        reasoning_effort = reasoning_effort or 'high'
        
        start_time = time.time()
        
        # Make request to Azure OpenAI
        response = await asyncio.to_thread(
            self.azure_client.chat.completions.create,
            model="o3-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            temperature=1.0,
        )
        
        response_text = response.choices[0].message.content
        
        # O3 provides separate token counts for reasoning and completion
        reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)
        completion_tokens = getattr(response.usage, 'completion_tokens', 0)
        total_tokens = reasoning_tokens + completion_tokens
        
        generation_time = time.time() - start_time
        
        # Calculate cost (O3 charges for all tokens)
        cost = total_tokens * config['cost_per_token']
        self.total_cost += cost
        self.total_generations += 1
        
        return GenerationResponse(
            text=response_text,
            tokens_used=total_tokens,
            generation_time=generation_time,
            success=True,
            model_name=model_name,
            timestamp=time.time(),
            cost_estimate=cost,
            reasoning_tokens=reasoning_tokens,
            completion_tokens=completion_tokens
        )
    
    def get_total_cost(self) -> float:
        """Get total cost of all remote model calls."""
        return self.total_cost
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_generations": self.total_generations,
            "total_cost": self.total_cost,
            "available_models": len(self.get_available_models()),
            "local_models": len([m for m in self.model_configs.values() if m['type'] == 'local']),
            "remote_models": len([m for m in self.model_configs.values() if m['type'] == 'remote'])
        }
    
    async def validate_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Validate that all models work with a simple test prompt and generous token limits."""
        test_prompt = "You are negotiating. Say only: I offer $50"
        results = {}
        
        for model_name in self.get_available_models():
            try:
                if model_name == 'o3-remote':
                    response = await self.generate_response(
                        model_name, test_prompt, max_completion_tokens=8000, reasoning_effort='high'
                    )
                else:
                    response = await self.generate_response(
                        model_name, test_prompt, max_tokens=4000
                    )
                
                results[model_name] = {
                    "success": response.success,
                    "response_length": len(response.text),
                    "tokens_used": response.tokens_used,
                    "generation_time": response.generation_time,
                    "response_preview": response.text[:100] if response.text else "",
                    "error": response.error,
                    "cost": response.cost_estimate,
                    "reasoning_tokens": getattr(response, 'reasoning_tokens', None)
                }
                
            except Exception as e:
                results[model_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    async def shutdown(self) -> None:
        """Clean shutdown of model manager."""
        logger.info(f"Shutting down UnifiedModelManager...")
        logger.info(f"Final stats: {self.total_generations} generations, ${self.total_cost:.4f} total cost")


def create_unified_model_manager(config: Optional[Dict[str, Any]] = None) -> UnifiedModelManager:
    """Factory function to create unified model manager with generous token limits."""
    return UnifiedModelManager(max_concurrent_models=2, config=config)