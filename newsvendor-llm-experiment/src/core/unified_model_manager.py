#!/usr/bin/env python3
"""
src/core/unified_model_manager.py
Unified Model Manager for Newsvendor Experiment - now includes Azure AI Grok support
Handles local Ollama models and remote models (Claude, O3, Azure AI Grok) with generous token limits
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
    """Manages local and remote models including Azure AI Grok with generous token limits."""
    
    def __init__(self, max_concurrent_models: int = 2, config: Optional[Dict[str, Any]] = None):
        """Initialize unified model manager with Azure AI Grok support."""
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
        
        logger.info(f"Initialized UnifiedModelManager with {len(self.get_available_models())} models (including Azure AI Grok)")
    
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
            },
            'grok-remote': {
                'provider': 'azure_ai_grok',
                'type': 'remote',
                'cost_per_token': 0.000020,  # Estimated cost per token for Grok
                'api_key': os.getenv('AZURE_AI_GROK3_MINI_API_KEY'),  # Updated env var name
                'base_url': 'https://newsvendor-playground-resource.services.ai.azure.com/models',  # Updated endpoint
                'api_version': '2024-05-01-preview',  # API version as specified
                'model_name': 'grok-3-mini'  # Updated model name
            }
        }
    
    def _init_remote_clients(self):
        """Initialize remote model clients including Azure AI Grok."""
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
        
        # Grok (Azure AI Services)
        try:
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential
            
            if self.remote_configs['grok-remote']['api_key']:
                self.grok_client = ChatCompletionsClient(
                    endpoint=self.remote_configs['grok-remote']['base_url'],
                    credential=AzureKeyCredential(self.remote_configs['grok-remote']['api_key']),
                    api_version=self.remote_configs['grok-remote']['api_version']
                )
                logger.info("✅ Initialized Azure AI client for Grok")
            else:
                logger.warning("AZURE_AI_GROK3_MINI_API_KEY not found in environment")
                self.grok_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Azure AI Grok client: {e}")
            self.grok_client = None
    
    def _load_unified_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load unified model configurations without restrictive token budgets."""
        
        # Base config without artificial token limits
        base_config = {
            'temperature': 0.5,
            'top_p': 0.9
        }
        
        configs = {}
        
        # Local models - no token limits, let them express naturally
        local_models = [
            "qwen2:1.5b", "gemma2:2b", "phi3:mini",
            "llama3.2:latest", "mistral:instruct", "qwen:7b", "qwen3:latest"
        ]
        
        for model in local_models:
            configs[model] = {
                'provider': 'ollama',
                'type': 'local',
                'cost_per_token': 0.0,
                'max_tokens': None,  # No limit - let models express naturally
                **base_config
            }
        
        # Remote models - no artificial limits, use service maximums
        for model_name, remote_config in self.remote_configs.items():
            # Use service maximum limits, not artificial restrictions
            if model_name == 'o3-remote':
                max_tokens = None  # Let O3 use its full reasoning capacity
            elif model_name == 'grok-remote':
                max_tokens = None  # Let Grok express fully
            elif model_name == 'claude-sonnet-4-remote':
                max_tokens = None  # Let Claude use full capacity
            else:
                max_tokens = None
            
            configs[model_name] = {
                **remote_config,
                **base_config,
                'max_tokens': max_tokens
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
            max_tokens: Maximum tokens (for local/Claude/Grok)
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
            elif config['provider'] == 'azure_ai_grok':
                return await self._generate_azure_ai_grok(model_name, prompt, max_tokens, **kwargs)
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
        """Generate response from Ollama model without token restrictions."""
        config = self.model_configs[model_name]
        
        # Don't impose artificial limits unless explicitly requested
        if max_tokens is None:
            # Let Ollama use its natural response length
            max_tokens = -1  # Ollama uses -1 for unlimited
        
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
        """Generate response from Claude via AWS Bedrock without artificial token limits."""
        if not self.bedrock_client:
            raise RuntimeError("Bedrock client not initialized")
        
        config = self.model_configs[model_name]
        
        # Use Claude's maximum (200k tokens) unless specifically limited
        if max_tokens is None:
            max_tokens = 200000  # Claude Sonnet 4's maximum
        
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
            modelId="us.anthropic.claude-sonnet-4-20250514-v1:0",
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
        """Generate response from O3 via Azure OpenAI without artificial token limits."""
        if not self.azure_client:
            raise RuntimeError("Azure client not initialized")
        
        config = self.model_configs[model_name]
        reasoning_effort = reasoning_effort or 'high'
        
        # Don't artificially limit O3's reasoning capacity
        if max_completion_tokens is None:
            max_completion_tokens = 100000  # Let O3 reason fully
        
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
    
    async def _generate_azure_ai_grok(self, model_name: str, prompt: str, max_tokens: Optional[int], **kwargs) -> GenerationResponse:
        """Generate response from Grok via Azure AI Services without artificial token limits."""
        if not self.grok_client:
            raise RuntimeError("Azure AI Grok client not initialized - check AZURE_AI_GROK3_MINI_API_KEY in .env")
        
        config = self.model_configs[model_name]
        
        # Don't artificially limit Grok unless explicitly requested
        if max_tokens is None:
            # Let Grok express itself fully - use a very high limit
            max_tokens = 100000  # Effectively unlimited for most responses
        
        start_time = time.time()
        
        # Import Azure AI message types
        from azure.ai.inference.models import UserMessage, SystemMessage
        
        # Prepare messages in Azure AI format
        prompt_messages = [
            SystemMessage(content="You are a helpful AI assistant in a negotiation scenario."),
            UserMessage(content=prompt)
        ]
        
        # Make request to Azure AI Grok
        response = await asyncio.to_thread(
            self.grok_client.complete,
            messages=prompt_messages,
            model=config['model_name'],  # Use grok-3-mini
            max_tokens=max_tokens,
            temperature=config.get('temperature', 1.0),  # Grok example used 1.0
            top_p=config.get('top_p', 1.0),  # Grok example used 1.0
            **kwargs
        )
        
        response_text = response.choices[0].message.content
        
        # Azure AI provides token usage information
        tokens_used = getattr(response.usage, 'total_tokens', 0)
        if not tokens_used:
            # Fallback: estimate tokens if not provided
            tokens_used = len(response_text.split()) * 1.3  # Rough estimate
        
        generation_time = time.time() - start_time
        
        # Calculate cost
        cost = tokens_used * config['cost_per_token']
        self.total_cost += cost
        self.total_generations += 1
        
        return GenerationResponse(
            text=response_text,
            tokens_used=int(tokens_used),
            generation_time=generation_time,
            success=True,
            model_name=model_name,
            timestamp=time.time(),
            cost_estimate=cost
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
        """Validate that all models work with a simple test prompt without token restrictions."""
        test_prompt = "You are negotiating. Say only: I offer $50"
        results = {}
        
        for model_name in self.get_available_models():
            try:
                if model_name == 'o3-remote':
                    response = await self.generate_response(
                        model_name, test_prompt, max_completion_tokens=None, reasoning_effort='high'
                    )
                else:
                    response = await self.generate_response(
                        model_name, test_prompt, max_tokens=None  # No token limit
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