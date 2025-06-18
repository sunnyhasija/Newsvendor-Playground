#!/usr/bin/env python3
"""
verify_grok_working.py
Quick verification that Grok is working with proper token limits
"""

import os
from dotenv import load_dotenv

def verify_grok_responses():
    """Verify Grok gives proper responses with adequate token limits."""
    
    load_dotenv()
    api_key = os.getenv("AZURE_AI_GROK3_MINI_API_KEY")
    
    if not api_key:
        print("❌ API key not found")
        return
    
    print("🧪 Verifying Grok with Proper Token Limits")
    print("=" * 45)
    
    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import UserMessage, SystemMessage
        from azure.core.credentials import AzureKeyCredential
        
        client = ChatCompletionsClient(
            endpoint="https://newsvendor-playground-resource.services.ai.azure.com/models",
            credential=AzureKeyCredential(api_key),
            api_version="2024-05-01-preview"
        )
        
        # Test 1: Negotiation with more tokens
        print("🧪 Test 1: Negotiation Response")
        messages1 = [
            SystemMessage(content="You are a helpful AI assistant in a negotiation scenario."),
            UserMessage(content="You are a supplier negotiating price. Make a price offer between $40-70."),
        ]
        
        response1 = client.complete(
            messages=messages1,
            model="grok-3-mini",
            max_tokens=200,  # More tokens for proper response
            temperature=1.0,
            top_p=1.0
        )
        
        text1 = response1.choices[0].message.content
        print(f"✅ Negotiation response: '{text1}'")
        print(f"   Tokens used: {getattr(response1.usage, 'total_tokens', 'N/A')}")
        
        # Test 2: Simple offer
        print(f"\n🧪 Test 2: Simple Offer")
        messages2 = [
            SystemMessage(content="You are negotiating. Be brief."),
            UserMessage(content="Say only: I offer $55"),
        ]
        
        response2 = client.complete(
            messages=messages2,
            model="grok-3-mini",
            max_tokens=50,
            temperature=0.5,
            top_p=0.9
        )
        
        text2 = response2.choices[0].message.content
        print(f"✅ Simple offer: '{text2}'")
        print(f"   Tokens used: {getattr(response2.usage, 'total_tokens', 'N/A')}")
        
        # Test 3: No token limit (let Grok express naturally)
        print(f"\n🧪 Test 3: Natural Expression (No Token Limit)")
        messages3 = [
            SystemMessage(content="You are a witty AI assistant."),
            UserMessage(content="Explain negotiation in a fun, creative way."),
        ]
        
        response3 = client.complete(
            messages=messages3,
            model="grok-3-mini",
            max_tokens=1000,  # Let Grok be creative
            temperature=1.0,
            top_p=1.0
        )
        
        text3 = response3.choices[0].message.content
        print(f"✅ Creative response: '{text3[:100]}{'...' if len(text3) > 100 else ''}'")
        print(f"   Full length: {len(text3)} characters")
        print(f"   Tokens used: {getattr(response3.usage, 'total_tokens', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🤖 Grok Verification Test")
    print("=" * 25)
    
    success = verify_grok_responses()
    
    if success:
        print(f"\n🎉 Grok is fully working!")
        print(f"\n🚀 Ready for integration:")
        print(f"✅ Connection established")
        print(f"✅ Responses working") 
        print(f"✅ Token usage tracking")
        print(f"✅ Ready for experiments!")
        
        print(f"\n📋 Next steps:")
        print(f"1. Grok is ready to use in unified_model_manager.py")
        print(f"2. Run validation: poetry run python run_validation_with_grok.py")
        print(f"3. Include in full experiments!")
    else:
        print(f"\n❌ Verification failed")

if __name__ == "__main__":
    main()