#!/usr/bin/env python3
"""
Fixed Success Rate Analysis - Properly distinguish completion vs agreement
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_success_rates(results_file: str = None):
    """Proper analysis of completion vs success rates."""
    
    # Find results file
    if not results_file:
        results_dir = Path("./experiment_results")
        files = list(results_dir.glob("complete_results_*.json"))
        if not files:
            logger.error("No results files found")
            return
        results_file = max(files, key=lambda f: f.stat().st_mtime)
    
    logger.info(f"Loading data from: {results_file}")
    
    # Load data
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    if 'results' in data:
        results_list = data['results']
    else:
        results_list = data
    
    df = pd.DataFrame(results_list)
    
    logger.info(f"Total negotiations: {len(df):,}")
    
    # Check completion vs success
    completed = df['completed'].astype(bool)
    has_price = pd.notna(df['agreed_price']) & (df['agreed_price'] != '') & (df['agreed_price'] != 0)
    
    # More detailed success criteria
    valid_prices = has_price & (pd.to_numeric(df['agreed_price'], errors='coerce') > 0)
    
    logger.info(f"Negotiations that 'completed': {completed.sum():,} ({completed.mean():.1%})")
    logger.info(f"Negotiations with price values: {has_price.sum():,} ({has_price.mean():.1%})")
    logger.info(f"Negotiations with valid prices > 0: {valid_prices.sum():,} ({valid_prices.mean():.1%})")
    
    # Cross-tabulation
    logger.info("\nDetailed breakdown:")
    logger.info("="*50)
    
    # Four categories
    completed_with_price = completed & valid_prices
    completed_no_price = completed & ~valid_prices
    not_completed_with_price = ~completed & valid_prices
    not_completed_no_price = ~completed & ~valid_prices
    
    logger.info(f"✅ Completed + Valid Price: {completed_with_price.sum():,} ({completed_with_price.mean():.1%})")
    logger.info(f"⚠️  Completed + No Price: {completed_no_price.sum():,} ({completed_no_price.mean():.1%})")
    logger.info(f"🔥 Not Completed + Price: {not_completed_with_price.sum():,} ({not_completed_with_price.mean():.1%})")
    logger.info(f"❌ Not Completed + No Price: {not_completed_no_price.sum():,} ({not_completed_no_price.mean():.1%})")
    
    # Look at actual price values
    logger.info("\nPrice value analysis:")
    logger.info("="*30)
    
    if 'agreed_price' in df.columns:
        # Convert to numeric, handling various formats
        prices_raw = df['agreed_price']
        prices_numeric = pd.to_numeric(prices_raw, errors='coerce')
        
        logger.info(f"Total price entries: {len(prices_raw):,}")
        logger.info(f"Non-null price entries: {pd.notna(prices_raw).sum():,}")
        logger.info(f"Numeric price entries: {pd.notna(prices_numeric).sum():,}")
        logger.info(f"Positive price entries: {(prices_numeric > 0).sum():,}")
        
        # Show distribution of price values
        if pd.notna(prices_numeric).sum() > 0:
            valid_prices_series = prices_numeric.dropna()
            if len(valid_prices_series) > 0:
                logger.info(f"\nPrice statistics (n={len(valid_prices_series):,}):")
                logger.info(f"  Mean: ${valid_prices_series.mean():.2f}")
                logger.info(f"  Median: ${valid_prices_series.median():.2f}")
                logger.info(f"  Min: ${valid_prices_series.min():.2f}")
                logger.info(f"  Max: ${valid_prices_series.max():.2f}")
                logger.info(f"  Std: ${valid_prices_series.std():.2f}")
        
        # Show some example non-numeric values
        non_numeric_mask = pd.isna(prices_numeric) & pd.notna(prices_raw)
        if non_numeric_mask.sum() > 0:
            logger.info(f"\nExamples of non-numeric price values:")
            examples = df[non_numeric_mask]['agreed_price'].unique()[:10]
            for example in examples:
                logger.info(f"  '{example}' (type: {type(example)})")
    
    # Termination type analysis if available
    if 'termination_type' in df.columns:
        logger.info("\nTermination type analysis:")
        logger.info("="*30)
        term_counts = df['termination_type'].value_counts()
        for term_type, count in term_counts.items():
            logger.info(f"  {term_type}: {count:,} ({count/len(df):.1%})")
    
    # Check if we have turns data to understand what happened
    if 'turns' in df.columns:
        logger.info("\nConversation analysis:")
        logger.info("="*25)
        
        has_turns = pd.notna(df['turns']) & (df['turns'] != '') & (df['turns'] != '[]')
        logger.info(f"Negotiations with conversation data: {has_turns.sum():,} ({has_turns.mean():.1%})")
        
        # Sample some conversations
        if has_turns.sum() > 0:
            sample_negotiations = df[has_turns].sample(min(3, has_turns.sum()))
            
            for idx, row in sample_negotiations.iterrows():
                logger.info(f"\nSample negotiation {row.get('negotiation_id', idx)}:")
                logger.info(f"  Completed: {row['completed']}")
                logger.info(f"  Price: {row.get('agreed_price', 'None')}")
                logger.info(f"  Termination: {row.get('termination_type', 'Unknown')}")
                
                # Parse turns if it's JSON string
                turns_data = row['turns']
                if isinstance(turns_data, str):
                    try:
                        turns_data = json.loads(turns_data)
                    except:
                        logger.info(f"  Turns: Unable to parse")
                        continue
                
                if isinstance(turns_data, list) and len(turns_data) > 0:
                    logger.info(f"  Rounds: {len(turns_data)}")
                    logger.info(f"  Last message: '{turns_data[-1].get('message', 'N/A')[:50]}...'")
    
    # Model-specific failure analysis
    logger.info("\n" + "="*60)
    logger.info("MODEL-SPECIFIC SUCCESS/FAILURE ANALYSIS")
    logger.info("="*60)
    
    # Add success indicators to dataframe
    df['true_success'] = completed_with_price
    df['has_valid_price'] = valid_prices
    
    # Analyze by buyer model
    logger.info("\n🛒 BUYER MODEL ANALYSIS:")
    logger.info("-" * 40)
    
    if 'buyer_model' in df.columns:
        buyer_analysis = df.groupby('buyer_model').agg({
            'true_success': ['count', 'sum', 'mean'],
            'has_valid_price': 'mean',
            'completed': 'mean'
        }).round(3)
        
        buyer_analysis.columns = ['Total', 'Successful', 'Success_Rate', 'Price_Rate', 'Completion_Rate']
        buyer_analysis = buyer_analysis.sort_values('Success_Rate', ascending=False)
        
        for model, row in buyer_analysis.iterrows():
            model_short = model.replace(':latest', '').replace('-remote', '')
            logger.info(f"  {model_short:20} | Success: {row['Success_Rate']:.1%} ({row['Successful']:>4.0f}/{row['Total']:>4.0f}) | Price: {row['Price_Rate']:.1%} | Complete: {row['Completion_Rate']:.1%}")
    
    # Analyze by supplier model
    logger.info("\n🏭 SUPPLIER MODEL ANALYSIS:")
    logger.info("-" * 40)
    
    if 'supplier_model' in df.columns:
        supplier_analysis = df.groupby('supplier_model').agg({
            'true_success': ['count', 'sum', 'mean'],
            'has_valid_price': 'mean',
            'completed': 'mean'
        }).round(3)
        
        supplier_analysis.columns = ['Total', 'Successful', 'Success_Rate', 'Price_Rate', 'Completion_Rate']
        supplier_analysis = supplier_analysis.sort_values('Success_Rate', ascending=False)
        
        for model, row in supplier_analysis.iterrows():
            model_short = model.replace(':latest', '').replace('-remote', '')
            logger.info(f"  {model_short:20} | Success: {row['Success_Rate']:.1%} ({row['Successful']:>4.0f}/{row['Total']:>4.0f}) | Price: {row['Price_Rate']:.1%} | Complete: {row['Completion_Rate']:.1%}")
    
    # Model pairing analysis
    logger.info("\n🤝 MODEL PAIRING FAILURE ANALYSIS:")
    logger.info("-" * 45)
    
    if 'buyer_model' in df.columns and 'supplier_model' in df.columns:
        # Find worst performing pairs
        pair_analysis = df.groupby(['buyer_model', 'supplier_model']).agg({
            'true_success': ['count', 'sum', 'mean']
        }).round(3)
        
        pair_analysis.columns = ['Total', 'Successful', 'Success_Rate']
        
        # Show pairs with lowest success rates (and sufficient sample size)
        low_success_pairs = pair_analysis[
            (pair_analysis['Total'] >= 10) & 
            (pair_analysis['Success_Rate'] < 0.5)
        ].sort_values('Success_Rate')
        
        if len(low_success_pairs) > 0:
            logger.info("Worst performing model pairs (success rate < 50%):")
            for (buyer, supplier), row in low_success_pairs.head(10).iterrows():
                buyer_short = buyer.replace(':latest', '').replace('-remote', '')
                supplier_short = supplier.replace(':latest', '').replace('-remote', '')
                logger.info(f"  {buyer_short:15} + {supplier_short:15} | {row['Success_Rate']:.1%} ({row['Successful']:>2.0f}/{row['Total']:>2.0f})")
        else:
            logger.info("No model pairs with particularly low success rates found.")
        
        # Show best performing pairs
        high_success_pairs = pair_analysis[
            (pair_analysis['Total'] >= 10) & 
            (pair_analysis['Success_Rate'] > 0.8)
        ].sort_values('Success_Rate', ascending=False)
        
        if len(high_success_pairs) > 0:
            logger.info("\nBest performing model pairs (success rate > 80%):")
            for (buyer, supplier), row in high_success_pairs.head(10).iterrows():
                buyer_short = buyer.replace(':latest', '').replace('-remote', '')
                supplier_short = supplier.replace(':latest', '').replace('-remote', '')
                logger.info(f"  {buyer_short:15} + {supplier_short:15} | {row['Success_Rate']:.1%} ({row['Successful']:>2.0f}/{row['Total']:>2.0f})")
    
    # Remote vs Local model analysis
    logger.info("\n🌐 REMOTE vs LOCAL MODEL ANALYSIS:")
    logger.info("-" * 40)
    
    if 'buyer_model' in df.columns:
        # Classify models
        df['buyer_type'] = df['buyer_model'].apply(lambda x: 'Remote' if 'remote' in str(x) else 'Local')
        df['supplier_type'] = df['supplier_model'].apply(lambda x: 'Remote' if 'remote' in str(x) else 'Local')
        
        # Analysis by model type
        type_analysis = df.groupby(['buyer_type', 'supplier_type']).agg({
            'true_success': ['count', 'sum', 'mean'],
            'has_valid_price': 'mean'
        }).round(3)
        
        type_analysis.columns = ['Total', 'Successful', 'Success_Rate', 'Price_Rate']
        
        for (buyer_type, supplier_type), row in type_analysis.iterrows():
            logger.info(f"  {buyer_type:6} + {supplier_type:6} | Success: {row['Success_Rate']:.1%} ({row['Successful']:>4.0f}/{row['Total']:>4.0f}) | Price: {row['Price_Rate']:.1%}")
    
    # Reflection pattern failure analysis
    logger.info("\n🤔 REFLECTION PATTERN FAILURE ANALYSIS:")
    logger.info("-" * 45)
    
    if 'reflection_pattern' in df.columns:
        reflection_names = {
            '00': 'No Reflection',
            '01': 'Buyer Only',
            '10': 'Supplier Only', 
            '11': 'Both Reflect'
        }
        
        reflection_analysis = df.groupby('reflection_pattern').agg({
            'true_success': ['count', 'sum', 'mean'],
            'has_valid_price': 'mean',
            'completed': 'mean'
        }).round(3)
        
        reflection_analysis.columns = ['Total', 'Successful', 'Success_Rate', 'Price_Rate', 'Completion_Rate']
        
        for pattern, row in reflection_analysis.iterrows():
            pattern_name = reflection_names.get(pattern, pattern)
            logger.info(f"  {pattern_name:15} | Success: {row['Success_Rate']:.1%} ({row['Successful']:>4.0f}/{row['Total']:>4.0f}) | Price: {row['Price_Rate']:.1%} | Complete: {row['Completion_Rate']:.1%}")
    
    # Sample failed negotiations by model
    logger.info("\n🔍 SAMPLE FAILED NEGOTIATIONS BY MODEL:")
    logger.info("-" * 45)
    
    failed_negotiations = df[~df['true_success']]
    if len(failed_negotiations) > 0:
        # Sample failures from different models
        if 'buyer_model' in df.columns:
            sample_failures = failed_negotiations.groupby('buyer_model').apply(
                lambda x: x.sample(min(2, len(x))) if len(x) > 0 else x
            ).reset_index(drop=True)
            
            for idx, row in sample_failures.head(10).iterrows():
                buyer_short = str(row.get('buyer_model', 'Unknown')).replace(':latest', '').replace('-remote', '')
                supplier_short = str(row.get('supplier_model', 'Unknown')).replace(':latest', '').replace('-remote', '')
                
                logger.info(f"\n  Negotiation: {row.get('negotiation_id', 'Unknown')}")
                logger.info(f"    Buyer: {buyer_short}, Supplier: {supplier_short}")
                logger.info(f"    Completed: {row.get('completed', 'Unknown')}")
                logger.info(f"    Price: {row.get('agreed_price', 'None')}")
                logger.info(f"    Termination: {row.get('termination_type', 'Unknown')}")
                logger.info(f"    Reflection: {row.get('reflection_pattern', 'Unknown')}")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("CORRECTED SUCCESS RATE SUMMARY")
    logger.info("="*60)
    
    true_success_rate = completed_with_price.mean()
    logger.info(f"📊 Total Negotiations: {len(df):,}")
    logger.info(f"✅ True Success Rate: {true_success_rate:.1%} ({completed_with_price.sum():,} negotiations)")
    logger.info(f"📝 Completion Rate: {completed.mean():.1%} ({completed.sum():,} negotiations)")
    logger.info(f"💰 Valid Price Rate: {valid_prices.mean():.1%} ({valid_prices.sum():,} negotiations)")
    
    if true_success_rate < 0.5:
        logger.warning("⚠️  Success rate is quite low - may indicate negotiation difficulties")
    elif true_success_rate < 0.8:
        logger.info("📈 Moderate success rate - room for improvement")
    else:
        logger.info("🎉 Good success rate!")
    
    # Model-specific recommendations
    if true_success_rate < 0.8:
        logger.info("\n💡 RECOMMENDATIONS:")
        logger.info("-" * 20)
        
        if 'buyer_model' in df.columns:
            # Find best and worst models
            buyer_success = df.groupby('buyer_model')['true_success'].mean()
            best_buyer = buyer_success.idxmax()
            worst_buyer = buyer_success.idxmin()
            
            logger.info(f"   Best buyer model: {best_buyer.replace(':latest', '').replace('-remote', '')} ({buyer_success.max():.1%} success)")
            logger.info(f"   Worst buyer model: {worst_buyer.replace(':latest', '').replace('-remote', '')} ({buyer_success.min():.1%} success)")
            
            supplier_success = df.groupby('supplier_model')['true_success'].mean()
            best_supplier = supplier_success.idxmax()
            worst_supplier = supplier_success.idxmin()
            
            logger.info(f"   Best supplier model: {best_supplier.replace(':latest', '').replace('-remote', '')} ({supplier_success.max():.1%} success)")
            logger.info(f"   Worst supplier model: {worst_supplier.replace(':latest', '').replace('-remote', '')} ({supplier_success.min():.1%} success)")

if __name__ == "__main__":
    analyze_success_rates()