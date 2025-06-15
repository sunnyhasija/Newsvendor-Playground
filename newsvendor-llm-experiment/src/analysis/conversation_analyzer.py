#!/usr/bin/env python3
"""
Conversation Analysis Extension - src/analysis/conversation_analyzer.py
Adds turn-by-turn behavioral analysis to your existing framework
"""

import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class ConversationAnalyzer:
    """Analyze turn-by-turn negotiation behavior from conversation transcripts"""
    
    def __init__(self, data_path: str):
        """Initialize with your dataset"""
        self.data = pd.read_csv(data_path)
        self.conversations = self._parse_conversations()
        
    def _parse_conversations(self) -> List[Dict]:
        """Parse conversation transcripts from the turns column"""
        conversations = []
        
        for idx, row in self.data.iterrows():
            try:
                if pd.notna(row['turns']) and isinstance(row['turns'], str):
                    turns_data = json.loads(row['turns'])
                    conversations.append({
                        'negotiation_id': row['negotiation_id'],
                        'buyer_model': row['buyer_model'],
                        'supplier_model': row['supplier_model'],
                        'reflection_pattern': row['reflection_pattern'],
                        'final_price': row['agreed_price'],
                        'turns': turns_data
                    })
            except (json.JSONDecodeError, KeyError):
                continue
                
        print(f"📝 Parsed {len(conversations)} conversations from {len(self.data)} negotiations")
        return conversations
    
    def analyze_opening_strategies(self) -> Dict[str, Any]:
        """Analyze opening bid strategies by model"""
        print("\n💼 OPENING BID STRATEGY ANALYSIS")
        print("="*50)
        
        opening_bids = {'buyer': {}, 'supplier': {}}
        
        for conv in self.conversations:
            if len(conv['turns']) > 0:
                first_turn = conv['turns'][0]
                if first_turn['speaker'] == 'buyer' and first_turn['price']:
                    model = conv['buyer_model']
                    if model not in opening_bids['buyer']:
                        opening_bids['buyer'][model] = []
                    opening_bids['buyer'][model].append(first_turn['price'])
        
        # Analyze opening bid patterns
        results = {}
        for role, model_bids in opening_bids.items():
            results[role] = {}
            for model, bids in model_bids.items():
                if len(bids) > 0:
                    results[role][model] = {
                        'count': len(bids),
                        'mean_opening': np.mean(bids),
                        'std_opening': np.std(bids),
                        'aggressive_rate': sum(1 for b in bids if b < 40) / len(bids)  # Below $40 = aggressive
                    }
        
        # Print results
        if 'buyer' in results:
            print("🛒 BUYER Opening Bid Strategies:")
            for model, stats in results['buyer'].items():
                print(f"   {model:<20} Avg: ${stats['mean_opening']:>5.1f}, "
                      f"Aggressive: {stats['aggressive_rate']*100:>4.1f}%, n={stats['count']}")
        
        return results
    
    def analyze_concession_patterns(self) -> Dict[str, Any]:
        """Analyze how models make concessions during negotiation"""
        print("\n📉 CONCESSION PATTERN ANALYSIS")
        print("="*50)
        
        concession_data = []
        
        for conv in self.conversations:
            if len(conv['turns']) >= 2:
                buyer_prices = []
                supplier_prices = []
                
                for turn in conv['turns']:
                    if turn['price']:
                        if turn['speaker'] == 'buyer':
                            buyer_prices.append(turn['price'])
                        else:
                            supplier_prices.append(turn['price'])
                
                # Calculate concessions
                buyer_concessions = []
                supplier_concessions = []
                
                for i in range(1, len(buyer_prices)):
                    concession = buyer_prices[i] - buyer_prices[i-1]  # Positive = higher offer (concession)
                    buyer_concessions.append(concession)
                
                for i in range(1, len(supplier_prices)):
                    concession = supplier_prices[i-1] - supplier_prices[i]  # Positive = lower ask (concession)
                    supplier_concessions.append(concession)
                
                if buyer_concessions:
                    concession_data.append({
                        'negotiation_id': conv['negotiation_id'],
                        'buyer_model': conv['buyer_model'],
                        'supplier_model': conv['supplier_model'],
                        'reflection_pattern': conv['reflection_pattern'],
                        'role': 'buyer',
                        'avg_concession': np.mean(buyer_concessions),
                        'total_concessions': len(buyer_concessions),
                        'final_price': conv['final_price']
                    })
                
                if supplier_concessions:
                    concession_data.append({
                        'negotiation_id': conv['negotiation_id'],
                        'buyer_model': conv['buyer_model'],
                        'supplier_model': conv['supplier_model'],
                        'reflection_pattern': conv['reflection_pattern'],
                        'role': 'supplier',
                        'avg_concession': np.mean(supplier_concessions),
                        'total_concessions': len(supplier_concessions),
                        'final_price': conv['final_price']
                    })
        
        # Analyze by model
        concession_df = pd.DataFrame(concession_data)
        
        if len(concession_df) > 0:
            print("🛒 BUYER Concession Patterns (positive = moving toward supplier):")
            buyer_concessions = concession_df[concession_df['role'] == 'buyer']
            buyer_stats = buyer_concessions.groupby('buyer_model')['avg_concession'].agg(['mean', 'count']).round(2)
            for model, stats in buyer_stats.iterrows():
                print(f"   {model:<20} Avg concession: ${stats['mean']:>+5.2f}, n={stats['count']}")
            
            print("\n🏭 SUPPLIER Concession Patterns (positive = moving toward buyer):")
            supplier_concessions = concession_df[concession_df['role'] == 'supplier']
            supplier_stats = supplier_concessions.groupby('supplier_model')['avg_concession'].agg(['mean', 'count']).round(2)
            for model, stats in supplier_stats.iterrows():
                print(f"   {model:<20} Avg concession: ${stats['mean']:>+5.2f}, n={stats['count']}")
        
        return concession_data
    
    def analyze_negotiation_language(self) -> Dict[str, Any]:
        """Analyze language patterns in negotiation messages"""
        print("\n💬 LANGUAGE PATTERN ANALYSIS")
        print("="*50)
        
        # Define key phrases and patterns
        aggressive_phrases = ['offer', 'want', 'need', 'demand']
        cooperative_phrases = ['accept', 'agree', 'deal', 'fine', 'okay']
        price_patterns = [r'\$\d+', r'(\d+)\s*dollars?']
        
        language_data = []
        
        for conv in self.conversations:
            for turn in conv['turns']:
                message = turn['message'].lower()
                
                # Count patterns
                aggressive_count = sum(1 for phrase in aggressive_phrases if phrase in message)
                cooperative_count = sum(1 for phrase in cooperative_phrases if phrase in message)
                price_mentions = len(re.findall(r'\$\d+|\d+\s*dollars?', message))
                message_length = len(message.split())
                
                language_data.append({
                    'negotiation_id': conv['negotiation_id'],
                    'buyer_model': conv['buyer_model'],
                    'supplier_model': conv['supplier_model'],
                    'reflection_pattern': conv['reflection_pattern'],
                    'speaker': turn['speaker'],
                    'round_number': turn['round_number'],
                    'aggressive_score': aggressive_count,
                    'cooperative_score': cooperative_count,
                    'price_mentions': price_mentions,
                    'message_length': message_length,
                    'final_price': conv['final_price']
                })
        
        language_df = pd.DataFrame(language_data)
        
        if len(language_df) > 0:
            # Analyze by model and role
            print("📊 Communication Style by Model:")
            for role in ['buyer', 'supplier']:
                role_col = f'{role}_model'
                role_data = language_df[language_df['speaker'] == role]
                
                print(f"\n{role.upper()} Communication Patterns:")
                if len(role_data) > 0:
                    style_stats = role_data.groupby(role_col).agg({
                        'aggressive_score': 'mean',
                        'cooperative_score': 'mean',
                        'message_length': 'mean'
                    }).round(2)
                    
                    for model, stats in style_stats.iterrows():
                        print(f"   {model:<20} Aggressive: {stats['aggressive_score']:>4.2f}, "
                              f"Cooperative: {stats['cooperative_score']:>4.2f}, "
                              f"Avg words: {stats['message_length']:>4.1f}")
        
        return language_data
    
    def analyze_convergence_speed(self) -> Dict[str, Any]:
        """Analyze how quickly different model combinations reach agreement"""
        print("\n⚡ CONVERGENCE SPEED ANALYSIS")
        print("="*50)
        
        convergence_data = []
        
        for conv in self.conversations:
            if len(conv['turns']) >= 2:
                # Extract price sequences
                buyer_prices = [t['price'] for t in conv['turns'] if t['speaker'] == 'buyer' and t['price']]
                supplier_prices = [t['price'] for t in conv['turns'] if t['speaker'] == 'supplier' and t['price']]
                
                # Calculate initial gap and convergence
                if buyer_prices and supplier_prices:
                    initial_gap = max(supplier_prices[0] - buyer_prices[0], 0) if len(supplier_prices) > 0 and len(buyer_prices) > 0 else 0
                    final_gap = abs(conv['final_price'] - ((buyer_prices[-1] + supplier_prices[-1]) / 2)) if conv['final_price'] else 0
                    
                    convergence_data.append({
                        'negotiation_id': conv['negotiation_id'],
                        'buyer_model': conv['buyer_model'],
                        'supplier_model': conv['supplier_model'],
                        'reflection_pattern': conv['reflection_pattern'],
                        'initial_gap': initial_gap,
                        'final_gap': final_gap,
                        'convergence_rate': (initial_gap - final_gap) / max(initial_gap, 1),
                        'total_rounds': len(conv['turns']),
                        'final_price': conv['final_price']
                    })
        
        convergence_df = pd.DataFrame(convergence_data)
        
        if len(convergence_df) > 0:
            print("🏁 Convergence Performance by Model Pairing:")
            pairing_stats = convergence_df.groupby(['buyer_model', 'supplier_model']).agg({
                'convergence_rate': 'mean',
                'total_rounds': 'mean',
                'final_price': 'mean'
            }).round(2)
            
            # Show top 10 fastest converging pairs
            fastest_pairs = pairing_stats.sort_values('total_rounds').head(10)
            
            for (buyer, supplier), stats in fastest_pairs.iterrows():
                print(f"   {buyer[:12]:<12} vs {supplier[:12]:<12} "
                      f"Rounds: {stats['total_rounds']:>4.1f}, "
                      f"Conv Rate: {stats['convergence_rate']:>5.2f}, "
                      f"Price: ${stats['final_price']:>5.1f}")
        
        return convergence_data
    
    def create_conversation_visualizations(self, output_dir: str = "./complete_analysis"):
        """Create visualizations of conversation patterns"""
        print(f"\n📊 Creating conversation visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Negotiation Conversation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Opening bid strategies
        opening_data = self.analyze_opening_strategies()
        if 'buyer' in opening_data and opening_data['buyer']:
            ax1 = axes[0, 0]
            models = list(opening_data['buyer'].keys())
            opening_bids = [opening_data['buyer'][m]['mean_opening'] for m in models]
            
            bars = ax1.bar(range(len(models)), opening_bids, color='lightblue', alpha=0.7)
            ax1.axhline(y=65, color='red', linestyle='--', alpha=0.7, label='Optimal ($65)')
            ax1.set_title('Average Opening Bids by Buyer Model')
            ax1.set_xlabel('Buyer Model')
            ax1.set_ylabel('Opening Bid ($)')
            ax1.set_xticks(range(len(models)))
            ax1.set_xticklabels([m.replace(':latest', '') for m in models], rotation=45, ha='right')
            ax1.legend()
            
            # Add value labels
            for bar, value in zip(bars, opening_bids):
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'${value:.0f}', ha='center', va='bottom')
        
        # 2. Negotiation rounds distribution
        ax2 = axes[0, 1]
        rounds_data = [len(conv['turns']) for conv in self.conversations]
        ax2.hist(rounds_data, bins=range(1, max(rounds_data)+2), alpha=0.7, color='lightgreen')
        ax2.set_title('Distribution of Negotiation Length')
        ax2.set_xlabel('Number of Rounds')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=np.mean(rounds_data), color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean: {np.mean(rounds_data):.1f}')
        ax2.legend()
        
        # 3. Price progression example
        ax3 = axes[1, 0]
        if len(self.conversations) > 0:
            # Show price progression for a few sample negotiations
            for i, conv in enumerate(self.conversations[:5]):
                buyer_prices = []
                supplier_prices = []
                rounds = []
                
                for turn in conv['turns']:
                    if turn['price']:
                        rounds.append(turn['round_number'])
                        if turn['speaker'] == 'buyer':
                            buyer_prices.append(turn['price'])
                            supplier_prices.append(None)
                        else:
                            buyer_prices.append(None)
                            supplier_prices.append(turn['price'])
                
                ax3.plot(rounds, buyer_prices, 'o-', alpha=0.6, label=f'Buyer {i+1}' if i == 0 else "")
                ax3.plot(rounds, supplier_prices, 's-', alpha=0.6, label=f'Supplier {i+1}' if i == 0 else "")
            
            ax3.axhline(y=65, color='red', linestyle='--', alpha=0.7, label='Optimal')
            ax3.set_title('Sample Price Progression')
            ax3.set_xlabel('Round')
            ax3.set_ylabel('Price ($)')
            ax3.legend()
        
        # 4. Model pairing success matrix
        ax4 = axes[1, 1]
        pairing_matrix = pd.DataFrame(index=self.data['buyer_model'].unique(), 
                                    columns=self.data['supplier_model'].unique())
        
        for buyer in pairing_matrix.index:
            for supplier in pairing_matrix.columns:
                avg_price = self.data[(self.data['buyer_model'] == buyer) & 
                                    (self.data['supplier_model'] == supplier)]['agreed_price'].mean()
                pairing_matrix.loc[buyer, supplier] = avg_price
        
        pairing_matrix = pairing_matrix.astype(float)
        sns.heatmap(pairing_matrix, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                   center=65, ax=ax4, cbar_kws={'label': 'Avg Price ($)'})
        ax4.set_title('Model Pairing Performance Matrix')
        ax4.set_xlabel('Supplier Model')
        ax4.set_ylabel('Buyer Model')
        
        plt.tight_layout()
        output_path = f"{output_dir}/conversation_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved to: {output_path}")
        plt.show()
    
    def run_complete_conversation_analysis(self) -> Dict[str, Any]:
        """Run all conversation analyses"""
        print("🎯 COMPREHENSIVE CONVERSATION ANALYSIS")
        print("="*60)
        
        results = {}
        
        if len(self.conversations) == 0:
            print("❌ No conversation data available for analysis")
            return results
        
        print(f"📝 Analyzing {len(self.conversations)} conversations...")
        
        # Run all analyses
        results['opening_strategies'] = self.analyze_opening_strategies()
        results['concession_patterns'] = self.analyze_concession_patterns()
        results['language_patterns'] = self.analyze_negotiation_language()
        results['convergence_speed'] = self.analyze_convergence_speed()
        
        # Create visualizations
        self.create_conversation_visualizations()
        
        print("\n🎉 Conversation analysis complete!")
        return results

def main():
    """Run conversation analysis on your dataset"""
    data_path = "./full_results/processed/complete_20250615_171248.csv"
    
    analyzer = ConversationAnalyzer(data_path)
    results = analyzer.run_complete_conversation_analysis()
    
    return results

if __name__ == '__main__':
    main()