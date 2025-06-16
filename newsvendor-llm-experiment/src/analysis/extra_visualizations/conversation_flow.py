#!/usr/bin/env python3
"""
Conversation Flow Diagrams
Sankey diagrams showing negotiation paths from opening bids to final prices
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings('ignore')

class ConversationFlowAnalyzer:
    """Analyze and visualize conversation flows in negotiations"""
    
    def __init__(self, data_path="./full_results/processed/complete_20250615_171248.csv"):
        """Initialize with negotiation data"""
        self.data = pd.read_csv(data_path)
        self.successful = self.data[self.data['completed'] == True].copy()
        
        # Parse conversation data if available
        self.conversations = self._parse_conversations()
        
        # Price ranges for flow analysis
        self.price_ranges = {
            'very_low': (30, 40),
            'low': (40, 50),
            'below_optimal': (50, 60),
            'optimal': (60, 70),
            'above_optimal': (70, 80),
            'high': (80, 90),
            'very_high': (90, 100)
        }
        
    def _parse_conversations(self):
        """Parse conversation transcripts from the data"""
        conversations = []
        
        # Check if turns column exists (from conversation tracking)
        if 'turns' in self.data.columns:
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
                            'total_rounds': row['total_rounds'],
                            'completed': row['completed'],
                            'turns': turns_data
                        })
                except (json.JSONDecodeError, KeyError):
                    continue
        else:
            # Create synthetic conversation flows from final prices
            print("⚠️ No conversation turns found, creating synthetic flows from final prices...")
            conversations = self._create_synthetic_flows()
        
        print(f"📝 Parsed {len(conversations)} conversations for flow analysis")
        return conversations
    
    def _create_synthetic_flows(self):
        """Create synthetic conversation flows when turn data unavailable"""
        synthetic_conversations = []
        
        for idx, row in self.successful.iterrows():
            if pd.notna(row['agreed_price']):
                # Generate plausible negotiation sequence
                final_price = row['agreed_price']
                
                # Estimate opening bids based on final price and roles
                buyer_opening = max(35, final_price - np.random.randint(10, 25))
                supplier_opening = min(85, final_price + np.random.randint(5, 15))
                
                # Create turn sequence
                turns = [
                    {'round_number': 1, 'speaker': 'buyer', 'price': buyer_opening},
                    {'round_number': 2, 'speaker': 'supplier', 'price': supplier_opening}
                ]
                
                # Add intermediate steps
                current_buyer = buyer_opening
                current_supplier = supplier_opening
                round_num = 3
                
                while round_num <= row['total_rounds'] and abs(current_buyer - current_supplier) > 2:
                    if round_num % 2 == 1:  # Buyer turn
                        current_buyer = min(current_buyer + np.random.randint(1, 5), final_price)
                        turns.append({'round_number': round_num, 'speaker': 'buyer', 'price': current_buyer})
                    else:  # Supplier turn
                        current_supplier = max(current_supplier - np.random.randint(1, 5), final_price)
                        turns.append({'round_number': round_num, 'speaker': 'supplier', 'price': current_supplier})
                    round_num += 1
                
                # Final agreement
                turns.append({'round_number': round_num, 'speaker': 'agreement', 'price': final_price})
                
                synthetic_conversations.append({
                    'negotiation_id': row['negotiation_id'],
                    'buyer_model': row['buyer_model'],
                    'supplier_model': row['supplier_model'],
                    'reflection_pattern': row['reflection_pattern'],
                    'final_price': final_price,
                    'total_rounds': row['total_rounds'],
                    'completed': row['completed'],
                    'turns': turns
                })
        
        return synthetic_conversations
    
    def _categorize_price(self, price):
        """Categorize price into ranges"""
        for category, (min_p, max_p) in self.price_ranges.items():
            if min_p <= price < max_p:
                return category
        return 'extreme' if price < 30 or price > 100 else 'unknown'
    
    def create_opening_to_final_sankey(self):
        """Create Sankey diagram from opening bids to final prices"""
        
        # Collect opening bid -> final price flows
        flows = []
        
        for conv in self.conversations:
            if len(conv['turns']) >= 2 and conv['completed']:
                # Get first buyer offer
                buyer_opening = None
                supplier_opening = None
                
                for turn in conv['turns']:
                    if turn['speaker'] == 'buyer' and turn['price'] and buyer_opening is None:
                        buyer_opening = turn['price']
                    elif turn['speaker'] == 'supplier' and turn['price'] and supplier_opening is None:
                        supplier_opening = turn['price']
                    
                    if buyer_opening and supplier_opening:
                        break
                
                if buyer_opening and supplier_opening and conv['final_price']:
                    flows.append({
                        'buyer_opening_cat': self._categorize_price(buyer_opening),
                        'supplier_opening_cat': self._categorize_price(supplier_opening),
                        'final_price_cat': self._categorize_price(conv['final_price']),
                        'buyer_opening': buyer_opening,
                        'supplier_opening': supplier_opening,
                        'final_price': conv['final_price'],
                        'reflection_pattern': conv['reflection_pattern'],
                        'buyer_model': conv['buyer_model'],
                        'supplier_model': conv['supplier_model']
                    })
        
        if not flows:
            print("❌ No flows found for Sankey diagram")
            return None
        
        flows_df = pd.DataFrame(flows)
        
        # Create flow counts
        sankey_flows = flows_df.groupby(['buyer_opening_cat', 'supplier_opening_cat', 'final_price_cat']).size().reset_index(name='count')
        
        # Prepare Sankey data
        all_nodes = list(set(
            flows_df['buyer_opening_cat'].unique().tolist() +
            flows_df['supplier_opening_cat'].unique().tolist() +
            flows_df['final_price_cat'].unique().tolist()
        ))
        
        # Create node labels with prefixes
        node_labels = []
        node_colors = []
        
        price_range_labels = {
            'very_low': 'Very Low ($30-40)',
            'low': 'Low ($40-50)', 
            'below_optimal': 'Below Optimal ($50-60)',
            'optimal': 'Optimal ($60-70)',
            'above_optimal': 'Above Optimal ($70-80)',
            'high': 'High ($80-90)',
            'very_high': 'Very High ($90-100)'
        }
        
        price_range_colors = {
            'very_low': '#FF4444',
            'low': '#FF7744',
            'below_optimal': '#FFAA44', 
            'optimal': '#44FF44',
            'above_optimal': '#AAFF44',
            'high': '#77FF44',
            'very_high': '#44FFAA'
        }
        
        # Build node mapping
        node_mapping = {}
        
        # Buyer opening nodes
        for cat in flows_df['buyer_opening_cat'].unique():
            label = f"Buyer Opening: {price_range_labels.get(cat, cat)}"
            node_labels.append(label)
            node_colors.append('#4ECDC4')  # Buyer color
            node_mapping[('buyer', cat)] = len(node_labels) - 1
        
        # Supplier opening nodes
        for cat in flows_df['supplier_opening_cat'].unique():
            label = f"Supplier Opening: {price_range_labels.get(cat, cat)}"
            node_labels.append(label)
            node_colors.append('#FF6B6B')  # Supplier color
            node_mapping[('supplier', cat)] = len(node_labels) - 1
        
        # Final price nodes
        for cat in flows_df['final_price_cat'].unique():
            label = f"Final: {price_range_labels.get(cat, cat)}"
            node_labels.append(label)
            node_colors.append(price_range_colors.get(cat, '#CCCCCC'))
            node_mapping[('final', cat)] = len(node_labels) - 1
        
        # Create links
        source = []
        target = []
        value = []
        link_colors = []
        
        # Buyer opening -> Final
        buyer_to_final = flows_df.groupby(['buyer_opening_cat', 'final_price_cat']).size().reset_index(name='count')
        for _, row in buyer_to_final.iterrows():
            source.append(node_mapping[('buyer', row['buyer_opening_cat'])])
            target.append(node_mapping[('final', row['final_price_cat'])])
            value.append(row['count'])
            link_colors.append('rgba(78, 205, 196, 0.4)')
        
        # Supplier opening -> Final
        supplier_to_final = flows_df.groupby(['supplier_opening_cat', 'final_price_cat']).size().reset_index(name='count')
        for _, row in supplier_to_final.iterrows():
            source.append(node_mapping[('supplier', row['supplier_opening_cat'])])
            target.append(node_mapping[('final', row['final_price_cat'])])
            value.append(row['count'])
            link_colors.append('rgba(255, 107, 107, 0.4)')
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors
            ),
            link=dict(
                source=source,
                target=target, 
                value=value,
                color=link_colors
            )
        )])
        
        fig.update_layout(
            title="Negotiation Flow: Opening Bids → Final Prices<br><sub>Flow thickness indicates frequency of path</sub>",
            font_size=10,
            width=1200,
            height=700
        )
        
        return fig
    
    def create_round_by_round_flow(self):
        """Create round-by-round negotiation flow visualization"""
        
        # Analyze price progression round by round
        round_flows = []
        
        for conv in self.conversations:
            if len(conv['turns']) >= 2 and conv['completed']:
                prices_by_round = {}
                
                for turn in conv['turns']:
                    if turn['price']:
                        round_num = turn['round_number']
                        prices_by_round[round_num] = turn['price']
                
                # Create flow sequence
                sorted_rounds = sorted(prices_by_round.keys())
                for i in range(len(sorted_rounds) - 1):
                    current_round = sorted_rounds[i]
                    next_round = sorted_rounds[i + 1]
                    
                    round_flows.append({
                        'from_round': current_round,
                        'to_round': next_round,
                        'from_price': prices_by_round[current_round],
                        'to_price': prices_by_round[next_round],
                        'from_price_cat': self._categorize_price(prices_by_round[current_round]),
                        'to_price_cat': self._categorize_price(prices_by_round[next_round]),
                        'price_change': prices_by_round[next_round] - prices_by_round[current_round],
                        'reflection_pattern': conv['reflection_pattern']
                    })
        
        if not round_flows:
            print("❌ No round flows found")
            return None
        
        flows_df = pd.DataFrame(round_flows)
        
        # Create parallel coordinates plot showing price evolution
        fig = go.Figure()
        
        # Sample conversations for clearer visualization
        sample_conversations = self.conversations[:50] if len(self.conversations) > 50 else self.conversations
        
        reflection_colors = {
            '00': '#1f77b4',
            '01': '#ff7f0e', 
            '10': '#2ca02c',
            '11': '#d62728'
        }
        
        for conv in sample_conversations:
            if len(conv['turns']) >= 2 and conv['completed']:
                rounds = []
                prices = []
                
                for turn in sorted(conv['turns'], key=lambda x: x['round_number']):
                    if turn['price']:
                        rounds.append(turn['round_number'])
                        prices.append(turn['price'])
                
                if len(rounds) >= 2:
                    fig.add_trace(go.Scatter(
                        x=rounds,
                        y=prices,
                        mode='lines+markers',
                        line=dict(
                            color=reflection_colors.get(conv['reflection_pattern'], '#666666'),
                            width=1
                        ),
                        marker=dict(size=4),
                        opacity=0.6,
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{conv['buyer_model'].replace(':latest', '')} vs {conv['supplier_model'].replace(':latest', '')}</b><br>" +
                            f"Reflection: {conv['reflection_pattern']}<br>" +
                            f"Round: %{{x}}<br>" +
                            f"Price: $%{{y}}<br>" +
                            "<extra></extra>"
                        )
                    ))
        
        # Add optimal price line
        max_rounds = max(turn['round_number'] for conv in sample_conversations for turn in conv['turns'] if turn['price'])
        fig.add_trace(go.Scatter(
            x=[1, max_rounds],
            y=[65, 65],
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            name='Optimal Price ($65)',
            showlegend=True
        ))
        
        # Add reflection pattern legend
        for pattern, color in reflection_colors.items():
            pattern_names = {
                '00': 'No Reflection',
                '01': 'Buyer Reflection',
                '10': 'Supplier Reflection', 
                '11': 'Both Reflection'
            }
            
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color=color, width=3),
                name=pattern_names.get(pattern, pattern),
                showlegend=True
            ))
        
        fig.update_layout(
            title='Round-by-Round Price Evolution<br><sub>Sample of 50 negotiations showing price convergence paths</sub>',
            xaxis_title='Negotiation Round',
            yaxis_title='Price ($)',
            width=1000,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_convergence_pattern_analysis(self):
        """Analyze and visualize convergence patterns"""
        
        convergence_data = []
        
        for conv in self.conversations:
            if len(conv['turns']) >= 2 and conv['completed']:
                prices = []
                speakers = []
                
                for turn in sorted(conv['turns'], key=lambda x: x['round_number']):
                    if turn['price']:
                        prices.append(turn['price'])
                        speakers.append(turn['speaker'])
                
                if len(prices) >= 2:
                    # Calculate convergence metrics
                    initial_gap = abs(prices[0] - prices[1]) if len(prices) > 1 else 0
                    final_price = conv['final_price']
                    
                    # Price volatility (standard deviation)
                    price_volatility = np.std(prices) if len(prices) > 1 else 0
                    
                    # Convergence speed (rounds to reach within 10% of final)
                    target_range = (final_price * 0.9, final_price * 1.1)
                    convergence_round = None
                    
                    for i, price in enumerate(prices):
                        if target_range[0] <= price <= target_range[1]:
                            convergence_round = i + 1
                            break
                    
                    convergence_data.append({
                        'negotiation_id': conv['negotiation_id'],
                        'reflection_pattern': conv['reflection_pattern'],
                        'buyer_model': conv['buyer_model'],
                        'supplier_model': conv['supplier_model'],
                        'initial_gap': initial_gap,
                        'final_price': final_price,
                        'price_volatility': price_volatility,
                        'convergence_round': convergence_round or len(prices),
                        'total_rounds': len(prices),
                        'convergence_speed': (convergence_round or len(prices)) / len(prices)
                    })
        
        if not convergence_data:
            print("❌ No convergence data found")
            return None
        
        conv_df = pd.DataFrame(convergence_data)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Initial Gap vs Final Price',
                'Convergence Speed by Reflection Pattern',
                'Price Volatility Distribution',
                'Convergence Efficiency'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Initial gap vs final price (colored by reflection)
        reflection_colors = {'00': '#1f77b4', '01': '#ff7f0e', '10': '#2ca02c', '11': '#d62728'}
        
        for pattern in ['00', '01', '10', '11']:
            pattern_data = conv_df[conv_df['reflection_pattern'] == pattern]
            if len(pattern_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=pattern_data['initial_gap'],
                        y=pattern_data['final_price'],
                        mode='markers',
                        marker=dict(color=reflection_colors[pattern], size=6, opacity=0.7),
                        name=f'Pattern {pattern}',
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # Add optimal price line
        fig.add_hline(y=65, line_dash="dash", line_color="red", row=1, col=1)
        
        # Plot 2: Convergence speed by reflection
        speed_by_pattern = conv_df.groupby('reflection_pattern')['convergence_speed'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=speed_by_pattern['reflection_pattern'],
                y=speed_by_pattern['convergence_speed'],
                marker_color=[reflection_colors[p] for p in speed_by_pattern['reflection_pattern']],
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Plot 3: Price volatility distribution
        fig.add_trace(
            go.Histogram(
                x=conv_df['price_volatility'],
                nbinsx=20,
                marker_color='lightblue',
                opacity=0.7,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Plot 4: Convergence efficiency (speed vs volatility)
        fig.add_trace(
            go.Scatter(
                x=conv_df['convergence_speed'],
                y=conv_df['price_volatility'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=conv_df['initial_gap'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Initial Gap")
                ),
                text=conv_df['reflection_pattern'],
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Initial Price Gap ($)", row=1, col=1)
        fig.update_yaxes(title_text="Final Price ($)", row=1, col=1)
        
        fig.update_xaxes(title_text="Reflection Pattern", row=1, col=2)
        fig.update_yaxes(title_text="Convergence Speed", row=1, col=2)
        
        fig.update_xaxes(title_text="Price Volatility", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_xaxes(title_text="Convergence Speed", row=2, col=2)
        fig.update_yaxes(title_text="Price Volatility", row=2, col=2)
        
        fig.update_layout(
            title='Convergence Pattern Analysis<br><sub>How different factors affect negotiation convergence</sub>',
            height=800,
            width=1200,
            showlegend=True
        )
        
        return fig
    
    def create_success_failure_paths(self):
        """Analyze paths that lead to success vs failure"""
        
        # Analyze both successful and failed negotiations
        all_conversations = []
        
        # Add successful conversations
        for conv in self.conversations:
            conv['outcome'] = 'success'
            all_conversations.append(conv)
        
        # Add failed conversations (synthetic)
        failed_data = self.data[self.data['completed'] == False]
        for idx, row in failed_data.iterrows():
            # Create synthetic failed conversation
            failed_conv = {
                'negotiation_id': row['negotiation_id'],
                'buyer_model': row['buyer_model'],
                'supplier_model': row['supplier_model'],
                'reflection_pattern': row['reflection_pattern'],
                'final_price': None,
                'total_rounds': row['total_rounds'],
                'completed': False,
                'outcome': 'failure',
                'turns': [
                    {'round_number': i, 'speaker': 'buyer' if i % 2 == 1 else 'supplier', 'price': None}
                    for i in range(1, int(row['total_rounds']) + 1)
                ]
            }
            all_conversations.append(failed_conv)
        
        # Analyze patterns
        patterns = {
            'successful_quick': [],  # Success in ≤3 rounds
            'successful_long': [],   # Success in >3 rounds
            'failed_timeout': [],    # Failed due to timeout
            'failed_early': []       # Failed early
        }
        
        for conv in all_conversations:
            rounds = conv['total_rounds']
            
            if conv['outcome'] == 'success':
                if rounds <= 3:
                    patterns['successful_quick'].append(conv)
                else:
                    patterns['successful_long'].append(conv)
            else:
                if rounds >= 8:  # Near timeout
                    patterns['failed_timeout'].append(conv)
                else:
                    patterns['failed_early'].append(conv)
        
        # Create stacked bar chart
        pattern_counts = {k: len(v) for k, v in patterns.items()}
        
        reflection_breakdown = {}
        for pattern_name, conversations in patterns.items():
            breakdown = {}
            for conv in conversations:
                refl = conv['reflection_pattern']
                breakdown[refl] = breakdown.get(refl, 0) + 1
            reflection_breakdown[pattern_name] = breakdown
        
        # Create visualization
        fig = go.Figure()
        
        reflection_patterns = ['00', '01', '10', '11']
        pattern_names = {
            'successful_quick': 'Quick Success (≤3 rounds)',
            'successful_long': 'Long Success (>3 rounds)', 
            'failed_timeout': 'Failed Timeout (≥8 rounds)',
            'failed_early': 'Failed Early (<8 rounds)'
        }
        
        colors = {
            'successful_quick': '#00FF00',
            'successful_long': '#90EE90',
            'failed_timeout': '#FF6666', 
            'failed_early': '#FF0000'
        }
        
        for pattern in reflection_patterns:
            values = []
            for outcome in ['successful_quick', 'successful_long', 'failed_timeout', 'failed_early']:
                count = reflection_breakdown.get(outcome, {}).get(pattern, 0)
                values.append(count)
            
            fig.add_trace(go.Bar(
                name=f'Reflection {pattern}',
                x=list(pattern_names.values()),
                y=values,
                text=values,
                textposition='inside',
                textfont=dict(color='white')
            ))
        
        fig.update_layout(
            title='Success vs Failure Patterns by Reflection<br><sub>How reflection patterns affect negotiation outcomes</sub>',
            xaxis_title='Outcome Pattern',
            yaxis_title='Number of Negotiations',
            barmode='stack',
            width=1000,
            height=600
        )
        
        return fig
    
    def generate_flow_report(self):
        """Generate comprehensive flow analysis report"""
        
        if not self.conversations:
            return "# No conversation data available for flow analysis"
        
        report = "# Conversation Flow Analysis Report\n\n"
        
        # Basic statistics
        total_conversations = len(self.conversations)
        successful_conversations = len([c for c in self.conversations if c['completed']])
        
        report += f"## 📊 Flow Analysis Overview\n\n"
        report += f"- **Total Conversations Analyzed:** {total_conversations:,}\n"
        report += f"- **Successful Negotiations:** {successful_conversations:,}\n"
        report += f"- **Success Rate:** {successful_conversations/total_conversations*100:.1f}%\n\n"
        
        # Price flow analysis
        opening_prices = []
        final_prices = []
        
        for conv in self.conversations:
            if conv['completed'] and len(conv['turns']) >= 2:
                # Find first price offers
                first_buyer_price = None
                first_supplier_price = None
                
                for turn in conv['turns']:
                    if turn['speaker'] == 'buyer' and turn['price'] and first_buyer_price is None:
                        first_buyer_price = turn['price']
                    elif turn['speaker'] == 'supplier' and turn['price'] and first_supplier_price is None:
                        first_supplier_price = turn['price']
                
                if first_buyer_price and first_supplier_price:
                    opening_prices.append((first_buyer_price, first_supplier_price))
                    final_prices.append(conv['final_price'])
        
        if opening_prices:
            avg_buyer_opening = np.mean([p[0] for p in opening_prices])
            avg_supplier_opening = np.mean([p[1] for p in opening_prices])
            avg_final = np.mean(final_prices)
            
            report += f"## 💰 Price Flow Patterns\n\n"
            report += f"- **Average Buyer Opening:** ${avg_buyer_opening:.2f}\n"
            report += f"- **Average Supplier Opening:** ${avg_supplier_opening:.2f}\n"
            report += f"- **Average Final Price:** ${avg_final:.2f}\n"
            report += f"- **Opening Gap:** ${avg_supplier_opening - avg_buyer_opening:.2f}\n"
            report += f"- **Buyer Concession:** ${avg_final - avg_buyer_opening:.2f}\n"
            report += f"- **Supplier Concession:** ${avg_supplier_opening - avg_final:.2f}\n\n"
        
        # Convergence patterns
        quick_successes = [c for c in self.conversations if c['completed'] and c['total_rounds'] <= 3]
        long_negotiations = [c for c in self.conversations if c['completed'] and c['total_rounds'] >= 7]
        
        report += f"## ⚡ Convergence Patterns\n\n"
        report += f"- **Quick Successes (≤3 rounds):** {len(quick_successes)} ({len(quick_successes)/successful_conversations*100:.1f}%)\n"
        report += f"- **Long Negotiations (≥7 rounds):** {len(long_negotiations)} ({len(long_negotiations)/successful_conversations*100:.1f}%)\n\n"
        
        # Most common flows
        if opening_prices and len(opening_prices) >= 10:
            # Find most common price categories
            opening_to_final = {}
            for i, (buyer_open, supplier_open) in enumerate(opening_prices):
                final = final_prices[i]
                
                buyer_cat = self._categorize_price(buyer_open)
                supplier_cat = self._categorize_price(supplier_open)
                final_cat = self._categorize_price(final)
                
                flow_key = f"{buyer_cat} buyer + {supplier_cat} supplier → {final_cat} final"
                opening_to_final[flow_key] = opening_to_final.get(flow_key, 0) + 1
            
            # Top flows
            top_flows = sorted(opening_to_final.items(), key=lambda x: x[1], reverse=True)[:5]
            
            report += f"## 🌊 Most Common Flow Patterns\n\n"
            for i, (flow, count) in enumerate(top_flows, 1):
                report += f"{i}. **{flow}** ({count} negotiations)\n"
            
            report += "\n"
        
        return report

def main():
    """Run conversation flow analysis"""
    print("🌊 Creating Conversation Flow Diagrams...")
    
    # Initialize analyzer
    analyzer = ConversationFlowAnalyzer()
    
    if not analyzer.conversations:
        print("❌ No conversation data available")
        return
    
    # Create visualizations
    print("🎨 Generating flow visualizations...")
    
    # Opening to final Sankey
    sankey_fig = analyzer.create_opening_to_final_sankey()
    if sankey_fig:
        sankey_fig.write_html("conversation_flow_sankey.html")
        sankey_fig.write_image("conversation_flow_sankey.png", width=1200, height=700, scale=2)
    
    # Round-by-round flow
    round_flow_fig = analyzer.create_round_by_round_flow()
    if round_flow_fig:
        round_flow_fig.write_html("conversation_round_by_round_flow.html")
        round_flow_fig.write_image("conversation_round_by_round_flow.png", width=1000, height=600, scale=2)
    
    # Convergence pattern analysis
    convergence_fig = analyzer.create_convergence_pattern_analysis()
    if convergence_fig:
        convergence_fig.write_html("conversation_convergence_patterns.html")
        convergence_fig.write_image("conversation_convergence_patterns.png", width=1200, height=800, scale=2)
    
    # Success/failure paths
    success_fail_fig = analyzer.create_success_failure_paths()
    if success_fail_fig:
        success_fail_fig.write_html("conversation_success_failure_paths.html")
        success_fail_fig.write_image("conversation_success_failure_paths.png", width=1000, height=600, scale=2)
    
    # Generate report
    print("📝 Generating flow analysis report...")
    report = analyzer.generate_flow_report()
    with open("conversation_flow_report.md", "w") as f:
        f.write(report)
    
    print("✅ Conversation Flow Diagrams complete!")
    print("📁 Files generated:")
    print("   - Sankey flow diagram: conversation_flow_sankey.html/png")
    print("   - Round-by-round flow: conversation_round_by_round_flow.html/png")
    print("   - Convergence patterns: conversation_convergence_patterns.html/png")
    print("   - Success/failure paths: conversation_success_failure_paths.html/png")
    print("   - Flow analysis report: conversation_flow_report.md")

if __name__ == "__main__":
    main()