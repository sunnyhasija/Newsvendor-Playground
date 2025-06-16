#!/usr/bin/env python3
"""
LLM Negotiation Tournament Bracket
March Madness style visualization of model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from math import sqrt, log2
import warnings
warnings.filterwarnings('ignore')

class TournamentBracketAnalyzer:
    """Create tournament-style analysis of LLM negotiation performance"""
    
    def __init__(self, data_path="./full_results/processed/complete_20250615_171248.csv"):
        """Initialize with negotiation data"""
        self.data = pd.read_csv(data_path)
        self.successful = self.data[self.data['completed'] == True].copy()
        
        # Model information
        self.models = list(self.data['buyer_model'].unique())
        
        # Model rankings and stats
        self.model_stats = self._calculate_model_performance()
        self.rankings = self._create_rankings()
        
        # Tournament structure
        self.bracket_structure = self._create_bracket_structure()
        
        # Colors for visualization
        self.tier_colors = {
            'tinyllama:latest': '#FF6B6B',
            'qwen2:1.5b': '#FF8E53', 
            'gemma2:2b': '#4ECDC4',
            'phi3:mini': '#45B7D1',
            'llama3.2:latest': '#96CEB4',
            'mistral:instruct': '#FFEAA7',
            'qwen:7b': '#DDA0DD',
            'qwen3:latest': '#98D8C8'
        }
    
    def _calculate_model_performance(self):
        """Calculate comprehensive performance metrics for each model"""
        
        stats = {}
        
        for model in self.models:
            # Combined performance (as buyer and supplier)
            model_data = self.successful[
                (self.successful['buyer_model'] == model) | 
                (self.successful['supplier_model'] == model)
            ]
            
            # Buyer-specific performance
            buyer_data = self.successful[self.successful['buyer_model'] == model]
            buyer_prices = buyer_data['agreed_price'].dropna()
            
            # Supplier-specific performance  
            supplier_data = self.successful[self.successful['supplier_model'] == model]
            supplier_prices = supplier_data['agreed_price'].dropna()
            
            # Overall statistics
            all_prices = model_data['agreed_price'].dropna()
            
            if len(model_data) > 0:
                stats[model] = {
                    # Overall performance
                    'total_negotiations': len(model_data),
                    'success_rate': len(model_data) / len(self.data[
                        (self.data['buyer_model'] == model) | 
                        (self.data['supplier_model'] == model)
                    ]),
                    
                    # Price performance
                    'avg_price': all_prices.mean() if len(all_prices) > 0 else 65,
                    'price_std': all_prices.std() if len(all_prices) > 1 else 0,
                    'distance_from_optimal': abs(all_prices.mean() - 65) if len(all_prices) > 0 else 0,
                    'optimal_rate': sum(abs(all_prices - 65) <= 5) / len(all_prices) if len(all_prices) > 0 else 0,
                    
                    # Efficiency metrics
                    'avg_rounds': model_data['total_rounds'].mean(),
                    'avg_tokens': model_data['total_tokens'].mean(),
                    'tokens_per_round': model_data['total_tokens'].sum() / model_data['total_rounds'].sum(),
                    
                    # Role-specific performance
                    'buyer_advantage': (65 - buyer_prices.mean()) if len(buyer_prices) > 0 else 0,  # Lower price = advantage
                    'supplier_advantage': (supplier_prices.mean() - 65) if len(supplier_prices) > 0 else 0,  # Higher price = advantage
                    
                    # Consistency
                    'consistency_score': 100 - (all_prices.std() * 4) if len(all_prices) > 1 else 100,
                    
                    # Head-to-head records
                    'wins': 0,  # Will be calculated in head-to-head analysis
                    'losses': 0,
                    'win_rate': 0.0
                }
        
        return stats
    
    def _create_rankings(self):
        """Create comprehensive rankings based on multiple criteria"""
        
        rankings = {}
        
        # Overall Performance Score (composite metric)
        overall_scores = {}
        for model, stats in self.model_stats.items():
            # Weighted composite score
            score = (
                stats['success_rate'] * 25 +  # 25% weight on success rate
                (100 - stats['distance_from_optimal'] * 3) * 0.25 +  # 25% weight on price optimality
                (100 - stats['avg_rounds'] * 10) * 0.20 +  # 20% weight on efficiency
                stats['optimal_rate'] * 20 +  # 20% weight on optimal outcomes
                stats['consistency_score'] * 0.10  # 10% weight on consistency
            )
            overall_scores[model] = max(0, score)
        
        rankings['overall'] = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Specific rankings
        rankings['price_optimality'] = sorted(
            [(m, 100 - s['distance_from_optimal'] * 3) for m, s in self.model_stats.items()],
            key=lambda x: x[1], reverse=True
        )
        
        rankings['efficiency'] = sorted(
            [(m, 100 - s['avg_rounds'] * 10) for m, s in self.model_stats.items()],
            key=lambda x: x[1], reverse=True
        )
        
        rankings['consistency'] = sorted(
            [(m, s['consistency_score']) for m, s in self.model_stats.items()],
            key=lambda x: x[1], reverse=True
        )
        
        rankings['buyer_performance'] = sorted(
            [(m, s['buyer_advantage']) for m, s in self.model_stats.items()],
            key=lambda x: x[1], reverse=True
        )
        
        rankings['supplier_performance'] = sorted(
            [(m, s['supplier_advantage']) for m, s in self.model_stats.items()],
            key=lambda x: x[1], reverse=True
        )
        
        return rankings
    
    def _create_bracket_structure(self):
        """Create tournament bracket structure"""
        
        # Use overall rankings for seeding
        seeded_models = [model for model, score in self.rankings['overall']]
        
        # Create bracket rounds
        bracket = {
            'quarterfinals': [
                {'match_id': 'QF1', 'team1': seeded_models[0], 'team2': seeded_models[7], 'winner': None},
                {'match_id': 'QF2', 'team1': seeded_models[3], 'team2': seeded_models[4], 'winner': None},
                {'match_id': 'QF3', 'team1': seeded_models[1], 'team2': seeded_models[6], 'winner': None},
                {'match_id': 'QF4', 'team1': seeded_models[2], 'team2': seeded_models[5], 'winner': None}
            ],
            'semifinals': [
                {'match_id': 'SF1', 'team1': None, 'team2': None, 'winner': None},
                {'match_id': 'SF2', 'team1': None, 'team2': None, 'winner': None}
            ],
            'final': [
                {'match_id': 'F1', 'team1': None, 'team2': None, 'winner': None}
            ]
        }
        
        # Simulate matchups based on head-to-head performance
        self._simulate_tournament(bracket)
        
        return bracket
    
    def _simulate_tournament(self, bracket):
        """Simulate tournament matchups based on actual performance data"""
        
        # Quarterfinals
        for match in bracket['quarterfinals']:
            winner = self._simulate_matchup(match['team1'], match['team2'])
            match['winner'] = winner
        
        # Semifinals
        bracket['semifinals'][0]['team1'] = bracket['quarterfinals'][0]['winner']
        bracket['semifinals'][0]['team2'] = bracket['quarterfinals'][1]['winner']
        bracket['semifinals'][1]['team1'] = bracket['quarterfinals'][2]['winner']
        bracket['semifinals'][1]['team2'] = bracket['quarterfinals'][3]['winner']
        
        for match in bracket['semifinals']:
            winner = self._simulate_matchup(match['team1'], match['team2'])
            match['winner'] = winner
        
        # Final
        bracket['final'][0]['team1'] = bracket['semifinals'][0]['winner']
        bracket['final'][0]['team2'] = bracket['semifinals'][1]['winner']
        bracket['final'][0]['winner'] = self._simulate_matchup(
            bracket['final'][0]['team1'], 
            bracket['final'][0]['team2']
        )
    
    def _simulate_matchup(self, model1, model2):
        """Simulate head-to-head matchup based on actual negotiation data"""
        
        # Find actual negotiations between these models
        head_to_head = self.successful[
            ((self.successful['buyer_model'] == model1) & (self.successful['supplier_model'] == model2)) |
            ((self.successful['buyer_model'] == model2) & (self.successful['supplier_model'] == model1))
        ]
        
        if len(head_to_head) > 0:
            # Use actual performance
            model1_scores = []
            model2_scores = []
            
            for _, neg in head_to_head.iterrows():
                price = neg['agreed_price']
                
                # Score based on how close to optimal and role advantage
                if neg['buyer_model'] == model1:
                    # Model1 as buyer - lower price is better
                    model1_score = (100 - price) + (abs(65 - price) < 5) * 10
                    model2_score = price + (abs(65 - price) < 5) * 10
                else:
                    # Model2 as buyer - lower price is better
                    model2_score = (100 - price) + (abs(65 - price) < 5) * 10
                    model1_score = price + (abs(65 - price) < 5) * 10
                
                model1_scores.append(model1_score)
                model2_scores.append(model2_score)
            
            avg1 = np.mean(model1_scores)
            avg2 = np.mean(model2_scores)
            
            return model1 if avg1 > avg2 else model2
        
        else:
            # Use overall performance scores
            score1 = dict(self.rankings['overall'])[model1]
            score2 = dict(self.rankings['overall'])[model2]
            return model1 if score1 > score2 else model2
    
    def create_tournament_bracket_viz(self):
        """Create visual tournament bracket"""
        
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # Colors for winners and losers
        winner_color = '#4CAF50'
        loser_color = '#FFCDD2'
        
        # Draw quarterfinals
        qf_y_positions = [1, 2.5, 5.5, 7]
        for i, match in enumerate(self.bracket_structure['quarterfinals']):
            y = qf_y_positions[i]
            
            # Team boxes
            team1_color = winner_color if match['winner'] == match['team1'] else loser_color
            team2_color = winner_color if match['winner'] == match['team2'] else loser_color
            
            # Team 1
            rect1 = patches.Rectangle((0.5, y), 1.5, 0.4, linewidth=1, 
                                    edgecolor='black', facecolor=team1_color)
            ax.add_patch(rect1)
            ax.text(1.25, y + 0.2, match['team1'].replace(':latest', '')[:8], 
                   ha='center', va='center', fontsize=8, weight='bold')
            
            # Team 2  
            rect2 = patches.Rectangle((0.5, y - 0.5), 1.5, 0.4, linewidth=1,
                                    edgecolor='black', facecolor=team2_color)
            ax.add_patch(rect2)
            ax.text(1.25, y - 0.3, match['team2'].replace(':latest', '')[:8],
                   ha='center', va='center', fontsize=8, weight='bold')
            
            # Connection line to semifinals
            ax.plot([2, 3], [y - 0.05, y - 0.05], 'k-', linewidth=2)
        
        # Draw semifinals
        sf_y_positions = [1.75, 6.25]
        for i, match in enumerate(self.bracket_structure['semifinals']):
            y = sf_y_positions[i]
            
            # Team boxes
            team1_color = winner_color if match['winner'] == match['team1'] else loser_color
            team2_color = winner_color if match['winner'] == match['team2'] else loser_color
            
            # Team 1
            rect1 = patches.Rectangle((3.5, y), 1.5, 0.4, linewidth=1,
                                    edgecolor='black', facecolor=team1_color)
            ax.add_patch(rect1)
            ax.text(4.25, y + 0.2, match['team1'].replace(':latest', '')[:8],
                   ha='center', va='center', fontsize=8, weight='bold')
            
            # Team 2
            rect2 = patches.Rectangle((3.5, y - 0.5), 1.5, 0.4, linewidth=1,
                                    edgecolor='black', facecolor=team2_color)
            ax.add_patch(rect2)
            ax.text(4.25, y - 0.3, match['team2'].replace(':latest', '')[:8],
                   ha='center', va='center', fontsize=8, weight='bold')
            
            # Connection line to final
            ax.plot([5, 6], [y - 0.05, y - 0.05], 'k-', linewidth=2)
        
        # Draw final
        final_match = self.bracket_structure['final'][0]
        y = 4
        
        # Championship teams
        team1_color = winner_color if final_match['winner'] == final_match['team1'] else loser_color
        team2_color = winner_color if final_match['winner'] == final_match['team2'] else loser_color
        
        # Team 1
        rect1 = patches.Rectangle((6.5, y), 1.5, 0.4, linewidth=2,
                                edgecolor='gold', facecolor=team1_color)
        ax.add_patch(rect1)
        ax.text(7.25, y + 0.2, final_match['team1'].replace(':latest', '')[:8],
               ha='center', va='center', fontsize=9, weight='bold')
        
        # Team 2
        rect2 = patches.Rectangle((6.5, y - 0.5), 1.5, 0.4, linewidth=2,
                                edgecolor='gold', facecolor=team2_color)
        ax.add_patch(rect2)
        ax.text(7.25, y - 0.3, final_match['team2'].replace(':latest', '')[:8],
               ha='center', va='center', fontsize=9, weight='bold')
        
        # Champion
        champion_rect = patches.Rectangle((8.5, y - 0.05), 1.2, 0.3, linewidth=3,
                                        edgecolor='gold', facecolor='#FFD700')
        ax.add_patch(champion_rect)
        ax.text(9.1, y + 0.1, '👑 ' + final_match['winner'].replace(':latest', '')[:6],
               ha='center', va='center', fontsize=10, weight='bold')
        
        # Labels
        ax.text(1.25, 0.2, 'QUARTERFINALS', ha='center', va='center', 
               fontsize=12, weight='bold')
        ax.text(4.25, 0.2, 'SEMIFINALS', ha='center', va='center',
               fontsize=12, weight='bold')
        ax.text(7.25, 0.2, 'FINAL', ha='center', va='center',
               fontsize=12, weight='bold')
        ax.text(9.1, 0.2, 'CHAMPION', ha='center', va='center',
               fontsize=12, weight='bold')
        
        plt.title('LLM Negotiation Tournament Bracket\n"March AI-ness" Championship', 
                 fontsize=16, weight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_bracket(self):
        """Create interactive tournament bracket with Plotly"""
        
        fig = go.Figure()
        
        # Define positions for interactive bracket
        rounds = {
            'quarterfinals': {'x': 1, 'matches': self.bracket_structure['quarterfinals']},
            'semifinals': {'x': 3, 'matches': self.bracket_structure['semifinals']},
            'final': {'x': 5, 'matches': self.bracket_structure['final']}
        }
        
        y_positions = {
            'quarterfinals': [1, 3, 5, 7],
            'semifinals': [2, 6],
            'final': [4]
        }
        
        # Add matches to plot
        for round_name, round_data in rounds.items():
            x = round_data['x']
            matches = round_data['matches']
            y_pos = y_positions[round_name]
            
            for i, match in enumerate(matches):
                y = y_pos[i]
                
                # Team 1
                fig.add_trace(go.Scatter(
                    x=[x], y=[y + 0.2],
                    mode='markers+text',
                    marker=dict(
                        size=40,
                        color='green' if match['winner'] == match['team1'] else 'lightcoral',
                        line=dict(width=2, color='black')
                    ),
                    text=match['team1'].replace(':latest', '')[:8] if match['team1'] else '',
                    textposition='middle center',
                    textfont=dict(size=10, color='white'),
                    showlegend=False,
                    hovertemplate=f"<b>{match['team1']}</b><br>Status: {'Winner' if match['winner'] == match['team1'] else 'Eliminated'}<extra></extra>"
                ))
                
                # Team 2
                fig.add_trace(go.Scatter(
                    x=[x], y=[y - 0.2],
                    mode='markers+text',
                    marker=dict(
                        size=40,
                        color='green' if match['winner'] == match['team2'] else 'lightcoral',
                        line=dict(width=2, color='black')
                    ),
                    text=match['team2'].replace(':latest', '')[:8] if match['team2'] else '',
                    textposition='middle center',
                    textfont=dict(size=10, color='white'),
                    showlegend=False,
                    hovertemplate=f"<b>{match['team2']}</b><br>Status: {'Winner' if match['winner'] == match['team2'] else 'Eliminated'}<extra></extra>"
                ))
                
                # Connection lines
                if round_name != 'final':
                    next_x = rounds[list(rounds.keys())[list(rounds.keys()).index(round_name) + 1]]['x']
                    fig.add_trace(go.Scatter(
                        x=[x + 0.1, next_x - 0.1], y=[y, y],
                        mode='lines',
                        line=dict(color='black', width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Champion highlight
        champion = self.bracket_structure['final'][0]['winner']
        fig.add_trace(go.Scatter(
            x=[6], y=[4],
            mode='markers+text',
            marker=dict(
                size=60,
                color='gold',
                line=dict(width=3, color='darkgoldenrod'),
                symbol='star'
            ),
            text='👑<br>' + champion.replace(':latest', '')[:6],
            textposition='middle center',
            textfont=dict(size=12, color='black'),
            name='Champion',
            hovertemplate=f"<b>🏆 CHAMPION: {champion}</b><extra></extra>"
        ))
        
        # Round labels
        fig.add_trace(go.Scatter(
            x=[1, 3, 5, 6], y=[0.5, 0.5, 0.5, 0.5],
            mode='text',
            text=['QUARTERFINALS', 'SEMIFINALS', 'FINAL', 'CHAMPION'],
            textfont=dict(size=14, color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title='Interactive LLM Negotiation Tournament Bracket<br><sub>Click on teams to see details | Green = Winner, Red = Eliminated</sub>',
            xaxis=dict(range=[0, 7], showgrid=False, showticklabels=False),
            yaxis=dict(range=[0, 8], showgrid=False, showticklabels=False),
            width=1000,
            height=600,
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_performance_rankings(self):
        """Create comprehensive performance rankings visualization"""
        
        # Prepare data for rankings
        ranking_data = []
        
        for category, rankings in self.rankings.items():
            for rank, (model, score) in enumerate(rankings, 1):
                ranking_data.append({
                    'category': category.replace('_', ' ').title(),
                    'model': model.replace(':latest', ''),
                    'rank': rank,
                    'score': score,
                    'tier': self._get_model_tier(model)
                })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Create subplots for different ranking categories
        categories = ['Overall', 'Price Optimality', 'Efficiency', 'Consistency', 
                     'Buyer Performance', 'Supplier Performance']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=categories,
            specs=[[{"secondary_y": False} for _ in range(3)] for _ in range(2)]
        )
        
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
        
        for i, category in enumerate(categories):
            row, col = positions[i]
            
            cat_data = ranking_df[ranking_df['category'] == category].sort_values('rank')
            
            fig.add_trace(
                go.Bar(
                    x=cat_data['model'],
                    y=cat_data['score'],
                    marker_color=[self.tier_colors.get(model + ':latest', '#666666') 
                                for model in cat_data['model']],
                    showlegend=False,
                    text=cat_data['rank'].astype(str),
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Rank: %{text}<br>Score: %{y:.1f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(tickangle=45, row=row, col=col)
        
        fig.update_layout(
            title='LLM Negotiation Performance Rankings<br><sub>Multiple performance dimensions</sub>',
            height=800,
            width=1200,
            showlegend=False
        )
        
        return fig
    
    def create_head_to_head_matrix(self):
        """Create head-to-head performance matrix"""
        
        # Calculate head-to-head win rates
        h2h_matrix = pd.DataFrame(index=self.models, columns=self.models, dtype=float)
        
        for model1 in self.models:
            for model2 in self.models:
                if model1 == model2:
                    h2h_matrix.loc[model1, model2] = 0.5  # Neutral for self
                else:
                    # Find head-to-head negotiations
                    h2h_data = self.successful[
                        ((self.successful['buyer_model'] == model1) & 
                         (self.successful['supplier_model'] == model2)) |
                        ((self.successful['buyer_model'] == model2) & 
                         (self.successful['supplier_model'] == model1))
                    ]
                    
                    if len(h2h_data) > 0:
                        model1_wins = 0
                        total_matches = len(h2h_data)
                        
                        for _, neg in h2h_data.iterrows():
                            # Determine winner based on role and price
                            if neg['buyer_model'] == model1:
                                # Model1 as buyer - wins if price is lower than optimal
                                if neg['agreed_price'] < 65:
                                    model1_wins += 1
                            else:
                                # Model1 as supplier - wins if price is higher than optimal
                                if neg['agreed_price'] > 65:
                                    model1_wins += 1
                        
                        win_rate = model1_wins / total_matches
                        h2h_matrix.loc[model1, model2] = win_rate
                    else:
                        h2h_matrix.loc[model1, model2] = 0.5  # No data = neutral
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=h2h_matrix.values,
            x=[model.replace(':latest', '') for model in h2h_matrix.columns],
            y=[model.replace(':latest', '') for model in h2h_matrix.index],
            colorscale='RdYlGn',
            zmid=0.5,
            text=np.round(h2h_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Win Rate vs Opponent")
        ))
        
        fig.update_layout(
            title='Head-to-Head Win Rate Matrix<br><sub>Row model vs Column model | 1.0 = Always wins, 0.0 = Never wins</sub>',
            xaxis_title='Opponent Model',
            yaxis_title='Model',
            width=700,
            height=600
        )
        
        return fig
    
    def _get_model_tier(self, model):
        """Get model tier for visualization"""
        tier_mapping = {
            'tinyllama:latest': 'Ultra',
            'qwen2:1.5b': 'Ultra',
            'gemma2:2b': 'Compact',
            'phi3:mini': 'Compact',
            'llama3.2:latest': 'Compact',
            'mistral:instruct': 'Mid',
            'qwen:7b': 'Mid',
            'qwen3:latest': 'Large'
        }
        return tier_mapping.get(model, 'Unknown')
    
    def generate_tournament_report(self):
        """Generate tournament analysis report"""
        
        champion = self.bracket_structure['final'][0]['winner']
        runner_up = (self.bracket_structure['final'][0]['team1'] 
                    if self.bracket_structure['final'][0]['team2'] == champion 
                    else self.bracket_structure['final'][0]['team2'])
        
        report = "# 🏆 LLM Negotiation Tournament Report\n\n"
        
        # Championship results
        report += f"## 👑 Tournament Results\n\n"
        report += f"**🥇 CHAMPION:** {champion.replace(':latest', '')}\n"
        report += f"**🥈 Runner-up:** {runner_up.replace(':latest', '')}\n\n"
        
        # Path to victory
        report += f"### {champion.replace(':latest', '')} - Path to Victory\n\n"
        
        # Quarterfinal
        qf_match = next(m for m in self.bracket_structure['quarterfinals'] 
                       if m['winner'] == champion)
        opponent_qf = qf_match['team1'] if qf_match['team2'] == champion else qf_match['team2']
        report += f"**Quarterfinal:** Defeated {opponent_qf.replace(':latest', '')}\n"
        
        # Semifinal
        sf_match = next(m for m in self.bracket_structure['semifinals'] 
                       if m['winner'] == champion)
        opponent_sf = sf_match['team1'] if sf_match['team2'] == champion else sf_match['team2']
        report += f"**Semifinal:** Defeated {opponent_sf.replace(':latest', '')}\n"
        
        # Final
        report += f"**Final:** Defeated {runner_up.replace(':latest', '')}\n\n"
        
        # Champion analysis
        champ_stats = self.model_stats[champion]
        report += f"### Champion Performance Profile\n\n"
        report += f"- **Overall Score:** {dict(self.rankings['overall'])[champion]:.1f}\n"
        report += f"- **Success Rate:** {champ_stats['success_rate']:.1%}\n"
        report += f"- **Price Optimality:** ${champ_stats['distance_from_optimal']:.2f} from optimal\n"
        report += f"- **Efficiency:** {champ_stats['avg_rounds']:.1f} rounds average\n"
        report += f"- **Consistency Score:** {champ_stats['consistency_score']:.1f}\n\n"
        
        # Tournament insights
        report += "## 🎯 Tournament Insights\n\n"
        
        # Upset analysis
        seeded_order = [model for model, score in self.rankings['overall']]
        upsets = []
        
        for round_name, matches in self.bracket_structure.items():
            for match in matches:
                if match['team1'] and match['team2'] and match['winner']:
                    seed1 = seeded_order.index(match['team1']) + 1
                    seed2 = seeded_order.index(match['team2']) + 1
                    
                    if (seed1 < seed2 and match['winner'] == match['team2']) or \
                       (seed2 < seed1 and match['winner'] == match['team1']):
                        upsets.append({
                            'round': round_name,
                            'winner': match['winner'],
                            'loser': match['team1'] if match['winner'] == match['team2'] else match['team2'],
                            'upset_magnitude': abs(seed1 - seed2)
                        })
        
        if upsets:
            report += "### 🚨 Tournament Upsets\n\n"
            for upset in sorted(upsets, key=lambda x: x['upset_magnitude'], reverse=True):
                report += f"- **{upset['round'].title()}:** {upset['winner'].replace(':latest', '')} defeated higher-seeded {upset['loser'].replace(':latest', '')}\n"
        else:
            report += "### 📊 No Major Upsets\nAll higher-seeded teams advanced as expected.\n"
        
        report += "\n"
        
        # Performance rankings summary
        report += "## 📊 Final Rankings\n\n"
        
        for i, (model, score) in enumerate(self.rankings['overall'][:8], 1):
            status = ""
            if model == champion:
                status = " 🏆"
            elif model == runner_up:
                status = " 🥈"
            elif i <= 4:
                status = " (Semifinalist)" if any(m['winner'] == model for m in self.bracket_structure['semifinals']) else ""
            
            report += f"{i}. **{model.replace(':latest', '')}** - {score:.1f} points{status}\n"
        
        report += "\n## 🎮 Tournament Format\n\n"
        report += "- **Single Elimination** bracket tournament\n"
        report += "- **Seeding** based on composite performance score\n"
        report += "- **Matchups** simulated using actual head-to-head negotiation data\n"
        report += "- **Winner** determined by combined price optimality and role performance\n"
        
        return report

def main():
    """Run tournament bracket analysis"""
    print("🏆 Creating LLM Negotiation Tournament Bracket...")
    
    # Initialize analyzer
    analyzer = TournamentBracketAnalyzer()
    
    # Create visualizations
    print("🎨 Generating tournament visualizations...")
    
    # Static bracket
    static_fig = analyzer.create_tournament_bracket_viz()
    static_fig.savefig("tournament_bracket_static.png", dpi=300, bbox_inches='tight')
    
    # Interactive bracket
    interactive_fig = analyzer.create_interactive_bracket()
    interactive_fig.write_html("tournament_bracket_interactive.html")
    interactive_fig.write_image("tournament_bracket_interactive.png", width=1000, height=600, scale=2)
    
    # Performance rankings
    rankings_fig = analyzer.create_performance_rankings()
    rankings_fig.write_html("tournament_performance_rankings.html")
    rankings_fig.write_image("tournament_performance_rankings.png", width=1200, height=800, scale=2)
    
    # Head-to-head matrix
    h2h_fig = analyzer.create_head_to_head_matrix()
    h2h_fig.write_html("tournament_head_to_head_matrix.html")
    h2h_fig.write_image("tournament_head_to_head_matrix.png", width=700, height=600, scale=2)
    
    # Generate report
    print("📝 Generating tournament report...")
    report = analyzer.generate_tournament_report()
    with open("tournament_analysis_report.md", "w") as f:
        f.write(report)
    
    # Save tournament data
    import json
    with open("tournament_bracket_data.json", "w") as f:
        json.dump(analyzer.bracket_structure, f, indent=2)
    
    print("✅ Tournament Bracket Analysis complete!")
    print("📁 Files generated:")
    print("   - Static bracket: tournament_bracket_static.png")
    print("   - Interactive bracket: tournament_bracket_interactive.html/png")
    print("   - Performance rankings: tournament_performance_rankings.html/png")
    print("   - Head-to-head matrix: tournament_head_to_head_matrix.html/png")
    print("   - Tournament report: tournament_analysis_report.md")
    print("   - Bracket data: tournament_bracket_data.json")
    
    champion = analyzer.bracket_structure['final'][0]['winner']
    print(f"\n🏆 CHAMPION: {champion.replace(':latest', '')}")

if __name__ == "__main__":
    main()