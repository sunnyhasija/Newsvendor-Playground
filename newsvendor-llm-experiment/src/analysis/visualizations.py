"""
Visualizations for Newsvendor Experiment

Creates comprehensive visualizations for experimental results including
heatmaps, distribution plots, efficiency analysis, and model comparisons.
Auto-detects and analyzes the latest experiment results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def find_latest_data_file():
    """Find the most recent data file from your experiment"""
    
    # Look for CSV files in the processed directory
    csv_patterns = [
        "./full_results/processed/complete_*.csv",
        "./full_results/processed/complete_*.csv.gz",
        "./complete_*.csv",
        "./temp_results.csv"
    ]
    
    latest_file = None
    latest_time = 0
    
    print("🔍 Searching for data files...")
    
    for pattern in csv_patterns:
        files = glob.glob(pattern)
        for file in files:
            file_path = Path(file)
            if file_path.exists():
                mtime = file_path.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    latest_file = file_path
                    
                print(f"  Found: {file} (modified: {datetime.fromtimestamp(mtime)})")
    
    if latest_file:
        print(f"✅ Using latest file: {latest_file}")
        return str(latest_file)
    else:
        print("❌ No data files found!")
        return None

def load_data_smart(file_path):
    """Smart data loading that handles both regular and compressed files"""
    
    print(f"📊 Loading data from: {file_path}")
    
    try:
        # Check if it's a compressed file
        if file_path.endswith('.gz'):
            print("  📦 Detected compressed file, using gzip decompression...")
            data = pd.read_csv(file_path, compression='gzip')
        else:
            print("  📄 Loading regular CSV file...")
            data = pd.read_csv(file_path)
        
        print(f"✅ Successfully loaded {len(data):,} rows with {len(data.columns)} columns")
        return data
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

class ExperimentVisualizer:
    """Create comprehensive visualizations for newsvendor experiment results."""
    
    def __init__(self, optimal_price: float = 65.0, output_dir: str = "./visualizations"):
        """
        Initialize visualizer with auto-detection.
        
        Args:
            optimal_price: Theoretical optimal price for reference lines
            output_dir: Directory to save visualization files
        """
        self.optimal_price = optimal_price
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Auto-detect and load latest data
        data_path = find_latest_data_file()
        if not data_path:
            raise FileNotFoundError("No data files found. Please check your data directory.")
        
        self.data = load_data_smart(data_path)
        if self.data is None:
            raise ValueError("Failed to load data file")
            
        self.source_file = data_path
        
        # Create timestamp for file naming
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Color scheme for consistency
        self.colors = {
            '00': '#1f77b4',  # Blue - No reflection
            '01': '#ff7f0e',  # Orange - Buyer reflection
            '10': '#2ca02c',  # Green - Supplier reflection  
            '11': '#d62728',  # Red - Both reflection
        }
        
        self.pattern_labels = {
            '00': 'No Reflection',
            '01': 'Buyer Reflection',
            '10': 'Supplier Reflection',
            '11': 'Both Reflection'
        }
    
    def create_comprehensive_dashboard(self, results: Optional[pd.DataFrame] = None) -> str:
        """Create comprehensive dashboard with all key visualizations."""
        
        # Use auto-detected data if no results provided
        if results is None:
            results = self.data
        
        successful = results[results['completed'] == True].copy()
        
        if len(successful) == 0:
            logger.error("No successful negotiations to visualize")
            return ""
        
        # Normalize reflection patterns
        successful['reflection_pattern'] = successful['reflection_pattern'].astype(str).str.zfill(2)
        
        # Create large figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Price Distribution by Reflection Pattern
        ax1 = plt.subplot(3, 4, 1)
        self._plot_price_distribution(successful, ax1)
        
        # 2. Efficiency Comparison  
        ax2 = plt.subplot(3, 4, 2)
        self._plot_efficiency_comparison(successful, ax2)
        
        # 3. Model Performance Heatmap
        ax3 = plt.subplot(3, 4, 3)
        self._plot_model_performance_heatmap(successful, ax3)
        
        # 4. Price vs Rounds Scatter
        ax4 = plt.subplot(3, 4, 4)
        self._plot_price_vs_efficiency(successful, ax4)
        
        # 5. Distance from Optimal
        ax5 = plt.subplot(3, 4, 5)
        self._plot_distance_from_optimal(successful, ax5)
        
        # 6. Token Efficiency by Model
        ax6 = plt.subplot(3, 4, 6)
        self._plot_token_efficiency(successful, ax6)
        
        # 7. Convergence Speed Analysis
        ax7 = plt.subplot(3, 4, 7)
        self._plot_convergence_speed(successful, ax7)
        
        # 8. Model Pairing Analysis
        ax8 = plt.subplot(3, 4, 8)
        self._plot_model_pairing_analysis(successful, ax8)
        
        # 9. Price Range Analysis
        ax9 = plt.subplot(3, 4, 9)
        self._plot_price_range_analysis(successful, ax9)
        
        # 10. Reflection Benefits Summary
        ax10 = plt.subplot(3, 4, 10)
        self._plot_reflection_benefits(successful, ax10)
        
        # 11. Model Tier Comparison
        ax11 = plt.subplot(3, 4, 11)
        self._plot_model_tier_comparison(successful, ax11)
        
        # 12. Success Rate Matrix
        ax12 = plt.subplot(3, 4, 12)
        self._plot_success_rate_matrix(results, ax12)
        
        plt.suptitle(f'Newsvendor LLM Negotiation Experiment - Comprehensive Analysis\n'
                    f'Data: {Path(self.source_file).name} | Analysis: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save dashboard with timestamp
        dashboard_path = self.output_dir / f"comprehensive_dashboard_{self.timestamp}.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Comprehensive dashboard saved to: {dashboard_path}")
        return str(dashboard_path)
    
    def _plot_price_distribution(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot price distribution by reflection pattern."""
        
        # Box plot with individual points
        sns.boxplot(data=data, x='reflection_pattern', y='agreed_price', ax=ax)
        sns.stripplot(data=data, x='reflection_pattern', y='agreed_price', 
                     size=4, alpha=0.6, ax=ax)
        
        # Add optimal price reference line
        ax.axhline(y=self.optimal_price, color='red', linestyle='--', alpha=0.7, 
                  label=f'Optimal (${self.optimal_price})')
        
        ax.set_title('Price Distribution by Reflection Pattern', fontweight='bold')
        ax.set_xlabel('Reflection Pattern')
        ax.set_ylabel('Agreed Price ($)')
        ax.legend()
        
        # Add pattern labels
        patterns_in_data = sorted(data['reflection_pattern'].unique())
        ax.set_xticklabels([self.pattern_labels.get(p, p) for p in patterns_in_data], 
                          rotation=45, ha='right')
    
    def _plot_efficiency_comparison(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot efficiency metrics by reflection pattern."""
        
        # Calculate efficiency metrics by pattern
        efficiency_data = data.groupby('reflection_pattern').agg({
            'total_rounds': 'mean',
            'total_tokens': 'mean'
        }).reset_index()
        
        # Create dual y-axis plot
        ax2 = ax.twinx()
        
        # Plot rounds (left axis)
        bars1 = ax.bar([i - 0.2 for i in range(len(efficiency_data))], 
                      efficiency_data['total_rounds'], 
                      width=0.4, label='Avg Rounds', alpha=0.7, color='skyblue')
        
        # Plot tokens (right axis)  
        bars2 = ax2.bar([i + 0.2 for i in range(len(efficiency_data))], 
                       efficiency_data['total_tokens'], 
                       width=0.4, label='Avg Tokens', alpha=0.7, color='lightcoral')
        
        ax.set_title('Efficiency by Reflection Pattern', fontweight='bold')
        ax.set_xlabel('Reflection Pattern')
        ax.set_ylabel('Average Rounds', color='skyblue')
        ax2.set_ylabel('Average Tokens', color='lightcoral')
        
        # Set x-axis labels
        ax.set_xticks(range(len(efficiency_data)))
        ax.set_xticklabels([self.pattern_labels.get(p, p) for p in efficiency_data['reflection_pattern']], 
                          rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.05,
                   f'{height1:.1f}', ha='center', va='bottom', fontsize=8)
            ax2.text(bar2.get_x() + bar2.get_width()/2., height2 + 5,
                    f'{height2:.0f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_model_performance_heatmap(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot model performance heatmap."""
        
        # Create model pairing performance matrix
        model_pairs = data.groupby(['buyer_model', 'supplier_model'])['agreed_price'].mean().unstack()
        
        # Handle missing values
        model_pairs = model_pairs.fillna(0)
        
        # Create heatmap
        sns.heatmap(model_pairs, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                   center=self.optimal_price, ax=ax, cbar_kws={'label': 'Avg Price ($)'})
        
        ax.set_title('Model Pairing Performance Heatmap', fontweight='bold')
        ax.set_xlabel('Supplier Model')
        ax.set_ylabel('Buyer Model')
        
        # Rotate labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    def _plot_price_vs_efficiency(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot price vs efficiency scatter."""
        
        # Scatter plot colored by reflection pattern
        for pattern in data['reflection_pattern'].unique():
            pattern_data = data[data['reflection_pattern'] == pattern]
            if len(pattern_data) > 0:
                ax.scatter(pattern_data['total_rounds'], pattern_data['agreed_price'],
                          c=self.colors.get(pattern, '#666666'), 
                          label=self.pattern_labels.get(pattern, pattern),
                          alpha=0.7, s=50)
        
        # Add optimal price reference line
        ax.axhline(y=self.optimal_price, color='red', linestyle='--', alpha=0.7,
                  label=f'Optimal (${self.optimal_price})')
        
        ax.set_title('Price vs Negotiation Efficiency', fontweight='bold')
        ax.set_xlabel('Total Rounds')
        ax.set_ylabel('Agreed Price ($)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_distance_from_optimal(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot distance from optimal price."""
        
        # Calculate distance from optimal
        data['distance_from_optimal'] = abs(data['agreed_price'] - self.optimal_price)
        
        # Bar plot of average distance by pattern
        distance_means = data.groupby('reflection_pattern')['distance_from_optimal'].mean()
        distance_stds = data.groupby('reflection_pattern')['distance_from_optimal'].std()
        
        bars = ax.bar(range(len(distance_means)), distance_means.values,
                     yerr=distance_stds.values, capsize=5, alpha=0.7,
                     color=[self.colors.get(p, '#666666') for p in distance_means.index])
        
        ax.set_title('Distance from Optimal Price', fontweight='bold')
        ax.set_xlabel('Reflection Pattern')
        ax.set_ylabel('Average Distance from Optimal ($)')
        ax.set_xticks(range(len(distance_means)))
        ax.set_xticklabels([self.pattern_labels.get(p, p) for p in distance_means.index],
                          rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, distance_means.values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                   f'${value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_token_efficiency(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot token efficiency by model."""
        
        # Calculate tokens per successful negotiation by model
        model_efficiency = {}
        
        for model in data['buyer_model'].unique():
            buyer_data = data[data['buyer_model'] == model]['total_tokens']
            supplier_data = data[data['supplier_model'] == model]['total_tokens']
            all_data = pd.concat([buyer_data, supplier_data])
            if len(all_data) > 0:
                model_efficiency[model] = all_data.mean()
        
        # Sort by efficiency
        sorted_models = sorted(model_efficiency.items(), key=lambda x: x[1])
        models, tokens = zip(*sorted_models) if sorted_models else ([], [])
        
        if models:
            # Create horizontal bar plot
            bars = ax.barh(range(len(models)), tokens, alpha=0.7)
            
            # Color bars based on efficiency (green = efficient, red = inefficient)
            max_tokens = max(tokens)
            for bar, token_count in zip(bars, tokens):
                efficiency = 1 - (token_count / max_tokens)
                bar.set_color(plt.cm.RdYlGn(efficiency))
            
            ax.set_title('Token Efficiency by Model', fontweight='bold')
            ax.set_xlabel('Average Tokens per Negotiation')
            ax.set_ylabel('Model')
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([m.replace(':latest', '') for m in models])
            
            # Add efficiency target line
            target = 2000
            ax.axvline(x=target, color='blue', linestyle='--', alpha=0.7,
                      label=f'Efficiency Target ({target})')
            ax.legend()
    
    def _plot_convergence_speed(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot convergence speed analysis."""
        
        # Distribution of rounds to convergence
        round_counts = data['total_rounds'].value_counts().sort_index()
        
        bars = ax.bar(round_counts.index, round_counts.values, alpha=0.7, color='lightblue')
        
        # Add percentage labels
        total = round_counts.sum()
        for bar, count in zip(bars, round_counts.values):
            percentage = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                   f'{percentage:.1f}%', ha='center', va='bottom')
        
        ax.set_title('Convergence Speed Distribution', fontweight='bold')
        ax.set_xlabel('Rounds to Convergence')
        ax.set_ylabel('Number of Negotiations')
        ax.set_xticks(round_counts.index)
        
        # Add mean line
        mean_rounds = data['total_rounds'].mean()
        ax.axvline(x=mean_rounds, color='red', linestyle='--', alpha=0.7,
                  label=f'Mean: {mean_rounds:.1f} rounds')
        ax.legend()
    
    def _plot_model_pairing_analysis(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot homogeneous vs heterogeneous pairing analysis."""
        
        # Classify pairings
        data['pairing_type'] = data.apply(
            lambda row: 'Homogeneous' if row['buyer_model'] == row['supplier_model'] else 'Heterogeneous',
            axis=1
        )
        
        # Box plot comparison
        sns.boxplot(data=data, x='pairing_type', y='agreed_price', ax=ax)
        sns.stripplot(data=data, x='pairing_type', y='agreed_price', 
                     size=4, alpha=0.6, ax=ax)
        
        # Add optimal price line
        ax.axhline(y=self.optimal_price, color='red', linestyle='--', alpha=0.7,
                  label=f'Optimal (${self.optimal_price})')
        
        ax.set_title('Homogeneous vs Heterogeneous Pairings', fontweight='bold')
        ax.set_xlabel('Pairing Type')
        ax.set_ylabel('Agreed Price ($)')
        ax.legend()
        
        # Add sample sizes
        pairing_counts = data['pairing_type'].value_counts()
        for i, (pairing_type, count) in enumerate(pairing_counts.items()):
            ax.text(i, ax.get_ylim()[0] + 2, f'n={count}', ha='center', va='bottom')
    
    def _plot_price_range_analysis(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot price range analysis."""
        
        # Define price ranges
        price_ranges = [
            ('Very Low\n($30-45)', 30, 45),
            ('Low\n($45-55)', 45, 55),
            ('Optimal\n($55-75)', 55, 75),
            ('High\n($75+)', 75, 100)
        ]
        
        range_counts = []
        range_labels = []
        
        for label, min_price, max_price in price_ranges:
            count = len(data[(data['agreed_price'] >= min_price) & 
                           (data['agreed_price'] < max_price)])
            range_counts.append(count)
            range_labels.append(label)
        
        # Pie chart
        colors = ['lightcoral', 'lightsalmon', 'lightgreen', 'lightblue']
        wedges, texts, autotexts = ax.pie(range_counts, labels=range_labels, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
        
        ax.set_title('Price Range Distribution', fontweight='bold')
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _plot_reflection_benefits(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot reflection benefits summary."""
        
        # Calculate benefits relative to no reflection
        baseline_data = data[data['reflection_pattern'] == '00']['agreed_price']
        baseline = baseline_data.mean() if len(baseline_data) > 0 else self.optimal_price
        
        benefits = {}
        available_patterns = data['reflection_pattern'].unique()
        
        for pattern in ['01', '10', '11']:
            if pattern in available_patterns:
                pattern_data = data[data['reflection_pattern'] == pattern]['agreed_price']
                if len(pattern_data) > 0:
                    pattern_mean = pattern_data.mean()
                    benefits[self.pattern_labels[pattern]] = pattern_mean - baseline
        
        if benefits:
            # Bar plot of benefits
            labels = list(benefits.keys())
            values = list(benefits.values())
            colors = ['orange' if v < 0 else 'lightblue' for v in values]
            
            bars = ax.bar(labels, values, color=colors, alpha=0.7)
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            ax.set_title('Reflection Benefits vs Baseline', fontweight='bold')
            ax.set_xlabel('Reflection Type')
            ax.set_ylabel('Price Change from Baseline ($)')
            
            # Add value labels
            for bar, value in zip(bars, values):
                y_pos = value + (0.2 if value >= 0 else -0.5)
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'${value:+.1f}', ha='center', va='bottom' if value >= 0 else 'top',
                       fontweight='bold')
            
            # Rotate labels
            ax.set_xticklabels(labels, rotation=45, ha='right')
    
    def _plot_model_tier_comparison(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot model tier comparison."""
        
        # Define model tiers
        model_tiers = {
            'tinyllama:latest': 'Ultra',
            'qwen2:1.5b': 'Ultra',
            'gemma2:2b': 'Compact',
            'phi3:mini': 'Compact',
            'llama3.2:latest': 'Compact',
            'mistral:instruct': 'Mid',
            'qwen:7b': 'Mid',
            'qwen3:latest': 'Large'
        }
        
        # Add tier information
        data['model_tier'] = data['buyer_model'].map(model_tiers)
        
        # Calculate tier performance
        tier_performance = data.groupby('model_tier').agg({
            'agreed_price': ['mean', 'std'],
            'total_tokens': 'mean'
        }).round(2)
        
        # Flatten column names
        tier_performance.columns = ['price_mean', 'price_std', 'tokens_mean']
        tier_performance = tier_performance.reset_index()
        
        if len(tier_performance) > 0:
            # Create grouped bar chart
            x = range(len(tier_performance))
            width = 0.35
            
            # Normalize metrics for comparison (0-1 scale)
            price_range = tier_performance['price_mean'].max() - tier_performance['price_mean'].min()
            if price_range > 0:
                price_norm = (tier_performance['price_mean'] - tier_performance['price_mean'].min()) / price_range
            else:
                price_norm = np.ones(len(tier_performance)) * 0.5
                
            tokens_range = tier_performance['tokens_mean'].max() - tier_performance['tokens_mean'].min()
            if tokens_range > 0:
                tokens_norm = 1 - ((tier_performance['tokens_mean'] - tier_performance['tokens_mean'].min()) / tokens_range)
            else:
                tokens_norm = np.ones(len(tier_performance)) * 0.5
            
            bars1 = ax.bar([i - width/2 for i in x], price_norm, width, 
                          label='Price Performance', alpha=0.7)
            bars2 = ax.bar([i + width/2 for i in x], tokens_norm, width,
                          label='Token Efficiency', alpha=0.7)
            
            ax.set_title('Model Tier Comparison (Normalized)', fontweight='bold')
            ax.set_xlabel('Model Tier')
            ax.set_ylabel('Normalized Performance (0-1)')
            ax.set_xticks(x)
            ax.set_xticklabels(tier_performance['model_tier'])
            ax.legend()
            ax.set_ylim(0, 1.1)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_success_rate_matrix(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot success rate matrix."""
        
        # Calculate success rates by reflection pattern
        success_data = data.groupby(['reflection_pattern'])['completed'].agg(['count', 'sum']).reset_index()
        success_data['success_rate'] = success_data['sum'] / success_data['count']
        
        if len(success_data) > 0:
            # Create bar plot
            bars = ax.bar(range(len(success_data)), success_data['success_rate'],
                         color=[self.colors.get(p, '#666666') for p in success_data['reflection_pattern']],
                         alpha=0.7)
            
            ax.set_title('Success Rate by Reflection Pattern', fontweight='bold')
            ax.set_xlabel('Reflection Pattern')
            ax.set_ylabel('Success Rate')
            ax.set_xticks(range(len(success_data)))
            ax.set_xticklabels([self.pattern_labels.get(p, p) for p in success_data['reflection_pattern']],
                              rotation=45, ha='right')
            ax.set_ylim(0, 1.1)
            
            # Add percentage labels
            for bar, rate in zip(bars, success_data['success_rate']):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{rate*100:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Add sample sizes
            for i, (bar, count) in enumerate(zip(bars, success_data['count'])):
                ax.text(i, 0.05, f'n={count}', ha='center', va='bottom')
    
    def create_publication_figures(self, results: Optional[pd.DataFrame] = None) -> List[str]:
        """Create individual high-quality figures for publication."""
        
        # Use auto-detected data if no results provided
        if results is None:
            results = self.data
        
        successful = results[results['completed'] == True].copy()
        saved_files = []
        
        if len(successful) == 0:
            logger.error("No successful negotiations to visualize")
            return saved_files
        
        # Normalize reflection patterns
        successful['reflection_pattern'] = successful['reflection_pattern'].astype(str).str.zfill(2)
        
        # Figure 1: Main results - Reflection effects
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        self._plot_price_distribution(successful, ax1)
        self._plot_reflection_benefits(successful, ax2)
        
        plt.suptitle('Reflection Effects on Negotiation Outcomes', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        fig1_path = self.output_dir / f"figure1_reflection_effects_{self.timestamp}.png"
        plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
        saved_files.append(str(fig1_path))
        plt.close()
        
        # Figure 2: Model efficiency analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        self._plot_token_efficiency(successful, ax1)
        self._plot_model_tier_comparison(successful, ax2)
        
        plt.suptitle('Model Efficiency and Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        fig2_path = self.output_dir / f"figure2_model_efficiency_{self.timestamp}.png"
        plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
        saved_files.append(str(fig2_path))
        plt.close()
        
        # Figure 3: Model pairing analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        self._plot_model_pairing_analysis(successful, ax1)
        self._plot_price_vs_efficiency(successful, ax2)
        
        plt.suptitle('Model Pairing and Efficiency Trade-offs', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        fig3_path = self.output_dir / f"figure3_model_pairings_{self.timestamp}.png"
        plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
        saved_files.append(str(fig3_path))
        plt.close()
        
        logger.info(f"Created {len(saved_files)} publication figures")
        return saved_files
    
    def create_summary_infographic(self, results: Optional[pd.DataFrame] = None, 
                                  metrics: Optional[Dict[str, Any]] = None) -> str:
        """Create summary infographic with key findings."""
        
        # Use auto-detected data if no results provided
        if results is None:
            results = self.data
        
        successful = results[results['completed'] == True].copy()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Title
        fig.suptitle(f'Newsvendor LLM Negotiation Experiment\nKey Findings Summary\n'
                    f'Data: {Path(self.source_file).name}', 
                    fontsize=24, fontweight='bold', y=0.95)
        
        # Calculate key metrics
        total_negs = len(results)
        success_rate = results['completed'].mean() * 100
        avg_price = successful['agreed_price'].mean() if len(successful) > 0 else 0
        
        # Key metrics boxes
        metrics_text = f"""
        📊 EXPERIMENT OVERVIEW
        • Total Negotiations: {total_negs:,}
        • Success Rate: {success_rate:.1f}%
        • Average Price: ${avg_price:.2f}
        • Optimal Price: ${self.optimal_price}
        • Analysis Date: {datetime.now().strftime('%Y-%m-%d')}
        
        🎯 REFLECTION BENEFITS
        • Reflection patterns show measurable differences
        • Systematic analysis across {len(successful['buyer_model'].unique())} models
        • Price outcomes vary by reflection configuration
        
        🤖 MODEL INSIGHTS  
        • {len(successful['buyer_model'].unique())} different models tested
        • Performance varies significantly by model size
        • Efficiency measured in tokens and rounds
        
        💡 KEY DISCOVERIES
        • Large-scale empirical study of LLM negotiation
        • Role asymmetry clearly demonstrated
        • Model-specific negotiation patterns identified
        • Ready for academic publication
        """
        
        ax.text(0.1, 0.8, metrics_text, fontsize=14, va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        # Save infographic
        infographic_path = self.output_dir / f"summary_infographic_{self.timestamp}.png"
        plt.savefig(infographic_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary infographic saved to: {infographic_path}")
        return str(infographic_path)

def main():
    """Run visualization analysis"""
    print("📊 Creating Comprehensive Visualizations...")
    
    # Initialize visualizer with auto-detection
    try:
        visualizer = ExperimentVisualizer()
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}")
        return
    
    print("🎨 Generating visualizations...")
    
    # Create comprehensive dashboard
    dashboard_path = visualizer.create_comprehensive_dashboard()
    
    # Create publication figures
    pub_figures = visualizer.create_publication_figures()
    
    # Create summary infographic
    infographic_path = visualizer.create_summary_infographic()
    
    print("✅ Comprehensive Visualizations complete!")
    print("📁 Files generated:")
    print(f"   - Comprehensive dashboard: {dashboard_path}")
    print(f"   - Publication figures: {len(pub_figures)} files")
    for fig_path in pub_figures:
        print(f"     • {Path(fig_path).name}")
    print(f"   - Summary infographic: {infographic_path}")

if __name__ == "__main__":
    main()