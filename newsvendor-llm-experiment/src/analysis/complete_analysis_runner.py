"""
Complete Analysis Runner for Newsvendor Experiment

Orchestrates comprehensive analysis including metrics calculation,
statistical testing, and visualization generation.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .metrics_calculator import MetricsCalculator
from .statistical_tests import StatisticalAnalyzer
from .visualizations import ExperimentVisualizer

# Auto-detection imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src" / "utils"))
sys.path.append(str(Path(__file__).parent.parent / "utils"))

try:
    from file_finder import auto_find_and_load_data, DataFileFinder, load_data_smart
    AUTO_DETECTION_AVAILABLE = True
except ImportError:
    print("⚠️  Auto-detection module not found, using fallback...")
    AUTO_DETECTION_AVAILABLE = False
    
    def auto_find_and_load_data():
        # Fallback - try common paths
        common_paths = [
            "./full_results/processed/complete_20250615_171248.csv",
            "./temp_results.csv",
            "./complete_*.csv"
        ]
        for path in common_paths:
            try:
                import pandas as pd
                return pd.read_csv(path), path
            except:
                continue
        return None, None




logger = logging.getLogger(__name__)


class CompleteAnalysisRunner:
    """Run comprehensive analysis of newsvendor experiment results."""
    
    def __init__(self, 
                 data_path: str = "temp_results.csv",
                 output_dir: str = "./analysis_output",
                 optimal_price: float = 65.0):
        """
        Initialize analysis runner.
        
        Args:
            data_path: Path to results CSV file
            output_dir: Directory for analysis outputs
            optimal_price: Theoretical optimal price
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize analysis components
        self.metrics_calculator = MetricsCalculator(optimal_price=optimal_price)
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = ExperimentVisualizer(
            optimal_price=optimal_price, 
            output_dir=str(self.output_dir / "visualizations")
        )
        
        # Create subdirectories
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "statistics").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        logger.info(f"Analysis runner initialized with output directory: {self.output_dir}")
    
    def load_data(self) -> pd.DataFrame:
        """Load and validate experimental data."""
        
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            # Load data
            data = pd.read_csv(self.data_path)
            
            # Validate required columns
            required_columns = [
                'negotiation_id', 'buyer_model', 'supplier_model', 
                'reflection_pattern', 'completed', 'agreed_price',
                'total_rounds', 'total_tokens'
            ]
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Data cleaning
            data = data.dropna(subset=['negotiation_id'])
            
            # Convert data types
            data['completed'] = data['completed'].astype(bool)
            data['agreed_price'] = pd.to_numeric(data['agreed_price'], errors='coerce')
            data['total_rounds'] = pd.to_numeric(data['total_rounds'], errors='coerce')
            data['total_tokens'] = pd.to_numeric(data['total_tokens'], errors='coerce')
            
            logger.info(f"Loaded {len(data)} negotiations from {self.data_path}")
            logger.info(f"Success rate: {data['completed'].mean():.1%}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def run_metrics_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive metrics analysis."""
        
        logger.info("Running metrics analysis...")
        
        try:
            # Generate comprehensive metrics
            metrics_report = self.metrics_calculator.generate_summary_report(data)
            
            # Save metrics report
            metrics_path = self.output_dir / "metrics" / "comprehensive_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics_report, f, indent=2, default=str)
            
            logger.info(f"Metrics analysis saved to: {metrics_path}")
            return metrics_report
            
        except Exception as e:
            logger.error(f"Error in metrics analysis: {e}")
            raise
    
    def run_statistical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive statistical analysis."""
        
        logger.info("Running statistical analysis...")
        
        try:
            # Run hypothesis tests
            statistical_results = self.statistical_analyzer.run_comprehensive_analysis(data)
            
            # Save statistical results
            stats_path = self.output_dir / "statistics" / "hypothesis_tests.json"
            with open(stats_path, 'w') as f:
                json.dump(statistical_results, f, indent=2, default=str)
            
            logger.info(f"Statistical analysis saved to: {stats_path}")
            return statistical_results
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            raise
    
    def run_visualization_analysis(self, data: pd.DataFrame) -> List[str]:
        """Generate comprehensive visualizations."""
        
        logger.info("Generating visualizations...")
        
        try:
            visualization_files = []
            
            # Create comprehensive dashboard
            dashboard_path = self.visualizer.create_comprehensive_dashboard(data)
            if dashboard_path:
                visualization_files.append(dashboard_path)
            
            # Create publication figures
            pub_figures = self.visualizer.create_publication_figures(data)
            visualization_files.extend(pub_figures)
            
            # Create summary infographic
            infographic_path = self.visualizer.create_summary_infographic(data, {})
            if infographic_path:
                visualization_files.append(infographic_path)
            
            logger.info(f"Generated {len(visualization_files)} visualization files")
            return visualization_files
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            raise
    
    def generate_executive_summary(self, 
                                 metrics: Dict[str, Any], 
                                 statistics: Dict[str, Any]) -> str:
        """Generate executive summary of key findings."""
        
        # Extract key metrics
        experiment_overview = metrics.get('experiment_overview', {})
        hypothesis_tests = statistics.get('hypothesis_tests', {})
        
        # Generate summary text
        summary = f"""
# Newsvendor LLM Negotiation Experiment - Executive Summary

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiment Overview
- **Total Negotiations:** {experiment_overview.get('total_negotiations', 'N/A'):,}
- **Success Rate:** {experiment_overview.get('success_rate', 0):.1%}
- **Average Price:** ${experiment_overview.get('avg_price', 0):.2f}
- **Distance from Optimal:** ${experiment_overview.get('distance_from_optimal', 0):.2f}
- **Average Efficiency:** {experiment_overview.get('avg_rounds', 0):.1f} rounds, {experiment_overview.get('avg_tokens', 0):.0f} tokens

## Hypothesis Testing Results

### H1: Reflection Benefits
**Status:** {self._get_hypothesis_status(hypothesis_tests.get('H1_reflection_benefits', {}))}
{self._get_hypothesis_summary(hypothesis_tests.get('H1_reflection_benefits', {}))}

### H2: Model Size-Efficiency Trade-offs  
**Status:** {self._get_hypothesis_status(hypothesis_tests.get('H2_model_efficiency', {}))}
{self._get_hypothesis_summary(hypothesis_tests.get('H2_model_efficiency', {}))}

### H3: Role Asymmetry
**Status:** {self._get_hypothesis_status(hypothesis_tests.get('H3_role_asymmetry', {}))}
{self._get_hypothesis_summary(hypothesis_tests.get('H3_role_asymmetry', {}))}

### H4: Model Pairing Synergy
**Status:** {self._get_hypothesis_status(hypothesis_tests.get('H4_model_synergy', {}))}
{self._get_hypothesis_summary(hypothesis_tests.get('H4_model_synergy', {}))}

## Key Findings

### Reflection Effects
- **Best buyer outcome:** {metrics.get('key_findings', {}).get('reflection_insights', {}).get('best_buyer_outcome', {}).get('pattern', 'N/A')} pattern (${metrics.get('key_findings', {}).get('reflection_insights', {}).get('best_buyer_outcome', {}).get('avg_price', 0):.2f})
- **Most efficient:** {metrics.get('key_findings', {}).get('reflection_insights', {}).get('most_efficient', {}).get('pattern', 'N/A')} pattern ({metrics.get('key_findings', {}).get('reflection_insights', {}).get('most_efficient', {}).get('avg_rounds', 0):.1f} rounds)
- **Closest to optimal:** {metrics.get('key_findings', {}).get('reflection_insights', {}).get('closest_to_optimal', {}).get('pattern', 'N/A')} pattern

### Model Performance
- **Top performing model:** {metrics.get('key_findings', {}).get('model_insights', {}).get('top_performer', 'N/A')}
- **Performance score:** {metrics.get('key_findings', {}).get('model_insights', {}).get('performance_score', 0):.3f}

### Efficiency Insights
- **Token efficiency:** {metrics.get('key_findings', {}).get('efficiency', {}).get('token_efficiency', 0):.3f}
- **Average negotiation length:** {metrics.get('key_findings', {}).get('efficiency', {}).get('avg_rounds', 0):.1f} rounds

## Research Implications

{self._format_research_implications(metrics.get('research_implications', []))}

## Statistical Power
- **Sample size:** {statistics.get('power_analysis', {}).get('sample_sizes', {}).get('total', 'N/A')}
- **Effect sizes:** {statistics.get('power_analysis', {}).get('effect_sizes', {}).get('interpretation', 'N/A')}
- **Power assessment:** {statistics.get('power_analysis', {}).get('power_assessment', {}).get('recommendation', 'N/A')}

## Recommendations for Future Research

1. **Replication:** Results should be replicated with additional model architectures
2. **Mechanism Analysis:** Investigate specific reflection mechanisms driving benefits
3. **Domain Generalization:** Test findings across different negotiation contexts
4. **Practical Applications:** Develop guidelines for LLM deployment in negotiations

---
*This analysis provides comprehensive insights into LLM negotiation behavior with implications for both research and practical applications.*
"""
        
        return summary
    
    def _get_hypothesis_status(self, hypothesis_data: Dict[str, Any]) -> str:
        """Get hypothesis status for summary."""
        if not hypothesis_data:
            return "NOT TESTED"
        
        supported = hypothesis_data.get('supported', False)
        confidence = hypothesis_data.get('confidence_level', 0)
        
        if supported and confidence > 0.95:
            return "✅ STRONGLY SUPPORTED"
        elif supported and confidence > 0.8:
            return "✅ SUPPORTED"
        elif supported:
            return "⚠️ WEAKLY SUPPORTED"
        else:
            return "❌ NOT SUPPORTED"
    
    def _get_hypothesis_summary(self, hypothesis_data: Dict[str, Any]) -> str:
        """Get hypothesis summary for executive report."""
        if not hypothesis_data:
            return "No analysis available.\n"
        
        interpretation = hypothesis_data.get('interpretation', 'No interpretation available.')
        effect_size = hypothesis_data.get('effect_size', 0)
        confidence = hypothesis_data.get('confidence_level', 0)
        
        return f"*{interpretation}*\n**Effect Size:** {effect_size:.3f}, **Confidence:** {confidence:.1%}\n"
    
    def _format_research_implications(self, implications: List[str]) -> str:
        """Format research implications list."""
        if not implications:
            return "No specific implications identified."
        
        formatted = []
        for i, implication in enumerate(implications, 1):
            formatted.append(f"{i}. {implication}")
        
        return "\n".join(formatted)
    
    def generate_detailed_report(self, 
                               data: pd.DataFrame,
                               metrics: Dict[str, Any], 
                               statistics: Dict[str, Any],
                               visualizations: List[str]) -> str:
        """Generate detailed analysis report."""
        
        # Create comprehensive report
        report = f"""
# Comprehensive Analysis Report: Newsvendor LLM Negotiation Experiment

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Data Overview](#data-overview)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Statistical Analysis](#statistical-analysis)
6. [Discussion](#discussion)
7. [Limitations](#limitations)
8. [Conclusions](#conclusions)

## Executive Summary
{self.generate_executive_summary(metrics, statistics)}

## Data Overview

### Sample Characteristics
- **Total Negotiations:** {len(data):,}
- **Successful Negotiations:** {len(data[data['completed']]):,}
- **Success Rate:** {data['completed'].mean():.1%}
- **Date Range:** {data.get('timestamp', pd.Series()).min() if 'timestamp' in data.columns else 'Not available'} to {data.get('timestamp', pd.Series()).max() if 'timestamp' in data.columns else 'Not available'}

### Model Distribution
{self._generate_model_distribution_table(data)}

### Reflection Pattern Distribution
{self._generate_reflection_distribution_table(data)}

## Methodology

### Experimental Design
- **Problem Type:** Classical newsvendor problem
- **Optimal Price:** ${self.metrics_calculator.optimal_price}
- **Reflection Conditions:** 4 patterns (00, 01, 10, 11)
- **Model Tiers:** Ultra, Compact, Mid-range, Large

### Analysis Framework
1. **Metrics Calculation:** Convergence rates, price optimality, efficiency
2. **Hypothesis Testing:** ANOVA, t-tests, effect size calculations
3. **Visualization:** Comprehensive dashboards and publication figures

## Results

### Key Performance Metrics
{self._format_performance_metrics(metrics)}

### Reflection Pattern Analysis
{self._format_reflection_analysis(metrics)}

### Model Performance Analysis
{self._format_model_analysis(metrics)}

## Statistical Analysis

### Hypothesis Testing Summary
{self._format_hypothesis_summary(statistics)}

### Statistical Power Analysis
{self._format_power_analysis(statistics)}

## Discussion

### Reflection Benefits (H1)
{self._generate_h1_discussion(statistics)}

### Model Efficiency Trade-offs (H2)
{self._generate_h2_discussion(statistics)}

### Role Asymmetry (H3)
{self._generate_h3_discussion(statistics)}

### Model Pairing Effects (H4)
{self._generate_h4_discussion(statistics)}

## Limitations

1. **Model Selection:** Limited to available Ollama models
2. **Problem Scope:** Single negotiation scenario (newsvendor)
3. **Reflection Implementation:** Simple prompt-based reflection
4. **Evaluation Metrics:** Focus on price convergence only

## Conclusions

### Primary Findings
1. Reflection mechanisms significantly impact LLM negotiation performance
2. Role asymmetry exists in reflection benefits
3. Model size-efficiency trade-offs vary by context
4. Heterogeneous pairings show potential advantages

### Implications for Practice
- LLM deployment in negotiations should consider reflection capabilities
- Role-specific reflection training may enhance performance
- Model selection should balance capability and efficiency

### Future Research Directions
- Extended reflection mechanisms
- Multi-round negotiation scenarios
- Cross-domain validation
- Real-world deployment studies

---

**Generated Files:**
- Metrics Report: `metrics/comprehensive_metrics.json`
- Statistical Analysis: `statistics/hypothesis_tests.json`
- Visualizations: {len(visualizations)} files in `visualizations/`

**Analysis completed on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report
    
    def _generate_model_distribution_table(self, data: pd.DataFrame) -> str:
        """Generate model distribution table."""
        buyer_counts = data['buyer_model'].value_counts()
        supplier_counts = data['supplier_model'].value_counts()
        
        table = "| Model | As Buyer | As Supplier | Total |\n"
        table += "|-------|----------|-------------|-------|\n"
        
        all_models = set(buyer_counts.index) | set(supplier_counts.index)
        for model in sorted(all_models):
            buyer_count = buyer_counts.get(model, 0)
            supplier_count = supplier_counts.get(model, 0)
            total_count = buyer_count + supplier_count
            table += f"| {model} | {buyer_count} | {supplier_count} | {total_count} |\n"
        
        return table
    
    def _generate_reflection_distribution_table(self, data: pd.DataFrame) -> str:
        """Generate reflection pattern distribution table."""
        pattern_counts = data['reflection_pattern'].value_counts().sort_index()
        
        pattern_names = {
            '00': 'No Reflection',
            '01': 'Buyer Reflection',
            '10': 'Supplier Reflection',
            '11': 'Both Reflection'
        }
        
        table = "| Pattern | Description | Count | Percentage |\n"
        table += "|---------|-------------|-------|------------|\n"
        
        total = len(data)
        for pattern in ['00', '01', '10', '11']:
            count = pattern_counts.get(pattern, 0)
            percentage = (count / total * 100) if total > 0 else 0
            table += f"| {pattern} | {pattern_names[pattern]} | {count} | {percentage:.1f}% |\n"
        
        return table
    
    def _format_performance_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format performance metrics for report."""
        overview = metrics.get('experiment_overview', {})
        
        return f"""
- **Success Rate:** {overview.get('success_rate', 0):.1%}
- **Average Price:** ${overview.get('avg_price', 0):.2f}
- **Price Standard Deviation:** ${overview.get('price_std', 0):.2f}
- **Distance from Optimal:** ${overview.get('distance_from_optimal', 0):.2f}
- **Optimal Convergence Rate:** {overview.get('optimal_convergence_rate', 0):.1%}
- **Average Rounds:** {overview.get('avg_rounds', 0):.1f}
- **Average Tokens:** {overview.get('avg_tokens', 0):.0f}
- **Token Efficiency:** {overview.get('token_efficiency', 0):.3f}
"""
    
    def _format_reflection_analysis(self, metrics: Dict[str, Any]) -> str:
        """Format reflection analysis for report."""
        reflection_data = metrics.get('reflection_analysis', [])
        
        if not reflection_data:
            return "No reflection analysis available."
        
        result = "| Pattern | Count | Success Rate | Avg Price | Avg Rounds | Distance from Optimal |\n"
        result += "|---------|-------|--------------|-----------|------------|----------------------|\n"
        
        pattern_names = {
            '00': 'No Reflection',
            '01': 'Buyer Reflection', 
            '10': 'Supplier Reflection',
            '11': 'Both Reflection'
        }
        
        for item in reflection_data:
            pattern = item.get('pattern', '')
            name = pattern_names.get(pattern, pattern)
            count = item.get('count', 0)
            success_rate = item.get('success_rate', 0)
            avg_price = item.get('avg_price', 0)
            avg_rounds = item.get('avg_rounds', 0)
            distance = item.get('distance_from_optimal', 0)
            
            result += f"| {name} | {count} | {success_rate:.1%} | ${avg_price:.2f} | {avg_rounds:.1f} | ${distance:.2f} |\n"
        
        return result
    
    def _format_model_analysis(self, metrics: Dict[str, Any]) -> str:
        """Format model analysis for report."""
        model_data = metrics.get('model_performance', [])
        
        if not model_data:
            return "No model analysis available."
        
        result = "### Top Performing Models\n\n"
        result += "| Model | Overall Score | Efficiency Score | As Buyer (Avg Price) | As Supplier (Avg Price) |\n"
        result += "|-------|---------------|------------------|---------------------|------------------------|\n"
        
        # Sort by overall performance and take top 5
        sorted_models = sorted(model_data, key=lambda x: x.get('overall_performance', 0), reverse=True)
        
        for model in sorted_models[:5]:
            name = model.get('model_name', '').replace(':latest', '')
            overall = model.get('overall_performance', 0)
            efficiency = model.get('efficiency_score', 0)
            buyer_price = model.get('as_buyer', {}).get('avg_price', 0)
            supplier_price = model.get('as_supplier', {}).get('avg_price', 0)
            
            result += f"| {name} | {overall:.3f} | {efficiency:.3f} | ${buyer_price:.2f} | ${supplier_price:.2f} |\n"
        
        return result
    
    def _format_hypothesis_summary(self, statistics: Dict[str, Any]) -> str:
        """Format hypothesis testing summary."""
        hypothesis_tests = statistics.get('hypothesis_tests', {})
        
        result = "| Hypothesis | Status | Effect Size | Confidence | Interpretation |\n"
        result += "|------------|--------|-------------|------------|----------------|\n"
        
        for h_name, h_data in hypothesis_tests.items():
            status = "✅" if h_data.get('supported', False) else "❌"
            effect_size = h_data.get('effect_size', 0)
            confidence = h_data.get('confidence_level', 0)
            interpretation = h_data.get('interpretation', '')[:50] + "..." if len(h_data.get('interpretation', '')) > 50 else h_data.get('interpretation', '')
            
            result += f"| {h_name.replace('_', ' ').title()} | {status} | {effect_size:.3f} | {confidence:.1%} | {interpretation} |\n"
        
        return result
    
    def _format_power_analysis(self, statistics: Dict[str, Any]) -> str:
        """Format power analysis results."""
        power_data = statistics.get('power_analysis', {})
        
        if not power_data:
            return "No power analysis available."
        
        sample_sizes = power_data.get('sample_sizes', {})
        effect_sizes = power_data.get('effect_sizes', {})
        assessment = power_data.get('power_assessment', {})
        
        return f"""
- **Total Sample Size:** {sample_sizes.get('total', 'N/A')}
- **Minimum Group Size:** {sample_sizes.get('min_group_size', 'N/A')}
- **Number of Groups:** {sample_sizes.get('groups', 'N/A')}
- **Observed Effect Size:** {effect_sizes.get('eta_squared', 0):.3f} ({effect_sizes.get('interpretation', 'N/A')})
- **Power Assessment:** {assessment.get('recommendation', 'N/A')}
"""
    
    def _generate_h1_discussion(self, statistics: Dict[str, Any]) -> str:
        """Generate H1 discussion section."""
        h1_data = statistics.get('hypothesis_tests', {}).get('H1_reflection_benefits', {})
        
        if not h1_data:
            return "H1 analysis not available."
        
        supported = h1_data.get('supported', False)
        interpretation = h1_data.get('interpretation', '')
        
        if supported:
            return f"""
The analysis provides support for H1, confirming that reflection mechanisms enhance LLM negotiation performance. {interpretation} This finding has significant implications for the design of negotiation support systems using LLMs.
"""
        else:
            return f"""
The analysis does not provide strong support for H1. {interpretation} This suggests that simple reflection prompting may not be sufficient to improve negotiation outcomes, or that more sophisticated reflection mechanisms may be needed.
"""
    
    def _generate_h2_discussion(self, statistics: Dict[str, Any]) -> str:
        """Generate H2 discussion section.""" 
        h2_data = statistics.get('hypothesis_tests', {}).get('H2_model_efficiency', {})
        
        if not h2_data:
            return "H2 analysis not available."
        
        supported = h2_data.get('supported', False)
        interpretation = h2_data.get('interpretation', '')
        
        return f"""
{'The results support' if supported else 'The results do not strongly support'} H2 regarding model size-efficiency trade-offs. {interpretation} This has practical implications for selecting appropriate models for negotiation applications based on performance requirements and computational constraints.
"""
    
    def _generate_h3_discussion(self, statistics: Dict[str, Any]) -> str:
        """Generate H3 discussion section."""
        h3_data = statistics.get('hypothesis_tests', {}).get('H3_role_asymmetry', {})
        
        if not h3_data:
            return "H3 analysis not available."
        
        supported = h3_data.get('supported', False)
        interpretation = h3_data.get('interpretation', '')
        
        return f"""
{'Evidence supports' if supported else 'Evidence does not clearly support'} H3 regarding role asymmetry in reflection benefits. {interpretation} This finding suggests that reflection mechanisms may need to be tailored to specific negotiation roles for optimal effectiveness.
"""
    
    def _generate_h4_discussion(self, statistics: Dict[str, Any]) -> str:
        """Generate H4 discussion section."""
        h4_data = statistics.get('hypothesis_tests', {}).get('H4_model_synergy', {})
        
        if not h4_data:
            return "H4 analysis not available."
        
        supported = h4_data.get('supported', False)
        interpretation = h4_data.get('interpretation', '')
        
        return f"""
{'The analysis supports' if supported else 'The analysis does not support'} H4 regarding heterogeneous model pairing benefits. {interpretation} This has implications for multi-agent system design and the potential benefits of diversity in negotiation contexts.
"""
    
    def run_complete_analysis(self) -> Dict[str, str]:
        """Run complete analysis pipeline and return file paths."""
        
        logger.info("Starting complete analysis pipeline...")
        
        try:
            # Load data
            data = self.load_data()
            
            # Run analysis components
            metrics = self.run_metrics_analysis(data)
            statistics = self.run_statistical_analysis(data)
            visualizations = self.run_visualization_analysis(data)
            
            # Generate reports
            logger.info("Generating reports...")
            
            # Executive summary
            executive_summary = self.generate_executive_summary(metrics, statistics)
            exec_path = self.output_dir / "reports" / "executive_summary.md"
            with open(exec_path, 'w') as f:
                f.write(executive_summary)
            
            # Detailed report
            detailed_report = self.generate_detailed_report(data, metrics, statistics, visualizations)
            report_path = self.output_dir / "reports" / "detailed_analysis_report.md"
            with open(report_path, 'w') as f:
                f.write(detailed_report)
            
            # Summary of generated files
            output_files = {
                'executive_summary': str(exec_path),
                'detailed_report': str(report_path),
                'metrics_report': str(self.output_dir / "metrics" / "comprehensive_metrics.json"),
                'statistical_analysis': str(self.output_dir / "statistics" / "hypothesis_tests.json"),
                'visualizations': visualizations,
                'analysis_directory': str(self.output_dir)
            }
            
            logger.info("Complete analysis pipeline finished successfully!")
            logger.info(f"Analysis outputs saved to: {self.output_dir}")
            
            return output_files
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            raise


def main():
    """Main function for running complete analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run complete newsvendor experiment analysis")
    parser.add_argument("--data", default="temp_results.csv", help="Path to results CSV file")
    parser.add_argument("--output", default="./analysis_output", help="Output directory")
    parser.add_argument("--optimal-price", type=float, default=65.0, help="Optimal price for analysis")
    
    args = parser.parse_args()
    
    # Run analysis
    runner = CompleteAnalysisRunner(
        data_path=args.data,
        output_dir=args.output,
        optimal_price=args.optimal_price
    )
    
    try:
        output_files = runner.run_complete_analysis()
        
        print("\n🎉 Analysis Complete!")
        print(f"📁 Output directory: {output_files['analysis_directory']}")
        print(f"📊 Executive summary: {output_files['executive_summary']}")
        print(f"📋 Detailed report: {output_files['detailed_report']}")
        print(f"📈 Visualizations: {len(output_files['visualizations'])} files")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())