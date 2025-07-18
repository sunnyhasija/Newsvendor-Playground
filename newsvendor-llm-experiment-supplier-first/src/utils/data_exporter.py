"""
Data Export and Storage for Newsvendor Experiment v0.6 - Turn Order Support

Handles saving experimental results in multiple formats with proper
organization and metadata preservation. Enhanced for turn order research.
"""

import json
import csv
import gzip
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import asdict
import pandas as pd

logger = logging.getLogger(__name__)


class DataExporter:
    """Handles data export and storage for experiment results with v0.6 turn order support."""
    
    def __init__(self, storage_config: Optional[Dict[str, Any]] = None):
        """
        Initialize data exporter with v0.6 turn order capabilities.
        
        Args:
            storage_config: Storage configuration dictionary
        """
        self.config = storage_config or {}
        
        # Storage configuration
        self.output_dir = Path(self.config.get('output_dir', './data'))
        self.backup_enabled = self.config.get('backup_enabled', True)
        self.compression_enabled = self.config.get('compression_enabled', True)
        self.export_formats = self.config.get('export_formats', ['csv', 'json'])
        
        # v0.6 Turn order tracking
        self.turn_order_tracking = self.config.get('turn_order_tracking', {
            'enabled': True,
            'separate_files': True,
            'compare_strategies': True
        })
        
        # Create directory structure
        self._create_directory_structure()
        
        logger.info(f"Initialized DataExporter v0.6 with output dir: {self.output_dir}")
        logger.info(f"Turn order tracking: {self.turn_order_tracking['enabled']}")
    
    def _create_directory_structure(self) -> None:
        """Create the required directory structure including v0.6 turn order directories."""
        directories = [
            self.output_dir,
            self.output_dir / 'raw',
            self.output_dir / 'processed', 
            self.output_dir / 'analysis',
            self.output_dir / 'visualizations',
            self.output_dir / 'backups',
            # v0.6 Turn order specific directories
            self.output_dir / 'turn_order_analysis',
            self.output_dir / 'buyer_first',
            self.output_dir / 'supplier_first',
            self.output_dir / 'comparative_analysis'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def save_results(
        self, 
        results: List[Any], 
        phase: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Save experimental results with v0.6 turn order tracking.
        
        Args:
            results: List of negotiation results
            phase: Experiment phase name
            metadata: Optional metadata to include
            
        Returns:
            Dictionary of saved file paths by format
        """
        
        if not results:
            logger.warning(f"No results to save for phase: {phase}")
            return {}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{phase}_{timestamp}"
        
        saved_files = {}
        
        try:
            # Convert results to dictionaries if needed
            processed_results = []
            for result in results:
                if hasattr(result, '__dict__'):
                    processed_results.append(asdict(result))
                elif isinstance(result, dict):
                    processed_results.append(result)
                else:
                    logger.warning(f"Unknown result type: {type(result)}")
                    continue
            
            # v0.6 Analyze turn order distribution
            turn_order_stats = self._analyze_turn_order_distribution(processed_results)
            
            # Add metadata with v0.6 turn order information
            export_data = {
                'metadata': {
                    'version': '0.6',
                    'phase': phase,
                    'timestamp': timestamp,
                    'total_results': len(processed_results),
                    'export_formats': self.export_formats,
                    'compression_enabled': self.compression_enabled,
                    'turn_order_stats': turn_order_stats,
                    **(metadata or {})
                },
                'results': processed_results
            }
            
            # Export in requested formats
            if 'json' in self.export_formats:
                json_path = await self._save_json(export_data, base_filename)
                saved_files['json'] = str(json_path)
            
            if 'csv' in self.export_formats:
                csv_path = await self._save_csv(processed_results, base_filename, phase)
                saved_files['csv'] = str(csv_path)
            
            if 'parquet' in self.export_formats:
                parquet_path = await self._save_parquet(processed_results, base_filename)
                if parquet_path:
                    saved_files['parquet'] = str(parquet_path)
            
            # v0.6 Save turn order specific files
            if self.turn_order_tracking['enabled']:
                turn_order_files = await self._save_by_turn_order(processed_results, base_filename, phase)
                saved_files.update(turn_order_files)
            
            # Create backup if enabled
            if self.backup_enabled:
                backup_path = await self._create_backup(export_data, base_filename)
                saved_files['backup'] = str(backup_path)
            
            logger.info(f"Saved {len(processed_results)} results for phase '{phase}' in {len(saved_files)} formats")
            logger.info(f"Turn order distribution: {turn_order_stats}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving results for phase {phase}: {e}")
            raise
    
    def _analyze_turn_order_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of turn orders in results (v0.6)."""
        
        turn_order_counts = {}
        first_speaker_counts = {}
        strategy_counts = {}
        
        for result in results:
            # Count turn order strategies
            strategy = result.get('turn_order_strategy', 'unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            # Count first speakers
            first_speaker = result.get('first_speaker', 'unknown')
            first_speaker_counts[first_speaker] = first_speaker_counts.get(first_speaker, 0) + 1
            
            # Count combined turn order info
            turn_order_key = f"{strategy}_{first_speaker}"
            turn_order_counts[turn_order_key] = turn_order_counts.get(turn_order_key, 0) + 1
        
        return {
            'total_negotiations': len(results),
            'turn_order_strategies': strategy_counts,
            'first_speakers': first_speaker_counts,
            'detailed_distribution': turn_order_counts,
            'buyer_first_percentage': first_speaker_counts.get('buyer', 0) / len(results) * 100 if results else 0,
            'supplier_first_percentage': first_speaker_counts.get('supplier', 0) / len(results) * 100 if results else 0
        }
    
    async def _save_by_turn_order(
        self, 
        results: List[Dict[str, Any]], 
        base_filename: str, 
        phase: str
    ) -> Dict[str, str]:
        """Save results separated by turn order strategy (v0.6)."""
        
        if not self.turn_order_tracking['separate_files']:
            return {}
        
        saved_files = {}
        
        # Separate results by turn order strategy
        buyer_first_results = [r for r in results if r.get('turn_order_strategy') == 'buyer_first']
        supplier_first_results = [r for r in results if r.get('turn_order_strategy') == 'supplier_first']
        
        # Save buyer-first results
        if buyer_first_results:
            buyer_first_path = self.output_dir / 'buyer_first' / f"{base_filename}_buyer_first.json"
            with open(buyer_first_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'turn_order_strategy': 'buyer_first',
                        'count': len(buyer_first_results),
                        'phase': phase
                    },
                    'results': buyer_first_results
                }, f, indent=2, default=str)
            saved_files['buyer_first_json'] = str(buyer_first_path)
        
        # Save supplier-first results
        if supplier_first_results:
            supplier_first_path = self.output_dir / 'supplier_first' / f"{base_filename}_supplier_first.json"
            with open(supplier_first_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'turn_order_strategy': 'supplier_first',
                        'count': len(supplier_first_results),
                        'phase': phase
                    },
                    'results': supplier_first_results
                }, f, indent=2, default=str)
            saved_files['supplier_first_json'] = str(supplier_first_path)
        
        # Create comparative analysis file
        if buyer_first_results and supplier_first_results:
            comparison_path = await self._create_turn_order_comparison(
                buyer_first_results, supplier_first_results, base_filename
            )
            saved_files['turn_order_comparison'] = str(comparison_path)
        
        return saved_files
    
    async def _create_turn_order_comparison(
        self, 
        buyer_first: List[Dict[str, Any]], 
        supplier_first: List[Dict[str, Any]], 
        base_filename: str
    ) -> Path:
        """Create a comparative analysis between turn order strategies (v0.6)."""
        
        comparison_data = {
            'metadata': {
                'comparison_type': 'turn_order_analysis',
                'buyer_first_count': len(buyer_first),
                'supplier_first_count': len(supplier_first),
                'generated_at': datetime.now().isoformat()
            },
            'buyer_first_summary': self._summarize_results(buyer_first),
            'supplier_first_summary': self._summarize_results(supplier_first),
            'comparative_metrics': self._calculate_comparative_metrics(buyer_first, supplier_first)
        }
        
        output_path = self.output_dir / 'comparative_analysis' / f"{base_filename}_comparison.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        logger.info(f"Created turn order comparison: {output_path}")
        return output_path
    
    def _summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize a set of results for comparison."""
        
        if not results:
            return {"count": 0}
        
        successful = [r for r in results if r.get('completed', False)]
        prices = [r.get('agreed_price') for r in successful if r.get('agreed_price')]
        
        summary = {
            "count": len(results),
            "success_count": len(successful),
            "success_rate": len(successful) / len(results) * 100,
            "avg_rounds": sum(r.get('total_rounds', 0) for r in successful) / len(successful) if successful else 0,
            "avg_tokens": sum(r.get('total_tokens', 0) for r in successful) / len(successful) if successful else 0
        }
        
        if prices:
            summary["price_statistics"] = {
                "count": len(prices),
                "mean": sum(prices) / len(prices),
                "median": sorted(prices)[len(prices)//2],
                "min": min(prices),
                "max": max(prices),
                "std": self._calculate_std(prices),
                "optimal_convergence_rate": sum(1 for p in prices if 60 <= p <= 70) / len(prices) * 100
            }
        
        return summary
    
    def _calculate_comparative_metrics(
        self, 
        buyer_first: List[Dict[str, Any]], 
        supplier_first: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comparative metrics between turn order strategies (v0.6)."""
        
        buyer_summary = self._summarize_results(buyer_first)
        supplier_summary = self._summarize_results(supplier_first)
        
        comparative_metrics = {
            "success_rate_difference": buyer_summary.get('success_rate', 0) - supplier_summary.get('success_rate', 0),
            "avg_rounds_difference": buyer_summary.get('avg_rounds', 0) - supplier_summary.get('avg_rounds', 0),
            "efficiency_difference": buyer_summary.get('avg_tokens', 0) - supplier_summary.get('avg_tokens', 0)
        }
        
        # Price comparisons
        buyer_prices = buyer_summary.get('price_statistics', {})
        supplier_prices = supplier_summary.get('price_statistics', {})
        
        if buyer_prices and supplier_prices:
            comparative_metrics["price_comparisons"] = {
                "mean_price_difference": buyer_prices.get('mean', 0) - supplier_prices.get('mean', 0),
                "median_price_difference": buyer_prices.get('median', 0) - supplier_prices.get('median', 0),
                "buyer_advantage_amount": buyer_prices.get('mean', 0) - supplier_prices.get('mean', 0),  # Negative = supplier advantage
                "convergence_rate_difference": buyer_prices.get('optimal_convergence_rate', 0) - supplier_prices.get('optimal_convergence_rate', 0)
            }
            
            # Research insights
            buyer_advantage = comparative_metrics["price_comparisons"]["buyer_advantage_amount"]
            comparative_metrics["research_insights"] = {
                "buyer_advantage_persists": buyer_advantage > 2.0,  # If supplier-first still has buyer advantage > $2
                "anchoring_effect_size": abs(buyer_advantage),
                "literature_bias_evidence": buyer_advantage > 0,  # Positive means buyers still advantaged even when going second
                "hypothesis_support": {
                    "H1_literature_bias": buyer_advantage > 5.0,  # Strong persistence suggests literature bias
                    "H2_anchoring_only": abs(buyer_advantage) < 2.0,  # Small difference suggests anchoring was main effect
                    "H3_mixed_effects": 2.0 <= abs(buyer_advantage) <= 5.0  # Moderate difference suggests both effects
                }
            }
        
        return comparative_metrics
    
    async def _save_json(self, data: Dict[str, Any], base_filename: str) -> Path:
        """Save data as JSON file."""
        
        output_path = self.output_dir / 'raw' / f"{base_filename}.json"
        
        if self.compression_enabled:
            output_path = output_path.with_suffix('.json.gz')
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        
        logger.debug(f"Saved JSON to: {output_path}")
        return output_path
    
    async def _save_csv(self, results: List[Dict[str, Any]], base_filename: str, phase: str) -> Path:
        """Save results as CSV file with flattened structure including v0.6 turn order fields."""
        
        if not results:
            return None
        
        # Flatten nested dictionaries for CSV format
        flattened_results = []
        for result in results:
            flattened = self._flatten_dict(result)
            flattened_results.append(flattened)
        
        # Create DataFrame
        df = pd.DataFrame(flattened_results)
        
        # Ensure v0.6 turn order columns are present
        v06_columns = ['turn_order_strategy', 'first_speaker']
        for col in v06_columns:
            if col not in df.columns:
                df[col] = 'unknown'
        
        # Save to CSV
        output_path = self.output_dir / 'processed' / f"{base_filename}.csv"
        
        if self.compression_enabled:
            output_path = output_path.with_suffix('.csv.gz')
            df.to_csv(output_path, index=False, compression='gzip')
        else:
            df.to_csv(output_path, index=False)
        
        logger.debug(f"Saved CSV with v0.6 turn order columns to: {output_path}")
        return output_path
    
    async def _save_parquet(self, results: List[Dict[str, Any]], base_filename: str) -> Optional[Path]:
        """Save results as Parquet file."""
        
        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning("PyArrow not available, skipping Parquet export")
            return None
        
        # Flatten and create DataFrame
        flattened_results = [self._flatten_dict(result) for result in results]
        df = pd.DataFrame(flattened_results)
        
        # Save to Parquet
        output_path = self.output_dir / 'processed' / f"{base_filename}.parquet"
        df.to_parquet(output_path, index=False)
        
        logger.debug(f"Saved Parquet to: {output_path}")
        return output_path
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to string representation for CSV
                if v and isinstance(v[0], dict):
                    # List of dictionaries - convert to JSON string
                    items.append((new_key, json.dumps(v, default=str)))
                else:
                    # Simple list - join as string
                    items.append((new_key, '|'.join(map(str, v))))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    async def _create_backup(self, data: Dict[str, Any], base_filename: str) -> Path:
        """Create compressed backup of data."""
        
        backup_path = self.output_dir / 'backups' / f"{base_filename}_backup.json.gz"
        
        with gzip.open(backup_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.debug(f"Created backup: {backup_path}")
        return backup_path
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    async def export_turn_order_analysis(self, results: List[Dict[str, Any]]) -> Path:
        """Export comprehensive turn order analysis (v0.6)."""
        
        analysis = {
            'metadata': {
                'analysis_type': 'comprehensive_turn_order',
                'version': '0.6',
                'generated_at': datetime.now().isoformat(),
                'total_negotiations': len(results)
            },
            'distribution_analysis': self._analyze_turn_order_distribution(results),
            'performance_by_strategy': {},
            'anchoring_vs_literature_analysis': {},
            'research_conclusions': {}
        }
        
        # Analyze performance by turn order strategy
        strategies = set(r.get('turn_order_strategy', 'unknown') for r in results)
        for strategy in strategies:
            strategy_results = [r for r in results if r.get('turn_order_strategy') == strategy]
            analysis['performance_by_strategy'][strategy] = self._summarize_results(strategy_results)
        
        # Anchoring vs Literature Bias Analysis
        buyer_first = [r for r in results if r.get('turn_order_strategy') == 'buyer_first']
        supplier_first = [r for r in results if r.get('turn_order_strategy') == 'supplier_first']
        
        if buyer_first and supplier_first:
            analysis['anchoring_vs_literature_analysis'] = self._calculate_comparative_metrics(buyer_first, supplier_first)
        
        # Research conclusions
        if 'anchoring_vs_literature_analysis' in analysis:
            comparative = analysis['anchoring_vs_literature_analysis']
            price_comp = comparative.get('price_comparisons', {})
            buyer_advantage = price_comp.get('buyer_advantage_amount', 0)
            
            analysis['research_conclusions'] = {
                'primary_finding': f"Buyer advantage of ${abs(buyer_advantage):.2f} {'persists' if buyer_advantage > 0 else 'disappears'} when suppliers go first",
                'anchoring_effect_size': price_comp.get('mean_price_difference', 0),
                'literature_bias_evidence': buyer_advantage > 2.0,
                'mixed_effects_likely': 2.0 <= abs(buyer_advantage) <= 8.0,
                'recommended_interpretation': self._interpret_results(buyer_advantage)
            }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self.output_dir / 'turn_order_analysis' / f"comprehensive_turn_order_analysis_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Exported comprehensive turn order analysis to: {output_path}")
        return output_path
    
    def _interpret_results(self, buyer_advantage: float) -> str:
        """Interpret research results based on buyer advantage amount."""
        
        if abs(buyer_advantage) < 1.0:
            return "No significant turn order effect detected. Results suggest minimal bias from either source."
        elif buyer_advantage > 8.0:
            return "Strong literature bias detected. Buyer advantage persists despite suppliers going first, indicating systematic bias in training data."
        elif buyer_advantage > 2.0:
            return "Mixed effects detected. Both literature bias and anchoring effects likely contribute to buyer advantage."
        elif buyer_advantage > 0:
            return "Weak literature bias detected. Small buyer advantage persists, suggesting minor systematic bias."
        else:
            return "Anchoring effect detected. Supplier advantage when going first suggests turn order was the primary factor."
    
    # Keep all existing methods from the original DataExporter
    async def export_analysis(self, analysis: Dict[str, Any], filename: str = None) -> Path:
        """Export analysis results."""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"analysis_{timestamp}.json"
        
        output_path = self.output_dir / 'analysis' / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Exported analysis to: {output_path}")
        return output_path
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage usage including v0.6 directories."""
        
        storage_info = {
            "output_directory": str(self.output_dir),
            "version": "0.6",
            "directories": {},
            "total_size_mb": 0
        }
        
        # Calculate directory sizes including v0.6 specific directories
        subdirs = ['raw', 'processed', 'analysis', 'visualizations', 'backups', 
                  'turn_order_analysis', 'buyer_first', 'supplier_first', 'comparative_analysis']
        
        for subdir in subdirs:
            dir_path = self.output_dir / subdir
            if dir_path.exists():
                size_bytes = sum(
                    f.stat().st_size for f in dir_path.rglob('*') if f.is_file()
                )
                size_mb = size_bytes / (1024 * 1024)
                storage_info["directories"][subdir] = {
                    "size_mb": round(size_mb, 2),
                    "file_count": len([f for f in dir_path.rglob('*') if f.is_file()])
                }
                storage_info["total_size_mb"] += size_mb
            else:
                storage_info["directories"][subdir] = {
                    "size_mb": 0,
                    "file_count": 0
                }
        
        storage_info["total_size_mb"] = round(storage_info["total_size_mb"], 2)
        return storage_info