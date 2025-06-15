"""
Data Export and Storage for Newsvendor Experiment

Handles saving experimental results in multiple formats with proper
organization and metadata preservation.
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
    """Handles data export and storage for experiment results."""
    
    def __init__(self, storage_config: Optional[Dict[str, Any]] = None):
        """
        Initialize data exporter.
        
        Args:
            storage_config: Storage configuration dictionary
        """
        self.config = storage_config or {}
        
        # Storage configuration
        self.output_dir = Path(self.config.get('output_dir', './data'))
        self.backup_enabled = self.config.get('backup_enabled', True)
        self.compression_enabled = self.config.get('compression_enabled', True)
        self.export_formats = self.config.get('export_formats', ['csv', 'json'])
        
        # Create directory structure
        self._create_directory_structure()
        
        logger.info(f"Initialized DataExporter with output dir: {self.output_dir}")
    
    def _create_directory_structure(self) -> None:
        """Create the required directory structure."""
        directories = [
            self.output_dir,
            self.output_dir / 'raw',
            self.output_dir / 'processed', 
            self.output_dir / 'analysis',
            self.output_dir / 'visualizations',
            self.output_dir / 'backups'
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
        Save experimental results to files.
        
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
            
            # Add metadata
            export_data = {
                'metadata': {
                    'phase': phase,
                    'timestamp': timestamp,
                    'total_results': len(processed_results),
                    'export_formats': self.export_formats,
                    'compression_enabled': self.compression_enabled,
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
            
            # Create backup if enabled
            if self.backup_enabled:
                backup_path = await self._create_backup(export_data, base_filename)
                saved_files['backup'] = str(backup_path)
            
            logger.info(f"Saved {len(processed_results)} results for phase '{phase}' in {len(saved_files)} formats")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving results for phase {phase}: {e}")
            raise
    
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
        """Save results as CSV file with flattened structure."""
        
        if not results:
            return None
        
        # Flatten nested dictionaries for CSV format
        flattened_results = []
        for result in results:
            flattened = self._flatten_dict(result)
            flattened_results.append(flattened)
        
        # Create DataFrame
        df = pd.DataFrame(flattened_results)
        
        # Save to CSV
        output_path = self.output_dir / 'processed' / f"{base_filename}.csv"
        
        if self.compression_enabled:
            output_path = output_path.with_suffix('.csv.gz')
            df.to_csv(output_path, index=False, compression='gzip')
        else:
            df.to_csv(output_path, index=False)
        
        logger.debug(f"Saved CSV to: {output_path}")
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
    
    def create_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary report from results."""
        
        if not results:
            return {"error": "No results to summarize"}
        
        # Basic statistics
        total_negotiations = len(results)
        successful = [r for r in results if r.get('completed', False)]
        
        summary = {
            "experiment_overview": {
                "total_negotiations": total_negotiations,
                "successful_negotiations": len(successful),
                "success_rate": len(successful) / total_negotiations if total_negotiations > 0 else 0,
                "generated_at": datetime.now().isoformat()
            }
        }
        
        if successful:
            # Price analysis
            prices = [r.get('agreed_price') for r in successful if r.get('agreed_price')]
            if prices:
                summary["price_statistics"] = {
                    "count": len(prices),
                    "mean": sum(prices) / len(prices),
                    "median": sorted(prices)[len(prices)//2],
                    "min": min(prices),
                    "max": max(prices),
                    "std": self._calculate_std(prices)
                }
            
            # Efficiency analysis
            tokens = [r.get('total_tokens', 0) for r in successful]
            rounds = [r.get('total_rounds', 0) for r in successful]
            
            summary["efficiency_statistics"] = {
                "avg_tokens": sum(tokens) / len(tokens) if tokens else 0,
                "avg_rounds": sum(rounds) / len(rounds) if rounds else 0,
                "tokens_per_round": sum(tokens) / sum(rounds) if sum(rounds) > 0 else 0
            }
            
            # Model analysis
            model_stats = {}
            for result in successful:
                buyer_model = result.get('buyer_model', 'unknown')
                supplier_model = result.get('supplier_model', 'unknown')
                
                for model in [buyer_model, supplier_model]:
                    if model not in model_stats:
                        model_stats[model] = {"count": 0, "success": 0}
                    model_stats[model]["count"] += 1
                    if result.get('completed'):
                        model_stats[model]["success"] += 1
            
            # Calculate success rates
            for model, stats in model_stats.items():
                stats["success_rate"] = stats["success"] / stats["count"] if stats["count"] > 0 else 0
            
            summary["model_statistics"] = model_stats
        
        return summary
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage usage."""
        
        storage_info = {
            "output_directory": str(self.output_dir),
            "directories": {},
            "total_size_mb": 0
        }
        
        # Calculate directory sizes
        for subdir in ['raw', 'processed', 'analysis', 'visualizations', 'backups']:
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
    
    def export_conversations(self, results: List[Dict[str, Any]], format: str = 'json') -> Path:
        """
        Export detailed conversation transcripts.
        
        Args:
            results: List of negotiation results with conversation data
            format: Export format ('json', 'csv', or 'txt')
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            output_path = self.output_dir / 'raw' / f"conversations_{timestamp}.json"
            
            conversations = []
            for result in results:
                if 'turns' in result:
                    conversation = {
                        'negotiation_id': result.get('negotiation_id'),
                        'buyer_model': result.get('buyer_model'),
                        'supplier_model': result.get('supplier_model'),
                        'reflection_pattern': result.get('reflection_pattern'),
                        'agreed_price': result.get('agreed_price'),
                        'turns': result['turns']
                    }
                    conversations.append(conversation)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, indent=2, default=str)
        
        elif format == 'csv':
            output_path = self.output_dir / 'processed' / f"conversations_{timestamp}.csv"
            
            # Flatten conversation turns for CSV
            rows = []
            for result in results:
                if 'turns' in result:
                    for turn in result['turns']:
                        row = {
                            'negotiation_id': result.get('negotiation_id'),
                            'buyer_model': result.get('buyer_model'),
                            'supplier_model': result.get('supplier_model'),
                            'reflection_pattern': result.get('reflection_pattern'),
                            'final_price': result.get('agreed_price'),
                            'round_number': turn.get('round_number'),
                            'speaker': turn.get('speaker'),
                            'message': turn.get('message'),
                            'price': turn.get('price'),
                            'tokens_used': turn.get('tokens_used'),
                            'generation_time': turn.get('generation_time')
                        }
                        rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        
        elif format == 'txt':
            output_path = self.output_dir / 'raw' / f"conversations_{timestamp}.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    if 'turns' in result:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"NEGOTIATION: {result.get('negotiation_id')}\n")
                        f.write(f"BUYER: {result.get('buyer_model')}\n")
                        f.write(f"SUPPLIER: {result.get('supplier_model')}\n")
                        f.write(f"REFLECTION: {result.get('reflection_pattern')}\n")
                        f.write(f"FINAL PRICE: ${result.get('agreed_price')}\n")
                        f.write(f"{'='*80}\n\n")
                        
                        for turn in result['turns']:
                            speaker = turn.get('speaker', 'unknown').upper()
                            message = turn.get('message', '')
                            price = turn.get('price')
                            round_num = turn.get('round_number')
                            
                            f.write(f"Round {round_num} - {speaker}:\n")
                            f.write(f"  Message: {message}\n")
                            if price:
                                f.write(f"  Price: ${price}\n")
                            f.write(f"\n")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported conversations to: {output_path}")
        return output_path
    
    def create_performance_dashboard(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a performance dashboard with key metrics."""
        
        if not results:
            return {"error": "No results for dashboard"}
        
        successful_results = [r for r in results if r.get('completed', False)]
        
        dashboard = {
            "overview": {
                "total_negotiations": len(results),
                "successful_rate": len(successful_results) / len(results) * 100,
                "avg_rounds": sum(r.get('total_rounds', 0) for r in successful_results) / len(successful_results) if successful_results else 0,
                "avg_tokens": sum(r.get('total_tokens', 0) for r in successful_results) / len(successful_results) if successful_results else 0
            },
            "price_metrics": {},
            "model_performance": {},
            "reflection_impact": {},
            "efficiency_metrics": {}
        }
        
        if successful_results:
            prices = [r.get('agreed_price') for r in successful_results if r.get('agreed_price')]
            
            if prices:
                dashboard["price_metrics"] = {
                    "mean_price": sum(prices) / len(prices),
                    "median_price": sorted(prices)[len(prices)//2],
                    "price_range": f"${min(prices)} - ${max(prices)}",
                    "optimal_convergence": sum(1 for p in prices if 60 <= p <= 70) / len(prices) * 100,
                    "price_std": self._calculate_std(prices)
                }
            
            # Model performance analysis
            model_performance = {}
            for result in successful_results:
                for role in ['buyer_model', 'supplier_model']:
                    model = result.get(role)
                    if model:
                        if model not in model_performance:
                            model_performance[model] = {
                                'negotiations': 0,
                                'successes': 0,
                                'avg_price': 0,
                                'avg_tokens': 0
                            }
                        
                        model_performance[model]['negotiations'] += 1
                        if result.get('completed'):
                            model_performance[model]['successes'] += 1
                        
                        if result.get('agreed_price'):
                            model_performance[model]['avg_price'] += result['agreed_price']
                        
                        model_performance[model]['avg_tokens'] += result.get('total_tokens', 0)
            
            # Calculate averages
            for model, stats in model_performance.items():
                if stats['negotiations'] > 0:
                    stats['success_rate'] = stats['successes'] / stats['negotiations'] * 100
                    stats['avg_price'] = stats['avg_price'] / stats['negotiations']
                    stats['avg_tokens'] = stats['avg_tokens'] / stats['negotiations']
            
            dashboard["model_performance"] = model_performance
            
            # Reflection impact analysis
            reflection_stats = {}
            for result in successful_results:
                pattern = result.get('reflection_pattern', '00')
                if pattern not in reflection_stats:
                    reflection_stats[pattern] = {
                        'count': 0,
                        'success_rate': 0,
                        'avg_price': 0,
                        'avg_rounds': 0
                    }
                
                reflection_stats[pattern]['count'] += 1
                if result.get('agreed_price'):
                    reflection_stats[pattern]['avg_price'] += result['agreed_price']
                reflection_stats[pattern]['avg_rounds'] += result.get('total_rounds', 0)
            
            # Calculate reflection averages
            for pattern, stats in reflection_stats.items():
                if stats['count'] > 0:
                    stats['avg_price'] = stats['avg_price'] / stats['count']
                    stats['avg_rounds'] = stats['avg_rounds'] / stats['count']
                    stats['success_rate'] = 100  # All in successful_results
            
            dashboard["reflection_impact"] = reflection_stats
        
        return dashboard
    
    async def export_dashboard(self, results: List[Dict[str, Any]]) -> Path:
        """Export performance dashboard as JSON."""
        dashboard = self.create_performance_dashboard(results)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self.output_dir / 'analysis' / f"dashboard_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dashboard, f, indent=2, default=str)
        
        logger.info(f"Exported dashboard to: {output_path}")
        return output_path
    
    def cleanup_old_files(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up files older than specified days."""
        
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(days=days_old)
        
        cleanup_stats = {
            "files_removed": 0,
            "space_freed_mb": 0
        }
        
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                file_path = Path(root) / file
                
                if file_path.stat().st_mtime < cutoff_time.timestamp():
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    
                    cleanup_stats["files_removed"] += 1
                    cleanup_stats["space_freed_mb"] += file_size / (1024 * 1024)
        
        cleanup_stats["space_freed_mb"] = round(cleanup_stats["space_freed_mb"], 2)
        logger.info(f"Cleanup complete: {cleanup_stats}")
        
        return cleanup_stats