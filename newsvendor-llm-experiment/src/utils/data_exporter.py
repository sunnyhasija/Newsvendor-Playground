"""
Data Export and Storage for Newsvendor Experiment

Handles saving experimental results in multiple formats with proper
organization and metadata preservation.
"""

import json
import csv
import gzip
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
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
    
    async def _save_parquet(self, results: List[Dict[str, Any]], base_filename: str) -> Path:
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
            dir_path =