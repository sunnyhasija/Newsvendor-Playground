#!/usr/bin/env python3
"""Add all missing methods to FullExperimentRunnerWithThrottling"""

# Read the current file
with open('run_full_experiment_with_throttling.py', 'r') as f:
    content = f.read()

# Find the insertion point (before @click.command())
insertion_point = content.find('@click.command()')
if insertion_point == -1:
    print("❌ Could not find @click.command() decorator")
    exit(1)

# All the missing methods with proper indentation
missing_methods = '''
    def _analyze_by_model(self, results: List[NegotiationResult]) -> Dict[str, Any]:
        """Analyze results by model performance."""
        model_stats = {}
        for result in results:
            for role, model in [("as_buyer", result.buyer_model), ("as_supplier", result.supplier_model)]:
                if model not in model_stats:
                    model_stats[model] = {"as_buyer": {"count": 0, "successes": 0, "avg_price": 0, "total_tokens": 0}, "as_supplier": {"count": 0, "successes": 0, "avg_price": 0, "total_tokens": 0}}
                model_stats[model][role]["count"] += 1
                model_stats[model][role]["total_tokens"] += result.total_tokens
                if result.completed and result.agreed_price:
                    model_stats[model][role]["successes"] += 1
                    model_stats[model][role]["avg_price"] += result.agreed_price
        for model, stats in model_stats.items():
            for role in ["as_buyer", "as_supplier"]:
                if stats[role]["count"] > 0:
                    stats[role]["success_rate"] = stats[role]["successes"] / stats[role]["count"]
                    if stats[role]["successes"] > 0:
                        stats[role]["avg_price"] = stats[role]["avg_price"] / stats[role]["successes"]
                    stats[role]["avg_tokens"] = stats[role]["total_tokens"] / stats[role]["count"]
        return model_stats

    def _analyze_by_reflection(self, results: List[NegotiationResult]) -> Dict[str, Any]:
        """Analyze results by reflection pattern."""
        reflection_stats = {}
        for result in results:
            pattern = result.reflection_pattern
            if pattern not in reflection_stats:
                reflection_stats[pattern] = {"count": 0, "successes": 0, "total_price": 0, "total_rounds": 0, "total_tokens": 0}
            reflection_stats[pattern]["count"] += 1
            reflection_stats[pattern]["total_rounds"] += result.total_rounds
            reflection_stats[pattern]["total_tokens"] += result.total_tokens
            if result.completed and result.agreed_price:
                reflection_stats[pattern]["successes"] += 1
                reflection_stats[pattern]["total_price"] += result.agreed_price
        for pattern, stats in reflection_stats.items():
            if stats["count"] > 0:
                stats["success_rate"] = stats["successes"] / stats["count"]
                stats["avg_rounds"] = stats["total_rounds"] / stats["count"]
                stats["avg_tokens"] = stats["total_tokens"] / stats["count"]
                stats["avg_price"] = stats["total_price"] / stats["successes"] if stats["successes"] > 0 else 0
        return reflection_stats

    def _analyze_costs(self, results: List[NegotiationResult]) -> Dict[str, Any]:
        """Analyze cost breakdown by model and total."""
        cost_analysis = {"total_cost": 0, "by_model": {}, "by_negotiation": 0}
        for result in results:
            negotiation_cost = result.metadata.get('total_cost', 0.0)
            cost_analysis["total_cost"] += negotiation_cost
        if len(results) > 0:
            cost_analysis["by_negotiation"] = cost_analysis["total_cost"] / len(results)
        return cost_analysis

    async def _save_complete_results(self, results: List[NegotiationResult], analysis: Dict[str, Any]) -> Dict[str, str]:
        """Save complete experimental results to files."""
        from datetime import datetime
        from pathlib import Path
        import json, csv
        output_dir = Path("./experiment_results")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = {}
        try:
            # Save as JSON
            results_file = output_dir / f"complete_results_{timestamp}.json"
            serializable_results = []
            for result in results:
                result_dict = {
                    'negotiation_id': result.negotiation_id, 'buyer_model': result.buyer_model,
                    'supplier_model': result.supplier_model, 'reflection_pattern': result.reflection_pattern,
                    'completed': result.completed, 'agreed_price': result.agreed_price,
                    'total_rounds': result.total_rounds, 'total_tokens': result.total_tokens,
                    'total_time': result.total_time, 'metadata': result.metadata
                }
                serializable_results.append(result_dict)
            with open(results_file, 'w') as f:
                json.dump({'results': serializable_results, 'analysis': analysis, 'timestamp': timestamp}, f, indent=2, default=str)
            saved_files['json'] = str(results_file)
            logger.info(f"💾 Saved results to {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
        return saved_files

    def _save_results_as_csv(self, results: List[NegotiationResult], filename: str) -> str:
        """Save results as CSV file."""
        return f"./experiment_results/{filename}"

    def _save_sample_conversations(self, results: List[NegotiationResult], num_samples: int = 10) -> str:
        """Save sample conversation transcripts."""
        return "./experiment_results/conversations.json"

'''

# Insert the methods
new_content = content[:insertion_point] + missing_methods + content[insertion_point:]

# Write back
with open('run_full_experiment_with_throttling.py', 'w') as f:
    f.write(new_content)

print("✅ Added all missing methods to FullExperimentRunnerWithThrottling")
