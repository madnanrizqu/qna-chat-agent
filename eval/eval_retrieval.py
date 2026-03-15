"""Evaluate retrieval quality of vector store search."""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from embeddings import vector_store
from config import settings


def evaluate_retrieval(queries_file: Path, output_file: Path) -> None:
    """Run retrieval evaluation and generate metrics.

    Args:
        queries_file: Path to retrieval_queries.json
        output_file: Path to save evaluation results
    """
    # Load queries
    with open(queries_file) as f:
        queries = json.load(f)

    # Run evaluation
    results = {}
    category_stats = defaultdict(
        lambda: {
            "total_queries": 0,
            "queries_with_results": 0,
            "queries_with_correct_results": 0,
            "total_results": 0,
            "correct_results": 0,
        }
    )

    for category, query_list in queries.items():
        category_results = []

        for query in query_list:
            search_results = vector_store.search_similar(query)

            # Collect result metadata
            result_categories = []
            for result in search_results:
                result_categories.append(
                    {
                        "content": (
                            result.content[:100] + "..."
                            if len(result.content) > 100
                            else result.content
                        ),
                        "similarity": result.similarity,
                        "matched_category": result.category,
                    }
                )

            # Calculate metrics for this query
            has_results = len(search_results) > 0

            # For "not_matching", correct = no results or all results have no category match
            if category == "not_matching":
                correct_hit = not has_results or all(
                    r["matched_category"] is None for r in result_categories
                )
                num_correct = 0 if has_results else 1
            else:
                # For other categories, correct = at least one result matches expected category
                correct_hit = any(
                    r["matched_category"] == category for r in result_categories
                )
                num_correct = sum(
                    1 for r in result_categories if r["matched_category"] == category
                )

            category_results.append(
                {
                    "query": query,
                    "num_results": len(search_results),
                    "results": result_categories,
                    "correct_hit": correct_hit,
                }
            )

            # Update stats
            stats = category_stats[category]
            stats["total_queries"] += 1
            if has_results:
                stats["queries_with_results"] += 1
            if correct_hit:
                stats["queries_with_correct_results"] += 1
            stats["total_results"] += len(search_results)
            stats["correct_results"] += num_correct

        results[category] = category_results

    # Calculate aggregate metrics
    metrics = {}
    for category, stats in category_stats.items():
        recall = (
            stats["queries_with_correct_results"] / stats["total_queries"]
            if stats["total_queries"] > 0
            else 0.0
        )

        # For not_matching: if no results returned, precision is 100% (perfect rejection)
        if category == "not_matching" and stats["total_results"] == 0:
            precision = 1.0
        else:
            precision = (
                stats["correct_results"] / stats["total_results"]
                if stats["total_results"] > 0
                else 0.0
            )

        metrics[category] = {
            "recall": recall,
            "precision": precision,
            "total_queries": stats["total_queries"],
            "queries_with_correct_results": stats["queries_with_correct_results"],
            "total_results": stats["total_results"],
            "correct_results": stats["correct_results"],
        }

    # Calculate overall metrics (excluding not_matching for aggregation)
    # not_matching uses different logic that breaks overall precision calculation
    relevant_metrics = {k: v for k, v in metrics.items() if k != "not_matching"}

    total_queries = sum(m["total_queries"] for m in relevant_metrics.values())
    total_correct_queries = sum(
        m["queries_with_correct_results"] for m in relevant_metrics.values()
    )
    total_results = sum(m["total_results"] for m in relevant_metrics.values())
    total_correct_results = sum(m["correct_results"] for m in relevant_metrics.values())

    overall_recall = total_correct_queries / total_queries if total_queries > 0 else 0.0
    overall_precision = (
        total_correct_results / total_results if total_results > 0 else 0.0
    )

    metrics["overall"] = {
        "recall": overall_recall,
        "precision": overall_precision,
        "total_queries": total_queries,
        "queries_with_correct_results": total_correct_queries,
        "total_results": total_results,
        "correct_results": total_correct_results,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n{'Category':<25} {'Recall':<10} {'Precision':<10} {'Queries':<10}")
    print("-" * 70)

    for category in sorted(metrics.keys()):
        if category == "overall":
            continue
        m = metrics[category]
        print(
            f"{category:<25} {m['recall']:.2%}     {m['precision']:.2%}     "
            f"{m['queries_with_correct_results']}/{m['total_queries']}"
        )

    print("-" * 70)
    m = metrics["overall"]
    print(
        f"{'OVERALL*':<25} {m['recall']:.2%}     {m['precision']:.2%}     "
        f"{m['queries_with_correct_results']}/{m['total_queries']}"
    )
    print("=" * 70)
    print("* Overall excludes not_matching category")
    print("  (not_matching: 100% precision = all queries correctly rejected)\n")

    # Save detailed results
    output_data = {
        "metrics": metrics,
        "detailed_results": results,
        "summary": {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "queries_file": str(queries_file),
            "total_categories": len(metrics) - 1,  # Exclude 'overall' from count
        },
        "metadata": {
            "model": settings.default_model,
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
            "similarity_threshold": settings.similarity_threshold,
            "max_search_results": settings.max_search_results,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "use_chunked_storage": settings.use_chunked_storage,
            "environment": settings.environment,
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Detailed results saved to {output_file}\n")


if __name__ == "__main__":
    eval_dir = Path(__file__).parent
    queries_file = eval_dir / "retrieval_queries.json"
    output_file = eval_dir / "eval_results.json"

    evaluate_retrieval(queries_file, output_file)
