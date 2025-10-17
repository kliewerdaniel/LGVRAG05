"""
Comprehensive evaluation framework for RAG system.

This module provides tools for evaluating retrieval quality, answer accuracy,
and overall system performance with industry-standard metrics.
"""

import os
import json
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from rag.retrieve import HybridRetriever
from rag.generate_answer import AnswerGenerator

logger = logging.getLogger(__name__)


@dataclass
class EvaluationQuery:
    """Represents a test query with ground truth."""
    id: str
    query: str
    ground_truth_answers: List[str]
    ground_truth_chunks: List[str]
    metadata: Dict[str, Any]


@dataclass
class EvaluationResult:
    """Results of evaluating a single query."""
    query_id: str
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    generated_answer: str
    retrieval_metrics: Dict[str, float]
    generation_metrics: Dict[str, float]
    execution_time: float


class RAGEvaluator:
    """
    Comprehensive evaluation framework for RAG systems.

    This class implements industry-standard metrics for evaluating
    retrieval quality, answer faithfulness, and overall performance.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the RAG evaluator."""
        self.config = get_config(config_path)
        self.retriever = HybridRetriever(config_path)
        self.answer_generator = AnswerGenerator(config_path)

        # Load evaluation dataset
        self.test_queries = self._load_evaluation_dataset()

    def _load_evaluation_dataset(self) -> List[EvaluationQuery]:
        """Load evaluation dataset from file or create synthetic one."""
        dataset_path = "evaluation/test_dataset.json"

        if os.path.exists(dataset_path):
            try:
                with open(dataset_path, 'r') as f:
                    data = json.load(f)

                queries = []
                for item in data.get("queries", []):
                    query = EvaluationQuery(
                        id=item["id"],
                        query=item["query"],
                        ground_truth_answers=item.get("ground_truth_answers", []),
                        ground_truth_chunks=item.get("ground_truth_chunks", []),
                        metadata=item.get("metadata", {})
                    )
                    queries.append(query)

                logger.info(f"Loaded {len(queries)} evaluation queries")
                return queries

            except Exception as e:
                logger.error(f"Error loading evaluation dataset: {e}")

        # Create synthetic evaluation dataset for testing
        logger.info("Creating synthetic evaluation dataset")
        return self._create_synthetic_dataset()

    def _create_synthetic_dataset(self) -> List[EvaluationQuery]:
        """Create a synthetic dataset for testing."""
        return [
            EvaluationQuery(
                id="q1",
                query="What is machine learning?",
                ground_truth_answers=[
                    "Machine learning is a subset of artificial intelligence",
                    "It enables computers to learn from data without explicit programming"
                ],
                ground_truth_chunks=["chunk_ml_definition"],
                metadata={"difficulty": "easy", "topic": "ml_basics"}
            ),
            EvaluationQuery(
                id="q2",
                query="How do neural networks work?",
                ground_truth_answers=[
                    "Neural networks consist of layers of interconnected nodes",
                    "They learn by adjusting weights through backpropagation"
                ],
                ground_truth_chunks=["chunk_nn_basics", "chunk_backprop"],
                metadata={"difficulty": "medium", "topic": "neural_networks"}
            ),
            EvaluationQuery(
                id="q3",
                query="What is the difference between supervised and unsupervised learning?",
                ground_truth_answers=[
                    "Supervised learning uses labeled data",
                    "Unsupervised learning finds patterns in unlabeled data"
                ],
                ground_truth_chunks=["chunk_supervised", "chunk_unsupervised"],
                metadata={"difficulty": "medium", "topic": "ml_types"}
            )
        ]

    def evaluate_retrieval(self, test_queries: Optional[List[EvaluationQuery]] = None) -> Dict[str, Any]:
        """
        Evaluate retrieval quality using standard IR metrics.

        Args:
            test_queries: List of queries to evaluate (uses default if None)

        Returns:
            Dictionary with retrieval evaluation metrics
        """
        if test_queries is None:
            test_queries = self.test_queries

        logger.info(f"Evaluating retrieval for {len(test_queries)} queries")

        all_metrics = {
            "recall@k": [],
            "precision@k": [],
            "mrr": [],  # Mean Reciprocal Rank
            "ndcg@k": [],  # Normalized Discounted Cumulative Gain
            "retrieval_time": []
        }

        for query in test_queries:
            start_time = time.time()

            # Perform retrieval
            retrieved_chunks = self.retriever.retrieve(query.query, top_k=10)

            retrieval_time = time.time() - start_time

            # Calculate metrics
            metrics = self._calculate_retrieval_metrics(query, retrieved_chunks)
            all_metrics["retrieval_time"].append(retrieval_time)

            for metric_name, value in metrics.items():
                if metric_name in all_metrics:
                    all_metrics[metric_name].append(value)

        # Calculate aggregate statistics
        results = {}
        for metric_name, values in all_metrics.items():
            if values:
                results[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }

        logger.info(f"Retrieval evaluation completed: {results}")
        return results

    def _calculate_retrieval_metrics(self, query: EvaluationQuery, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate retrieval metrics for a single query."""
        # Get ground truth chunk IDs
        ground_truth_ids = set(query.ground_truth_chunks)

        # Get retrieved chunk IDs
        retrieved_ids = set(chunk.get("chunk_id", chunk.get("id", "")) for chunk in retrieved_chunks)

        # Calculate Recall@k (k=10)
        relevant_retrieved = len(ground_truth_ids.intersection(retrieved_ids))
        recall = relevant_retrieved / len(ground_truth_ids) if ground_truth_ids else 0.0

        # Calculate Precision@k (k=10)
        precision = relevant_retrieved / len(retrieved_chunks) if retrieved_chunks else 0.0

        # Calculate Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for i, chunk in enumerate(retrieved_chunks):
            chunk_id = chunk.get("chunk_id", chunk.get("id", ""))
            if chunk_id in ground_truth_ids:
                mrr = 1.0 / (i + 1)
                break

        # Calculate NDCG@k (simplified version)
        ndcg = self._calculate_ndcg(ground_truth_ids, retrieved_chunks)

        return {
            "recall@k": recall,
            "precision@k": precision,
            "mrr": mrr,
            "ndcg@k": ndcg
        }

    def _calculate_ndcg(self, ground_truth_ids: set, retrieved_chunks: List[Dict[str, Any]], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not ground_truth_ids:
            return 0.0

        # Create relevance scores (1 for relevant, 0 for non-relevant)
        relevance_scores = []
        for chunk in retrieved_chunks[:k]:
            chunk_id = chunk.get("chunk_id", chunk.get("id", ""))
            relevance = 1.0 if chunk_id in ground_truth_ids else 0.0
            relevance_scores.append(relevance)

        # Calculate DCG
        dcg = 0.0
        for i, relevance in enumerate(relevance_scores):
            dcg += relevance / np.log2(i + 2)  # +2 because i starts from 0

        # Calculate IDCG (Ideal DCG)
        ideal_relevance = [1.0] * min(len(ground_truth_ids), k)
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevance):
            idcg += relevance / np.log2(i + 2)

        # Return NDCG
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_generation(self, test_queries: Optional[List[EvaluationQuery]] = None) -> Dict[str, Any]:
        """
        Evaluate answer generation quality.

        Args:
            test_queries: List of queries to evaluate

        Returns:
            Dictionary with generation evaluation metrics
        """
        if test_queries is None:
            test_queries = self.test_queries

        logger.info(f"Evaluating generation for {len(test_queries)} queries")

        results = []

        for query in test_queries:
            try:
                # Perform full RAG pipeline
                retrieval_results = self.retriever.retrieve_with_entities(query.query, top_k=5)

                # Generate answer
                answer = self.answer_generator.generate_answer_sync(
                    query=query.query,
                    context_chunks=retrieval_results["chunks"],
                    entities=retrieval_results.get("entities", []),
                    relationships=retrieval_results.get("relationships", [])
                )

                # Evaluate answer quality
                quality_metrics = self._evaluate_answer_quality(query, answer)

                result = {
                    "query_id": query.id,
                    "query": query.query,
                    "generated_answer": answer.answer,
                    "quality_metrics": quality_metrics,
                    "execution_time": answer.generation_time,
                    "tokens_used": answer.tokens_used
                }

                results.append(result)

            except Exception as e:
                logger.error(f"Error evaluating query {query.id}: {e}")
                results.append({
                    "query_id": query.id,
                    "error": str(e)
                })

        # Aggregate results
        aggregated = self._aggregate_generation_metrics(results)
        logger.info(f"Generation evaluation completed: {aggregated}")

        return aggregated

    def _evaluate_answer_quality(self, query: EvaluationQuery, answer) -> Dict[str, float]:
        """Evaluate the quality of a generated answer."""
        metrics = {}

        # 1. Answer Length (completeness indicator)
        word_count = len(answer.answer.split())
        metrics["answer_length"] = min(1.0, word_count / 100.0)  # Optimal around 100 words

        # 2. Ground Truth Coverage (how many ground truth answers are covered)
        ground_truth_coverage = 0.0
        if query.ground_truth_answers:
            covered_answers = 0
            answer_lower = answer.answer.lower()

            for gt_answer in query.ground_truth_answers:
                # Simple substring matching (could be improved with semantic similarity)
                if gt_answer.lower() in answer_lower:
                    covered_answers += 1

            ground_truth_coverage = covered_answers / len(query.ground_truth_answers)

        metrics["ground_truth_coverage"] = ground_truth_coverage

        # 3. Context Utilization (uses retrieved information)
        context_usage = 0.0
        if hasattr(answer, 'context_chunks') and answer.context_chunks:
            # Check if answer contains content from retrieved chunks
            answer_lower = answer.answer.lower()
            context_overlap = 0

            for chunk in answer.context_chunks:
                chunk_text = chunk.get("text", "").lower()
                # Count significant word overlaps
                chunk_words = set(chunk_text.split())
                answer_words = set(answer_lower.split())
                overlap = len(chunk_words.intersection(answer_words))
                if overlap > 3:  # Threshold for meaningful overlap
                    context_overlap += 1

            context_usage = min(1.0, context_overlap / len(answer.context_chunks))

        metrics["context_utilization"] = context_usage

        # 4. Overall Quality Score (weighted combination)
        weights = {"answer_length": 0.2, "ground_truth_coverage": 0.5, "context_utilization": 0.3}
        overall_quality = sum(metrics[key] * weights[key] for key in weights)

        metrics["overall_quality"] = overall_quality

        return metrics

    def _aggregate_generation_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate generation evaluation results."""
        successful_results = [r for r in results if "error" not in r]

        if not successful_results:
            return {"error": "No successful evaluations"}

        # Collect all metrics
        all_metrics = {
            "overall_quality": [],
            "answer_length": [],
            "ground_truth_coverage": [],
            "context_utilization": [],
            "execution_time": [],
            "tokens_used": []
        }

        for result in successful_results:
            metrics = result.get("quality_metrics", {})

            for metric_name in all_metrics:
                if metric_name in metrics:
                    all_metrics[metric_name].append(metrics[metric_name])

                # Also collect from top-level fields
                if metric_name in result:
                    all_metrics[metric_name].append(result[metric_name])

        # Calculate statistics
        aggregated = {}
        for metric_name, values in all_metrics.items():
            if values:
                aggregated[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }

        aggregated["total_queries"] = len(results)
        aggregated["successful_queries"] = len(successful_results)
        aggregated["success_rate"] = len(successful_results) / len(results)

        return aggregated

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of the entire RAG system.

        Returns:
            Complete evaluation results
        """
        logger.info("Starting comprehensive RAG evaluation")

        start_time = time.time()

        # Run retrieval evaluation
        retrieval_results = self.evaluate_retrieval()

        # Run generation evaluation
        generation_results = self.evaluate_generation()

        total_time = time.time() - start_time

        # Combine results
        comprehensive_results = {
            "evaluation_timestamp": start_time,
            "total_evaluation_time": total_time,
            "retrieval_evaluation": retrieval_results,
            "generation_evaluation": generation_results,
            "system_summary": {
                "total_queries": len(self.test_queries),
                "avg_retrieval_time": retrieval_results.get("retrieval_time", {}).get("mean", 0),
                "avg_generation_time": generation_results.get("execution_time", {}).get("mean", 0),
                "overall_quality_score": generation_results.get("overall_quality", {}).get("mean", 0)
            }
        }

        # Save results
        self._save_evaluation_results(comprehensive_results)

        logger.info(f"Comprehensive evaluation completed in {total_time:.2f}s")
        return comprehensive_results

    def _save_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to file."""
        results_dir = "evaluation/results"
        os.makedirs(results_dir, exist_ok=True)

        timestamp = int(time.time())
        filename = f"evaluation_results_{timestamp}.json"

        filepath = os.path.join(results_dir, filename)

        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Evaluation results saved to: {filepath}")

        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")

    def generate_evaluation_report(self) -> str:
        """Generate a human-readable evaluation report."""
        results = self.run_comprehensive_evaluation()

        report = []
        report.append("# RAG System Evaluation Report")
        report.append(f"Generated: {time.ctime(results['evaluation_timestamp'])}")
        report.append("")

        # System Summary
        summary = results.get("system_summary", {})
        report.append("## System Summary")
        report.append(f"- Total Queries Evaluated: {summary.get('total_queries', 0)}")
        report.append(f"- Average Retrieval Time: {summary.get('avg_retrieval_time', 0):.3f}s")
        report.append(f"- Average Generation Time: {summary.get('avg_generation_time', 0):.3f}s")
        report.append(f"- Overall Quality Score: {summary.get('overall_quality_score', 0):.3f}")
        report.append("")

        # Retrieval Metrics
        retrieval = results.get("retrieval_evaluation", {})
        report.append("## Retrieval Metrics")
        if "recall@k" in retrieval:
            recall_stats = retrieval["recall@k"]
            report.append(f"- Recall@k: {recall_stats.get('mean', 0):.3f} ± {recall_stats.get('std', 0):.3f}")

        if "precision@k" in retrieval:
            precision_stats = retrieval["precision@k"]
            report.append(f"- Precision@k: {precision_stats.get('mean', 0):.3f} ± {precision_stats.get('std', 0):.3f}")

        if "mrr" in retrieval:
            mrr_stats = retrieval["mrr"]
            report.append(f"- Mean Reciprocal Rank: {mrr_stats.get('mean', 0):.3f} ± {mrr_stats.get('std', 0):.3f}")

        if "ndcg@k" in retrieval:
            ndcg_stats = retrieval["ndcg@k"]
            report.append(f"- NDCG@k: {ndcg_stats.get('mean', 0):.3f} ± {ndcg_stats.get('std', 0):.3f}")
        report.append("")

        # Generation Metrics
        generation = results.get("generation_evaluation", {})
        report.append("## Generation Metrics")
        if "overall_quality" in generation:
            quality_stats = generation["overall_quality"]
            report.append(f"- Overall Quality: {quality_stats.get('mean', 0):.3f} ± {quality_stats.get('std', 0):.3f}")

        if "ground_truth_coverage" in generation:
            coverage_stats = generation["ground_truth_coverage"]
            report.append(f"- Ground Truth Coverage: {coverage_stats.get('mean', 0):.3f} ± {coverage_stats.get('std', 0):.3f}")

        if "context_utilization" in generation:
            context_stats = generation["context_utilization"]
            report.append(f"- Context Utilization: {context_stats.get('mean', 0):.3f} ± {context_stats.get('std', 0):.3f}")

        return "\n".join(report)


def main():
    """Example usage of the RAG evaluator."""
    logging.basicConfig(level=logging.INFO)

    try:
        evaluator = RAGEvaluator()

        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation()

        print("Evaluation Results Summary:")
        print(f"Total queries: {results['system_summary']['total_queries']}")
        print(f"Avg retrieval time: {results['system_summary']['avg_retrieval_time']:.3f}s")
        print(f"Avg generation time: {results['system_summary']['avg_generation_time']:.3f}s")
        print(f"Overall quality score: {results['system_summary']['overall_quality_score']:.3f}")

        # Generate and print report
        report = evaluator.generate_evaluation_report()
        print("\n" + "="*50)
        print(report)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
