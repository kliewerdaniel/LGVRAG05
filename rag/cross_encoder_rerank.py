"""
Cross-encoder re-ranking for improved retrieval quality.

This module implements cross-encoder models for re-ranking retrieved
candidates, providing more accurate relevance scoring than bi-encoders.
"""

import os
import logging
import sys
from typing import List, Dict, Any, Optional
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("sentence-transformers not available. Cross-encoder re-ranking will use fallback method.")

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder model for re-ranking retrieved candidates.

    This class uses cross-encoder models (like MS MARCO) to provide
    more accurate relevance scores for query-document pairs.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.model = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = CrossEncoder(model_name)
                logger.info(f"Loaded cross-encoder model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model {model_name}: {e}")
                logger.info("Falling back to simple similarity-based re-ranking")
        else:
            logger.info("Using fallback re-ranking method (no sentence-transformers)")

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Re-rank candidates using cross-encoder model.

        Args:
            query: Search query
            candidates: List of candidate documents/chunks
            top_k: Number of top results to return (None for all)

        Returns:
            Re-ranked candidates with cross-encoder scores
        """
        if not candidates:
            return []

        if self.model is None:
            # Fallback: use simple similarity-based re-ranking
            return self._fallback_rerank(query, candidates, top_k)

        try:
            logger.info(f"Cross-encoder re-ranking {len(candidates)} candidates")

            # Prepare query-document pairs
            pairs = [(query, candidate.get("text", "")) for candidate in candidates]

            # Get cross-encoder scores
            scores = self.model.predict(pairs)

            # Update candidates with cross-encoder scores
            for i, candidate in enumerate(candidates):
                candidate["cross_encoder_score"] = float(scores[i])
                candidate["rerank_score"] = float(scores[i])

            # Sort by cross-encoder score (descending)
            reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)

            # Add rank positions
            for i, candidate in enumerate(reranked):
                candidate["rerank_position"] = i + 1

            # Return top-k results
            if top_k is not None:
                reranked = reranked[:top_k]

            logger.info(f"Cross-encoder re-ranking completed, top score: {reranked[0].get('rerank_score', 0):.4f}")
            return reranked

        except Exception as e:
            logger.error(f"Error in cross-encoder re-ranking: {e}")
            # Fallback to simple method
            return self._fallback_rerank(query, candidates, top_k)

    def _fallback_rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fallback re-ranking method when cross-encoder is not available."""
        logger.info("Using fallback re-ranking method")

        # Simple re-ranking based on existing scores and text similarity
        for candidate in candidates:
            # Use existing hybrid/vector score as base
            base_score = candidate.get("hybrid_score", candidate.get("similarity", 0))

            # Boost score if query terms appear in the text
            query_terms = set(query.lower().split())
            text = candidate.get("text", "").lower()

            # Count query term matches
            term_matches = sum(1 for term in query_terms if term in text)
            term_boost = term_matches / max(len(query_terms), 1)

            # Combine base score with term boost
            fallback_score = base_score * 0.7 + term_boost * 0.3

            candidate["rerank_score"] = fallback_score
            candidate["cross_encoder_score"] = fallback_score  # For compatibility

        # Sort by fallback score
        reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)

        # Add rank positions
        for i, candidate in enumerate(reranked):
            candidate["rerank_position"] = i + 1

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the cross-encoder model."""
        if self.model is None:
            return {
                "model_name": self.model_name,
                "status": "unavailable",
                "fallback_mode": True
            }

        return {
            "model_name": self.model_name,
            "status": "loaded",
            "fallback_mode": False,
            "model_type": "cross-encoder"
        }


def main():
    """Example usage of cross-encoder re-ranker."""
    logging.basicConfig(level=logging.INFO)

    # Example candidates (normally these would come from retrieval)
    candidates = [
        {
            "id": "chunk1",
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
            "source": "ml_overview.md",
            "hybrid_score": 0.85
        },
        {
            "id": "chunk2",
            "text": "Neural networks are computational models inspired by biological neural networks, consisting of layers of interconnected nodes.",
            "source": "neural_networks.md",
            "hybrid_score": 0.72
        },
        {
            "id": "chunk3",
            "text": "Deep learning uses multiple layers of neural networks to learn hierarchical representations of data.",
            "source": "deep_learning.md",
            "hybrid_score": 0.68
        }
    ]

    query = "What is machine learning and how does it relate to artificial intelligence?"

    # Initialize reranker
    reranker = CrossEncoderReranker()

    # Re-rank candidates
    reranked = reranker.rerank(query, candidates, top_k=2)

    print(f"Query: {query}")
    print(f"Re-ranked {len(reranked)} candidates:")

    for i, candidate in enumerate(reranked, 1):
        print(f"\n{i}. Score: {candidate.get('rerank_score', 0):.4f}")
        print(f"   Text: {candidate.get('text', '')[:100]}...")
        print(f"   Source: {candidate.get('source', '')}")

    # Print model info
    model_info = reranker.get_model_info()
    print(f"\nModel info: {model_info}")


if __name__ == "__main__":
    main()
