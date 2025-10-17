"""
Enhanced hybrid retrieval module with BM25 and cross-encoder re-ranking.

This module implements sophisticated retrieval strategies that combine
vector similarity search, BM25 sparse search, and knowledge graph traversal
with cross-encoder re-ranking for enhanced contextual understanding and retrieval accuracy.
"""

import os
import json
import logging
import sys
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from db.helix_interface import ChromaDBInterface
from ingestion.embeddings import EmbeddingGenerator
from .bm25_search import BM25Search, create_bm25_from_chunks
from .cross_encoder_rerank import CrossEncoderReranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Enhanced hybrid retriever with BM25 and cross-encoder re-ranking.

    This class implements sophisticated retrieval strategies that combine
    vector similarity search, BM25 sparse search, and knowledge graph traversal
    with cross-encoder re-ranking for enhanced contextual understanding and retrieval accuracy.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the enhanced hybrid retriever.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.db_interface = ChromaDBInterface(config_path)
        self.embedding_generator = EmbeddingGenerator(config_path)

        # Initialize BM25 search
        self.bm25_search = None
        self._initialize_bm25()

        # Initialize cross-encoder reranker
        self.cross_encoder = CrossEncoderReranker()

        # Enhanced retrieval parameters
        self.vector_top_k = self.config.retrieval.vector_top_k
        self.bm25_top_k = getattr(self.config.retrieval, 'bm25_top_k', 50)
        self.graph_top_k = getattr(self.config.retrieval, 'graph_top_k', 50)
        self.graph_depth = self.config.retrieval.graph_depth
        self.hybrid_weights = getattr(self.config.retrieval, 'hybrid_weights', [0.5, 0.3, 0.2])
        self.rerank_top_k = self.config.retrieval.rerank_top_k
        self.final_top_k = getattr(self.config.retrieval, 'final_top_k', 5)

    def _initialize_bm25(self) -> None:
        """Initialize BM25 search with existing chunks."""
        try:
            # Get all chunks from database for BM25 indexing
            all_chunks = self.db_interface.get_all_chunks()

            if all_chunks:
                self.bm25_search = create_bm25_from_chunks(all_chunks)
                logger.info(f"BM25 index initialized with {len(all_chunks)} chunks")
            else:
                logger.warning("No chunks available for BM25 initialization")

        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")
            self.bm25_search = None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Enhanced hybrid retrieval with BM25 and cross-encoder re-ranking.

        Args:
            query: Search query string
            top_k: Number of results to return (uses config default if None)

        Returns:
            List of retrieved chunks with relevance scores
        """
        if top_k is None:
            top_k = self.final_top_k

        logger.info(f"Performing enhanced hybrid retrieval for query: {query}")

        # Generate embedding for query
        query_embedding = self.embedding_generator._generate_embeddings([query])
        if len(query_embedding) == 0:
            logger.error("Failed to generate query embedding")
            return []

        query_embedding = query_embedding[0]

        # Stage 1: Multi-strategy initial retrieval
        vector_results = self._vector_search(query_embedding, self.vector_top_k)
        bm25_results = self._bm25_search(query, self.bm25_top_k)
        graph_results = self._graph_search(query, self.graph_top_k)

        # Stage 2: Fusion with Reciprocal Rank Fusion (RRF)
        fused_results = self._reciprocal_rank_fusion(
            [vector_results, bm25_results, graph_results],
            weights=self.hybrid_weights
        )

        # Stage 3: Cross-encoder re-ranking
        reranked_results = self.cross_encoder.rerank(query, fused_results, self.rerank_top_k)

        # Stage 4: Final selection
        final_results = reranked_results[:top_k]

        logger.info(f"Enhanced retrieval completed: {len(final_results)} final results")
        return final_results

    def retrieve_with_entities(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve results with entity information.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            Dictionary containing chunks, entities, and relationships
        """
        # Get basic retrieval results
        chunks = self.retrieve(query, top_k)

        # Extract entities from query and results
        query_entities = self._extract_query_entities(query)

        # Find related entities in results
        result_entities = []
        relationships = []

        for chunk in chunks:
            # Find entities mentioned in this chunk
            chunk_entities = self._find_entities_in_chunk(chunk)
            result_entities.extend(chunk_entities)

            # Find relationships involving these entities
            chunk_relationships = self._find_relationships_in_chunk(chunk)
            relationships.extend(chunk_relationships)

        return {
            "query": query,
            "query_entities": query_entities,
            "chunks": chunks,
            "entities": result_entities,
            "relationships": relationships
        }

    def _vector_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        try:
            results = self.db_interface.search_similar_chunks(query_embedding, top_k)

            # Add retrieval method metadata
            for result in results:
                result["retrieval_method"] = "vector_similarity"
                result["retrieval_score"] = result.get("similarity", 0.0)

            return results

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform BM25 sparse search."""
        if self.bm25_search is None:
            logger.warning("BM25 search not available")
            return []

        try:
            results = self.bm25_search.search(query, top_k)

            # Add retrieval method metadata
            for result in results:
                result["retrieval_method"] = "bm25"
                result["retrieval_score"] = result.get("bm25_score", 0.0)

            return results

        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []

    def _graph_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform graph-based search using entity relationships."""
        try:
            # Extract entities from query
            query_entities = self._extract_query_entities(query)

            if not query_entities:
                return []

            graph_results = []

            # For each query entity, find related chunks through graph traversal
            for entity in query_entities:
                entity_name = entity.get("name", "")

                # Get related entities through graph traversal
                related_entities = self.db_interface.get_entity_relationships(
                    entity_name,
                    max_depth=self.graph_depth
                )

                # Find chunks that mention related entities
                for related in related_entities:
                    related_entity_name = related.get("related_entity", "")

                    # Search for chunks containing this entity
                    chunks = self._find_chunks_with_entity(related_entity_name, top_k=3)

                    for chunk in chunks:
                        chunk["retrieval_method"] = "graph_traversal"
                        chunk["retrieval_score"] = related.get("weight", 1.0) * 0.5  # Scale down graph scores
                        chunk["traversal_path"] = [
                            entity_name,
                            related.get("relationship", ""),
                            related_entity_name
                        ]
                        graph_results.append(chunk)

            # Deduplicate results based on chunk_id
            seen_chunks = set()
            unique_results = []

            for result in graph_results:
                chunk_id = result.get("chunk_id", "")
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    unique_results.append(result)

            return unique_results[:top_k]

        except Exception as e:
            logger.error(f"Error in graph search: {e}")
            return []

    def _reciprocal_rank_fusion(self, result_lists: List[List[Dict[str, Any]]],
                               weights: List[float], k: int = 60) -> List[Dict[str, Any]]:
        """Combine multiple retrieval results using Reciprocal Rank Fusion."""
        # Create a map of chunk_id to RRF score
        rrf_scores = defaultdict(float)

        for list_idx, results in enumerate(result_lists):
            weight = weights[list_idx] if list_idx < len(weights) else 1.0

            for rank, result in enumerate(results):
                chunk_id = result.get("chunk_id", result.get("id", ""))

                # RRF score: weight * sum(1 / (k + rank)) for each occurrence
                rrf_score = weight * (1.0 / (k + rank + 1))  # +1 because rank is 0-indexed
                rrf_scores[chunk_id] += rrf_score

        # Sort by RRF score (descending)
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Reconstruct results with RRF scores
        fused_results = []
        seen_chunks = set()

        for chunk_id, rrf_score in sorted_chunks:
            # Find the original result data
            result = None
            for results in result_lists:
                for r in results:
                    if r.get("chunk_id") == chunk_id or r.get("id") == chunk_id:
                        result = r.copy()
                        break
                if result:
                    break

            if result and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                result["rrf_score"] = rrf_score
                result["retrieval_score"] = rrf_score  # For compatibility
                fused_results.append(result)

        return fused_results

    def _extract_query_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities from query text."""
        # This is a simplified entity extraction for queries
        # In a full implementation, you might use the same entity extractor

        # For now, we'll do simple pattern matching for common entity types
        entities = []

        # Simple patterns for entity extraction
        patterns = {
            "ORGANIZATION": r'\b[A-Z][a-zA-Z0-9\s]*(Inc|Corp|LLC|Ltd|Company|Corp|Inc)\b',
            "TECHNOLOGY": r'\b(Python|Java|JavaScript|C\+\+|AI|ML|Machine Learning|Deep Learning)\b',
            "CONCEPT": r'\b(algorithm|model|network|system|framework|platform)\b'
        }

        for entity_type, pattern in patterns.items():
            import re
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "name": match.strip(),
                    "type": entity_type,
                    "confidence": 0.8  # Default confidence for pattern matching
                })

        return entities

    def _find_chunks_with_entity(self, entity_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find chunks that contain a specific entity."""
        # This is a simplified implementation
        # In practice, you might want to search the database for chunks containing the entity

        # For now, return empty list - this would need to be implemented
        # based on your specific entity storage strategy
        return []

    def _find_entities_in_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find entities mentioned in a chunk."""
        # Simplified implementation
        return []

    def _find_relationships_in_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find relationships mentioned in a chunk."""
        # Simplified implementation
        return []

    def update_bm25_index(self) -> None:
        """Update BM25 index with latest chunks from database."""
        try:
            all_chunks = self.db_interface.get_all_chunks()
            if all_chunks:
                self.bm25_search = create_bm25_from_chunks(all_chunks)
                logger.info(f"BM25 index updated with {len(all_chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to update BM25 index: {e}")

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get enhanced retrieval statistics."""
        stats = {
            "vector_top_k": self.vector_top_k,
            "bm25_top_k": self.bm25_top_k,
            "graph_top_k": self.graph_top_k,
            "graph_depth": self.graph_depth,
            "hybrid_weights": self.hybrid_weights,
            "rerank_top_k": self.rerank_top_k,
            "final_top_k": self.final_top_k,
            "embedding_dimension": self.embedding_generator.get_embedding_dimension()
        }

        if self.bm25_search:
            stats["bm25_stats"] = self.bm25_search.get_index_stats()

        stats["cross_encoder_info"] = self.cross_encoder.get_model_info()

        return stats


def main():
    """Example usage of the HybridRetriever."""
    logging.basicConfig(level=logging.INFO)

    try:
        retriever = HybridRetriever()

        # Example query
        query = "What is machine learning and how does it relate to artificial intelligence?"

        # Perform retrieval
        results = retriever.retrieve(query, top_k=5)

        print(f"Query: {query}")
        print(f"Retrieved {len(results)} results:")

        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result.get('retrieval_methods', ['unknown'])}]")
            print(f"   Score: {result.get('hybrid_score', 0):.4f}")
            print(f"   Text: {result.get('text', '')[:200]}...")
            print(f"   Source: {result.get('source', 'unknown')}")

        # Get retrieval with entities
        detailed_results = retriever.retrieve_with_entities(query, top_k=3)
        print(f"\nDetailed results include {len(detailed_results.get('entities', []))} entities")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
