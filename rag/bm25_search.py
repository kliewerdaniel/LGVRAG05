"""
BM25 sparse search implementation for hybrid retrieval.

This module implements the BM25 (Best Matching 25) algorithm for
sparse/keyword-based document ranking, complementing vector similarity search.
"""

import os
import json
import logging
import math
import sys
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import re

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class BM25Search:
    """
    BM25 (Best Matching 25) sparse search implementation.

    This class implements the BM25 ranking function for keyword-based
    document retrieval, providing an alternative to dense vector search.
    """

    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 search.

        Args:
            documents: List of documents to index (optional, can be added later)
            k1: Term frequency saturation parameter (typically 1.2-2.0)
            b: Length normalization parameter (typically 0.75)
        """
        self.documents = documents or []
        self.k1 = k1
        self.b = b

        # BM25 statistics
        self.doc_freq = Counter()  # Document frequency for each term
        self.doc_lengths = {}  # Length (in tokens) for each document
        self.avg_doc_length = 0
        self.total_docs = 0
        self.term_to_docs = defaultdict(set)  # Which documents contain each term

        # Vocabulary and IDF cache
        self.vocabulary = set()
        self.idf_cache = {}

        if documents:
            self._build_index()

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the index."""
        self.documents.extend(documents)
        self._build_index()

    def _build_index(self) -> None:
        """Build the BM25 index from documents."""
        logger.info(f"Building BM25 index for {len(self.documents)} documents")

        # Reset statistics
        self.doc_freq = Counter()
        self.doc_lengths = {}
        self.term_to_docs = defaultdict(set)
        self.vocabulary = set()
        self.idf_cache = {}

        total_length = 0

        for doc in self.documents:
            doc_id = doc.get("id", "")
            text = doc.get("text", "")

            # Tokenize document
            tokens = self._tokenize(text)

            # Store document length
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            # Count term frequencies and document frequencies
            term_freq = Counter(tokens)

            for term, freq in term_freq.items():
                self.vocabulary.add(term)
                self.term_to_docs[term].add(doc_id)
                self.doc_freq[term] += 1

        # Calculate average document length
        self.total_docs = len(self.documents)
        self.avg_doc_length = total_length / max(self.total_docs, 1)

        logger.info(f"BM25 index built: {len(self.vocabulary)} unique terms, "
                   f"avg doc length: {self.avg_doc_length:.1f}")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words/terms."""
        if not text:
            return []

        # Convert to lowercase and split on non-alphanumeric characters
        # Keep alphanumeric characters and apostrophes (for contractions)
        tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())

        # Remove very short tokens (less than 2 characters)
        tokens = [token for token in tokens if len(token) >= 2]

        return tokens

    def _get_idf(self, term: str) -> float:
        """Calculate Inverse Document Frequency for a term."""
        if term in self.idf_cache:
            return self.idf_cache[term]

        if term not in self.vocabulary:
            idf = 0.0
        else:
            # IDF = log((N - df + 0.5) / (df + 0.5))
            # where N is total documents, df is document frequency
            df = self.doc_freq[term]
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5))

        self.idf_cache[term] = idf
        return idf

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query using BM25.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of documents ranked by BM25 score
        """
        if not self.documents:
            logger.warning("No documents in BM25 index")
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        if not query_tokens:
            logger.warning("Empty query after tokenization")
            return []

        logger.info(f"BM25 search for '{query}' with {len(query_tokens)} tokens")

        # Calculate BM25 scores for all documents
        doc_scores = {}

        for doc in self.documents:
            doc_id = doc.get("id", "")
            doc_text = doc.get("text", "")

            # Calculate BM25 score for this document
            score = self._calculate_bm25_score(query_tokens, doc_id, doc_text)

            if score > 0:
                doc_scores[doc_id] = score

        # Sort documents by score (descending)
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top-k results with metadata
        results = []
        for doc_id, score in ranked_docs[:top_k]:
            # Find the original document
            doc = next((d for d in self.documents if d.get("id") == doc_id), None)
            if doc:
                result = doc.copy()
                result["bm25_score"] = score
                result["retrieval_method"] = "bm25"
                result["retrieval_score"] = score
                results.append(result)

        logger.info(f"BM25 search returned {len(results)} results")
        return results

    def _calculate_bm25_score(self, query_tokens: List[str], doc_id: str, doc_text: str) -> float:
        """Calculate BM25 score for a document given query tokens."""
        doc_length = self.doc_lengths.get(doc_id, 0)

        if doc_length == 0:
            return 0.0

        # Tokenize document for term frequency calculation
        doc_tokens = self._tokenize(doc_text)
        doc_term_freq = Counter(doc_tokens)

        score = 0.0

        for term in query_tokens:
            if term not in self.vocabulary:
                continue

            # Term frequency in document
            tf = doc_term_freq.get(term, 0)

            # Inverse document frequency
            idf = self._get_idf(term)

            # BM25 term score
            # BM25 = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (doc_length / avg_doc_length)))
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

            if denominator > 0:
                term_score = idf * (numerator / denominator)
                score += term_score

        return score

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the BM25 index."""
        return {
            "total_documents": self.total_docs,
            "vocabulary_size": len(self.vocabulary),
            "avg_document_length": self.avg_doc_length,
            "k1": self.k1,
            "b": self.b,
            "idf_cache_size": len(self.idf_cache)
        }


def create_bm25_from_chunks(chunks: List[Dict[str, Any]]) -> BM25Search:
    """
    Create a BM25 search index from document chunks.

    Args:
        chunks: List of document chunks with 'id', 'text', and other metadata

    Returns:
        BM25Search instance ready for querying
    """
    # Prepare documents for BM25
    documents = []
    for chunk in chunks:
        doc = {
            "id": chunk.get("id", ""),
            "text": chunk.get("text", ""),
            "chunk_id": chunk.get("id", ""),
            "source": chunk.get("source", ""),
            "metadata": chunk.get("metadata", {})
        }
        documents.append(doc)

    return BM25Search(documents)


def main():
    """Example usage of BM25 search."""
    logging.basicConfig(level=logging.INFO)

    # Example documents
    documents = [
        {
            "id": "doc1",
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
            "source": "ml_overview.md"
        },
        {
            "id": "doc2",
            "text": "Neural networks are computational models inspired by biological neural networks, consisting of layers of interconnected nodes.",
            "source": "neural_networks.md"
        },
        {
            "id": "doc3",
            "text": "Deep learning uses multiple layers of neural networks to learn hierarchical representations of data.",
            "source": "deep_learning.md"
        }
    ]

    # Create BM25 index
    bm25 = BM25Search(documents)

    # Example queries
    queries = [
        "machine learning artificial intelligence",
        "neural networks deep learning",
        "computer programming algorithms"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = bm25.search(query, top_k=2)

        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.get('bm25_score', 0):.4f}")
            print(f"   Text: {result.get('text', '')[:100]}...")
            print(f"   Source: {result.get('source', '')}")

    # Print index statistics
    stats = bm25.get_index_stats()
    print(f"\nIndex stats: {stats}")


if __name__ == "__main__":
    main()
