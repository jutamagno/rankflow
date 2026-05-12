"""Tests for dense retrieval — no real model loaded, uses mocks."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.retrieval.embedder import Item, DenseRetriever, RetrievalResult


def make_items(n: int = 10) -> list[Item]:
    return [
        Item(id=f"item_{i}", title=f"Produto {i}", description=f"Descrição {i}", category="moda")
        for i in range(n)
    ]


class TestItem:
    def test_to_text_concatenates(self):
        item = Item("x", "Fone Bluetooth", "Som cristalino", "eletrônicos")
        text = item.to_text()
        assert "Fone Bluetooth" in text
        assert "Som cristalino" in text

    def test_default_language_pt(self):
        item = Item("x", "t", "d", "cat")
        assert item.language == "pt"


class TestDenseRetriever:
    @patch("src.retrieval.embedder.SentenceTransformer")
    @patch("src.retrieval.embedder.faiss")
    def test_retrieve_before_build_raises(self, mock_faiss, mock_st):
        retriever = DenseRetriever()
        with pytest.raises(RuntimeError, match="build_index"):
            retriever.retrieve("fone de ouvido")

    @patch("src.retrieval.embedder.SentenceTransformer")
    def test_build_index_sets_items(self, mock_st_cls):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(5, 64).astype(np.float32)
        mock_st_cls.return_value = mock_model

        retriever = DenseRetriever()
        items = make_items(5)

        with patch("src.retrieval.embedder.faiss") as mock_faiss:
            mock_index = MagicMock()
            mock_faiss.IndexFlatIP.return_value = mock_index
            mock_index.search.return_value = (
                np.array([[0.9, 0.8, 0.7]]),
                np.array([[0, 1, 2]]),
            )
            retriever.index = mock_index
            retriever.items = items
            retriever._dim = 64

            results = retriever.retrieve("produto", top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].score == pytest.approx(0.9)

    @patch("src.retrieval.embedder.SentenceTransformer")
    def test_retrieve_filters_negative_indices(self, mock_st_cls):
        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model
        mock_model.encode.return_value = np.zeros((1, 64), dtype=np.float32)

        retriever = DenseRetriever()
        retriever.items = make_items(5)
        retriever._dim = 64

        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8]]),
            np.array([[0, -1]]),  # -1 = FAISS "no result"
        )
        retriever.index = mock_index

        results = retriever.retrieve("teste")
        assert len(results) == 1
        assert results[0].item_id == "item_0"
