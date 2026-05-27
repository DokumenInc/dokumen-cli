"""Tests for CLI tree search fast path (Step 4 of #544)."""

import json
import os
import pytest
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from dokumen.tree_search import CacheTreeDataSource


@pytest.fixture
def tree_index_data():
    """Sample tree index JSON data."""
    return {
        "file_path": "docs/api.md",
        "file_type": "md",
        "title": "API Reference",
        "description": "API documentation for the platform",
        "total_nodes": 2,
        "total_tokens": 300,
        "nodes": [
            {
                "node_id": "n1",
                "title": "Authentication",
                "level": 0,
                "summary": "How auth works",
                "text": "OAuth and API keys",
            },
            {
                "node_id": "n2",
                "title": "Endpoints",
                "level": 0,
                "text": "GET /users",
            },
        ],
    }


class TestCacheTreeDataSource:
    """Tests for CacheTreeDataSource."""

    @pytest.mark.asyncio
    async def test_get_indexed_documents_returns_cached_trees(self, tree_index_data):
        """Loads tree indexes from JSON files in cache dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a tree index file
            with open(os.path.join(tmpdir, "abc123.json"), "w") as f:
                json.dump(tree_index_data, f)

            ds = CacheTreeDataSource(cache_dir=tmpdir)
            docs = await ds.get_indexed_documents()

        assert len(docs) == 1
        assert docs[0].file_path == "docs/api.md"
        assert docs[0].title == "API Reference"
        assert len(docs[0].tree.nodes) == 2

    @pytest.mark.asyncio
    async def test_get_indexed_documents_empty_when_no_dir(self):
        """Returns empty list when cache dir doesn't exist."""
        ds = CacheTreeDataSource(cache_dir="/nonexistent/path")
        docs = await ds.get_indexed_documents()
        assert docs == []

    @pytest.mark.asyncio
    async def test_get_indexed_documents_skips_invalid_json(self, tree_index_data):
        """Skips files with invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid file
            with open(os.path.join(tmpdir, "valid.json"), "w") as f:
                json.dump(tree_index_data, f)
            # Invalid file
            with open(os.path.join(tmpdir, "invalid.json"), "w") as f:
                f.write("not json")

            ds = CacheTreeDataSource(cache_dir=tmpdir)
            docs = await ds.get_indexed_documents()

        assert len(docs) == 1

    @pytest.mark.asyncio
    async def test_get_indexed_documents_skips_non_json(self, tree_index_data):
        """Ignores non-.json files in cache dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "abc.json"), "w") as f:
                json.dump(tree_index_data, f)
            with open(os.path.join(tmpdir, "readme.txt"), "w") as f:
                f.write("not a tree index")

            ds = CacheTreeDataSource(cache_dir=tmpdir)
            docs = await ds.get_indexed_documents()

        assert len(docs) == 1

    @pytest.mark.asyncio
    async def test_get_tree_finds_matching_file(self, tree_index_data):
        """Returns tree for matching file_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "abc.json"), "w") as f:
                json.dump(tree_index_data, f)

            ds = CacheTreeDataSource(cache_dir=tmpdir)
            tree = await ds.get_tree("docs/api.md")

        assert tree is not None
        assert tree.title == "API Reference"

    @pytest.mark.asyncio
    async def test_get_tree_returns_none_when_not_found(self, tree_index_data):
        """Returns None when no matching file_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "abc.json"), "w") as f:
                json.dump(tree_index_data, f)

            ds = CacheTreeDataSource(cache_dir=tmpdir)
            tree = await ds.get_tree("docs/missing.md")

        assert tree is None


class TestExploreAgentTreeSearch:
    """Tests for ExploreAgent._try_tree_search integration."""

    @pytest.mark.asyncio
    async def test_tree_search_disabled_returns_none(self):
        """Returns None when pageindex.enabled=false."""
        from dokumen.explore_agent import ExploreAgent

        agent = ExploreAgent(query_runner=MagicMock())

        with patch("dokumen.config.load_config") as mock_config:
            config = MagicMock()
            config.pageindex.enabled = False
            mock_config.return_value = config

            result = await agent._try_tree_search("test query")

        assert result is None

    @pytest.mark.asyncio
    async def test_tree_search_no_cached_trees_returns_none(self):
        """Returns None when no cached tree indexes exist."""
        from dokumen.explore_agent import ExploreAgent

        agent = ExploreAgent(query_runner=MagicMock())

        with patch("dokumen.config.load_config") as mock_config:
            config = MagicMock()
            config.pageindex.enabled = True
            config.pageindex.model = "claude-haiku-4-5-20251001"
            mock_config.return_value = config

            with patch("dokumen.tree_search.CacheTreeDataSource") as mock_ds_cls:
                mock_ds = AsyncMock()
                mock_ds.get_indexed_documents = AsyncMock(return_value=[])
                mock_ds_cls.return_value = mock_ds

                result = await agent._try_tree_search("test query")

        assert result is None

    @pytest.mark.asyncio
    async def test_tree_search_error_returns_none(self):
        """Returns None on exception (graceful fallback)."""
        from dokumen.explore_agent import ExploreAgent

        agent = ExploreAgent(query_runner=MagicMock())

        with patch("dokumen.config.load_config", side_effect=RuntimeError("config fail")):
            result = await agent._try_tree_search("test query")

        assert result is None
