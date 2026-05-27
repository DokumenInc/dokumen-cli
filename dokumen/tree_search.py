"""CLI tree search data source backed by .dokumen-cache/tree_indexes/.

Provides CacheTreeDataSource implementing TreeDataSourceProtocol
for the TreeSearchService to query cached tree indexes from disk.
"""

import json
import os
from typing import Optional

from dokumen_pageindex import DocumentTree, IndexedDocument

from .logging_config import get_logger

logger = get_logger(__name__)

TREE_INDEX_DIR = ".dokumen-cache/tree_indexes"


class CacheTreeDataSource:
    """Data source that loads indexed documents from local file cache.

    Implements TreeDataSourceProtocol for CLI use. Reads tree indexes
    from .dokumen-cache/tree_indexes/{file_hash}.json files on disk.
    """

    def __init__(self, cache_dir: str = TREE_INDEX_DIR) -> None:
        self._cache_dir = cache_dir

    async def get_indexed_documents(self, company_id: str = "") -> list[IndexedDocument]:
        """Load all cached tree indexes from disk.

        Args:
            company_id: Unused in CLI (single-tenant).

        Returns:
            List of IndexedDocument from cached tree files.
        """
        logger.info("tree_search.cache.get_indexed_documents.start", cache_dir=self._cache_dir)

        if not os.path.isdir(self._cache_dir):
            logger.info("tree_search.cache.no_cache_dir", cache_dir=self._cache_dir)
            return []

        documents: list[IndexedDocument] = []
        for filename in os.listdir(self._cache_dir):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self._cache_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                tree = DocumentTree.from_dict(data)
                documents.append(
                    IndexedDocument(
                        file_path=tree.file_path,
                        title=tree.title or tree.file_path,
                        summary=tree.description[:300] if tree.description else tree.file_path,
                        tree=tree,
                    )
                )
            except Exception as e:
                logger.warning(
                    "tree_search.cache.skip_invalid",
                    filename=filename,
                    error=str(e),
                )
                continue

        logger.info(
            "tree_search.cache.get_indexed_documents.complete",
            count=len(documents),
        )
        return documents

    async def get_tree(self, file_path: str, company_id: str = "") -> Optional[DocumentTree]:
        """Load a single tree from the cache by scanning for matching file_path.

        Args:
            file_path: Path of the document.
            company_id: Unused in CLI.

        Returns:
            DocumentTree if found, None otherwise.
        """
        logger.debug("tree_search.cache.get_tree.start", file_path=file_path)

        if not os.path.isdir(self._cache_dir):
            return None

        for filename in os.listdir(self._cache_dir):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self._cache_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                if data.get("file_path") == file_path:
                    tree = DocumentTree.from_dict(data)
                    logger.debug(
                        "tree_search.cache.get_tree.found",
                        file_path=file_path,
                        nodes=tree.total_nodes,
                    )
                    return tree
            except Exception:
                continue

        logger.debug("tree_search.cache.get_tree.not_found", file_path=file_path)
        return None
