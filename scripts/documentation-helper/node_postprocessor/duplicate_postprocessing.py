from abc import abstractmethod
from typing import List, Optional

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore


class DuplicateRemoverNodePostProcessor:
    """Node postprocessor."""

    @abstractmethod
    def postprocess_nodes(
        self, nodes: List[NodeWithScore], quey_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""

        print("_pstprocess_nodes entered")

        unique_hashes = set()
        unique_nodes = []

        for node in nodes:
            node_hash = node.node.hash
            if node_hash not in unique_hashes:
                unique_hashes.add(node_hash)
                unique_nodes.append(node)

        return unique_nodes
