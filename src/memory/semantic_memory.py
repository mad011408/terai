"""
Semantic memory for long-term knowledge storage.
Organizes information in a graph-like structure.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid
from pathlib import Path


class NodeType(Enum):
    """Types of memory nodes."""
    CONCEPT = "concept"
    FACT = "fact"
    PROCEDURE = "procedure"
    EPISODE = "episode"
    ENTITY = "entity"
    RELATION = "relation"


class RelationType(Enum):
    """Types of relations between nodes."""
    IS_A = "is_a"
    HAS_A = "has_a"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    INSTANCE_OF = "instance_of"


@dataclass
class MemoryNode:
    """A node in semantic memory."""
    node_id: str
    node_type: NodeType
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 0.5
    decay_rate: float = 0.1
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
            "decay_rate": self.decay_rate,
            "tags": list(self.tags)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNode":
        return cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            content=data["content"],
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            accessed_at=datetime.fromisoformat(data["accessed_at"]) if "accessed_at" in data else datetime.now(),
            access_count=data.get("access_count", 0),
            importance=data.get("importance", 0.5),
            decay_rate=data.get("decay_rate", 0.1),
            tags=set(data.get("tags", []))
        )

    def update_access(self) -> None:
        """Update access statistics."""
        self.accessed_at = datetime.now()
        self.access_count += 1
        # Increase importance with access
        self.importance = min(1.0, self.importance + 0.01)


@dataclass
class MemoryEdge:
    """An edge connecting two memory nodes."""
    edge_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEdge":
        return cls(
            edge_id=data["edge_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now()
        )


class SemanticMemory:
    """
    Long-term semantic memory organized as a knowledge graph.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: Dict[str, MemoryEdge] = {}
        self.adjacency: Dict[str, List[str]] = {}  # node_id -> list of edge_ids
        self.persist_path = persist_path

        if persist_path:
            self._load_from_disk()

    def add_node(self, content: str, node_type: NodeType = NodeType.CONCEPT,
                metadata: Optional[Dict] = None, tags: Optional[Set[str]] = None,
                importance: float = 0.5) -> MemoryNode:
        """Add a node to memory."""
        node_id = str(uuid.uuid4())

        node = MemoryNode(
            node_id=node_id,
            node_type=node_type,
            content=content,
            metadata=metadata or {},
            tags=tags or set(),
            importance=importance
        )

        self.nodes[node_id] = node
        self.adjacency[node_id] = []

        if self.persist_path:
            self._save_to_disk()

        return node

    def add_edge(self, source_id: str, target_id: str,
                relation_type: RelationType = RelationType.RELATED_TO,
                weight: float = 1.0,
                metadata: Optional[Dict] = None) -> Optional[MemoryEdge]:
        """Add an edge between nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        edge_id = str(uuid.uuid4())

        edge = MemoryEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            metadata=metadata or {}
        )

        self.edges[edge_id] = edge
        self.adjacency[source_id].append(edge_id)
        self.adjacency[target_id].append(edge_id)

        if self.persist_path:
            self._save_to_disk()

        return edge

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a node by ID."""
        node = self.nodes.get(node_id)
        if node:
            node.update_access()
        return node

    def get_related_nodes(self, node_id: str,
                         relation_type: Optional[RelationType] = None,
                         direction: str = "both") -> List[Tuple[MemoryNode, MemoryEdge]]:
        """Get nodes related to a given node."""
        if node_id not in self.nodes:
            return []

        results = []
        edge_ids = self.adjacency.get(node_id, [])

        for edge_id in edge_ids:
            edge = self.edges.get(edge_id)
            if not edge:
                continue

            if relation_type and edge.relation_type != relation_type:
                continue

            if direction == "outgoing" and edge.source_id != node_id:
                continue
            if direction == "incoming" and edge.target_id != node_id:
                continue

            # Get the related node
            related_id = edge.target_id if edge.source_id == node_id else edge.source_id
            related_node = self.nodes.get(related_id)

            if related_node:
                related_node.update_access()
                results.append((related_node, edge))

        return results

    def search_by_content(self, query: str, limit: int = 10) -> List[MemoryNode]:
        """Search nodes by content."""
        query_lower = query.lower()
        results = []

        for node in self.nodes.values():
            if query_lower in node.content.lower():
                node.update_access()
                results.append(node)

        # Sort by importance and access count
        results.sort(key=lambda n: (n.importance, n.access_count), reverse=True)
        return results[:limit]

    def search_by_tags(self, tags: Set[str], match_all: bool = False) -> List[MemoryNode]:
        """Search nodes by tags."""
        results = []

        for node in self.nodes.values():
            if match_all:
                if tags.issubset(node.tags):
                    results.append(node)
            else:
                if tags & node.tags:  # Any overlap
                    results.append(node)

        return results

    def search_by_type(self, node_type: NodeType) -> List[MemoryNode]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_path(self, start_id: str, end_id: str,
                max_depth: int = 5) -> Optional[List[Tuple[MemoryNode, Optional[MemoryEdge]]]]:
        """Find a path between two nodes using BFS."""
        if start_id not in self.nodes or end_id not in self.nodes:
            return None

        if start_id == end_id:
            return [(self.nodes[start_id], None)]

        # BFS
        from collections import deque
        queue = deque([(start_id, [(self.nodes[start_id], None)])])
        visited = {start_id}

        while queue:
            current_id, path = queue.popleft()

            if len(path) > max_depth:
                continue

            for edge_id in self.adjacency.get(current_id, []):
                edge = self.edges.get(edge_id)
                if not edge:
                    continue

                next_id = edge.target_id if edge.source_id == current_id else edge.source_id

                if next_id in visited:
                    continue

                next_node = self.nodes.get(next_id)
                if not next_node:
                    continue

                new_path = path + [(next_node, edge)]

                if next_id == end_id:
                    return new_path

                visited.add(next_id)
                queue.append((next_id, new_path))

        return None

    def get_subgraph(self, center_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get a subgraph around a center node."""
        if center_id not in self.nodes:
            return {"nodes": [], "edges": []}

        collected_nodes = {center_id}
        collected_edges = set()
        frontier = {center_id}

        for _ in range(depth):
            new_frontier = set()

            for node_id in frontier:
                for edge_id in self.adjacency.get(node_id, []):
                    edge = self.edges.get(edge_id)
                    if not edge:
                        continue

                    collected_edges.add(edge_id)
                    other_id = edge.target_id if edge.source_id == node_id else edge.source_id

                    if other_id not in collected_nodes:
                        collected_nodes.add(other_id)
                        new_frontier.add(other_id)

            frontier = new_frontier

        return {
            "nodes": [self.nodes[nid].to_dict() for nid in collected_nodes if nid in self.nodes],
            "edges": [self.edges[eid].to_dict() for eid in collected_edges if eid in self.edges]
        }

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its edges."""
        if node_id not in self.nodes:
            return False

        # Remove edges
        edge_ids_to_remove = self.adjacency.get(node_id, []).copy()
        for edge_id in edge_ids_to_remove:
            self.delete_edge(edge_id)

        # Remove node
        del self.nodes[node_id]
        del self.adjacency[node_id]

        if self.persist_path:
            self._save_to_disk()

        return True

    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge."""
        edge = self.edges.get(edge_id)
        if not edge:
            return False

        # Remove from adjacency
        if edge.source_id in self.adjacency:
            self.adjacency[edge.source_id] = [e for e in self.adjacency[edge.source_id] if e != edge_id]
        if edge.target_id in self.adjacency:
            self.adjacency[edge.target_id] = [e for e in self.adjacency[edge.target_id] if e != edge_id]

        del self.edges[edge_id]

        if self.persist_path:
            self._save_to_disk()

        return True

    def decay_memories(self, time_factor: float = 1.0) -> int:
        """Apply decay to memory importance based on time."""
        decayed_count = 0
        now = datetime.now()

        for node in self.nodes.values():
            # Calculate time since last access
            time_diff = (now - node.accessed_at).total_seconds() / 3600  # Hours
            decay = node.decay_rate * time_factor * time_diff / 24  # Daily decay

            old_importance = node.importance
            node.importance = max(0.0, node.importance - decay)

            if node.importance < old_importance:
                decayed_count += 1

        if self.persist_path:
            self._save_to_disk()

        return decayed_count

    def consolidate(self, min_importance: float = 0.1) -> int:
        """Remove low-importance memories."""
        to_remove = [
            node_id for node_id, node in self.nodes.items()
            if node.importance < min_importance
        ]

        for node_id in to_remove:
            self.delete_node(node_id)

        return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes_by_type": {
                nt.value: sum(1 for n in self.nodes.values() if n.node_type == nt)
                for nt in NodeType
            },
            "edges_by_type": {
                rt.value: sum(1 for e in self.edges.values() if e.relation_type == rt)
                for rt in RelationType
            },
            "average_importance": sum(n.importance for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0,
            "most_accessed": max(self.nodes.values(), key=lambda n: n.access_count).node_id if self.nodes else None
        }

    def _save_to_disk(self) -> None:
        """Save memory to disk."""
        if not self.persist_path:
            return

        path = Path(self.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": {eid: edge.to_dict() for eid, edge in self.edges.items()},
            "adjacency": self.adjacency
        }

        with open(path, 'w') as f:
            json.dump(data, f)

    def _load_from_disk(self) -> None:
        """Load memory from disk."""
        if not self.persist_path:
            return

        path = Path(self.persist_path)
        if not path.exists():
            return

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            self.nodes = {
                nid: MemoryNode.from_dict(node_data)
                for nid, node_data in data.get("nodes", {}).items()
            }
            self.edges = {
                eid: MemoryEdge.from_dict(edge_data)
                for eid, edge_data in data.get("edges", {}).items()
            }
            self.adjacency = data.get("adjacency", {})

        except Exception:
            pass
