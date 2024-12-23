import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
from time_utils import TimeUtils

class KnowledgeNode:
    def __init__(self, content: str, node_type: str, metadata: Dict = None):
        self.content = content
        self.node_type = node_type  # concept, rule, fact, procedure, etc.
        self.metadata = metadata or {}
        self.creation_time = TimeUtils.get_current_time()
        self.last_accessed = self.creation_time
        self.access_count = 0
        self.confidence = 1.0
        self.embedding = None
        self.relationships = []  # List of (node_id, relationship_type, strength)

    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'node_type': self.node_type,
            'metadata': self.metadata,
            'creation_time': TimeUtils.format_timestamp(self.creation_time),
            'last_accessed': TimeUtils.format_timestamp(self.last_accessed),
            'access_count': self.access_count,
            'confidence': self.confidence,
            'relationships': self.relationships
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeNode':
        node = cls(data['content'], data['node_type'], data['metadata'])
        node.creation_time = TimeUtils.parse_timestamp(data['creation_time'])
        node.last_accessed = TimeUtils.parse_timestamp(data['last_accessed'])
        node.access_count = data['access_count']
        node.confidence = data['confidence']
        node.relationships = data['relationships']
        return node

class EnhancedKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_embeddings = {}
        self.relationship_types = {
            'is_a': 0.8,           # hierarchical relationship
            'part_of': 0.7,        # compositional relationship
            'leads_to': 0.6,       # causal relationship
            'similar_to': 0.5,     # similarity relationship
            'depends_on': 0.7,     # dependency relationship
            'contradicts': -0.5,   # contradictory relationship
            'enhances': 0.6        # enhancement relationship
        }

    def add_node(self, node_id: str, node: KnowledgeNode) -> None:
        """Add a node to the knowledge graph"""
        self.graph.add_node(node_id, data=node)
        node.last_accessed = TimeUtils.get_current_time()
        node.access_count += 1

    def add_relationship(self, source_id: str, target_id: str, relationship_type: str, 
                        strength: float = None) -> None:
        """Add a relationship between nodes with automatic strength calculation"""
        if strength is None:
            strength = self.relationship_types.get(relationship_type, 0.5)
        
        self.graph.add_edge(source_id, target_id, 
                           type=relationship_type, 
                           strength=strength,
                           timestamp=TimeUtils.get_current_time())

    def get_node_neighbors(self, node_id: str, relationship_type: Optional[str] = None) -> List[Tuple[str, str, float]]:
        """Get neighboring nodes with optional relationship type filter"""
        neighbors = []
        for _, target, data in self.graph.edges(node_id, data=True):
            if relationship_type is None or data['type'] == relationship_type:
                neighbors.append((target, data['type'], data['strength']))
        return neighbors

    def find_path(self, start_id: str, end_id: str, min_strength: float = 0.0) -> List[str]:
        """Find the strongest path between two nodes"""
        def weight_function(u, v, data):
            return 1 - data['strength']  # Convert strength to distance (higher strength = shorter path)

        try:
            path = nx.shortest_path(self.graph, start_id, end_id, weight=weight_function)
            return path
        except nx.NetworkXNoPath:
            return []

    def get_subgraph(self, node_id: str, depth: int = 2) -> nx.DiGraph:
        """Get a subgraph centered around a node with specified depth"""
        nodes = {node_id}
        for _ in range(depth):
            new_nodes = set()
            for node in nodes:
                new_nodes.update(self.graph.predecessors(node))
                new_nodes.update(self.graph.successors(node))
            nodes.update(new_nodes)
        return self.graph.subgraph(nodes)

    def prune_weak_relationships(self, threshold: float = 0.2) -> None:
        """Remove weak relationships from the graph"""
        edges_to_remove = [(u, v) for u, v, data in self.graph.edges(data=True)
                          if data['strength'] < threshold]
        self.graph.remove_edges_from(edges_to_remove)

    def merge_nodes(self, node_id1: str, node_id2: str, new_node_id: str) -> None:
        """Merge two nodes into a new node, combining their relationships"""
        node1 = self.graph.nodes[node_id1]['data']
        node2 = self.graph.nodes[node_id2]['data']
        
        # Create new merged node
        merged_content = f"{node1.content} + {node2.content}"
        merged_metadata = {**node1.metadata, **node2.metadata}
        merged_node = KnowledgeNode(merged_content, "merged", merged_metadata)
        
        # Add merged node and transfer relationships
        self.add_node(new_node_id, merged_node)
        for pred in self.graph.predecessors(node_id1):
            edge_data = self.graph.edges[pred, node_id1]
            self.add_relationship(pred, new_node_id, edge_data['type'], edge_data['strength'])
        for pred in self.graph.predecessors(node_id2):
            edge_data = self.graph.edges[pred, node_id2]
            self.add_relationship(pred, new_node_id, edge_data['type'], edge_data['strength'])
            
        # Remove old nodes
        self.graph.remove_node(node_id1)
        self.graph.remove_node(node_id2)

    def save_to_file(self, filepath: str) -> None:
        """Save the knowledge graph to a file"""
        data = {
            'nodes': {
                node_id: self.graph.nodes[node_id]['data'].to_dict()
                for node_id in self.graph.nodes
            },
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'type': data['type'],
                    'strength': data['strength'],
                    'timestamp': TimeUtils.format_timestamp(data['timestamp'])
                }
                for u, v, data in self.graph.edges(data=True)
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str) -> None:
        """Load the knowledge graph from a file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.graph.clear()
        
        # Add nodes
        for node_id, node_data in data['nodes'].items():
            node = KnowledgeNode.from_dict(node_data)
            self.add_node(node_id, node)
        
        # Add edges
        for edge in data['edges']:
            self.add_relationship(
                edge['source'],
                edge['target'],
                edge['type'],
                edge['strength']
            )
