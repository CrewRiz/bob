from typing import Dict, List, Any, Optional
from knowledge_management import EnhancedKnowledgeGraph, KnowledgeNode
from time_utils import TimeUtils
import numpy as np
from collections import defaultdict
import asyncio
import logging

class LearningManager:
    def __init__(self, knowledge_graph: EnhancedKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.learning_history = []
        self.learning_stats = defaultdict(float)
        self.current_focus = None
        self.learning_thresholds = {
            'confidence': 0.7,
            'relevance': 0.6,
            'novelty': 0.3
        }

    async def learn_from_interaction(self, interaction_data: Dict[str, Any]) -> None:
        """Process and learn from new interaction data"""
        # Extract key information
        content = interaction_data.get('content', '')
        context = interaction_data.get('context', {})
        timestamp = TimeUtils.get_current_time()

        # Create knowledge nodes
        main_node = KnowledgeNode(
            content=content,
            node_type='interaction',
            metadata={'context': context, 'source': 'interaction'}
        )
        node_id = f"interaction_{timestamp.timestamp()}"
        
        # Add to knowledge graph
        self.knowledge_graph.add_node(node_id, main_node)
        
        # Find related nodes
        related_nodes = await self._find_related_nodes(content)
        
        # Create relationships
        for related_id, similarity in related_nodes:
            if similarity > self.learning_thresholds['relevance']:
                self.knowledge_graph.add_relationship(
                    node_id, 
                    related_id, 
                    'similar_to',
                    similarity
                )

        # Update learning history
        self.learning_history.append({
            'timestamp': timestamp,
            'node_id': node_id,
            'type': 'interaction',
            'related_nodes': len(related_nodes)
        })

    async def _find_related_nodes(self, content: str) -> List[tuple]:
        """Find nodes related to the given content"""
        # This is a placeholder for more sophisticated similarity calculation
        related = []
        for node_id in self.knowledge_graph.graph.nodes():
            node = self.knowledge_graph.graph.nodes[node_id]['data']
            # Simple string similarity for demonstration
            similarity = self._calculate_similarity(content, node.content)
            if similarity > self.learning_thresholds['relevance']:
                related.append((node_id, similarity))
        return sorted(related, key=lambda x: x[1], reverse=True)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # This is a simple implementation - replace with more sophisticated method
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0

    async def consolidate_knowledge(self) -> None:
        """Consolidate and optimize the knowledge graph"""
        timestamp = TimeUtils.get_current_time()
        
        # Prune weak relationships
        self.knowledge_graph.prune_weak_relationships(
            threshold=self.learning_thresholds['relevance']
        )

        # Find and merge similar nodes
        nodes = list(self.knowledge_graph.graph.nodes())
        for i, node_id1 in enumerate(nodes[:-1]):
            for node_id2 in nodes[i+1:]:
                node1 = self.knowledge_graph.graph.nodes[node_id1]['data']
                node2 = self.knowledge_graph.graph.nodes[node_id2]['data']
                
                similarity = self._calculate_similarity(node1.content, node2.content)
                if similarity > 0.9:  # Very high similarity threshold for merging
                    new_node_id = f"merged_{timestamp.timestamp()}"
                    self.knowledge_graph.merge_nodes(node_id1, node_id2, new_node_id)

    async def generate_insights(self) -> List[Dict[str, Any]]:
        """Generate insights from the knowledge graph"""
        insights = []
        
        # Find highly connected nodes (hubs of knowledge)
        for node_id in self.knowledge_graph.graph.nodes():
            neighbors = self.knowledge_graph.get_node_neighbors(node_id)
            if len(neighbors) > 5:  # Arbitrary threshold
                node = self.knowledge_graph.graph.nodes[node_id]['data']
                insights.append({
                    'type': 'hub',
                    'node_id': node_id,
                    'content': node.content,
                    'connections': len(neighbors)
                })

        # Find strong relationship patterns
        for u, v, data in self.knowledge_graph.graph.edges(data=True):
            if data['strength'] > 0.8:  # Strong relationship threshold
                insights.append({
                    'type': 'strong_relationship',
                    'source': u,
                    'target': v,
                    'relationship_type': data['type'],
                    'strength': data['strength']
                })

        return insights

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning process"""
        return {
            'total_nodes': len(self.knowledge_graph.graph.nodes()),
            'total_relationships': len(self.knowledge_graph.graph.edges()),
            'learning_events': len(self.learning_history),
            'average_node_connections': np.mean([
                len(list(self.knowledge_graph.graph.neighbors(node)))
                for node in self.knowledge_graph.graph.nodes()
            ]) if self.knowledge_graph.graph.nodes() else 0,
            'last_consolidation': self.learning_stats['last_consolidation'],
            'knowledge_density': len(self.knowledge_graph.graph.edges()) / 
                               len(self.knowledge_graph.graph.nodes()) 
                               if len(self.knowledge_graph.graph.nodes()) > 0 else 0
        }

    async def adaptive_learning_rate(self) -> float:
        """Calculate adaptive learning rate based on current state"""
        stats = self.get_learning_statistics()
        base_rate = 0.1
        
        # Adjust based on knowledge density
        density_factor = min(stats['knowledge_density'] / 5, 1.0)  # Cap at 1.0
        
        # Adjust based on recent learning events
        recent_events = len([
            event for event in self.learning_history
            if TimeUtils.get_time_diff(
                TimeUtils.get_current_time(),
                event['timestamp']
            ) < 3600  # Last hour
        ])
        recency_factor = np.exp(-recent_events / 10)  # Decay with more recent events
        
        return base_rate * density_factor * recency_factor
