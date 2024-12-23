from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import deque
from time_utils import TimeUtils
import json
import logging
from dataclasses import dataclass, asdict
import asyncio

@dataclass
class MemorySegment:
    content: Any
    importance: float
    timestamp: str
    context: Dict
    access_count: int = 0
    last_accessed: Optional[str] = None
    decay_rate: float = 0.1
    tags: List[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemorySegment':
        return cls(**data)

class MemoryManager:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.short_term_memory = deque(maxlen=100)
        self.working_memory = {}
        self.long_term_memory = {}
        self.memory_index = {}
        self.importance_threshold = 0.5
        self.consolidation_threshold = 0.8
        self.memory_stats = {
            'total_memories': 0,
            'consolidated_memories': 0,
            'forgotten_memories': 0
        }

    async def add_memory(self, content: Any, context: Dict, importance: float = 0.5) -> str:
        """Add a new memory segment with automatic importance assessment"""
        timestamp = TimeUtils.format_timestamp(TimeUtils.get_current_time())
        
        # Create memory segment
        memory = MemorySegment(
            content=content,
            importance=importance,
            timestamp=timestamp,
            context=context,
            tags=self._generate_tags(content, context)
        )

        # Add to short-term memory first
        self.short_term_memory.append(memory)
        
        # Generate memory ID
        memory_id = f"mem_{timestamp}"
        self.memory_index[memory_id] = 'short_term'
        
        # Update stats
        self.memory_stats['total_memories'] += 1
        
        # Trigger memory consolidation if needed
        if len(self.short_term_memory) >= self.short_term_memory.maxlen * 0.8:
            await self.consolidate_memories()
            
        return memory_id

    async def consolidate_memories(self) -> None:
        """Consolidate short-term memories into long-term storage"""
        consolidation_candidates = []
        
        # Analyze short-term memories
        for memory in self.short_term_memory:
            importance = self._calculate_importance(memory)
            if importance >= self.consolidation_threshold:
                consolidation_candidates.append(memory)
                
        # Move important memories to long-term storage
        for memory in consolidation_candidates:
            memory_id = f"ltm_{TimeUtils.format_timestamp(TimeUtils.get_current_time())}"
            self.long_term_memory[memory_id] = memory
            self.memory_index[memory_id] = 'long_term'
            self.memory_stats['consolidated_memories'] += 1
            
        # Clear consolidated memories from short-term
        self.short_term_memory = deque(
            [m for m in self.short_term_memory if m not in consolidation_candidates],
            maxlen=self.short_term_memory.maxlen
        )

    def _calculate_importance(self, memory: MemorySegment) -> float:
        """Calculate memory importance based on various factors"""
        base_importance = memory.importance
        
        # Factor in access frequency
        access_factor = min(memory.access_count / 10.0, 1.0)
        
        # Factor in recency
        if memory.last_accessed:
            time_diff = TimeUtils.get_time_diff(
                TimeUtils.get_current_time(),
                TimeUtils.parse_timestamp(memory.last_accessed)
            )
            recency_factor = np.exp(-time_diff / 86400)  # Decay over 24 hours
        else:
            recency_factor = 1.0
            
        # Factor in context richness
        context_factor = min(len(memory.context) / 5.0, 1.0)
        
        return (base_importance * 0.4 + 
                access_factor * 0.3 + 
                recency_factor * 0.2 + 
                context_factor * 0.1)

    async def retrieve_memory(self, query: Dict[str, Any], limit: int = 5) -> List[Tuple[str, MemorySegment]]:
        """Retrieve relevant memories based on query parameters"""
        relevant_memories = []
        
        # Search in both short-term and long-term memories
        all_memories = list(self.short_term_memory) + list(self.long_term_memory.values())
        
        for memory in all_memories:
            relevance = self._calculate_relevance(memory, query)
            if relevance > self.importance_threshold:
                # Update access statistics
                memory.access_count += 1
                memory.last_accessed = TimeUtils.format_timestamp(TimeUtils.get_current_time())
                relevant_memories.append((relevance, memory))
        
        # Sort by relevance and return top matches
        relevant_memories.sort(reverse=True, key=lambda x: x[0])
        return relevant_memories[:limit]

    def _calculate_relevance(self, memory: MemorySegment, query: Dict[str, Any]) -> float:
        """Calculate relevance of a memory to the query"""
        relevance_score = 0.0
        
        # Check content similarity
        if 'content' in query:
            content_similarity = self._text_similarity(
                str(memory.content), 
                str(query['content'])
            )
            relevance_score += content_similarity * 0.4
            
        # Check context match
        if 'context' in query:
            context_match = sum(
                memory.context.get(k) == v 
                for k, v in query['context'].items()
            ) / len(query['context']) if query['context'] else 0
            relevance_score += context_match * 0.3
            
        # Check tag match
        if 'tags' in query and memory.tags:
            tag_match = len(set(query['tags']) & set(memory.tags)) / len(query['tags'])
            relevance_score += tag_match * 0.3
            
        return relevance_score

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0

    def _generate_tags(self, content: Any, context: Dict) -> List[str]:
        """Generate relevant tags for a memory segment"""
        tags = []
        
        # Add context-based tags
        tags.extend(context.keys())
        
        # Add content-based tags (if content is string)
        if isinstance(content, str):
            # Add significant words as tags
            words = content.lower().split()
            tags.extend([w for w in words if len(w) > 4])  # Simple but effective
            
        return list(set(tags))  # Remove duplicates

    async def forget_old_memories(self) -> None:
        """Remove old, unimportant memories to free up space"""
        current_time = TimeUtils.get_current_time()
        
        # Analyze long-term memories
        memories_to_forget = []
        for memory_id, memory in self.long_term_memory.items():
            # Calculate current importance
            importance = self._calculate_importance(memory)
            
            # Check if memory should be forgotten
            if importance < self.importance_threshold:
                memories_to_forget.append(memory_id)
                
        # Remove forgotten memories
        for memory_id in memories_to_forget:
            del self.long_term_memory[memory_id]
            del self.memory_index[memory_id]
            self.memory_stats['forgotten_memories'] += 1

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        return {
            **self.memory_stats,
            'short_term_count': len(self.short_term_memory),
            'working_memory_count': len(self.working_memory),
            'long_term_count': len(self.long_term_memory),
            'total_active_memories': (
                len(self.short_term_memory) + 
                len(self.working_memory) + 
                len(self.long_term_memory)
            )
        }

    def save_state(self, filepath: str) -> None:
        """Save memory state to file"""
        state = {
            'short_term': [m.to_dict() for m in self.short_term_memory],
            'working_memory': {k: v.to_dict() for k, v in self.working_memory.items()},
            'long_term': {k: v.to_dict() for k, v in self.long_term_memory.items()},
            'memory_index': self.memory_index,
            'stats': self.memory_stats
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str) -> None:
        """Load memory state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        # Restore memories
        self.short_term_memory = deque(
            [MemorySegment.from_dict(m) for m in state['short_term']],
            maxlen=100
        )
        self.working_memory = {
            k: MemorySegment.from_dict(v) 
            for k, v in state['working_memory'].items()
        }
        self.long_term_memory = {
            k: MemorySegment.from_dict(v) 
            for k, v in state['long_term'].items()
        }
        self.memory_index = state['memory_index']
        self.memory_stats = state['stats']
