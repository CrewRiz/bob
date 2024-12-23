import time
import logging
import asyncio
import numpy as np
import re
from datetime import datetime
from typing import Dict, Any, List
import inspect
import ast
import threading
import copy
import sys
import os
import faiss
import networkx as nx
import json
import subprocess
import pyautogui
from selenium import webdriver
import google.generativeai as genai
import streamlit as st

# Configure logging
logging.basicConfig(filename='learning_system.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Gemini API Key and Model Configuration
GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY'  # Replace with your Gemini API key
GEMINI_MODEL_TEXT = 'gemini-pro'
GEMINI_MODEL_EMBEDDING = 'embedding-001'
GEMINI_MODEL_CHAT = 'gemini-pro'
GENAI_CONFIGURED = False

# Rate Limiting Configuration
API_CALLS_PER_MINUTE = 2
CALL_HISTORY = []

def configure_gemini():
    global GENAI_CONFIGURED
    if not GENAI_CONFIGURED:
        genai.configure(api_key=GEMINI_API_KEY)
        GENAI_CONFIGURED = True

def can_make_api_call():
    global CALL_HISTORY
    now = time.time()
    CALL_HISTORY = [t for t in CALL_HISTORY if now - t < 60]
    return len(CALL_HISTORY) < API_CALLS_PER_MINUTE

def record_api_call():
    global CALL_HISTORY
    CALL_HISTORY.append(time.time())

async def safe_gemini_call(api_call):
    while not can_make_api_call():
        wait_time = 60 - (time.time() - CALL_HISTORY[0]) if CALL_HISTORY else 60
        logging.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds.")
        await asyncio.sleep(max(1, wait_time))  # Wait at least 1 second
    record_api_call()
    return await api_call

# --- UtilityFunction Class ---
class UtilityFunction:
    def __init__(self):
        self.history = []
        self.desired_threshold = 0.8  # Example threshold

    def evaluate(self, system_state) -> float:
        task_success_rate = self.calculate_task_success_rate(system_state)
        resource_efficiency = self.calculate_resource_efficiency(system_state)
        learning_efficiency = self.calculate_learning_efficiency(system_state)
        time_efficiency = self.calculate_time_efficiency(system_state)
        utility = (
            0.4 * task_success_rate +
            0.3 * resource_efficiency +
            0.2 * learning_efficiency +
            0.1 * time_efficiency
        )
        self.history.append(utility)
        return utility

    def calculate_task_success_rate(self, system_state):
        completed_tasks = system_state.metrics.get('completed_tasks', 0)
        total_tasks = system_state.metrics.get('total_tasks', 1)
        return completed_tasks / total_tasks if total_tasks > 0 else 0

    def calculate_resource_efficiency(self, system_state):
        cpu_usage = system_state.metrics.get('cpu_usage', 0.5)
        memory_usage = system_state.metrics.get('memory_usage', 0.5)
        return 1.0 - (cpu_usage + memory_usage) / 2

    def calculate_learning_efficiency(self, system_state):
        new_rules = system_state.metrics.get('new_rules_generated', 0)
        return min(new_rules / 10.0, 1.0)

    def calculate_time_efficiency(self, system_state):
        avg_time = system_state.metrics.get('average_task_time', 1.0)
        return 1.0 / avg_time

# --- Rule Class ---
class Rule:
    def __init__(self, rule_text, condition=None, action=None, confidence=1.0):
        self.rule_text = rule_text
        self.condition = condition
        self.action = action
        self.usage_count = 0
        self.strength = 1.0
        self.embedding = None
        self.confidence = confidence
        self.creation_time = time.time()
        self.last_used = time.time()
        self.connections = []
        self._needs_embedding = True

    async def ensure_embedding(self):
        if self._needs_embedding:
            await self._get_embedding()
            self._needs_embedding = False

    def update_embedding(self, new_embedding):
        self.embedding = 0.9 * self.embedding + 0.1 * new_embedding

    def increment_usage(self):
        self.usage_count += 1
        self.last_used = time.time()
        self.strength *= 1.1

    def decay(self, current_time, decay_rate=0.1):
        time_factor = np.exp(-decay_rate * (current_time - self.last_used))
        self.strength *= time_factor

    async def _get_embedding(self) -> np.array:
        configure_gemini()
        model = genai.GenerativeModel(GEMINI_MODEL_EMBEDDING)
        try:
            response = await safe_gemini_call(model.generate_embeddings(
                contents=[self.rule_text]
            ))
            if response and response.embeddings:
                self.embedding = np.array(response.embeddings[0].values)
                return self.embedding
            else:
                logging.error(f"Embedding API call failed with empty response: {response}")
                return np.zeros(768)  # Default embedding size for embedding-001
        except Exception as e:
            logging.error(f"Embedding API call failed: {e}")
            return np.zeros(768)

    def __str__(self):
        return self.rule_text

# --- KnowledgeNode Class ---
class KnowledgeNode:
    def __init__(self, rule: Rule, strength=1.0):
        self.rule = rule
        self.strength = strength
        self.entangled_nodes = []

    def reinforce(self, amount=0.1):
        self.strength += amount

    def decay(self, amount=0.05):
        self.strength -= amount if self.strength > amount else 0

# --- ComputerInteractionSystem Class ---
class ComputerInteractionSystem:
    def __init__(self, config=None):
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 1.0
        logging.basicConfig(filename='system_actions.log', level=logging.INFO)

        self.config = config or {
            'action_limits': {
                'file_operations_per_minute': 10,
                'mouse_moves_per_minute': 60,
                'web_requests_per_minute': 30
            },
            'unsafe_commands': ['rm -rf', 'format', 'del', 'shutdown'],
            'unsafe_domains': ['malware', 'phishing'],
            'unsafe_paths': ['/system', 'C:\\Windows'],
            'unsafe_content': ['password', 'credit card']
        }

        self.action_history = []
        self.driver = None

    def safe_mouse_move(self, x, y):
        try:
            screen_width, screen_height = pyautogui.size()
            if 0 <= x <= screen_width and 0 <= y <= screen_height:
                pyautogui.moveTo(x, y, duration=0.5)
                self.log_action(f"Mouse moved to {x}, {y}")
            else:
                raise ValueError("Coordinates out of screen bounds")
        except Exception as e:
            self.log_action(f"Mouse move failed: {str(e)}", level="ERROR")

    def click_element(self, image_path=None, coordinates=None):
        try:
            if image_path:
                location = pyautogui.locateOnScreen(image_path)
                if location:
                    pyautogui.click(location)
                    self.log_action(f"Clicked element at {location}")
                else:
                    raise ValueError(f"Element not found: {image_path}")
            elif coordinates:
                self.safe_mouse_move(*coordinates)
                pyautogui.click()
                self.log_action(f"Clicked at coordinates {coordinates}")
        except Exception as e:
            self.log_action(f"Click failed: {str(e)}", level="ERROR")

    def type_text(self, text, interval=0.1):
        try:
            pyautogui.typewrite(text, interval=interval)
            self.log_action(f"Typed text: {text[:20]}...")
        except Exception as e:
            self.log_action(f"Type failed: {str(e)}", level="ERROR")

    def execute_command(self, command):
        try:
            if self.is_safe_command(command):
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                self.log_action(f"Executed command: {command}")
                return result.stdout
            else:
                raise ValueError(f"Unsafe command: {command}")
        except Exception as e:
            self.log_action(f"Command execution failed: {str(e)}", level="ERROR")

    def browse_web(self, url):
        try:
            if self.is_safe_url(url):
                if not self.driver:
                    self.driver = webdriver.Chrome()
                self.driver.get(url)
                self.log_action(f"Browsed to: {url}")
            else:
                raise ValueError(f"Unsafe URL: {url}")
        except Exception as e:
            self.log_action(f"Web browsing failed: {str(e)}", level="ERROR")

    def is_safe_command(self, command):
        return not any(cmd in command.lower() for cmd in self.config['unsafe_commands'])

    def is_safe_url(self, url):
        return not any(domain in url.lower() for domain in self.config['unsafe_domains'])

    def is_safe_path(self, path):
        return not any(p in path for p in self.config['unsafe_paths'])

    def is_safe_content(self, content):
        return not any(c in content.lower() for c in self.config['unsafe_content'])

    def log_action(self, message, level="INFO"):
        self.action_history.append({
            'timestamp': time.time(),
            'action': message,
            'level': level
        })
        if level == "INFO":
            logging.info(message)
        elif level == "ERROR":
            logging.error(message)

    def execute_action(self, action):
        try:
            if action['type'] == 'mouse_move':
                x, y = action['coordinates']
                self.safe_mouse_move(x, y)
                return True
            elif action['type'] == 'click':
                self.click_element(coordinates=action['coordinates'])
                return True
            elif action['type'] == 'type_text':
                self.type_text(action['text'])
                return True
            elif action['type'] == 'execute_command':
                output = self.execute_command(action['command'])
                return True
            else:
                logging.error(f"Unknown action type: {action['type']}")
                return False
        except Exception as e:
            logging.error(f"Failed to execute action {action}: {e}")
            return False

    def cleanup(self):
        if self.driver:
            self.driver.quit()

    def __del__(self):
        self.cleanup()

# --- RAGMemory Class ---
class RAGMemory:
    def __init__(self, dimension=768):  # Adjusted dimension for Gemini embeddings
        self.index = faiss.IndexFlatL2(dimension)
        self.memories = []

    def add_memory(self, memory: Dict):
        embedding = self._get_embedding(memory['content'])
        if embedding is not None:
            self.index.add(np.array([embedding]))
            self.memories.append(memory)

    def get_relevant_context(self, query: str = None, k: int = 5):
        if not query or self.index.ntotal == 0:
            return self.memories[-k:]
        query_embedding = self._get_embedding(query)
        if query_embedding is not None:
            D, I = self.index.search(np.array([query_embedding]), k)
            return [self.memories[i] for i in I[0]]
        return []

    def _get_embedding(self, text: str) -> np.array:
        configure_gemini()
        model = genai.GenerativeModel(GEMINI_MODEL_EMBEDDING)
        try:
            response = asyncio.run(safe_gemini_call(model.generate_embeddings(
                contents=[text]
            )))
            if response and response.embeddings:
                return np.array(response.embeddings[0].values)
            else:
                logging.error(f"Embedding API call failed with empty response: {response}")
                return None
        except Exception as e:
            logging.error(f"Embedding API call failed: {e}")
            return None

# --- Persona Class ---
class Persona:
    def __init__(self):
        self.name = "Alice"
        self.pronouns = ("she", "her")
        self.personality_traits = ["curious", "helpful", "analytical"]
        self.communication_style = "formal"

    def introduce_self(self):
        return f"Hello, I am {self.name}, your enhanced learning assistant."

    def generate_response(self, base_response: str, state):
        response = f"{self.introduce_self()} {base_response}"
        if state.current_task:
            response += f" Currently, I am focused on the task: '{state.current_task}'."
        last_task = state.get_last_task()
        if last_task and last_task != state.current_task:
            response += f" Previously, I worked on '{last_task}'."
        return response

# --- SystemState Class ---
class SystemState:
    def __init__(self):
        self.metrics = {
            'complexity': 0,
            'novelty': 0,
            'confidence': 1.0,
            'resource_usage': {},
            'completed_tasks': 0,
            'total_tasks': 0,
            'cpu_usage': 0.5,
            'memory_usage': 0.5,
            'new_rules_generated': 0,
            'average_task_time': 1.0
        }
        self.active_rules = []
        self.pending_actions = []
        self.knowledge_graph = nx.DiGraph()
        self.current_task = None
        self.past_interactions = []
        self.emotions = {}
        self.operational_state = {}
        self.time_limits = {
            'reasoning': 5.0,
            'self_improvement': 10.0
        }

    def update(self, new_data: Dict):
        self.metrics.update(new_data.get('metrics', {}))
        self.active_rules.extend(new_data.get('new_rules', []))
        self.pending_actions.extend(new_data.get('actions', []))
        if 'current_task' in new_data:
            self.current_task = new_data['current_task']
        if 'emotions' in new_data:
            self.emotions.update(new_data['emotions'])

    def get_summary(self) -> Dict:
        return {
            'metrics': self.metrics,
            'active_rules_count': len(self.active_rules),
            'pending_actions_count': len(self.pending_actions),
            'current_task': self.current_task,
            'emotions': self.emotions
        }

    def get_last_task(self):
        if len(self.past_interactions) > 1:
            return self.past_interactions[-2]['task']
        return None

    def inspect_self(self):
        local_vars = locals()
        global_vars = globals()
        current_module = inspect.getmodule(inspect.currentframe())
        functions = inspect.getmembers(current_module, inspect.isfunction)
        classes = inspect.getmembers(current_module, inspect.isclass)
        self.operational_state = {
            'local_variables': local_vars,
            'global_variables': global_vars,
            'functions': functions,
            'classes': classes
        }

    def update_time_constraints(self):
        if self.metrics['cpu_usage'] > 0.8:
            self.time_limits['reasoning'] = max(1.0, self.time_limits['reasoning'] - 1.0)
        else:
            self.time_limits['reasoning'] = min(10.0, self.time_limits['reasoning'] + 1.0)

# --- PatternDetectionAgent Class ---
class PatternDetectionAgent:
    def __init__(self):
        pass

    async def detect_patterns(self, data: Dict, time_limit: float = 5.0) -> Dict:
        configure_gemini()
        model = genai.GenerativeModel(GEMINI_MODEL_TEXT)
        prompt = f"""Analyze the following data for patterns:
1. Recurring elements
2. Structural similarities
3. Temporal patterns
4. Causal relationships

Data: {data}
"""
        try:
            response = await asyncio.wait_for(
                safe_gemini_call(model.generate_content(
                    prompt
                )),
                timeout=time_limit
            )
            if response.parts:
                return self._parse_pattern_response(response.text)
            else:
                logging.warning("Pattern detection received empty response.")
                return {}
        except asyncio.TimeoutError:
            logging.warning("Pattern detection timed out.")
            return {}

    def _parse_pattern_response(self, content: str) -> Dict:
        patterns = {}
        sections = re.split(r'\n(?=\d+\.)', content.strip())
        for section in sections:
            match = re.match(r'\d+\.\s*(.*)', section)
            if match:
                category = match.group(1).strip().lower().replace(' ', '_')
                details = section.split('\n', 1)[1] if '\n' in section else ''
                patterns[category] = details.strip()
        return patterns

# --- RuleGenerationAgent Class ---
class RuleGenerationAgent:
    def __init__(self):
        pass

    async def generate_rules(self, data: Dict, time_limit: float = 5.0) -> List[Rule]:
        configure_gemini()
        model = genai.GenerativeModel(GEMINI_MODEL_TEXT)
        prompt = f"""Generate formal rules based on these patterns:
Data: {data}

Format:
1. Condition: [when this occurs]
   Action: [system should do this]
   Confidence: [0-1 score]
"""
        try:
            response = await asyncio.wait_for(
                safe_gemini_call(model.generate_content(
                    prompt
                )),
                timeout=time_limit
            )
            if response.parts:
                return self._parse_rules(response.text)
            else:
                logging.warning("Rule generation received empty response.")
                return []
        except asyncio.TimeoutError:
            logging.warning("Rule generation timed out.")
            return []

    def _parse_rules(self, content: str) -> List[Rule]:
        rules = []
        rule_texts = re.split(r'\n(?=\d+\.)', content.strip())
        for rule_text in rule_texts:
            condition_match = re.search(r'Condition:\s*\[(.+?)\]', rule_text)
            action_match = re.search(r'Action:\s*\[(.+?)\]', rule_text)
            confidence_match = re.search(r'Confidence:\s*\[(.+?)\]', rule_text)

            condition = condition_match.group(1).strip() if condition_match else None
            action = action_match.group(1).strip() if action_match else None
            try:
                confidence = float(confidence_match.group(1).strip()) if confidence_match else 1.0
            except ValueError:
                confidence = 1.0

            if condition and action:
                rule_full_text = f"When {condition}, then {action}."
                rule = Rule(rule_text=rule_full_text, condition=condition, action=action, confidence=confidence)
                rules.append(rule)
            else:
                logging.warning(f"Incomplete rule skipped: {rule_text}")
        return rules

# --- AnalysisAgent Class ---
class AnalysisAgent:
    def __init__(self):
        pass

    async def analyze_state(self, state: Dict, time_limit: float = 5.0) -> Dict:
        configure_gemini()
        model = genai.GenerativeModel(GEMINI_MODEL_CHAT)
        prompt = f"""Analyze the current system state:
1. Key metrics evaluation
2. Areas needing improvement
3. Recommended actions
4. Resource requirements

State: {state}
"""
        try:
            response = await asyncio.wait_for(
                safe_gemini_call(model.generate_content(
                    prompt
                )),
                timeout=time_limit
            )
            if response.parts:
                return self._parse_analysis(response.text)
            else:
                logging.warning("State analysis received empty response.")
                return {}
        except asyncio.TimeoutError:
            logging.warning("State analysis timed out.")
            return {}

    def _parse_analysis(self, content: str) -> Dict:
        analysis = {}
        sections = re.split(r'\n(?=\d+\.)', content.strip())
        for section in sections:
            match = re.match(r'\d+\.\s*(.*)', section)
            if match:
                category = match.group(1).strip().lower().replace(' ', '_')
                details = section.split('\n', 1)[1] if '\n' in section else ''
                analysis[category] = details.strip()
        return analysis

# --- NoveltySeekingAlgorithm Class ---
class NoveltySeekingAlgorithm:
    def __init__(self, knowledge_graph=None):
        self.knowledge_graph = knowledge_graph if knowledge_graph else nx.Graph()
        self.meta_rules = ["seek_new_methods", "evaluate_incompleteness"]
        self.rule_generation_agent = None
        self.pattern_detection_agent = None
        self.computer_interaction_system = None
        self.utility_function = None

    async def apply_rules(self, task):
        start_time = time.time()
        time_limit = 5.0
        if await self.assess_incompleteness(task):
            logging.info(f"Incompleteness detected for task '{task}'")

        reasoning_paths = self.apply_advanced_reasoning(task)
        for path in reasoning_paths:
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit:
                logging.warning("Rule application timed out.")
                break
            for node_name in path:
                node_data = self.knowledge_graph.nodes[node_name]['data']
                success = self.execute_rule(node_data.rule)
                if success:
                    node_data.reinforce()
                else:
                    node_data.decay()

    async def assess_incompleteness(self, task):
        relevant_nodes = self.get_relevant_nodes(task)
        if not relevant_nodes:
            await self.seek_novel_information(task)
            return True
        return False

    def get_relevant_nodes(self, task):
        relevant_nodes = []
        for node in self.knowledge_graph.nodes:
            node_data = self.knowledge_graph.nodes[node]['data']
            similarity = self.compute_similarity(task, node_data.rule.rule_text)
            if similarity > 0.7:
                relevant_nodes.append(node)
        return relevant_nodes

    async def seek_novel_information(self, task):
        if self.pattern_detection_agent:
            patterns = await self.pattern_detection_agent.detect_patterns({'task': task})
            if self.rule_generation_agent:
                new_rules = await self.rule_generation_agent.generate_rules(patterns)
                for rule in new_rules:
                    await rule.ensure_embedding()
                    node_name = f"rule_{rule.rule_text}"
                    self.add_node(node_name, rule)
                logging.info(f"Seeking new knowledge: Generated new rules for '{task}'")
        else:
            logging.warning("PatternDetectionAgent or RuleGenerationAgent not set.")

    def add_node(self, name, rule: Rule):
        node = KnowledgeNode(rule)
        self.knowledge_graph.add_node(name, data=node)

    def execute_rule(self, rule: Rule):
        try:
            if rule.action and self.computer_interaction_system:
                result = self.computer_interaction_system.execute_action(rule.action)
                return result
            else:
                return False
        except Exception as e:
            logging.error(f"Failed to execute rule {rule.rule_text}: {e}")
            return False

    def compute_similarity(self, text1, text2):
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        if embedding1 is None or embedding2 is None:
            return 0.0
        epsilon = 1e-8
        denominator = (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)) + epsilon
        similarity = np.dot(embedding1, embedding2) / denominator
        return similarity

    def get_embedding(self, text: str) -> np.array:
        configure_gemini()
        model = genai.GenerativeModel(GEMINI_MODEL_EMBEDDING)
        try:
            response = asyncio.run(safe_gemini_call(model.generate_embeddings(
                contents=[text]
            )))
            if response and response.embeddings:
                return np.array(response.embeddings[0].values)
            else:
                logging.error(f"Embedding API call failed with empty response: {response}")
                return None
        except Exception as e:
            logging.error(f"Embedding API call failed: {e}")
            return None

    def apply_advanced_reasoning(self, task):
        relevant_nodes = self.get_relevant_nodes(task)
        reasoning_paths = []
        for node in relevant_nodes:
            paths = nx.single_source_shortest_path(self.knowledge_graph, node, cutoff=2)
            reasoning_paths.extend(paths.values())
        return reasoning_paths

# --- PersistentStorage Class ---
class PersistentStorage:
    def __init__(self, base_dir='./persistent_memory'):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(f"{base_dir}/rag_memory", exist_ok=True)
        os.makedirs(f"{base_dir}/rules", exist_ok=True)
        os.makedirs(f"{base_dir}/system_state", exist_ok=True)
        os.makedirs(f"{base_dir}/knowledge_graph", exist_ok=True)

    def save_rag_memory(self, memory: RAGMemory):
        faiss.write_index(memory.index, f"{self.base_dir}/rag_memory/index.index")
        with open(f"{self.base_dir}/rag_memory/memories.json", 'w') as f:
            json.dump(memory.memories, f)

    def load_rag_memory(self) -> RAGMemory:
        memory = RAGMemory()
        index_path = f"{self.base_dir}/rag_memory/index.index"
        memories_path = f"{self.base_dir}/rag_memory/memories.json"
        if os.path.exists(index_path):
            memory.index = faiss.read_index(index_path)
        if os.path.exists(memories_path):
            with open(memories_path, 'r') as f:
                memory.memories = json.load(f)
        return memory

    def save_rules(self, rules: List[Rule]):
        rules_data = [{
            'rule_text': rule.rule_text,
            'condition': rule.condition,
            'action': rule.action,
            'usage_count': rule.usage_count,
            'strength': rule.strength,
            'embedding': rule.embedding.tolist() if rule.embedding is not None else None,
            'confidence': rule.confidence,
            'creation_time': rule.creation_time,
            'last_used': rule.last_used,
            'connections': rule.connections
        } for rule in rules]
        with open(f"{self.base_dir}/rules/rules.json", 'w') as f:
            json.dump(rules_data, f)

    def load_rules(self) -> List[Rule]:
        if os.path.exists(f"{self.base_dir}/rules/rules.json"):
            with open(f"{self.base_dir}/rules/rules.json", 'r') as f:
                rules_data = json.load(f)
            rules = []
            for data in rules_data:
                rule = Rule(
                    rule_text=data['rule_text'],
                    condition=data['condition'],
                    action=data['action'],
                    confidence=data['confidence']
                )
                rule.usage_count = data['usage_count']
                rule.strength = data['strength']
                if data.get('embedding') is not None:
                    rule.embedding = np.array(data['embedding'])
                rule.creation_time = data['creation_time']
                rule.last_used = data['last_used']
                rule.connections = data['connections']
                rules.append(rule)
            return rules
        return []

    def save_system_state(self, state: SystemState):
        state_data = {
            'metrics': state.metrics,
            'active_rules': [rule.rule_text for rule in state.active_rules],
            'pending_actions': state.pending_actions,
            'knowledge_graph': nx.node_link_data(state.knowledge_graph),
            'current_task': state.current_task,
            'past_interactions': state.past_interactions,
            'emotions': state.emotions,
            'operational_state': state.operational_state
        }
        with open(f"{self.base_dir}/system_state/state.json", 'w') as f:
            json.dump(state_data, f)

    def load_system_state(self) -> SystemState:
        state = SystemState()
        if os.path.exists(f"{self.base_dir}/system_state/state.json"):
            with open(f"{self.base_dir}/system_state/state.json", 'r') as f:
                state_data = json.load(f)
            state.metrics = state_data['metrics']
            state.pending_actions = state_data['pending_actions']
            state.current_task = state_data.get('current_task')
            state.past_interactions = state_data.get('past_interactions', [])
            state.emotions = state_data.get('emotions', {})
            state.operational_state = state_data.get('operational_state', {})
            # Load knowledge graph
            state.knowledge_graph = nx.node_link_graph(state_data['knowledge_graph'])
        return state

    def save_knowledge_graph(self, knowledge_graph):
        data = nx.node_link_data(knowledge_graph)
        with open(f"{self.base_dir}/knowledge_graph/graph.json", 'w') as f:
            json.dump(data, f)

    def load_knowledge_graph(self):
        path = f"{self.base_dir}/knowledge_graph/graph.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            return nx.node_link_graph(data)
        else:
            return nx.Graph()

# --- WebInteractionAgent Class# --- WebInteractionAgent Class ---
class WebInteractionAgent:
    def __init__(self, system_state):
        self.browser = ComputerInteractionSystem()
        self.system_state = system_state
        self.action_history = []
        self.known_elements = {}

    async def interact_with_webpage(self, task_description):
        try:
            # Basic task interpretation (you can make this more sophisticated)
            if "browse" in task_description.lower():
                url = re.search(r'browse\s+(https?://[^\s]+)', task_description, re.IGNORECASE)
                if url:
                    url = url.group(1)
                    self.browser.browse_web(url)
                    self.action_history.append(f"Browsed to {url}")
                    return True
            if "click" in task_description.lower():
                 element_identifier = re.search(r'click\s+(.+)', task_description, re.IGNORECASE)
                 if element_identifier:
                    element_identifier = element_identifier.group(1)
                    coordinates = self.find_element_coordinates(element_identifier)
                    if coordinates:
                        self.browser.click_element(coordinates=coordinates)
                        self.action_history.append(f"Clicked on element {element_identifier}")
                        return True
                    else:
                        logging.error(f"Could not find element {element_identifier}")
                        return False
            if "type" in task_description.lower():
                type_match = re.search(r'type\s+"([^"]+)"', task_description, re.IGNORECASE)
                if type_match:
                    text_to_type = type_match.group(1)
                    self.browser.type_text(text_to_type)
                    self.action_history.append(f"Typed text: {text_to_type}")
                    return True
            return False

        except Exception as e:
            logging.error(f"Web interaction failed: {e}")
            return False

    def find_element_coordinates(self, element_identifier):
        # Placeholder: actual implementation needs to be more robust using Selenium or image recognition
        # This is a very basic implementation and will likely fail in most real-world scenarios.
         try:
            if isinstance(element_identifier,str):
                logging.info("Attempting to locate on screen with image")
                location = pyautogui.locateOnScreen(element_identifier)
                if location:
                    return pyautogui.center(location)
            logging.info(f"returning placeholder for {element_identifier}")
            return (100,100)
         except Exception as e:
              logging.error(f"element find failed: {e}")
              return None


# --- SelfModifier Class ---
class SelfModifier:
    def __init__(self, utility_function):
        self.utility_function = utility_function

    def should_modify_logic(self, system_state):
        current_utility = self.utility_function.evaluate(system_state)
        if current_utility < self.utility_function.desired_threshold:
            return True
        return False

    def generate_new_code(self, task_description):
        configure_gemini()
        model = genai.GenerativeModel(GEMINI_MODEL_TEXT)
        try:
            response = asyncio.run(safe_gemini_call(model.generate_content(
                f"Write Python code to {task_description}"
            )))
            if response.parts:
                new_code = response.text
                return new_code
            else:
                logging.error("Code generation failed: empty response from Gemini.")
                return ""
        except Exception as e:
            logging.error(f"Code generation failed: {e}")
            return ""

    def is_valid_code(self, code_str):
        try:
            ast.parse(code_str)
            return True
        except SyntaxError as e:
            logging.error(f"Invalid code: {e}")
            return False

    def modify_logic(self, module_name, object_name, new_code):
        exec_globals = {}
        try:
            exec(new_code, exec_globals)
            new_object = exec_globals.get(object_name)
            if new_object:
                setattr(sys.modules[__name__], object_name, new_object) # Modified to use __name__
                logging.info(f"Modified {object_name} in {module_name}")
            else:
                logging.warning(f"{object_name} not found in the new code.")
        except Exception as e:
            logging.error(f"Failed to modify logic: {e}")

# --- EnhancedLearningSystem Class ---
class EnhancedLearningSystem:
    def __init__(self):
        configure_gemini()
        # Initialize storage
        self.storage = PersistentStorage()

        # Load persistent components
        self.system_state = self.storage.load_system_state()
        self.memory = self.storage.load_rag_memory()

        # Load rules
        self.system_state.active_rules = self.storage.load_rules()

        # Initialize agents
        self.web_agent = WebInteractionAgent(self.system_state)
        self.pattern_agent = PatternDetectionAgent()
        self.rule_agent = RuleGenerationAgent()
        self.analysis_agent = AnalysisAgent()

        # Initialize NoveltySeekingAlgorithm
        self.knowledge_graph = self.storage.load_knowledge_graph()
        self.novelty_agent = NoveltySeekingAlgorithm(self.knowledge_graph)
        self.novelty_agent.rule_generation_agent = self.rule_agent
        self.novelty_agent.pattern_detection_agent = self.pattern_agent
        self.novelty_agent.computer_interaction_system = self.web_agent.browser
        self.novelty_agent.utility_function = UtilityFunction()

        # Add the persona
        self.persona = Persona()

        # Initialize utility function
        self.utility_function = UtilityFunction()
        self.self_modifier = SelfModifier(self.utility_function)

    def save_state(self):
        """Save all persistent data"""
        self.storage.save_system_state(self.system_state)
        self.storage.save_rag_memory(self.memory)
        self.storage.save_rules(self.system_state.active_rules)
        self.storage.save_knowledge_graph(self.novelty_agent.knowledge_graph)

    async def process_task(self, task: str) -> Dict:
        try:
            # Add task to memory
            self.memory.add_memory({'content': task})

            # Update current task in system state
            self.system_state.current_task = task
            self.system_state.past_interactions.append({
                'timestamp': time.time(),
                'task': task
            })
            self.system_state.metrics['total_tasks'] += 1
            
            # Interact with the webpage if needed
            if await self.web_agent.interact_with_webpage(task):
                self.system_state.metrics['completed_tasks'] += 1 # consider webpage actions a task completion
            else:
                # Use the novelty agent to apply rules
                await self.novelty_agent.apply_rules(task)

                # Detect patterns in the task
                patterns = await self.pattern_agent.detect_patterns({'task': task}, time_limit=self.system_state.time_limits['reasoning'])

                # Generate new rules based on patterns
                new_rules = await self.rule_agent.generate_rules(patterns, time_limit=self.system_state.time_limits['reasoning'])
                self.system_state.active_rules.extend(new_rules)
                self.system_state.metrics['new_rules_generated'] += len(new_rules)

                # Analyze system state
                analysis = await self.analysis_agent.analyze_state(self.system_state.get_summary(), time_limit=self.system_state.time_limits['reasoning'])

                # Update system metrics
                self.system_state.update({'metrics': analysis.get('key_metrics', {})})
                
            # Save state after processing
            self.save_state()

            # Base response from Alice's processing
            base_response = f"I have processed the task '{task}'."
            response = self.persona.generate_response(base_response, self.system_state)


            # Update average task time
            task_time = time.time() - self.system_state.past_interactions[-1]['timestamp']
            previous_avg_time = self.system_state.metrics.get('average_task_time', 1.0)
            self.system_state.metrics['average_task_time'] = (previous_avg_time + task_time) / 2

            # Consider self-improvement
            await self.self_improve()

            return {
                'status': 'success',
                'response': response,
                'system_state': self.system_state.get_summary()
            }

        except Exception as e:
            logging.error(f"Task processing failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    async def self_improve(self):
        if self.self_modifier.should_modify_logic(self.system_state):
            # Generate new code
            task_description = "optimize the rule execution process to be faster"
            new_code = self.self_modifier.generate_new_code(task_description)
            if self.self_modifier.is_valid_code(new_code):
                # Verify the modification with a time limit
                verifier = FormalVerifier(self.utility_function)
                if verifier.verify_modification(new_code, self.system_state, time_limit=5.0):
                    # Apply modification with a time limit
                    await asyncio.wait_for(
                        self.apply_modification(new_code),
                        timeout=2.0
                    )
                    logging.info("Self-improvement applied successfully.")
                else:
                    logging.warning("Modification rejected during verification.")
            else:
                logging.error("Generated code is invalid.")
        else:
            logging.info("No modification necessary at this time.")

    async def apply_modification(self, code_str):
        # Apply the code modification
        self.self_modifier.modify_logic('NoveltySeekingAlgorithm', 'NoveltySeekingAlgorithm', code_str)

# --- Streamlit Interface ---
def create_interface():
    st.title("Alice - Enhanced Learning System Interface")

    # Initialize system
    if 'system' not in st.session_state:
        st.session_state.system = EnhancedLearningSystem()

    # Task input
    st.header("Task Input")
    task = st.text_area("Enter task description:")

    if st.button("Process Task"):
        if task:
            with st.spinner("Processing task..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(st.session_state.system.process_task(task))
                loop.close()

                # Display Alice's response
                st.header("Alice's Response")
                st.write(results.get('response', 'No response'))

                # Display system state
                st.subheader("System State")
                st.json(results.get('system_state', {}))
        else:
            st.warning("Please enter a task description.")

    # System metrics
    st.header("System Metrics")
    metrics = st.session_state.system.system_state.get_summary()
    st.json(metrics)

# --- Main Execution ---
if __name__ == "__main__":
    # Run interface
    create_interface()

# --- FormalVerifier Class ---
class FormalVerifier:
    def __init__(self, utility_function: UtilityFunction):
        self.utility_function = utility_function

    def verify_modification(self, modification_code: str, system_state: SystemState, time_limit: float) -> bool:
        start_time = time.time()
        simulated_state = self.simulate_modification(modification_code, system_state)
        expected_utility = self.utility_function.evaluate(simulated_state)
        current_utility = self.utility_function.evaluate(system_state)
        if time.time() - start_time > time_limit:
            logging.warning("Verification timed out.")
            return False
        return expected_utility > current_utility

    def simulate_modification(self, code_str: str, system_state: SystemState) -> SystemState:
        simulated_state = copy.deepcopy(system_state)
        exec_globals = globals()  # Use globals to access existing classes and functions
        exec_locals = {'system_state': simulated_state}
        try:
            compiled_code = compile(code_str, '<string>', 'exec')
            exec(compiled_code, exec_globals, exec_locals)
            return simulated_state
        except Exception as e:
            logging.error(f"Simulation failed: {e}")
            return system_state