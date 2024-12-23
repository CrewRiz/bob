import asyncio
from typing import Dict, Any, List, Callable, Optional
import time
import psutil
import numpy as np
from dataclasses import dataclass
from time_utils import TimeUtils
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import cProfile
import pstats
import io

@dataclass
class PerformanceMetrics:
    cpu_usage: float
    memory_usage: float
    execution_time: float
    task_success_rate: float
    throughput: float
    latency: float
    error_rate: float
    resource_efficiency: float

class PerformanceOptimizer:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or psutil.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.performance_history = []
        self.optimization_thresholds = {
            'cpu_threshold': 80.0,  # percentage
            'memory_threshold': 75.0,  # percentage
            'latency_threshold': 1.0,  # seconds
            'error_rate_threshold': 0.05  # 5%
        }
        self.profiler = cProfile.Profile()
        self.current_metrics = None
        self.optimization_strategies = {}

    async def optimize_task(self, task_func: Callable, *args, **kwargs) -> Any:
        """Optimize and execute a task with performance monitoring"""
        start_time = time.time()
        
        # Start profiling
        self.profiler.enable()
        
        try:
            # Get current system metrics
            initial_metrics = self._get_system_metrics()
            
            # Choose optimization strategy
            strategy = self._select_optimization_strategy(initial_metrics)
            
            # Execute task with chosen strategy
            if strategy == 'parallel':
                result = await self._execute_parallel(task_func, *args, **kwargs)
            elif strategy == 'batch':
                result = await self._execute_batch(task_func, *args, **kwargs)
            else:
                result = await self._execute_sequential(task_func, *args, **kwargs)
            
            # Calculate final metrics
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Update performance metrics
            self._update_metrics(execution_time, result)
            
            return result
            
        finally:
            # Stop profiling
            self.profiler.disable()
            self._save_profile_stats()

    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        return {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'memory_available': memory.available,
            'swap_usage': psutil.swap_memory().percent
        }

    def _select_optimization_strategy(self, metrics: Dict[str, float]) -> str:
        """Select the best optimization strategy based on current metrics"""
        if metrics['cpu_usage'] > self.optimization_thresholds['cpu_threshold']:
            return 'batch'  # Batch processing for high CPU usage
        elif metrics['memory_usage'] > self.optimization_thresholds['memory_threshold']:
            return 'sequential'  # Sequential for high memory usage
        else:
            return 'parallel'  # Parallel processing for normal conditions

    async def _execute_parallel(self, func: Callable, *args, **kwargs) -> Any:
        """Execute task using parallel processing"""
        if self._is_cpu_bound(func):
            # Use process pool for CPU-bound tasks
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.process_pool,
                partial(func, *args, **kwargs)
            )
        else:
            # Use thread pool for I/O-bound tasks
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.thread_pool,
                partial(func, *args, **kwargs)
            )

    async def _execute_batch(self, func: Callable, *args, **kwargs) -> Any:
        """Execute task using batch processing"""
        batch_size = self._calculate_optimal_batch_size()
        results = []
        
        # Split task into batches
        batches = self._create_batches(args, batch_size)
        
        for batch in batches:
            # Process each batch
            batch_result = await self._execute_sequential(func, *batch, **kwargs)
            results.append(batch_result)
            
            # Allow system to recover between batches
            await asyncio.sleep(0.1)
            
        return self._combine_results(results)

    async def _execute_sequential(self, func: Callable, *args, **kwargs) -> Any:
        """Execute task sequentially"""
        return func(*args, **kwargs)

    def _is_cpu_bound(self, func: Callable) -> bool:
        """Determine if a function is CPU-bound"""
        # Profile the function with sample data
        self.profiler.enable()
        func(*self._generate_sample_data(func))
        self.profiler.disable()
        
        # Analyze profile stats
        stats = pstats.Stats(self.profiler).sort_stats('cumulative')
        cpu_time = stats.total_tt
        
        return cpu_time > 0.1  # Consider CPU-bound if takes more than 100ms

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on system resources"""
        available_memory = psutil.virtual_memory().available
        cpu_usage = psutil.cpu_percent()
        
        # Base batch size on available resources
        base_size = 1000
        memory_factor = min(1.0, available_memory / (1024 * 1024 * 1024))  # Scale based on GB
        cpu_factor = max(0.1, 1.0 - (cpu_usage / 100.0))
        
        return int(base_size * memory_factor * cpu_factor)

    def _create_batches(self, data: tuple, batch_size: int) -> List[tuple]:
        """Split data into batches"""
        if not data:
            return []
            
        batches = []
        for i in range(0, len(data[0]), batch_size):
            batch = tuple(d[i:i + batch_size] for d in data)
            batches.append(batch)
            
        return batches

    def _combine_results(self, results: List[Any]) -> Any:
        """Combine results from batch processing"""
        if not results:
            return None
            
        if isinstance(results[0], (list, tuple)):
            return type(results[0])(item for result in results for item in result)
        elif isinstance(results[0], dict):
            combined = {}
            for result in results:
                combined.update(result)
            return combined
        else:
            return results[-1]  # Return last result for scalar values

    def _update_metrics(self, execution_time: float, result: Any) -> None:
        """Update performance metrics"""
        metrics = self._get_system_metrics()
        
        self.current_metrics = PerformanceMetrics(
            cpu_usage=metrics['cpu_usage'],
            memory_usage=metrics['memory_usage'],
            execution_time=execution_time,
            task_success_rate=1.0 if result is not None else 0.0,
            throughput=1.0 / execution_time if execution_time > 0 else 0.0,
            latency=execution_time,
            error_rate=0.0 if result is not None else 1.0,
            resource_efficiency=self._calculate_resource_efficiency(metrics)
        )
        
        self.performance_history.append(self.current_metrics)

    def _calculate_resource_efficiency(self, metrics: Dict[str, float]) -> float:
        """Calculate resource efficiency score"""
        cpu_efficiency = 1.0 - (metrics['cpu_usage'] / 100.0)
        memory_efficiency = 1.0 - (metrics['memory_usage'] / 100.0)
        
        return (cpu_efficiency + memory_efficiency) / 2.0

    def _save_profile_stats(self) -> None:
        """Save profiling statistics"""
        s = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        stats.print_stats()
        
        logging.debug(f"Profile stats:\n{s.getvalue()}")

    def _generate_sample_data(self, func: Callable) -> tuple:
        """Generate sample data for function profiling"""
        # Basic sample data generation - extend based on function signature
        return tuple()

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not self.performance_history:
            return {}
            
        metrics = [asdict(m) for m in self.performance_history]
        return {
            'average_cpu_usage': np.mean([m['cpu_usage'] for m in metrics]),
            'average_memory_usage': np.mean([m['memory_usage'] for m in metrics]),
            'average_execution_time': np.mean([m['execution_time'] for m in metrics]),
            'average_throughput': np.mean([m['throughput'] for m in metrics]),
            'average_latency': np.mean([m['latency'] for m in metrics]),
            'error_rate': np.mean([m['error_rate'] for m in metrics]),
            'resource_efficiency': np.mean([m['resource_efficiency'] for m in metrics])
        }

    def cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
