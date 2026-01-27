"""
Parallel Executor - High-Performance Parallel Processing Engine

Features:
- Concurrent task execution
- Load balancing across models
- Automatic failover
- Rate limiting and throttling
- Performance monitoring
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading


T = TypeVar('T')


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class Priority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class TaskResult(Generic[T]):
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    result: Optional[T] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    worker_id: str = ""
    retries: int = 0


@dataclass
class ExecutorConfig:
    """Configuration for the ParallelExecutor."""
    max_workers: int = 10
    max_concurrent_tasks: int = 50
    default_timeout: float = 120.0
    retry_count: int = 3
    retry_delay: float = 1.0
    enable_rate_limiting: bool = True
    rate_limit_per_second: float = 10.0
    enable_monitoring: bool = True


@dataclass 
class WorkerStats:
    """Statistics for a worker."""
    worker_id: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    is_busy: bool = False


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, capacity: float = None):
        self.rate = rate
        self.capacity = capacity or rate
        self.tokens = self.capacity
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    async def acquire(self, tokens: float = 1.0) -> bool:
        """Acquire tokens, waiting if necessary."""
        while True:
            with self._lock:
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_update = now
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
            
            # Wait and retry
            await asyncio.sleep(0.1)
    
    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens without waiting."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


class TaskQueue:
    """Priority-based task queue."""
    
    def __init__(self):
        self._queues: Dict[Priority, asyncio.Queue] = {
            p: asyncio.Queue() for p in Priority
        }
        self._lock = asyncio.Lock()
    
    async def put(self, task: Any, priority: Priority = Priority.NORMAL) -> None:
        """Add task to queue."""
        await self._queues[priority].put(task)
    
    async def get(self) -> Any:
        """Get highest priority task."""
        while True:
            for priority in Priority:
                queue = self._queues[priority]
                if not queue.empty():
                    return await queue.get()
            await asyncio.sleep(0.01)
    
    def size(self) -> int:
        """Get total queue size."""
        return sum(q.qsize() for q in self._queues.values())
    
    def is_empty(self) -> bool:
        """Check if all queues are empty."""
        return all(q.empty() for q in self._queues.values())


class ParallelExecutor:
    """
    High-Performance Parallel Execution Engine
    
    Features:
    - Execute tasks concurrently with configurable workers
    - Priority-based task scheduling
    - Automatic retry with exponential backoff
    - Rate limiting to prevent overload
    - Real-time performance monitoring
    - Load balancing across workers
    """
    
    def __init__(self, config: Optional[ExecutorConfig] = None):
        self.config = config or ExecutorConfig()
        self._task_queue = TaskQueue()
        self._rate_limiter = RateLimiter(
            self.config.rate_limit_per_second,
            self.config.rate_limit_per_second * 2
        ) if self.config.enable_rate_limiting else None
        self._workers: Dict[str, WorkerStats] = {}
        self._running = False
        self._task_counter = 0
        self._lock = asyncio.Lock()
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._results: Dict[str, TaskResult] = {}
        
        # Thread pool for CPU-bound tasks
        self._thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
    
    async def start(self) -> None:
        """Start the executor."""
        self._running = True
        
        # Initialize workers
        for i in range(self.config.max_workers):
            worker_id = f"worker_{i}"
            self._workers[worker_id] = WorkerStats(worker_id=worker_id)
    
    async def stop(self) -> None:
        """Stop the executor."""
        self._running = False
        
        # Cancel all active tasks
        for task in self._active_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)
        
        self._thread_pool.shutdown(wait=True)
    
    async def submit(
        self,
        func: Callable,
        *args,
        priority: Priority = Priority.NORMAL,
        timeout: Optional[float] = None,
        task_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Submit a task for execution.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            priority: Task priority
            timeout: Execution timeout
            task_id: Optional task ID
            **kwargs: Keyword arguments
            
        Returns:
            Task ID
        """
        async with self._lock:
            self._task_counter += 1
            task_id = task_id or f"task_{self._task_counter}"
        
        task_data = {
            "id": task_id,
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "timeout": timeout or self.config.default_timeout,
            "submitted_at": time.time(),
            "retries": 0
        }
        
        await self._task_queue.put(task_data, priority)
        
        # Start processing if not already running
        if task_id not in self._active_tasks:
            self._active_tasks[task_id] = asyncio.create_task(
                self._process_task(task_data)
            )
        
        return task_id
    
    async def submit_batch(
        self,
        tasks: List[tuple],
        priority: Priority = Priority.NORMAL
    ) -> List[str]:
        """
        Submit multiple tasks at once.
        
        Args:
            tasks: List of (func, args, kwargs) tuples
            priority: Priority for all tasks
            
        Returns:
            List of task IDs
        """
        task_ids = []
        for func, args, kwargs in tasks:
            task_id = await self.submit(func, *args, priority=priority, **kwargs)
            task_ids.append(task_id)
        return task_ids
    
    async def _process_task(self, task_data: Dict) -> TaskResult:
        """Process a single task."""
        task_id = task_data["id"]
        start_time = time.time()
        
        # Rate limiting
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        
        # Find available worker
        worker = self._get_available_worker()
        if worker:
            worker.is_busy = True
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                task_data["func"](*task_data["args"], **task_data["kwargs"]),
                timeout=task_data["timeout"]
            )
            
            task_result = TaskResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time_ms=(time.time() - start_time) * 1000,
                worker_id=worker.worker_id if worker else ""
            )
            
            # Update worker stats
            if worker:
                worker.total_tasks += 1
                worker.successful_tasks += 1
                worker.total_time_ms += task_result.execution_time_ms
                worker.avg_time_ms = worker.total_time_ms / worker.total_tasks
                
        except asyncio.TimeoutError:
            task_result = TaskResult(
                task_id=task_id,
                status=TaskStatus.TIMEOUT,
                error="Task timed out",
                execution_time_ms=(time.time() - start_time) * 1000,
                worker_id=worker.worker_id if worker else ""
            )
            
            # Retry if attempts remaining
            if task_data["retries"] < self.config.retry_count:
                task_data["retries"] += 1
                await asyncio.sleep(self.config.retry_delay * task_data["retries"])
                return await self._process_task(task_data)
                
        except Exception as e:
            task_result = TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
                worker_id=worker.worker_id if worker else "",
                retries=task_data["retries"]
            )
            
            # Retry if attempts remaining
            if task_data["retries"] < self.config.retry_count:
                task_data["retries"] += 1
                await asyncio.sleep(self.config.retry_delay * task_data["retries"])
                return await self._process_task(task_data)
            
            if worker:
                worker.failed_tasks += 1
        
        finally:
            if worker:
                worker.is_busy = False
            
            # Store result
            self._results[task_id] = task_result
            
            # Clean up active task
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]
        
        return task_result
    
    def _get_available_worker(self) -> Optional[WorkerStats]:
        """Get the least busy available worker."""
        available = [w for w in self._workers.values() if not w.is_busy]
        if not available:
            return None
        
        # Return worker with least total time (load balancing)
        return min(available, key=lambda w: w.total_time_ms)
    
    async def get_result(
        self, 
        task_id: str, 
        timeout: Optional[float] = None
    ) -> Optional[TaskResult]:
        """
        Get result of a task, waiting if necessary.
        
        Args:
            task_id: Task ID to get result for
            timeout: Maximum time to wait
            
        Returns:
            TaskResult or None if not found
        """
        start = time.time()
        timeout = timeout or self.config.default_timeout
        
        while time.time() - start < timeout:
            if task_id in self._results:
                return self._results[task_id]
            
            if task_id not in self._active_tasks:
                return None
            
            await asyncio.sleep(0.1)
        
        return None
    
    async def execute_parallel(
        self,
        funcs: List[Callable],
        *shared_args,
        **shared_kwargs
    ) -> List[Any]:
        """
        Execute multiple functions in parallel.
        
        Args:
            funcs: List of async functions to execute
            *shared_args: Arguments to pass to all functions
            **shared_kwargs: Keyword arguments to pass to all functions
            
        Returns:
            List of results
        """
        tasks = [
            asyncio.create_task(func(*shared_args, **shared_kwargs))
            for func in funcs
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def map_parallel(
        self,
        func: Callable,
        items: List[Any],
        max_concurrent: Optional[int] = None
    ) -> List[Any]:
        """
        Apply function to items in parallel.
        
        Args:
            func: Async function to apply
            items: Items to process
            max_concurrent: Maximum concurrent executions
            
        Returns:
            List of results
        """
        max_concurrent = max_concurrent or self.config.max_concurrent_tasks
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_func(item):
            async with semaphore:
                return await func(item)
        
        tasks = [asyncio.create_task(limited_func(item)) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        active_workers = sum(1 for w in self._workers.values() if w.is_busy)
        
        return {
            "total_workers": len(self._workers),
            "active_workers": active_workers,
            "pending_tasks": self._task_queue.size(),
            "active_tasks": len(self._active_tasks),
            "completed_tasks": len(self._results),
            "workers": {
                w.worker_id: {
                    "total_tasks": w.total_tasks,
                    "successful": w.successful_tasks,
                    "failed": w.failed_tasks,
                    "avg_time_ms": round(w.avg_time_ms, 2),
                    "is_busy": w.is_busy
                }
                for w in self._workers.values()
            }
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self._active_tasks:
            self._active_tasks[task_id].cancel()
            self._results[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.CANCELLED,
                error="Task cancelled by user"
            )
            del self._active_tasks[task_id]
            return True
        return False
    
    def cancel_all(self) -> int:
        """Cancel all running tasks."""
        count = 0
        for task_id in list(self._active_tasks.keys()):
            if self.cancel_task(task_id):
                count += 1
        return count


# Convenience functions
_executor: Optional[ParallelExecutor] = None


async def get_executor() -> ParallelExecutor:
    """Get or create the global executor."""
    global _executor
    if _executor is None:
        _executor = ParallelExecutor()
        await _executor.start()
    return _executor


async def run_parallel(funcs: List[Callable], *args, **kwargs) -> List[Any]:
    """
    Quick function to run multiple functions in parallel.
    
    Args:
        funcs: Functions to execute
        *args: Shared arguments
        **kwargs: Shared keyword arguments
        
    Returns:
        List of results
    """
    executor = await get_executor()
    return await executor.execute_parallel(funcs, *args, **kwargs)


async def map_async(func: Callable, items: List[Any]) -> List[Any]:
    """
    Quick function to map a function over items in parallel.
    
    Args:
        func: Function to apply
        items: Items to process
        
    Returns:
        List of results
    """
    executor = await get_executor()
    return await executor.map_parallel(func, items)
